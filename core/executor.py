"""
Layer 4 — Execution System for nvagent.

Components
──────────
Sandbox          Path/command safety validator (blocks writes outside workspace,
                 dangerous shell patterns, respects safe_mode flag).

CommandResult    Structured output from any shell command: exit_code, stdout,
                 stderr, duration, timed_out, and helpers to extract errors /
                 format for the agent.

TestCase         A single test result (name, status, error_message, duration).
TestSuiteResult  Aggregate: framework, pass/fail/skip/error counts, failing
                 test list.  Produced by parse_*_output() and the unified
                 parse_test_output() dispatcher.

Framework & formatter detection:
  detect_test_framework(workspace)  →  str | None
  build_test_command(framework, path, extra_args)  →  list[str]
  detect_formatters(workspace)  →  list[str]
  build_formatter_command(formatter, path)  →  list[str]

These are all pure (no I/O) so they are trivially testable.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox
# ─────────────────────────────────────────────────────────────────────────────

# Shell command patterns that are unconditionally blocked
_BLOCKED_CMD_PATTERNS: list[str] = [
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f\s+/",        # rm -rf /
    r"sudo\s+rm",                               # sudo rm anything
    r"\bmkfs\b",                                # format filesystem
    r"\bdd\b.*\bif=",                           # dd if= (disk overwrite)
    r":\(\)\s*\{.*\}",                          # fork bomb
    r"chmod\s+-[rR]\s+777\s+/",                # chmod 777 everything
    r">\s*/etc/passwd",                         # clobber system files
    r">\s*/etc/shadow",
    r"curl\s+.*\|\s*(ba)?sh",                  # curl | bash
    r"wget\s+.*\|\s*(ba)?sh",                  # wget | bash
    r"\|\s*(ba)?sh\s*<",                        # pipe to shell
    r"nc\s+-[a-zA-Z]*e\b",                     # netcat exec
]


class Sandbox:
    """
    Validates file paths and shell commands for safety.

    • Path validation: all absolute paths must be under *workspace*.
    • Command validation: blocks dangerous shell patterns.
    • Both checks are bypassed when safe_mode=False.
    """

    def __init__(self, workspace: Path, safe_mode: bool = True) -> None:
        self.workspace = workspace.resolve()
        self.safe_mode = safe_mode
        self._blocked: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in _BLOCKED_CMD_PATTERNS
        ]

    def validate_path(self, path: Path) -> tuple[bool, str]:
        """Returns (ok, reason).  Path must be inside workspace."""
        if not self.safe_mode:
            return True, ""
        try:
            path.resolve().relative_to(self.workspace)
            return True, ""
        except ValueError:
            return False, (
                f"Path '{path}' is outside the workspace '{self.workspace}'. "
                "Refusing to modify it."
            )

    def validate_command(self, command: str) -> tuple[bool, str]:
        """Returns (ok, reason).  Blocks shell patterns considered dangerous."""
        if not self.safe_mode:
            return True, ""
        for p in self._blocked:
            if p.search(command):
                return False, (
                    f"Command blocked by safe_mode — matches dangerous pattern. "
                    f"Offending segment: {p.pattern!r}"
                )
        return True, ""

    def describe(self) -> str:
        return (
            f"Sandbox(workspace={self.workspace}, safe_mode={self.safe_mode})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CommandResult
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that tag a line as error-bearing when scanning combined output
_ERROR_KEYWORDS = frozenset(
    ["error", "exception", "traceback", "fatal", "failed", "failure",
     "assert", "panic", "aborted", "segfault", "killed"]
)
_WARN_KEYWORDS = frozenset(["warning", "warn", "deprecated"])


@dataclass
class CommandResult:
    """Structured result from a shell command execution."""

    command:    str
    exit_code:  int
    stdout:     str
    stderr:     str
    duration_s: float
    timed_out:  bool = False

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout)
        if self.stderr.strip():
            parts.append(self.stderr)
        return "\n".join(parts)

    # ── Error extraction ───────────────────────────────────────────────────

    def extract_errors(self, max_lines: int = 30) -> list[str]:
        """
        Return lines from combined output that look like errors or tracebacks.
        Preserves context: if a 'Traceback' header is found the next 10 lines
        are kept too.
        """
        lines = self.combined_output.splitlines()
        result: list[str] = []
        capture_remaining = 0

        for line in lines:
            ll = line.lower()
            is_error = any(kw in ll for kw in _ERROR_KEYWORDS)

            if is_error or capture_remaining > 0:
                result.append(line)
                if capture_remaining > 0:
                    capture_remaining -= 1
                # Start a traceback capture window
                if "traceback" in ll or "most recent call" in ll:
                    capture_remaining = 10

            if len(result) >= max_lines:
                result.append(f"  … ({len(lines) - len(result)} more lines)")
                break

        return result

    def extract_warnings(self) -> list[str]:
        lines = self.combined_output.splitlines()
        return [l for l in lines if any(kw in l.lower() for kw in _WARN_KEYWORDS)]

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_agent_str(self, max_chars: int = 8000) -> str:
        """
        Render for injection into the agent conversation.  Truncates long
        output while preserving the most actionable parts (errors sink to top).
        """
        if self.timed_out:
            return (
                f"$ {self.command}\n"
                f"⏱ TIMED OUT after {self.duration_s:.1f}s\n"
                f"{self.stderr[:500]}"
            )

        status = "✓" if self.success else "✗"
        parts = [
            f"$ {self.command}",
            f"{status} exit={self.exit_code}  ({self.duration_s:.2f}s)",
        ]

        # Error lines get precedence
        errors = self.extract_errors()
        if errors and not self.success:
            parts.append("Errors / tracebacks:")
            parts.extend(f"  {e}" for e in errors)

        # Full combined output (truncated)
        combined = self.combined_output
        if combined:
            if len(combined) > max_chars:
                combined = combined[:max_chars] + f"\n… [{len(combined) - max_chars:,} chars truncated]"
            parts.append(combined)

        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Test result model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name:          str
    status:        str          # "passed" | "failed" | "error" | "skipped"
    file:          str = ""
    duration_s:    float = 0.0
    error_message: str = ""     # failure / error body

    def render(self) -> str:
        icon = {"passed": "✓", "failed": "✗", "error": "!", "skipped": "⊘"}.get(self.status, "?")
        loc  = f"  [{self.file}]" if self.file else ""
        base = f"  {icon} {self.name}{loc}"
        if self.error_message:
            excerpt = self.error_message[:200].replace("\n", "\n      ")
            base += f"\n      {excerpt}"
        return base


@dataclass
class TestSuiteResult:
    """Aggregate result from a test run."""

    framework:    str
    passed:       int = 0
    failed:       int = 0
    errors:       int = 0
    skipped:      int = 0
    duration_s:   float = 0.0
    failing_tests: list[TestCase] = field(default_factory=list)
    raw_output:   str = ""

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0

    def summary_line(self) -> str:
        parts = []
        if self.passed:
            parts.append(f"{self.passed} passed")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} error{'s' if self.errors != 1 else ''}")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        total_str = f"  ({self.total} total)" if self.total > 0 else ""
        time_str  = f"  in {self.duration_s:.2f}s" if self.duration_s > 0 else ""
        icon = "✓" if self.success else "✗"
        return f"{icon} [{self.framework}]  {',  '.join(parts) or 'no tests ran'}{total_str}{time_str}"

    def to_agent_str(self) -> str:
        lines = [self.summary_line()]
        if self.failing_tests:
            lines.append("\nFailing tests:")
            for tc in self.failing_tests[:20]:
                lines.append(tc.render())
            if len(self.failing_tests) > 20:
                lines.append(f"  … {len(self.failing_tests) - 20} more failures")
        if not self.failing_tests and not self.success:
            # No structured failures but run failed — show raw tail
            tail = "\n".join(self.raw_output.splitlines()[-40:])
            lines.append(f"\nOutput tail:\n{tail}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Test output parsers
# ─────────────────────────────────────────────────────────────────────────────

# ── pytest ────────────────────────────────────────────────────────────────────

# Match each count independently (pytest order varies: "3 failed, 12 passed in 0.42s")
_PY_PASSED  = re.compile(r"(\d+)\s+passed")
_PY_FAILED  = re.compile(r"(\d+)\s+failed")
_PY_ERRORS  = re.compile(r"(\d+)\s+error")
_PY_SKIPPED = re.compile(r"(\d+)\s+skipped")
_PY_DURATION= re.compile(r"in\s+([\d.]+)s")
_PY_FAIL_HEAD = re.compile(r"^FAILED\s+(.+?)(?:\s+-\s+(.+))?$", re.MULTILINE)
_PY_SHORT_TEST= re.compile(r"^(.+?)::\S+\s+(PASSED|FAILED|ERROR|SKIPPED)", re.MULTILINE)

def parse_pytest_output(output: str) -> TestSuiteResult:
    result = TestSuiteResult(framework="pytest", raw_output=output)

    m_p = _PY_PASSED.search(output)
    m_f = _PY_FAILED.search(output)
    m_e = _PY_ERRORS.search(output)
    m_s = _PY_SKIPPED.search(output)
    m_d = _PY_DURATION.search(output)

    if m_p:
        result.passed   = int(m_p.group(1))
    if m_f:
        result.failed   = int(m_f.group(1))
    if m_e:
        result.errors   = int(m_e.group(1))
    if m_s:
        result.skipped  = int(m_s.group(1))
    if m_d:
        result.duration_s = float(m_d.group(1))

    # Collect FAILED lines
    for m2 in _PY_FAIL_HEAD.finditer(output):
        name = m2.group(1).strip()
        msg  = (m2.group(2) or "").strip()
        result.failing_tests.append(TestCase(name=name, status="failed", error_message=msg))

    # If FAILED lines not found, fall back to short-form
    if not result.failing_tests:
        for m3 in _PY_SHORT_TEST.finditer(output):
            status_str = m3.group(2).lower()
            if status_str in ("failed", "error"):
                name = m3.group(1).strip()
                result.failing_tests.append(TestCase(name=name, status=status_str))

    # Extract error bodies for each failing test (between FAIL: header and next FAIL:/=== separator)
    if result.failing_tests:
        sections = re.split(r"_{3,}.*?_{3,}", output)
        err_map: dict[str, str] = {}
        for section in sections:
            # Try to match a test name in the section header
            header = section.splitlines()[0] if section.strip() else ""
            for tc in result.failing_tests:
                short = tc.name.split("::")[-1]
                if short and short in header:
                    # Grab relevant error lines
                    err_lines = [
                        l for l in section.splitlines()
                        if any(kw in l.lower() for kw in ("assertionerror", "error:", "assert ", "e   "))
                    ]
                    if err_lines and not tc.error_message:
                        tc.error_message = "\n".join(err_lines[:8])

    return result


# ── jest / vitest ─────────────────────────────────────────────────────────────

_JEST_SUMMARY = re.compile(
    r"Tests?:\s+(?:(\d+) failed[,\s]*)?(?:(\d+) passed[,\s]*)?(?:(\d+) skipped[,\s]*)?(\d+) total"
)
_JEST_FAIL    = re.compile(r"^\s+●\s+(.+)$", re.MULTILINE)
_JEST_TIME    = re.compile(r"Time:\s+([\d.]+)\s*s")

def parse_jest_output(output: str) -> TestSuiteResult:
    result = TestSuiteResult(framework="jest", raw_output=output)
    m = _JEST_SUMMARY.search(output)
    if m:
        result.failed  = int(m.group(1) or 0)
        result.passed  = int(m.group(2) or 0)
        result.skipped = int(m.group(3) or 0)
        total          = int(m.group(4) or 0)
        result.errors  = max(0, total - result.passed - result.failed - result.skipped)
    mt = _JEST_TIME.search(output)
    if mt:
        result.duration_s = float(mt.group(1))
    for m2 in _JEST_FAIL.finditer(output):
        name = m2.group(1).strip()
        if name:
            result.failing_tests.append(TestCase(name=name, status="failed"))
    return result


# ── cargo test ────────────────────────────────────────────────────────────────

_CARGO_SUMMARY = re.compile(r"test result:\s+(ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored.*?([\d.]+)s")
_CARGO_FAIL    = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED$", re.MULTILINE)

def parse_cargo_output(output: str) -> TestSuiteResult:
    result = TestSuiteResult(framework="cargo-test", raw_output=output)
    m = _CARGO_SUMMARY.search(output)
    if m:
        result.passed     = int(m.group(2))
        result.failed     = int(m.group(3))
        result.skipped    = int(m.group(4))
        result.duration_s = float(m.group(5))
    for m2 in _CARGO_FAIL.finditer(output):
        result.failing_tests.append(TestCase(name=m2.group(1), status="failed"))
    return result


# ── go test ───────────────────────────────────────────────────────────────────

_GO_PASS  = re.compile(r"^--- PASS:\s+(\S+)\s+\(([\d.]+)s\)", re.MULTILINE)
_GO_FAIL  = re.compile(r"^--- FAIL:\s+(\S+)\s+\(([\d.]+)s\)", re.MULTILINE)
_GO_FINAL = re.compile(r"^(ok|FAIL)\s+\S+\s+([\d.]+)s", re.MULTILINE)

def parse_go_test_output(output: str) -> TestSuiteResult:
    result = TestSuiteResult(framework="go-test", raw_output=output)
    for m in _GO_PASS.finditer(output):
        result.passed += 1
    for m in _GO_FAIL.finditer(output):
        result.failed += 1
        result.failing_tests.append(
            TestCase(name=m.group(1), status="failed", duration_s=float(m.group(2)))
        )
    mf = _GO_FINAL.search(output)
    if mf:
        result.duration_s = float(mf.group(2))
    return result


# ── vitest specific ───────────────────────────────────────────────────────────

_VITEST_SUMMARY = re.compile(r"Tests\s+(\d+) failed\s+\|\s+(\d+) passed")

def parse_vitest_output(output: str) -> TestSuiteResult:
    result = parse_jest_output(output)   # structure is similar
    result.framework = "vitest"
    m = _VITEST_SUMMARY.search(output)
    if m:
        result.failed = int(m.group(1))
        result.passed = int(m.group(2))
    return result


# ── Unified dispatcher ────────────────────────────────────────────────────────

def parse_test_output(output: str, framework: str) -> TestSuiteResult:
    """Dispatch raw test output to the appropriate parser."""
    fw = framework.lower()
    if fw in ("pytest", "python", "py"):
        return parse_pytest_output(output)
    if fw in ("jest", "node", "npm", "yarn", "bun"):
        return parse_jest_output(output)
    if fw in ("vitest",):
        return parse_vitest_output(output)
    if fw in ("cargo", "cargo-test", "rust"):
        return parse_cargo_output(output)
    if fw in ("go", "go-test"):
        return parse_go_test_output(output)
    # Best-effort auto-detect from output content
    if "FAILED" in output and "::" in output:
        return parse_pytest_output(output)
    if "● " in output:
        return parse_jest_output(output)
    if "cargo test" in output or "running " in output and "tests" in output:
        return parse_cargo_output(output)
    if "--- FAIL:" in output:
        return parse_go_test_output(output)
    # Generic fallback
    result = TestSuiteResult(framework=framework, raw_output=output)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test framework detection & command building
# ─────────────────────────────────────────────────────────────────────────────

def detect_test_framework(workspace: Path) -> Optional[str]:
    """
    Identify the test framework used by inspecting project files.
    Returns a canonical framework string or None.
    """
    # Go
    if (workspace / "go.mod").exists():
        return "go"

    # Rust / Cargo
    if (workspace / "Cargo.toml").exists():
        return "cargo"

    # Node / JS / TS
    pkg = workspace / "package.json"
    if pkg.exists():
        try:
            import json
            data = json.loads(pkg.read_text())
            scripts  = data.get("scripts", {})
            dev_deps = {**data.get("devDependencies", {}), **data.get("dependencies", {})}
            if "vitest" in dev_deps:
                return "vitest"
            if "jest" in dev_deps or "jest" in scripts.get("test", ""):
                return "jest"
            if scripts.get("test"):
                return "npm"
        except Exception:
            pass
        return "npm"

    # Python — check for pytest
    for pyfile in ("pytest.ini", "setup.cfg", "tox.ini"):
        if (workspace / pyfile).exists():
            return "pytest"
    pyproject = workspace / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(errors="replace")
        if "pytest" in text or "tool.pytest" in text:
            return "pytest"
        if "unittest" in text:
            return "unittest"

    # Fallback: any test files present?
    test_files = list(workspace.rglob("test_*.py")) + list(workspace.rglob("*_test.py"))
    if test_files:
        if shutil.which("pytest"):
            return "pytest"
        return "unittest"

    return None


def build_test_command(
    framework: str,
    path: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """
    Build the shell command list for running tests with *framework*.
    *path* narrows the run to a file or directory.
    """
    extra = extra_args or []
    fw = framework.lower()

    if fw == "pytest":
        cmd = [shutil.which("pytest") or "pytest", "-v", "--tb=short", "--no-header", "-q"]
        if path:
            cmd.append(path)
        cmd.extend(extra)
        return cmd

    if fw == "unittest":
        cmd = ["python", "-m", "unittest", "discover"]
        if path:
            cmd += ["-s", path]
        cmd.extend(extra)
        return cmd

    if fw in ("jest", "npm"):
        return ["npm", "test", "--", "--forceExit"] + extra

    if fw == "vitest":
        return ["npx", "vitest", "run"] + ([path] if path else []) + extra

    if fw == "cargo":
        return ["cargo", "test"] + extra

    if fw == "go":
        target = f"./{path}/..." if path else "./..."
        return ["go", "test", "-v", target] + extra

    # Generic fallback
    return ["npm", "test"] + extra


# ─────────────────────────────────────────────────────────────────────────────
# Formatter detection & command building
# ─────────────────────────────────────────────────────────────────────────────

def detect_formatters(workspace: Path) -> list[str]:
    """Return available formatters for this workspace, in preference order."""
    found: list[str] = []

    # Python formatters
    if shutil.which("ruff"):
        found.append("ruff-format")
    # black — check venv bin too
    black = shutil.which("black") or str(workspace / ".venv" / "bin" / "black")
    if Path(black).exists():
        found.append("black")

    # JS/TS formatters
    prettier_local = workspace / "node_modules" / ".bin" / "prettier"
    if prettier_local.exists() or shutil.which("prettier"):
        found.append("prettier")

    # Systems
    if shutil.which("gofmt"):
        found.append("gofmt")
    if shutil.which("rustfmt"):
        found.append("rustfmt")

    return found


def build_formatter_command(
    formatter: str,
    path: Optional[str] = None,
    check_only: bool = False,
) -> list[str]:
    """
    Build the shell command list to run *formatter* on *path*.
    *check_only=True* runs in dry-run/diff mode (no writes).
    """
    fmt = formatter.lower()

    if fmt == "ruff-format":
        cmd = [shutil.which("ruff") or "ruff", "format"]
        if check_only:
            cmd.append("--diff")
        if path:
            cmd.append(path)
        else:
            cmd.append(".")
        return cmd

    if fmt == "black":
        black = shutil.which("black") or "black"
        cmd = [black]
        if check_only:
            cmd += ["--check", "--diff"]
        if path:
            cmd.append(path)
        else:
            cmd.append(".")
        return cmd

    if fmt == "prettier":
        local = Path("node_modules") / ".bin" / "prettier"
        prettier = str(local) if local.exists() else (shutil.which("prettier") or "prettier")
        cmd = [prettier]
        if not check_only:
            cmd.append("--write")
        if path:
            cmd.append(path)
        else:
            cmd += ["**/*.{js,jsx,ts,tsx,json,css,md}"]
        return cmd

    if fmt == "gofmt":
        cmd = ["gofmt"]
        if not check_only:
            cmd.append("-w")
        if path:
            cmd.append(path)
        else:
            cmd.append(".")
        return cmd

    if fmt == "rustfmt":
        cmd = ["rustfmt"]
        if check_only:
            cmd.append("--check")
        if path:
            cmd.append(path)
        return cmd

    # Unknown formatter — return as-is
    return [fmt] + ([path] if path else [])


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Defines how many times a failing command should be retried and the
    conditions under which a failure is considered *transient* (worth retrying)
    vs *permanent* (not worth retrying).
    """

    max_retries:      int   = 2
    # Patterns in combined output that indicate a permanent failure
    permanent_patterns: list[str] = field(default_factory=lambda: [
        r"syntax error",
        r"modulenotfounderror",
        r"importerror",
        r"nameerror",
        r"typeerror",
        r"no such file",
        r"command not found",
        r"permission denied",
    ])
    # Patterns that indicate a transient failure worth retrying
    transient_patterns: list[str] = field(default_factory=lambda: [
        r"timeout",
        r"connection refused",
        r"temporarily unavailable",
        r"try again",
        r"rate limit",
        r"lock.*held",
    ])

    def __post_init__(self) -> None:
        self._perm_re = [re.compile(p, re.IGNORECASE) for p in self.permanent_patterns]
        self._trans_re = [re.compile(p, re.IGNORECASE) for p in self.transient_patterns]

    def is_permanent(self, result: "CommandResult") -> bool:
        combined = result.combined_output.lower()
        return any(p.search(combined) for p in self._perm_re)

    def is_transient(self, result: "CommandResult") -> bool:
        combined = result.combined_output.lower()
        return any(p.search(combined) for p in self._trans_re)

    def should_retry(self, result: "CommandResult", attempt: int) -> bool:
        """True when we should run the command again."""
        if result.success:
            return False
        if attempt >= self.max_retries:
            return False
        if self.is_permanent(result):
            return False
        return True
