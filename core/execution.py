"""Execution utilities — command running, test parsing, formatter detection.

Consolidates what was in core/execution/executor.py into a single module.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Process group kill ─────────────────────────────────────────────────────────

def _kill_proc_group(proc) -> None:
    """Kill the process and its entire process group (Unix) or just the process."""
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except Exception:
        pass


# ── CommandResult ──────────────────────────────────────────────────────────────

_ERROR_KEYWORDS = frozenset(["error", "exception", "traceback", "fatal", "failed", "failure", "assert", "panic", "aborted", "segfault", "killed"])
_WARN_KEYWORDS = frozenset(["warning", "warn", "deprecated"])


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

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

    def extract_errors(self, max_lines: int = 30) -> list[str]:
        lines = self.combined_output.splitlines()
        result: list[str] = []
        capture = 0
        for line in lines:
            ll = line.lower()
            is_err = any(kw in ll for kw in _ERROR_KEYWORDS)
            if is_err or capture > 0:
                result.append(line)
                if capture > 0:
                    capture -= 1
                if "traceback" in ll or "most recent call" in ll:
                    capture = 10
            if len(result) >= max_lines:
                result.append(f"  … ({len(lines) - len(result)} more lines)")
                break
        return result

    def to_agent_str(self, max_chars: int = 8000) -> str:
        if self.timed_out:
            return f"$ {self.command}\n⏱ TIMED OUT after {self.duration_s:.1f}s\n{self.stderr[:500]}"
        status = "✓" if self.success else "✗"
        parts = [f"$ {self.command}", f"{status} exit={self.exit_code}  ({self.duration_s:.2f}s)"]
        errors = self.extract_errors()
        if errors and not self.success:
            parts.append("Errors / tracebacks:")
            parts.extend(f"  {e}" for e in errors)
        combined = self.combined_output
        if combined:
            if len(combined) > max_chars:
                combined = combined[:max_chars] + f"\n… [{len(combined) - max_chars:,} chars truncated]"
            parts.append(combined)
        return "\n".join(parts)


# ── Test result model ──────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    status: str
    file: str = ""
    duration_s: float = 0.0
    error_message: str = ""

    def render(self) -> str:
        icon = {"passed": "✓", "failed": "✗", "error": "!", "skipped": "⊘"}.get(self.status, "?")
        base = f"  {icon} {self.name}"
        if self.error_message:
            excerpt = self.error_message[:200].replace("\n", "\n      ")
            base += f"\n      {excerpt}"
        return base


@dataclass
class TestSuiteResult:
    framework: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    failing_tests: list[TestCase] = field(default_factory=list)
    raw_output: str = ""

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
        icon = "✓" if self.success else "✗"
        time_str = f"  in {self.duration_s:.2f}s" if self.duration_s > 0 else ""
        return f"{icon} [{self.framework}]  {',  '.join(parts) or 'no tests ran'}{time_str}"

    def to_agent_str(self) -> str:
        lines = [self.summary_line()]
        if self.failing_tests:
            lines.append("\nFailing tests:")
            for tc in self.failing_tests[:20]:
                lines.append(tc.render())
        elif not self.success:
            tail = "\n".join(self.raw_output.splitlines()[-40:])
            lines.append(f"\nOutput tail:\n{tail}")
        return "\n".join(lines)


# ── Test output parsers ────────────────────────────────────────────────────────

def parse_pytest_output(output: str) -> TestSuiteResult:
    result = TestSuiteResult(framework="pytest", raw_output=output)
    for pat, attr in [
        (r"(\d+)\s+passed", "passed"), (r"(\d+)\s+failed", "failed"),
        (r"(\d+)\s+error", "errors"), (r"(\d+)\s+skipped", "skipped"),
    ]:
        m = re.search(pat, output)
        if m:
            setattr(result, attr, int(m.group(1)))
    m = re.search(r"in\s+([\d.]+)s", output)
    if m:
        result.duration_s = float(m.group(1))
    for m2 in re.finditer(r"^FAILED\s+(.+?)(?:\s+-\s+(.+))?$", output, re.MULTILINE):
        result.failing_tests.append(TestCase(name=m2.group(1).strip(), status="failed", error_message=(m2.group(2) or "").strip()))
    return result


def parse_jest_output(output: str) -> TestSuiteResult:
    result = TestSuiteResult(framework="jest", raw_output=output)
    m = re.search(r"Tests?:\s+(?:(\d+) failed[,\s]*)?(?:(\d+) passed[,\s]*)?(?:(\d+) skipped[,\s]*)?(\d+) total", output)
    if m:
        result.failed = int(m.group(1) or 0)
        result.passed = int(m.group(2) or 0)
        result.skipped = int(m.group(3) or 0)
    for m2 in re.finditer(r"^\s+●\s+(.+)$", output, re.MULTILINE):
        name = m2.group(1).strip()
        if name:
            result.failing_tests.append(TestCase(name=name, status="failed"))
    return result


def parse_test_output(output: str, framework: str) -> TestSuiteResult:
    fw = framework.lower()
    if fw in ("pytest", "python"):
        return parse_pytest_output(output)
    if fw in ("jest", "npm", "yarn", "vitest"):
        return parse_jest_output(output)
    # Auto-detect fallback
    if "FAILED" in output and "::" in output:
        return parse_pytest_output(output)
    if "● " in output:
        return parse_jest_output(output)
    return TestSuiteResult(framework=framework, raw_output=output)


# ── Test framework detection ───────────────────────────────────────────────────

def detect_test_framework(workspace: Path) -> str | None:
    if (workspace / "go.mod").exists():
        return "go"
    if (workspace / "Cargo.toml").exists():
        return "cargo"
    pkg = workspace / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text())
            deps = {**data.get("devDependencies", {}), **data.get("dependencies", {})}
            if "vitest" in deps:
                return "vitest"
            if "jest" in deps:
                return "jest"
        except Exception:
            pass
        return "npm"
    if any((workspace / f).exists() for f in ("pytest.ini", "setup.cfg", "tox.ini")):
        return "pytest"
    pyproject = workspace / "pyproject.toml"
    if pyproject.exists() and "pytest" in pyproject.read_text(errors="replace"):
        return "pytest"
    if list(workspace.rglob("test_*.py")) or list(workspace.rglob("*_test.py")):
        return "pytest" if shutil.which("pytest") else "unittest"
    return None


def build_test_command(framework: str, path: str | None = None, extra_args: list[str] | None = None) -> list[str]:
    extra = extra_args or []
    fw = framework.lower()
    if fw == "pytest":
        cmd = [shutil.which("pytest") or "pytest", "-v", "--tb=short", "-q"]
        if path:
            cmd.append(path)
        return cmd + extra
    if fw == "unittest":
        cmd = ["python", "-m", "unittest", "discover"]
        if path:
            cmd += ["-s", path]
        return cmd + extra
    if fw in ("jest", "npm"):
        return ["npm", "test", "--", "--forceExit"] + extra
    if fw == "vitest":
        return ["npx", "vitest", "run"] + ([path] if path else []) + extra
    if fw == "cargo":
        return ["cargo", "test"] + extra
    if fw == "go":
        return ["go", "test", "-v", f"./{path}/..." if path else "./..."] + extra
    return ["npm", "test"] + extra


# ── Formatter detection ────────────────────────────────────────────────────────

def detect_formatters(workspace: Path) -> list[str]:
    found = []
    if shutil.which("ruff"):
        found.append("ruff-format")
    if shutil.which("black"):
        found.append("black")
    if (workspace / "node_modules" / ".bin" / "prettier").exists() or shutil.which("prettier"):
        found.append("prettier")
    if shutil.which("gofmt"):
        found.append("gofmt")
    if shutil.which("rustfmt"):
        found.append("rustfmt")
    return found


def build_formatter_command(formatter: str, path: str | None = None, check_only: bool = False) -> list[str]:
    fmt = formatter.lower()
    if fmt == "ruff-format":
        cmd = [shutil.which("ruff") or "ruff", "format"]
        if check_only:
            cmd.append("--diff")
        cmd.append(path or ".")
        return cmd
    if fmt == "black":
        cmd = [shutil.which("black") or "black"]
        if check_only:
            cmd += ["--check", "--diff"]
        cmd.append(path or ".")
        return cmd
    if fmt == "prettier":
        prettier = shutil.which("prettier") or "prettier"
        cmd = [prettier]
        if not check_only:
            cmd.append("--write")
        cmd.append(path or ".")
        return cmd
    if fmt == "gofmt":
        cmd = ["gofmt"]
        if not check_only:
            cmd.append("-w")
        cmd.append(path or ".")
        return cmd
    return [fmt] + ([path] if path else [])
