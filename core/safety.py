"""
Layer 5 — Safety & Stability for nvagent.

Components
──────────
SafetyConfig     Dataclass holding all tunable safety limits.

GitCheckpointer  Auto-commits workspace state before the agent starts
                 modifying files, providing a hard rollback point.
                 • is_git_repo()
                 • checkpoint(message) → commit sha | None
                 • restore(ref)        → bool
                 • uncommitted_files() → list[str]

ChangeValidator  Validates a file after every write:
                 • Syntax check  (Python ast, JSON, TypeScript tsc --noEmit)
                 • Optional linter run (ruff / eslint) on the changed file
                 Returns a ValidationResult with ok/errors/warnings.

LoopDetector     Detects when the agent is stuck repeating the same tool
                 calls.  Fingerprints each tool invocation and raises a flag
                 when the same pattern recurs within a rolling window.
                 • record(tool_name, args_json)
                 • is_looping() → bool
                 • description() → human-readable explanation

ResourceGuard    Enforces hard limits on:
                 • Wall-clock seconds   (max_wall_seconds)
                 • Total tokens used    (max_tokens_per_task)
                 • Files changed        (max_files_per_task)
                 • Individual output    (max_output_bytes per tool call)
                 • Tool calls executed  (max_tool_calls)
                 Returns a Violation on the first breach, or None.

All components are designed to fail-safe: if git is unavailable, if a
linter is not installed, etc., they degrade gracefully without raising.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import subprocess
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# SafetyConfig lives in config.py to avoid circular imports
# (config ← safety ← executor ← tools ← config)
from nvagent.config import SafetyConfig  # noqa: F401 – re-exported for convenience

# ─────────────────────────────────────────────────────────────────────────────
# Violation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Violation:
    """A safety limit that was breached."""

    kind: str  # "loop" | "resource" | "validation" | "tests"
    message: str
    fatal: bool = True  # fatal → abort task; non-fatal → warn only

    def __str__(self) -> str:
        tag = "FATAL" if self.fatal else "WARNING"
        return f"[Safety {tag} — {self.kind}] {self.message}"


# ─────────────────────────────────────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    ok: bool
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_str(self) -> str:
        if self.ok and not self.warnings:
            return f"✓ {self.path}: validation passed"
        lines = [f"{'✓' if self.ok else '✗'} {self.path}:"]
        for e in self.errors:
            lines.append(f"  ✗ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GitCheckpointer
# ─────────────────────────────────────────────────────────────────────────────


def _git(args: list[str], cwd: Path, timeout: int = 15) -> tuple[str, str, int]:
    """Run a git command; return (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return "", str(exc), -1


class GitCheckpointer:
    """
    Creates a git commit capturing the workspace state before the agent
    starts writing files.  Provides a safe rollback target.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._checkpoint_ref: Optional[str] = None  # SHA of checkpoint commit
        self._stash_ref: Optional[str] = None  # stash entry when dirty

    # ── Public API ─────────────────────────────────────────────────────────

    def is_git_repo(self) -> bool:
        _, _, rc = _git(["rev-parse", "--git-dir"], self.workspace)
        return rc == 0

    def uncommitted_files(self) -> list[str]:
        """Return list of modified / untracked files (short status)."""
        out, _, rc = _git(["status", "--short"], self.workspace)
        if rc != 0:
            return []
        return [line[3:].strip() for line in out.splitlines() if line.strip()]

    def current_sha(self) -> Optional[str]:
        out, _, rc = _git(["rev-parse", "HEAD"], self.workspace)
        return out if rc == 0 else None

    async def checkpoint(self, message: str = "nvagent: pre-task checkpoint") -> Optional[str]:
        """
        Create a checkpoint commit of all current changes without disturbing
        the working tree.  Returns the commit SHA, or None if git is unavailable.

        Strategy:
          1. If repo is clean → HEAD is already the safe point, return its SHA.
          2. Otherwise → stage all changes and commit them on the current branch.
             The working tree is NOT stashed or reset — the user's files are
             left exactly as they are.  The commit just records the state so
             /rollback can hard-reset back to it if needed.
        """
        if not self.is_git_repo():
            return None

        dirty_files = self.uncommitted_files()

        if not dirty_files:
            sha = self.current_sha()
            self._checkpoint_ref = sha
            return sha

        # Stage everything (tracked modifications + new untracked files)
        _git(["add", "-A"], self.workspace)
        # Commit — working tree is unchanged, we're just recording the snapshot
        _, _, rc = _git(["commit", "-m", message, "--no-verify"], self.workspace)
        if rc == 0:
            sha = self.current_sha()
            self._checkpoint_ref = sha
            return sha
        else:
            # Commit failed (e.g. nothing to commit after add) — use HEAD
            self._checkpoint_ref = self.current_sha()
            return self._checkpoint_ref

    async def restore(self, ref: Optional[str] = None) -> tuple[bool, str]:
        """
        Restore the workspace to *ref* (defaults to checkpoint_ref).
        Returns (success, message).
        """
        if not self.is_git_repo():
            return False, "Not a git repository."

        target = ref or self._checkpoint_ref
        if not target:
            return False, "No checkpoint reference recorded."

        # Hard-reset to the target commit
        _, err, rc = _git(["reset", "--hard", target], self.workspace)
        if rc != 0:
            return False, f"git reset failed: {err}"

        # Clean untracked files that the agent may have created
        _git(["clean", "-fd"], self.workspace)

        return True, f"Restored to {target[:8]}."

    def describe_checkpoint(self) -> str:
        if not self._checkpoint_ref:
            return "No git checkpoint recorded."
        out, _, _ = _git(
            ["log", "--oneline", "-1", self._checkpoint_ref],
            self.workspace,
        )
        return f"Checkpoint: {out or self._checkpoint_ref[:8]}"


# ─────────────────────────────────────────────────────────────────────────────
# ChangeValidator
# ─────────────────────────────────────────────────────────────────────────────


# Syntax checkers: extension → callable(path: Path) → list[str] errors
def _check_python_syntax(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=str(path))
        return []
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]
    except Exception as e:
        return [f"Parse error: {e}"]


def _check_json_syntax(path: Path) -> list[str]:
    try:
        json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return []
    except json.JSONDecodeError as e:
        return [f"JSON error at line {e.lineno}: {e.msg}"]


def _check_typescript_syntax(path: Path, workspace: Path) -> list[str]:
    """Quick tsc type-check for a single file (no-emit)."""
    local_tsc = workspace / "node_modules" / ".bin" / "tsc"
    tsc = str(local_tsc) if local_tsc.exists() else shutil.which("tsc")
    if not tsc:
        return []  # tsc not available — skip
    try:
        r = subprocess.run(
            [tsc, "--noEmit", "--allowJs", "--checkJs", str(path)],
            capture_output=True,
            text=True,
            cwd=str(workspace),
            timeout=20,
        )
        errors = []
        for line in (r.stdout + r.stderr).splitlines():
            if "error TS" in line:
                errors.append(line.strip()[:200])
        return errors[:5]  # cap at 5
    except Exception:
        return []


_SYNTAX_CHECKERS: dict[str, object] = {
    ".py": _check_python_syntax,
    ".json": _check_json_syntax,
}
_TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}


class ChangeValidator:
    """
    Validate a file immediately after it is written.
    Returns a ValidationResult so the agent can see and fix issues quickly
    rather than discovering them at test-run time.
    """

    def __init__(
        self,
        workspace: Path,
        run_linter: bool = False,
    ) -> None:
        self.workspace = workspace
        self.run_linter = run_linter

    def validate_file(self, path: Path) -> ValidationResult:
        """Synchronous validation (called from async context via executor)."""
        if not path.exists():
            return ValidationResult(ok=False, path=str(path), errors=[f"File not found: {path}"])

        errors: list[str] = []
        warnings: list[str] = []
        ext = path.suffix.lower()

        # ── Syntax check ─────────────────────────────────────────────────────
        if ext in _SYNTAX_CHECKERS:
            checker = _SYNTAX_CHECKERS[ext]
            errors.extend(checker(path))  # type: ignore[operator]
        elif ext in _TS_EXTENSIONS:
            errors.extend(_check_typescript_syntax(path, self.workspace))

        # ── Ruff linter (Python) ──────────────────────────────────────────────
        if self.run_linter and ext == ".py" and not errors:
            ruff = shutil.which("ruff")
            if ruff:
                try:
                    r = subprocess.run(
                        [ruff, "check", "--output-format=concise", "--select=E,F,W", str(path)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    for line in (r.stdout + r.stderr).splitlines():
                        if line.strip() and "error" in line.lower():
                            errors.append(line.strip()[:200])
                        elif line.strip():
                            warnings.append(line.strip()[:200])
                    errors = errors[:8]
                    warnings = warnings[:8]
                except Exception:
                    pass

        return ValidationResult(
            ok=len(errors) == 0,
            path=str(
                path.relative_to(self.workspace) if path.is_relative_to(self.workspace) else path
            ),
            errors=errors,
            warnings=warnings,
        )

    async def validate_file_async(self, path: Path) -> ValidationResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.validate_file, path)


# ─────────────────────────────────────────────────────────────────────────────
# LoopDetector
# ─────────────────────────────────────────────────────────────────────────────


def _fingerprint(tool_name: str, args: dict) -> str:
    """Stable hash of a tool call — insensitive to key order."""
    try:
        canonical = json.dumps({"n": tool_name, "a": args}, sort_keys=True)
    except (TypeError, ValueError):
        canonical = f"{tool_name}:{str(args)}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class LoopDetector:
    """
    Detect when the agent is stuck repeating the same tool calls.

    Two detection strategies:
    1. Exact repeat — the same fingerprint appears ≥ max_identical times
       within the rolling window.
    2. Cycle — the last N/2 calls are an exact repeat of the N/2 before them
       (e.g., [A, B, A, B] → detected).
    """

    def __init__(self, max_identical: int = 3, window: int = 8) -> None:
        self._max_identical = max_identical
        self._window = window
        self._history: deque[tuple[str, str]] = deque(maxlen=window * 2)
        self._loop_desc: str = ""

    def record(self, tool_name: str, args: dict) -> None:
        fp = _fingerprint(tool_name, args)
        self._history.append((tool_name, fp))

    def is_looping(self) -> bool:
        recent = list(self._history)[-self._window :]

        # Strategy 1: exact repeat count
        from collections import Counter

        counts = Counter(fp for _, fp in recent)
        for fp, count in counts.items():
            if count >= self._max_identical:
                tool = next(n for n, f in recent if f == fp)
                self._loop_desc = (
                    f"Tool '{tool}' called {count} times with identical arguments "
                    f"in the last {len(recent)} calls — infinite loop detected."
                )
                return True

        # Strategy 2: half-window cycle detection
        n = len(recent)
        if n >= 4:
            half = n // 2
            first_half = [fp for _, fp in recent[:half]]
            second_half = [fp for _, fp in recent[half : half * 2]]
            if first_half == second_half:
                tools = [n for n, _ in recent[:half]]
                self._loop_desc = (
                    f"Repeating cycle detected: [{', '.join(tools)}] × 2 — "
                    "agent is not making progress."
                )
                return True

        return False

    def description(self) -> str:
        return self._loop_desc or "Repeated tool call pattern detected."

    def reset(self) -> None:
        self._history.clear()
        self._loop_desc = ""


# ─────────────────────────────────────────────────────────────────────────────
# ResourceGuard
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ResourceState:
    tokens_used: int = 0
    turns: int = 0
    tool_calls: int = 0
    files_changed: int = 0
    output_bytes: int = 0
    elapsed_s: float = 0.0


class ResourceGuard:
    """
    Enforces hard resource limits.  Call start() once at the beginning of a
    task, then check(...) after each turn.  Returns a Violation when any
    limit is exceeded (or None when all is well).
    """

    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg
        self._start: float = 0.0
        self.state = ResourceState()

    def start(self) -> None:
        self._start = time.monotonic()
        self.state = ResourceState()

    def update(
        self,
        *,
        tokens: int = 0,
        tool_calls: int = 0,
        files_changed: int = 0,
        output_bytes: int = 0,
    ) -> None:
        self.state.tokens_used += tokens
        self.state.tool_calls += tool_calls
        self.state.files_changed = files_changed  # set (not increment)
        self.state.output_bytes += output_bytes
        self.state.elapsed_s = time.monotonic() - self._start

    def check(self) -> Optional[Violation]:
        s = self.state
        c = self._cfg

        if s.elapsed_s >= c.max_wall_seconds:
            return Violation(
                kind="resource",
                message=(
                    f"Wall-clock limit reached: {s.elapsed_s:.0f}s ≥ {c.max_wall_seconds:.0f}s. "
                    "Task is taking too long — stopping to prevent runaway execution."
                ),
                fatal=True,
            )
        if s.tokens_used >= c.max_tokens_per_task:
            return Violation(
                kind="resource",
                message=(
                    f"Token budget exhausted: {s.tokens_used:,} ≥ {c.max_tokens_per_task:,}. "
                    "Stopping to control cost."
                ),
                fatal=True,
            )
        if s.tool_calls >= c.max_tool_calls:
            return Violation(
                kind="resource",
                message=(
                    f"Too many tool calls: {s.tool_calls} ≥ {c.max_tool_calls}. "
                    "The agent may be stuck — stopping."
                ),
                fatal=True,
            )
        if s.files_changed >= c.max_files_per_task:
            return Violation(
                kind="resource",
                message=(
                    f"Too many files modified: {s.files_changed} ≥ {c.max_files_per_task}. "
                    "Pausing — please review changes before continuing."
                ),
                fatal=False,  # warning, not abort
            )
        return None

    def summary(self) -> str:
        s = self.state
        return (
            f"tokens={s.tokens_used:,}  tool_calls={s.tool_calls}"
            f"  files={s.files_changed}  elapsed={s.elapsed_s:.1f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestEnforcer
# ─────────────────────────────────────────────────────────────────────────────


class TestEnforcer:
    """
    After code changes, run the test suite and block task completion
    when require_tests_pass=True and tests fail.
    """

    def __init__(self, workspace: Path, cfg: SafetyConfig) -> None:
        self.workspace = workspace
        self._cfg = cfg

    async def run_and_check(
        self,
        changed_files: list[str],
    ) -> Optional[Violation]:
        """
        Returns a Violation if tests fail and require_tests_pass is True.
        Returns None if tests pass, aren't required, or couldn't be run.
        """
        if not self._cfg.require_tests_pass:
            return None
        if not changed_files:
            return None

        # Import here to avoid circular deps
        from nvagent.core.executor import (
            detect_test_framework,
            build_test_command,
            parse_test_output,
        )

        fw = detect_test_framework(self.workspace)
        if not fw:
            return None  # can't enforce without a known framework

        cmd_override = self._cfg.test_command.strip()
        if cmd_override:
            cmd = cmd_override.split()
        else:
            cmd = build_test_command(fw, extra_args=["--tb=no", "-q"] if fw == "pytest" else None)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self._cfg.test_timeout
            )
        except asyncio.TimeoutError:
            return Violation(
                kind="tests",
                message=f"Test suite timed out ({self._cfg.test_timeout}s). "
                "Cannot verify correctness — task blocked.",
                fatal=True,
            )
        except FileNotFoundError:
            return None  # runner not installed — skip

        raw = (
            stdout_b.decode("utf-8", errors="replace")
            + "\n"
            + stderr_b.decode("utf-8", errors="replace")
        )
        suite = parse_test_output(raw, fw)

        if not suite.success:
            failures_str = ""
            if suite.failing_tests:
                failures_str = "\n" + "\n".join(f"  ✗ {t.name}" for t in suite.failing_tests[:10])
            return Violation(
                kind="tests",
                message=(
                    f"Tests failed after your changes — task blocked until tests pass.\n"
                    f"{suite.summary_line()}{failures_str}"
                ),
                fatal=True,
            )

        return None
