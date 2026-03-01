"""Safety layer — git checkpointing, loop detection, resource limits."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from nvagent.config import SafetyConfig


@dataclass
class Violation:
    kind: str
    message: str
    fatal: bool = True

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


def _git(args: list[str], cwd: Path, timeout: int = 15) -> tuple[str, str, int]:
    try:
        r = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=str(cwd), timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return "", str(e), -1


class GitCheckpointer:
    """Creates a git commit before agent modifications for safe rollback."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._checkpoint_ref: str | None = None

    def is_git_repo(self) -> bool:
        _, _, rc = _git(["rev-parse", "--git-dir"], self.workspace)
        return rc == 0

    def current_sha(self) -> str | None:
        out, _, rc = _git(["rev-parse", "HEAD"], self.workspace)
        return out if rc == 0 else None

    async def checkpoint(self, message: str = "nvagent: pre-task checkpoint") -> str | None:
        if not self.is_git_repo():
            return None
        dirty_out, _, _ = _git(["status", "--short"], self.workspace)
        if not dirty_out:
            sha = self.current_sha()
            self._checkpoint_ref = sha
            return sha
        _git(["add", "-A"], self.workspace)
        _, _, rc = _git(["commit", "-m", message, "--no-verify"], self.workspace)
        sha = self.current_sha()
        self._checkpoint_ref = sha
        return sha

    async def restore(self, ref: str | None = None) -> tuple[bool, str]:
        if not self.is_git_repo():
            return False, "Not a git repository."
        target = ref or self._checkpoint_ref
        if not target:
            return False, "No checkpoint reference recorded."
        _, err, rc = _git(["reset", "--hard", target], self.workspace)
        if rc != 0:
            return False, f"git reset failed: {err}"
        _git(["clean", "-fd"], self.workspace)
        return True, f"Restored to {target[:8]}."


class LoopDetector:
    """Detect when the agent is stuck repeating the same tool calls."""

    def __init__(self, max_identical: int = 3, window: int = 8) -> None:
        self._max_identical = max_identical
        self._window = window
        self._history: deque[tuple[str, str]] = deque(maxlen=window * 2)
        self._loop_desc = ""

    def _fp(self, tool_name: str, args: dict) -> str:
        try:
            canonical = json.dumps({"n": tool_name, "a": args}, sort_keys=True)
        except (TypeError, ValueError):
            canonical = f"{tool_name}:{args}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def record(self, tool_name: str, args: dict) -> None:
        self._history.append((tool_name, self._fp(tool_name, args)))

    def is_looping(self) -> bool:
        recent = list(self._history)[-self._window:]
        counts = Counter(fp for _, fp in recent)
        for fp, count in counts.items():
            if count >= self._max_identical:
                tool = next(n for n, f in recent if f == fp)
                self._loop_desc = f"Tool '{tool}' called {count}× with identical args — loop detected."
                return True
        n = len(recent)
        if n >= 4:
            half = n // 2
            if [fp for _, fp in recent[:half]] == [fp for _, fp in recent[half:half * 2]]:
                tools = [name for name, _ in recent[:half]]
                self._loop_desc = f"Repeating cycle: [{', '.join(tools)}] — not making progress."
                return True
        return False

    def description(self) -> str:
        return self._loop_desc or "Repeated tool call pattern detected."

    def reset(self) -> None:
        self._history.clear()
        self._loop_desc = ""


class ResourceGuard:
    """Enforce hard resource limits per task."""

    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg
        self._start = 0.0
        self._tokens = 0
        self._tool_calls = 0
        self._files_changed = 0

    def start(self) -> None:
        self._start = time.monotonic()
        self._tokens = self._tool_calls = self._files_changed = 0

    def update(self, *, tokens: int = 0, tool_calls: int = 0, files_changed: int = 0) -> None:
        self._tokens += tokens
        self._tool_calls += tool_calls
        self._files_changed = files_changed

    def check(self) -> Violation | None:
        elapsed = time.monotonic() - self._start
        cfg = self._cfg
        if elapsed >= cfg.max_wall_seconds:
            return Violation("resource", f"Wall-clock limit reached: {elapsed:.0f}s — stopping.", fatal=True)
        if self._tokens >= cfg.max_tokens_per_task:
            return Violation("resource", f"Token budget exhausted: {self._tokens:,} — stopping.", fatal=True)
        if self._tool_calls >= cfg.max_tool_calls:
            return Violation("resource", f"Too many tool calls: {self._tool_calls} — stopping.", fatal=True)
        return None
