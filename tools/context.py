"""Shared context object passed to all tool handlers."""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

_BLOCKED_CMD_PATTERNS = [
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f\s+/", r"sudo\s+rm", r"\bmkfs\b",
    r"\bdd\b.*\bif=", r":\(\)\s*\{.*\}", r"chmod\s+-[rR]\s+777\s+/",
    r">\s*/etc/passwd", r">\s*/etc/shadow",
    r"curl\s+.*\|\s*(ba)?sh", r"wget\s+.*\|\s*(ba)?sh",
    r"\|\s*(ba)?sh\s*<", r"nc\s+-[a-zA-Z]*e\b",
]


class Sandbox:
    def __init__(self, workspace: Path, safe_mode: bool = True) -> None:
        self.workspace = workspace.resolve()
        self.safe_mode = safe_mode
        self._blocked = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_CMD_PATTERNS]

    def validate_path(self, path: Path) -> tuple[bool, str]:
        if not self.safe_mode:
            return True, ""
        try:
            path.resolve().relative_to(self.workspace)
            return True, ""
        except ValueError:
            return False, f"Path '{path}' is outside the workspace '{self.workspace}'."

    def validate_command(self, command: str) -> tuple[bool, str]:
        if not self.safe_mode:
            return True, ""
        for p in self._blocked:
            if p.search(command):
                return False, f"Command blocked by safe_mode — matches pattern: {p.pattern!r}"
        return True, ""


class _SimpleRetryPolicy:
    """Minimal retry policy: retry if exit_code != 0 and attempt < max_attempts."""

    def should_retry(self, result: object, attempt: int, max_attempts: int = 3) -> bool:
        exit_code = getattr(result, "exit_code", 0)
        return exit_code != 0 and attempt < max_attempts


class ToolContext:
    """Mutable session state shared by all tool handlers."""

    def __init__(
        self,
        workspace: Path,
        max_file_bytes: int = 102400,
        confirm_fn: Callable[[str, str], Awaitable[bool]] | None = None,
        safe_mode: bool = True,
        dry_run: bool = False,
        stream_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.workspace = workspace
        self.max_file_bytes = max_file_bytes
        self.confirm_fn = confirm_fn
        self.dry_run = dry_run
        self.stream_fn = stream_fn  # called with partial output lines during run_command
        self.changed_files: list[str] = []
        self.undo_stack: list[dict[str, str | None]] = []
        self._current_turn_backups: dict[str, str | None] = {}
        self.sandbox = Sandbox(workspace, safe_mode=safe_mode)
        self._checkpoints: dict[str, dict[str, str | None]] = {}
        self._read_mtimes: dict[str, int] = {}
        self._path_locks: dict[str, asyncio.Lock] = {}
        self._read_result_cache: SimpleCache = SimpleCache()
        self._rg_path: str | None = shutil.which("rg")
        self._patch_bin: str | None = shutil.which("patch")
        self._todos: list[dict] = []
        self._retry_policy = _SimpleRetryPolicy()
        self.active_proc: object | None = None  # currently running subprocess, for interrupt

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.workspace / p).resolve()

    def _get_path_lock(self, fpath: Path) -> asyncio.Lock:
        return self._path_locks.setdefault(str(fpath), asyncio.Lock())

    def _check_stale(self, fpath: Path) -> str | None:
        key = str(fpath)
        recorded = self._read_mtimes.get(key)
        if recorded is None or not fpath.exists():
            return None
        try:
            current = fpath.stat().st_mtime_ns
        except OSError:
            return None
        if current != recorded:
            rel = fpath.relative_to(self.workspace) if fpath.is_relative_to(self.workspace) else fpath
            return f"⚠ [STALE READ] '{rel}' was modified since last read. Re-read before writing."
        return None

    def last_read_mtime(self, abs_path: str) -> int | None:
        return self._read_mtimes.get(abs_path)


class SimpleCache:
    """In-memory read cache: (path, mtime_ns) -> content."""

    def __init__(self, max_size: int = 200) -> None:
        self._data: dict[tuple[str, int], str] = {}
        self._max_size = max_size

    def get(self, path: str, mtime_ns: int) -> str | None:
        return self._data.get((path, mtime_ns))

    def put(self, path: str, mtime_ns: int, content: str) -> None:
        if len(self._data) >= self._max_size:
            # Evict first 50 entries
            for k in list(self._data.keys())[:50]:
                self._data.pop(k, None)
        self._data[(path, mtime_ns)] = content

    # Compat shim for old-style cache.get(path, mtime) calls from file.py
    def __call__(self, path: str, mtime_ns: int) -> str | None:
        return self.get(path, mtime_ns)
