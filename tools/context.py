"""
Shared context object passed to all tool handlers.

ToolContext centralises the mutable state (workspace, locks, caches, etc.)
that was previously scattered across ToolExecutor attributes, allowing each
handler module to stay focused on its own domain without referencing the
monolithic executor.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Awaitable, Callable, Optional

from nvagent.core.execution import Sandbox
from nvagent.core.execution.executor import RetryPolicy
from nvagent.core.state import get_tool_cache, ToolCache


class ToolContext:
    """Mutable session state + path utilities shared by all handlers."""

    def __init__(
        self,
        workspace: Path,
        max_file_bytes: int = 102400,
        confirm_fn: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        safe_mode: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.workspace = workspace
        self.max_file_bytes = max_file_bytes
        self.confirm_fn = confirm_fn
        self.dry_run = dry_run

        # Per-turn tracking
        self.changed_files: list[str] = []
        # Undo: stack of {abs_path -> old_content_or_None_if_new}
        self.undo_stack: list[dict[str, str | None]] = []
        self._current_turn_backups: dict[str, str | None] = {}

        # Sandbox: path + command safety validator
        self.sandbox = Sandbox(workspace, safe_mode=safe_mode)

        # Named checkpoints for rollback
        self._checkpoints: dict[str, dict[str, str | None]] = {}

        # Retry policy for run_tests / run_command
        self._retry_policy = RetryPolicy(max_retries=2)

        # File mtime tracking: abs_path_str → mtime_ns at last read
        self._read_mtimes: dict[str, int] = {}

        # Per-path asyncio locks (single-threaded asyncio: setdefault is atomic)
        self._path_locks: dict[str, asyncio.Lock] = {}

        # Two-level read cache (L1 in-memory + L2 SQLite)
        self._read_result_cache: ToolCache = get_tool_cache(workspace)

        # Binary paths cached once at init
        self._rg_path: Optional[str] = shutil.which("rg")
        self._patch_bin: Optional[str] = shutil.which("patch")

        # In-session structured task list
        self._todos: list[dict] = []

    # ── Path utilities ────────────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> Path:
        """Resolve *path* relative to workspace, or return as-is if absolute."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.workspace / p).resolve()

    def _get_path_lock(self, fpath: Path) -> asyncio.Lock:
        """Return (creating if necessary) a per-path asyncio.Lock."""
        key = str(fpath)
        return self._path_locks.setdefault(key, asyncio.Lock())

    def _check_stale(self, fpath: Path) -> Optional[str]:
        """
        Return a warning string if *fpath* was modified externally since the
        agent last read it, or None if the file is fresh / not previously read.
        """
        key = str(fpath)
        recorded_mtime = self._read_mtimes.get(key)
        if recorded_mtime is None:
            return None
        if not fpath.exists():
            return None
        try:
            current_mtime = fpath.stat().st_mtime_ns
        except OSError:
            return None
        if current_mtime != recorded_mtime:
            try:
                rel = fpath.relative_to(self.workspace)
            except ValueError:
                rel = fpath
            return (
                f"⚠ [STALE READ] '{rel}' was modified externally after the agent last read it "
                f"(recorded mtime differs from current). "
                f"The agent's edits are based on an outdated version of this file. "
                f"Re-read the file before writing to avoid overwriting unseen changes."
            )
        return None

    def last_read_mtime(self, abs_path: str) -> int | None:
        """Return the mtime_ns recorded when this file was last read, or None."""
        return self._read_mtimes.get(abs_path)
