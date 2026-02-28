"""Async file modification watcher using polling."""

from __future__ import annotations

import asyncio
from pathlib import Path


class FileWatcher:
    """Async background task that polls file mtimes every 2 s.
    
    Tracked paths are registered with track(); changed paths are consumed
    with pop_changed() before each new agent turn.
    """

    POLL_INTERVAL = 2.0

    def __init__(self) -> None:
        self._mtimes: dict[Path, float] = {}
        self._changed: set[Path] = set()
        self._task: asyncio.Task | None = None

    def track(self, paths: list[Path]) -> None:
        """Register new paths for watching (already-tracked paths are updated)."""
        for p in paths:
            try:
                if p.exists() and p.is_file():
                    self._mtimes[p] = p.stat().st_mtime
            except OSError:
                pass

    def pop_changed(self) -> list[Path]:
        """Return and clear the set of paths that changed since last pop."""
        changed, self._changed = list(self._changed), set()
        return changed

    def start(self) -> None:
        """Start the polling task if not already running."""
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._poll())

    async def stop(self) -> None:
        """Stop the polling task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll(self) -> None:
        """Main polling loop (mtime check every POLL_INTERVAL seconds)."""
        while True:
            await asyncio.sleep(self.POLL_INTERVAL)
            for p, old_mtime in list(self._mtimes.items()):
                try:
                    new_mtime = p.stat().st_mtime
                    if new_mtime != old_mtime:
                        self._mtimes[p] = new_mtime
                        self._changed.add(p)
                except OSError:
                    pass
