"""Async spinner for showing progress."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from nvagent.tui.app.ansi import VIOLET, RESET, cols

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.08


class Spinner:
    """Async braille spinner that writes in-place using \r."""
    
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._frame_idx = 0
        self._msg = ""
        self._tokens = 0
    
    def _write_frame(self) -> None:
        """Write the current spinner frame."""
        frame = _SPINNER_FRAMES[self._frame_idx % len(_SPINNER_FRAMES)]
        token_info = f" · {self._tokens} tokens" if self._tokens > 0 else ""
        line = f"\r{VIOLET}{frame}{RESET} {self._msg}{token_info}"
        sys.stdout.write(line)
        sys.stdout.flush()
    
    def add_tokens(self, n: int = 1) -> None:
        """Increment the live token counter shown in the spinner."""
        self._tokens += n
    
    def _clear(self) -> None:
        """Clear the spinner line."""
        sys.stdout.write("\r" + " " * cols() + "\r")
        sys.stdout.flush()
    
    async def _run(self) -> None:
        """Run the spinner animation loop."""
        while self._running:
            self._write_frame()
            self._frame_idx += 1
            await asyncio.sleep(_SPINNER_INTERVAL)
    
    def start(self, msg: str) -> None:
        """Start the spinner with the given message."""
        self._msg = msg
        self._tokens = 0
        self._running = True
        self._task = asyncio.create_task(self._run())
    
    def update(self, msg: str) -> None:
        """Change label without stopping."""
        self._msg = msg
    
    async def stop(self) -> None:
        """Async stop — awaits the task cancellation so the line is fully clear before calling code continues."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._clear()
        self._tokens = 0
