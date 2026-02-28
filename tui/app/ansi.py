"""ANSI color codes and terminal utilities."""

from __future__ import annotations

import fcntl as _fcntl
import os
import sys
import time
from typing import Callable

# ─────────────────────────────────────────────────────────────────────────────
# Blocking-stdout guard
#
# When the kernel buffer is momentarily full, direct write() calls can raise
# BlockingIOError (EAGAIN/errno 35 on macOS) if the fd was set to O_NONBLOCK.
# Patch sys.stdout.write / flush to temporarily restore blocking mode on that
# specific error and retry — this affects only the file object, not the fd.
# ─────────────────────────────────────────────────────────────────────────────

_orig_stdout_write = sys.stdout.write
_orig_stdout_flush = sys.stdout.flush


def _safe_write(text: str) -> int:
    """Write to stdout with blocking error handling."""
    try:
        return _orig_stdout_write(text)
    except (BlockingIOError, OSError):
        fd = sys.stdout.fileno()
        flags = _fcntl.fcntl(fd, _fcntl.F_GETFL)
        _fcntl.fcntl(fd, _fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        try:
            return _orig_stdout_write(text)
        finally:
            _fcntl.fcntl(fd, _fcntl.F_SETFL, flags)


def _safe_flush() -> None:
    """Flush stdout with blocking error handling."""
    try:
        _orig_stdout_flush()
    except (BlockingIOError, OSError):
        pass


# Patch stdout at module load time
sys.stdout.write = _safe_write  # type: ignore[method-assign]
sys.stdout.flush = _safe_flush  # type: ignore[method-assign]


# ─────────────────────────────────────────────────────────────────────────────
# ANSI color codes
# ─────────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

# Palette
GREEN = "\033[38;2;118;185;0m"  # NVIDIA green
BLUE = "\033[38;2;121;192;255m"  # user blue
WHITE = "\033[38;2;230;237;243m"
ORANGE = "\033[38;2;240;136;62m"  # tool calls
GRAY = "\033[38;2;139;148;158m"  # previews / dim
YELLOW = "\033[33m"
RED = "\033[31m"
BRED = "\033[1;31m"
VIOLET = "\033[38;2;167;139;250m"  # spinner / thinking


def c(*parts: str) -> str:
    """Concatenate color codes and reset at the end."""
    return "".join(parts) + RESET


# ─────────────────────────────────────────────────────────────────────────────
# Terminal utilities
# ─────────────────────────────────────────────────────────────────────────────

# Terminal-width cache (refreshed at most every 0.5 s)
_cols_cache: tuple[float, int] = (0.0, 80)


def cols() -> int:
    """Return current terminal column count, cached for 0.5 s."""
    global _cols_cache
    now = time.time()
    if now - _cols_cache[0] < 0.5:
        return _cols_cache[1]
    try:
        import shutil

        width = shutil.get_terminal_size().columns
    except Exception:
        width = 80
    _cols_cache = (now, width)
    return width


def rule(char: str = "─", color: str = DIM) -> str:
    """Return a horizontal rule spanning the terminal width."""
    return color + char * cols() + RESET


def ts() -> str:
    """Return current timestamp as HH:MM:SS."""
    return time.strftime("%H:%M:%S")


def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    import re

    _ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*m")
    return _ANSI_STRIP_RE.sub("", s)


def out(text: str = "") -> None:
    """Write a line to stdout and flush immediately."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
