"""Readline integration for input history and editing."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import readline as _readline

    _READLINE_AVAILABLE = True
except ImportError:
    _readline = None  # type: ignore[assignment]
    _READLINE_AVAILABLE = False

_HISTORY_FILE = Path.home() / ".nvagent" / "input_history"
_MAX_HISTORY = 1_000


def setup_readline() -> None:
    """Initialise readline: load history file and configure sensible defaults."""
    if not _READLINE_AVAILABLE or not _readline:
        return
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _HISTORY_FILE.exists():
            _readline.read_history_file(str(_HISTORY_FILE))
        _readline.set_history_length(_MAX_HISTORY)
        # vi-style or emacs (default emacs — matches bash)
        _readline.parse_and_bind("set editing-mode emacs")
        # Tab completion: complete from history on double-tab
        _readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def save_readline_history() -> None:
    """Persist history to disk. Called on clean exit."""
    if not _READLINE_AVAILABLE or not _readline:
        return
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _readline.write_history_file(str(_HISTORY_FILE))
    except Exception:
        pass


def rl_prompt(colored: str) -> str:
    """Wrap ANSI escape sequences in readline non-printing markers (\001...\002).

    Without these markers readline miscounts the visible prompt width, causing
    the cursor to land in the wrong column after up-arrow edits.

    Args:
        colored: Prompt string with ANSI color codes.

    Returns:
        Prompt string with readline non-printing markers.
    """
    return re.sub(r"(\x1b\[[0-9;]*m)", r"\001\1\002", colored)
