"""nvagent — Claude Code-style terminal REPL.

Writes directly to stdout (ANSI colors). All output is native terminal text:
fully scrollable, selectable, and copyable like any shell command.

No TUI framework — no custom canvas — no copy/paste friction.
"""

from __future__ import annotations

from nvagent.tui.app.repl import NVAgentREPL, launch_tui

__all__ = [
    "NVAgentREPL",
    "launch_tui",
]
