"""Backward-compatibility shim for nvagent.core.loop.

The implementation has been split into focused modules:
  core/events.py     — AgentEvent dataclass
  core/pricing.py    — cost_usd() and price table
  core/compaction.py — compact_history() and thresholds
  core/feedback.py   — classify_tool_error() and RETRY_HINTS
  core/agent.py      — Agent class

All public names remain importable from this module so existing code
(cli.py, tui/app/repl.py, tests) continues to work without changes.
"""

from nvagent.core.events import AgentEvent
from nvagent.core.agent import Agent, TOOL_SCHEMAS_BY_NAME

__all__ = ["Agent", "AgentEvent", "TOOL_SCHEMAS_BY_NAME"]
