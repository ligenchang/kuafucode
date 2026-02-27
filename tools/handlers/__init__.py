"""
Handler base class and registry.

Each handler class owns a slice of the ToolExecutor's behaviour.
All handlers receive a shared ToolContext that holds workspace state.
"""

from __future__ import annotations

from nvagent.tools.context import ToolContext


class BaseHandler:
    """Base class for all tool handlers.

    Subclasses receive the shared *ctx* and implement async handler methods
    that formerly lived on ToolExecutor.
    """

    __slots__ = ("ctx",)

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
