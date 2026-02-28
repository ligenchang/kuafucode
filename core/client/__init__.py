"""LLM Client API integration and model routing.

Exports:
  - NIMClient - NIM API client with streaming and tool support
  - TaskType, classify_task - Model routing (fast/code/default classification)
  - Tool helpers - build_tool_system_addon, normalize_for_text_tools, inject_tool_prompt
  - Error detection - is_glm_model, is_retryable, cost_usd
"""

from .core import NIMClient, StreamEvent
from .utils import (
    TaskType,
    classify_task,
    build_tool_system_addon,
    normalize_for_text_tools,
    inject_tool_prompt,
    is_glm_model,
    is_retryable,
)
from .pricing import cost_usd

__all__ = [
    "NIMClient",
    "StreamEvent",
    "TaskType",
    "classify_task",
    "build_tool_system_addon",
    "normalize_for_text_tools",
    "inject_tool_prompt",
    "is_glm_model",
    "is_retryable",
    "cost_usd",
]
