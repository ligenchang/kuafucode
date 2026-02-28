"""LLM stream processing for a single agent turn.

:class:`LLMStream` is an async iterator that wraps a single ``stream_chat``
call, applies budget/truncation logic, and exposes processed
:class:`~nvagent.core.events.AgentEvent` objects to the caller while
accumulating its own :attr:`LLMStream.result` for the post-stream handler.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional

from nvagent.core.events import AgentEvent

if TYPE_CHECKING:
    from nvagent.core.client import NIMClient, TaskType
    from nvagent.core.safety import ResourceGuard


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class StreamResult:
    """Mutable result that :class:`LLMStream` fills as iteration proceeds."""

    assistant_content: str = ""
    """Accumulated visible (non-think) assistant text."""

    tool_calls_data: Optional[List[dict]] = None
    """Parsed tool calls, if the model issued any."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    truncated: bool = False
    """True when the stream was cut short due to a token-limit error."""

    think_budget_exceeded: bool = False
    """True when the think-token budget was exhausted before any visible content."""

    fatal_error: bool = False
    """True when the stream ended with an unrecoverable error."""

    perf: dict = field(default_factory=dict)
    """Per-turn timing keys, e.g. ``llm_first_token_turn1``, ``think_turn1``."""


# ──────────────────────────────────────────────────────────────────────────────
# Async iterator
# ──────────────────────────────────────────────────────────────────────────────


class LLMStream:
    """Async-iterable processor for a single LLM chat turn.

    Usage::

        stream = LLMStream(client, messages, schemas, task, ...)
        async for event in stream:
            yield event          # forward to TUI
        result = stream.result   # inspect accumulated data
    """

    def __init__(
        self,
        client: "NIMClient",
        messages: list,
        schemas: list,
        task: "TaskType",
        *,
        force_tool_use: bool = False,
        max_think_chars: int = 0,
        cancelled_fn: Callable[[], bool] = lambda: False,
        resource_guard: Optional["ResourceGuard"] = None,
        turn_num: int = 1,
    ) -> None:
        self._client = client
        self._messages = messages
        self._schemas = schemas
        self._task = task
        self._force_tool_use = force_tool_use
        self._max_think_chars = max_think_chars
        self._cancelled_fn = cancelled_fn
        self._guard = resource_guard
        self._turn = turn_num

        self.result = StreamResult()

    def __aiter__(self):
        return self._process()

    async def _process(self):  # noqa: C901 (complex but linear)
        """Core generator — yields AgentEvent objects, fills self.result."""
        result = self.result
        perf = result.perf

        _first_token_logged = False
        _t_llm_call = time.monotonic()
        _t_think_start: Optional[float] = None
        _think_char_count = 0
        _think_phase_done = False

        async for event in self._client.stream_chat(
            messages=self._messages,
            tools=self._schemas,
            task=self._task,
            force_tool_use=self._force_tool_use,
        ):
            if self._cancelled_fn():
                yield AgentEvent(type="error", data={"message": "Cancelled."})
                result.fatal_error = True
                return

            # ── Visible token ─────────────────────────────────────────────
            if event.type == "token":
                if not _first_token_logged:
                    _first_token_logged = True
                    perf[f"llm_first_token_turn{self._turn}"] = time.monotonic() - _t_llm_call
                if _t_think_start is not None and not _think_phase_done:
                    _think_phase_done = True
                    _think_dur = time.monotonic() - _t_think_start
                    perf[f"think_turn{self._turn}"] = _think_dur
                    perf[f"think_chars_turn{self._turn}"] = float(_think_char_count)
                    yield AgentEvent(
                        type="status",
                        data=(
                            f"⏱ think {_think_dur:.1f}s"
                            f"  {_think_char_count:,} chars"
                            f" (~{_think_char_count // 4:,} tok)"
                        ),
                    )
                result.assistant_content += event.data
                yield AgentEvent(type="token", data=event.data)
                result.total_tokens += 1
                if self._guard:
                    self._guard.update(tokens=1)

            # ── Think token ───────────────────────────────────────────────
            elif event.type == "think_token":
                if not _first_token_logged:
                    _first_token_logged = True
                    perf[f"llm_first_token_turn{self._turn}"] = time.monotonic() - _t_llm_call
                if _t_think_start is None:
                    _t_think_start = time.monotonic()
                _think_char_count += len(event.data)

                # Budget check — abort think phase early
                if (
                    self._max_think_chars > 0
                    and not result.assistant_content
                    and _think_char_count > self._max_think_chars
                ):
                    result.think_budget_exceeded = True
                    perf[f"think_chars_turn{self._turn}"] = float(_think_char_count)
                    yield AgentEvent(
                        type="status",
                        data=(
                            f"⚠ Think budget exceeded ({_think_char_count:,} chars / "
                            f"{_think_char_count // 4:,} tok) — retrying with concise directive"
                        ),
                    )
                    return

                yield AgentEvent(type="think_token", data=event.data)

            # ── Usage metadata ────────────────────────────────────────────
            elif event.type == "usage":
                if isinstance(event.data, dict):
                    actual_total = event.data.get("total_tokens", 0)
                    actual_input = event.data.get("input_tokens", 0)
                    actual_output = event.data.get("output_tokens", 0)
                    if actual_total > 0:
                        result.total_tokens = actual_total
                        result.input_tokens += actual_input
                        result.output_tokens += actual_output
                        if self._guard:
                            self._guard.update(tokens=actual_total)
                        if actual_input > 25_600:
                            yield AgentEvent(
                                type="status",
                                data=(
                                    f"⚠ Context pressure: {actual_input:,} input tokens "
                                    f"(+{actual_output:,} output). "
                                    "Consider /compact if the session grows further."
                                ),
                            )

            # ── Tool calls ────────────────────────────────────────────────
            elif event.type == "tool_calls":
                result.tool_calls_data = event.data

            # ── Pass-through status (capture API timing into perf) ────────
            elif event.type == "status":
                _smsg = str(event.data)
                if _smsg.startswith("⏱ api-connect "):
                    try:
                        perf[f"api_connect_turn{self._turn}"] = float(_smsg.split()[2].rstrip("s"))
                    except Exception:
                        pass
                elif _smsg.startswith("⏱ api-first-chunk "):
                    try:
                        perf[f"api_first_chunk_turn{self._turn}"] = float(
                            _smsg.split()[2].rstrip("s")
                        )
                    except Exception:
                        pass
                yield AgentEvent(type="status", data=event.data)

            # ── Error handling ────────────────────────────────────────────
            elif event.type == "error":
                msg = (
                    event.data.get("message", "")
                    if isinstance(event.data, dict)
                    else str(event.data)
                )
                if "truncated" in msg or "finish_reason=length" in msg:
                    result.truncated = True
                    yield AgentEvent(
                        type="status",
                        data="[truncated] Response exceeded token limit — continuing…",
                    )
                    return
                # Fatal error — propagate and signal caller to stop
                result.fatal_error = True
                yield AgentEvent(type="error", data=event.data)
                return

            # ── Done ──────────────────────────────────────────────────────
            elif event.type == "done":
                if isinstance(event.data, dict):
                    actual_total = event.data.get("total_tokens", 0)
                    if actual_total > 0:
                        result.total_tokens = actual_total
                        if self._guard:
                            self._guard.update(tokens=actual_total)
                return
