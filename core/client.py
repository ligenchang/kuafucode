"""NVIDIA NIM API client — AsyncOpenAI wrapper with streaming and model routing.

Tool calling strategy (auto-detected per account):
  1. Native tools API  (preferred, OpenAI-compatible)
  2. Prompt-based fallback  (works with any model/account)
     — triggered automatically when the API returns 404 "Function not found"
     — the model is given tool schemas as text and asked to emit
       <tool_call>{"name":..., "args":{...}}</tool_call> blocks
"""

from __future__ import annotations

import json
import re
import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI

try:
    from openai import (
        RateLimitError as _OAIRateLimit,
        APIStatusError as _OAIStatus,
        APITimeoutError as _OAITimeout,
    )
except ImportError:
    _OAIRateLimit = Exception  # type: ignore
    _OAIStatus = Exception  # type: ignore
    _OAITimeout = Exception  # type: ignore

# HTTP/2 support via h2 package — enables multiplexing and HPACK header compression.
# Falls back to HTTP/1.1 if h2 is not installed (pip install h2).
try:
    import h2  # noqa: F401

    _HTTP2_ENABLED = True
except ImportError:
    _HTTP2_ENABLED = False

# Shared httpx connection pool — reused across all requests from this process.
# max_connections=10 is ample for a single-user CLI (streaming uses one at a time).
# keepalive_expiry=60s means idle sockets stay warm between turns instead of
# doing a full TLS handshake on every request.
_HTTP_LIMITS = httpx.Limits(
    max_connections=10,
    max_keepalive_connections=5,
    keepalive_expiry=60.0,
)
_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None or _SHARED_HTTP_CLIENT.is_closed:
        _SHARED_HTTP_CLIENT = httpx.AsyncClient(
            limits=_HTTP_LIMITS,
            timeout=httpx.Timeout(connect=10.0, read=45.0, write=30.0, pool=5.0),
            http2=_HTTP2_ENABLED,
        )
    return _SHARED_HTTP_CLIENT


from nvagent.config import Config


class TaskType(Enum):
    FAST = "fast"
    DEFAULT = "default"
    CODE = "code"


FAST_KEYWORDS = {
    # Questions / explanations
    "explain",
    "what is",
    "what does",
    "what are",
    "what do",
    "summarize",
    "describe",
    "list",
    "show me",
    "tell me",
    "why is",
    "why does",
    "when did",
    "who is",
    "where is",
    "how does",
    "how do",
    "how are",
    "define",
    "show",
    # Read-only analysis — no file writes
    "check",
    "analyze",
    "analyse",
    "review",
    "find",
    "search",
    "look",
    "inspect",
    "diagnose",
    "trace",
    "investigate",
    "print",
    "display",
    "read",
    "view",
    "debug",
    "profile",
}
CODE_KEYWORDS = {
    # Explicit code-generation / write intent
    "implement",
    "build",
    "create",
    "architect",
    "design",
    "rewrite",
    "refactor",
    "from scratch",
    "complex",
    "add feature",
    "add support",
    "integrate",
    "migrate",
    "write",
    "generate",
    "scaffold",
    "setup",
    "configure",
    # Fixes that require writing code
    "fix",
    "patch",
    "resolve",
    "correct",
    # Modification intent
    "update",
    "modify",
    "change",
    "edit",
    "extend",
    "improve",
    "optimize",
}

# Pre-compiled single-pass matchers for classify_task (faster than any(kw in s ...))
_RE_FAST = re.compile(
    "|".join(re.escape(kw) for kw in sorted(FAST_KEYWORDS, key=len, reverse=True))
)
_RE_CODE = re.compile(
    "|".join(re.escape(kw) for kw in sorted(CODE_KEYWORDS, key=len, reverse=True))
)

# No regex needed for tool-call parsing — str.find() scan in _parse_tool_calls
# is O(n) with zero backtracking (see below).


@dataclass
class StreamEvent:
    type: str  # "token" | "tool_calls" | "error" | "done"
    data: object


def classify_task(prompt: str) -> TaskType:
    lower = prompt.lower()
    # CODE wins when explicit write/create/fix intent is present
    if _RE_CODE.search(lower):
        # But FAST wins back for pure read-only analysis that mentions a code word
        # only incidentally (e.g. "check why fix didn't work")
        if _RE_FAST.search(lower) and not any(
            kw in lower
            for kw in (
                "implement",
                "build",
                "create",
                "write",
                "generate",
                "refactor",
                "scaffold",
                "rewrite",
            )
        ):
            return TaskType.FAST
        return TaskType.CODE
    # Default: FAST — let the model decide what tools to call
    return TaskType.FAST


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-based tool calling helpers
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_CALL_TAG = "tool_call"
_TOOL_CALL_OPEN = f"<{_TOOL_CALL_TAG}>"
_TOOL_CALL_CLOSE = f"</{_TOOL_CALL_TAG}>"


def _build_tool_system_addon(tools: list[dict]) -> str:
    """Build the system-prompt appendix that teaches a model to call tools."""
    lines = [
        "",
        "─────────────────────────────────────────────",
        "TOOLS",
        "─────────────────────────────────────────────",
        "You may call tools by outputting a JSON block wrapped in XML tags.",
        "Use EXACTLY this format (one tool per block, call only one at a time):",
        "",
        f"{_TOOL_CALL_OPEN}",
        '{"name": "tool_name", "args": {"param": "value"}}',
        f"{_TOOL_CALL_CLOSE}",
        "",
        "After you emit a tool_call block, STOP and wait for the result.",
        "Available tools:",
        "",
    ]
    for t in tools:
        fn = t["function"]
        name = fn["name"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_strs = []
        for p, meta in params.items():
            req = " (required)" if p in required else ""
            param_strs.append(
                f"    {p}: {meta.get('type','any')}{req} — {meta.get('description','')}"
            )
        lines.append(f"  {name}: {desc}")
        lines.extend(param_strs)
        lines.append("")
    return "\n".join(lines)


def _normalize_for_text_tools(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI tool-call message format to plain-text equivalents so any
    model can understand the conversation history even without the tools API.

    Transforms:
      assistant {tool_calls: [...]}  →  assistant with <tool_call> text
      tool {tool_call_id, content}   →  user with <tool_result> text
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")

        if role == "assistant" and m.get("tool_calls"):
            # Rebuild as plain-text tool calls
            parts = [m.get("content") or ""]
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if raw else {}
                    body = json.dumps({"name": name, "args": args})
                except Exception:
                    body = json.dumps({"name": name, "args": {}})
                parts.append(f"{_TOOL_CALL_OPEN}\n{body}\n{_TOOL_CALL_CLOSE}")
            out.append({"role": "assistant", "content": "\n".join(p for p in parts if p)})

        elif role == "tool":
            # Convert to a user message so the model sees the result inline
            result = m.get("content", "")
            out.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})

        else:
            out.append(m)

    return out


def _inject_tool_prompt(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Return a copy of messages with tool instructions appended to the system message."""
    addon = _build_tool_system_addon(tools)
    patched = []
    found_system = False
    for m in messages:
        if m["role"] == "system" and not found_system:
            patched.append({**m, "content": m["content"] + "\n" + addon})
            found_system = True
        else:
            patched.append(m)
    if not found_system:
        patched.insert(0, {"role": "system", "content": addon.strip()})
    return patched


# UUID pattern: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (tool call IDs sent to NIM)
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

# Models that use the GLM thinking API (extra_body + delta.reasoning_content)
_GLM_MODEL_PREFIXES = ("z-ai/glm", "glm")


def _is_glm_model(model: str) -> bool:
    """Return True for GLM-family models that require extra_body thinking params."""
    m = model.lower()
    return any(m.startswith(p) or f"/{p.split('/')[-1]}" in m for p in _GLM_MODEL_PREFIXES)


def _is_nim_toolcall_routing_404(error: Exception) -> bool:
    """Return True for NIM 404s where a specific tool *call* UUID wasn't routed.

    Pattern: {"detail": "Function '<uuid>': Not found for account '...'"}
    This is a transient NIM backend error — the model DOES support tools.
    It should be retried, NOT trigger the text-fallback path.
    """
    if not isinstance(error, _OAIStatus):
        return False
    if getattr(error, "status_code", 0) != 404:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = str(body.get("detail", ""))
    # Must contain both a UUID and "Not found for account" to be a routing 404
    return bool(_UUID_RE.search(detail)) and "not found for account" in detail.lower()


def _is_tool_404(error: Exception) -> bool:
    """Return True when the NIM API signals the tools API is *unsupported* (404).

    Excludes UUID-keyed routing 404s (transient NIM errors on individual tool
    call invocations) — those are retryable and do NOT mean the model lacks
    native tool support.  Use _is_nim_toolcall_routing_404() to detect those.

    str(APIStatusError) only returns "Error code: 404" — the body dict is NOT
    included.  We must inspect status_code + body directly for typed SDK errors.
    """
    # Routing 404s (UUID in detail) are NOT tool-unsupported signals
    if _is_nim_toolcall_routing_404(error):
        return False
    # Typed OpenAI SDK error — check status code + body detail
    if isinstance(error, _OAIStatus):
        if getattr(error, "status_code", 0) != 404:
            return False
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            detail = str(body.get("detail", ""))
            title = str(body.get("title", ""))
            # Only flip to text fallback for endpoint-level "not supported" messages
            return (
                ("tool" in detail.lower() and "not supported" in detail.lower())
                or ("Function not found" in detail or "function not found" in detail)
                or (
                    "Not Found" in title
                    and not _UUID_RE.search(detail)
                    and ("tool" in detail.lower() or "function" in detail.lower())
                )
            )
        # Fall through to string check for plain-text 404 bodies
    # String-based fallback (httpx errors, non-SDK wrappers)
    msg = str(error)
    return "404" in msg and (
        ("Function" in msg or "function" in msg)
        and not _UUID_RE.search(msg)  # exclude UUID-keyed routing errors
    )


_RETRY_MAX = 3
_RETRY_STATUSES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> tuple[bool, float]:
    """
    Returns (should_retry, wait_seconds).
    Handles RateLimitError (429) and transient server errors (5xx).
    NIM tool-call routing 404s are handled separately in stream_chat's inner
    try/except and never reach this function.
    """
    msg = str(exc)
    # OpenAI SDK typed errors
    if isinstance(exc, _OAIRateLimit):
        return True, 5.0
    if isinstance(exc, _OAITimeout):
        return False, 0.0  # timeout = server overloaded; skip retry, switch model
    if isinstance(exc, _OAIStatus):
        code = getattr(exc, "status_code", 0)
        if code in _RETRY_STATUSES:
            return True, 2.0 if code != 429 else 5.0
    # String-based fallback (e.g. httpx errors, wrapped exceptions)
    for status in _RETRY_STATUSES:
        if str(status) in msg:
            return True, 5.0 if status == 429 else 2.0
    msg_lower = msg.lower()
    if "timed out" in msg_lower or "timeout" in msg_lower:
        return False, 0.0  # timeout = server overloaded; skip retry, switch model
    if "connection" in msg_lower:
        return True, 3.0
    return False, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


class NIMClient:
    """Async NVIDIA NIM client with streaming support."""

    def __init__(self, config: Config):
        self.config = config
        api_key = config.api_key
        if not api_key:
            raise ValueError(
                "NVIDIA API key not set. Run: nvagent config set api_key nvapi-...\n"
                "Or: export NVIDIA_API_KEY=nvapi-..."
            )
        self._client = AsyncOpenAI(
            base_url=config.api.base_url,
            api_key=api_key,
            http_client=_get_http_client(),
        )
        # Flips to False on first 404 "Function not found" → uses text fallback
        self._tool_api_supported: bool = True

    def get_model(self, task: TaskType = TaskType.DEFAULT, use_tools: bool = False) -> str:
        if task == TaskType.FAST:
            return self.config.models.fast
        elif task == TaskType.CODE:
            return self.config.models.code
        return self.config.models.default

    # ── Public entry point ────────────────────────────────────────────────────

    async def stream_chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        task: TaskType = TaskType.DEFAULT,
        force_tool_use: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        from nvagent.config import SUPPORTED_MODELS

        model = self.get_model(task, use_tools=bool(tools))

        # Build fallback chain: current model first, then remaining SUPPORTED_MODELS
        # in their listed order.  Each model gets _RETRY_MAX attempts, then we
        # move on.  At most one model switch per stream_chat() call.
        _fallbacks = [model] + [m for m in SUPPORTED_MODELS if m != model]
        _model_idx = 0
        attempt = 0

        while True:
            attempt += 1
            try:
                if tools and self._tool_api_supported:
                    try:
                        async for ev in self._stream_native(
                            messages, tools, model, force_tool_use=force_tool_use
                        ):
                            yield ev
                        return
                    except Exception as e:
                        if _is_nim_toolcall_routing_404(e) or _is_tool_404(e):
                            self._tool_api_supported = False
                            yield StreamEvent(
                                type="status",
                                data="Native tool API unavailable — switching to prompt-based mode",
                            )
                            # Fall through to text-tools block below (no raise)
                        else:
                            raise  # fall through to retry logic below

                if tools and not self._tool_api_supported:
                    async for ev in self._stream_text_tools(messages, tools, model):
                        yield ev
                    return

                # No tools — plain streaming
                async for ev in self._stream_native(messages, None, model):
                    yield ev
                return

            except Exception as exc:
                retryable, wait = _is_retryable(exc)
                if retryable and attempt < _RETRY_MAX:
                    yield StreamEvent(
                        type="status",
                        data=f"Retrying after error ({attempt}/{_RETRY_MAX - 1}): {exc}",
                    )
                    await asyncio.sleep(wait * attempt)  # exponential: 1x, 2x, …
                    continue

                # Retries exhausted — try the next fallback model if one exists
                _model_idx += 1
                if _model_idx < len(_fallbacks):
                    next_model = _fallbacks[_model_idx]
                    short_name = next_model.split("/")[-1]
                    yield StreamEvent(
                        type="status",
                        data=f"Model unavailable ({exc or 'no response'}) — switching to {short_name}",
                    )
                    model = next_model
                    attempt = 0
                    self._tool_api_supported = True  # reset: new model may support native tools
                    continue

                # All fallbacks exhausted
                yield StreamEvent(type="error", data={"message": str(exc)})
                return

    # ── Native tools (OpenAI tools API) ──────────────────────────────────────

    async def _stream_native(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        model: str,
        force_tool_use: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.agent.max_tokens,
            "temperature": self.config.agent.temperature,
            "stream": True,
            # Request token usage in the final streaming chunk.
            # Supported by OpenAI-compatible APIs; ignored if unsupported.
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "required" if force_tool_use else "auto"
        # GLM5 / GLM4-Thinking require extra_body to enable the reasoning field.
        # Without this, delta.reasoning_content is never populated.
        if _is_glm_model(model):
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "clear_thinking": False,
                }
            }

        tool_calls_acc: dict[int, dict] = {}
        finish_reason: Optional[str] = None
        think_stripper = _ThinkStripper()
        # Accumulate real token usage from the final API chunk
        _usage_input: int = 0
        _usage_output: int = 0

        _t_req = time.monotonic()
        stream = await self._client.chat.completions.create(**kwargs)
        _t_connected = time.monotonic() - _t_req
        yield StreamEvent(type="status", data=f"⏱ api-connect {_t_connected:.2f}s")
        yield StreamEvent(type="status", data="streaming")

        _first_chunk_logged = False
        async for chunk in stream:
            if not _first_chunk_logged:
                _first_chunk_logged = True
                _t_first = time.monotonic() - _t_req
                yield StreamEvent(type="status", data=f"⏱ api-first-chunk {_t_first:.2f}s")
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            finish_reason = choice.finish_reason or finish_reason

            # Capture real token counts from the usage chunk (final chunk,
            # choices may be empty but usage is populated).
            if hasattr(chunk, "usage") and chunk.usage is not None:
                _usage_input = getattr(chunk.usage, "prompt_tokens", 0) or 0
                _usage_output = getattr(chunk.usage, "completion_tokens", 0) or 0

            if delta.content is not None or getattr(delta, "reasoning_content", None) is not None:
                # GLM5/GLM4: reasoning arrives via delta.reasoning_content (separate field).
                # Other models (DeepSeek, MiniMax, Qwen): reasoning is inline in
                # delta.content wrapped in <think>...</think> — handled by ThinkStripper.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield StreamEvent(type="think_token", data=reasoning)
                if delta.content is not None:
                    visible, think = think_stripper.feed(delta.content)
                    if think:
                        yield StreamEvent(type="think_token", data=think)
                    if visible:
                        yield StreamEvent(type="token", data=visible)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_acc[idx]["args"] += tc.function.arguments

            if finish_reason == "stop":
                # Flush any chars the ThinkStripper buffered to detect partial
                # opening tags before yielding done — same as the tool_calls path.
                _flush_visible, _flush_think = think_stripper.flush()
                if _flush_think:
                    yield StreamEvent(type="think_token", data=_flush_think)
                if _flush_visible:
                    yield StreamEvent(type="token", data=_flush_visible)
                yield StreamEvent(
                    type="done",
                    data={
                        "finish": "stop",
                        "input_tokens": _usage_input,
                        "output_tokens": _usage_output,
                        "total_tokens": _usage_input + _usage_output,
                    },
                )
                return
            if finish_reason == "length":
                # Response was truncated — surface as an error so the loop
                # can inject a recovery hint rather than silently dropping data.
                yield StreamEvent(
                    type="error",
                    data={
                        "message": "Response truncated (finish_reason=length). "
                        "The output exceeded max_tokens. "
                        "Try splitting the task into smaller steps or "
                        "reduce the size of individual file writes."
                    },
                )
                return
            if finish_reason == "tool_calls":
                break

        # ── Flush any text the ThinkStripper held back ────────────────────
        # The stripper holds up to len("<think>")-1 = 6 chars to detect partial
        # opening tags. When the stream ends on tool_calls (not "stop"), flush()
        # is never called from the normal done path, so those chars are silently
        # dropped — causing the truncated "opportuni…" effect seen in the TUI.
        _flush_visible, _flush_think = think_stripper.flush()
        if _flush_think:
            yield StreamEvent(type="think_token", data=_flush_think)
        if _flush_visible:
            yield StreamEvent(type="token", data=_flush_visible)

        if tool_calls_acc:
            calls = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc["args"]}
                calls.append(
                    {
                        "id": tc["id"] or str(uuid.uuid4()),
                        "name": tc["name"],
                        "args": args,
                        "args_raw": tc["args"],
                    }
                )
            yield StreamEvent(type="tool_calls", data=calls)
            # Emit a usage event after tool_calls so the agent loop can
            # update its token budget with real numbers.
            if _usage_input or _usage_output:
                yield StreamEvent(
                    type="usage",
                    data={
                        "input_tokens": _usage_input,
                        "output_tokens": _usage_output,
                        "total_tokens": _usage_input + _usage_output,
                    },
                )
        else:
            # No tool calls and no explicit finish_reason=stop/length — stream
            # ended unexpectedly (empty response or unknown finish_reason).
            if not _first_chunk_logged:
                # Zero chunks received — the connection succeeded but the model
                # returned nothing.  Surface as a transient error so the retry
                # logic in loop.py can re-run the turn.
                yield StreamEvent(type="error", data={"message": ""})
            else:
                # Some chunks arrived but the stream terminated without a
                # recognised finish_reason.  Treat like a normal stop.
                yield StreamEvent(
                    type="done",
                    data={
                        "finish": finish_reason or "unknown",
                        "input_tokens": _usage_input,
                        "output_tokens": _usage_output,
                        "total_tokens": _usage_input + _usage_output,
                    },
                )

    # ── Prompt-based tool calling (text parsing fallback) ─────────────────────

    async def _stream_text_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream without the tools API.  Tool schemas are embedded in the system
        prompt.  The model emits <tool_call>{json}</tool_call> blocks which
        we parse and yield as StreamEvent(type="tool_calls").
        """
        patched = _inject_tool_prompt(_normalize_for_text_tools(messages), tools)
        kwargs: dict = {
            "model": model,
            "messages": patched,
            "max_tokens": self.config.agent.max_tokens,
            "temperature": self.config.agent.temperature,
            "stream": True,
        }
        if _is_glm_model(model):
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "clear_thinking": False,
                }
            }

        full_text = ""
        think_stripper = _ThinkStripper()
        buffering_call = False  # True once we see _TOOL_CALL_OPEN

        try:
            _t_req = time.monotonic()
            stream = await self._client.chat.completions.create(**kwargs)
            _t_connected = time.monotonic() - _t_req
            yield StreamEvent(type="status", data=f"⏱ api-connect {_t_connected:.2f}s")
            yield StreamEvent(type="status", data="streaming")

            for_done = False
            _first_chunk_logged = False
            async for chunk in stream:
                if not _first_chunk_logged:
                    _first_chunk_logged = True
                    _t_first = time.monotonic() - _t_req
                    yield StreamEvent(type="status", data=f"⏱ api-first-chunk {_t_first:.2f}s")
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                token = choice.delta.content or ""
                # GLM5: reasoning arrives via reasoning_content, not content
                reasoning = getattr(choice.delta, "reasoning_content", None)
                if reasoning:
                    yield StreamEvent(type="think_token", data=reasoning)
                full_text += token

                if not buffering_call:
                    if _TOOL_CALL_OPEN in full_text:
                        # About to buffer a tool call — tell the UI
                        buffering_call = True
                        yield StreamEvent(type="status", data="preparing tool call…")
                    else:
                        # Normal text — split think vs visible
                        visible, think = think_stripper.feed(token)
                        if think:
                            yield StreamEvent(type="think_token", data=think)
                        if visible:
                            yield StreamEvent(type="token", data=visible)
                # else: silently accumulate the tool_call block

                # Once we have a complete tool_call block, stop streaming
                if buffering_call and _TOOL_CALL_CLOSE in full_text:
                    break

                if choice.finish_reason == "stop":
                    for_done = True
                    break

            # After streaming, flush any remaining buffered text
            visible, think = think_stripper.flush()
            if think:
                yield StreamEvent(type="think_token", data=think)
            if visible:
                yield StreamEvent(type="token", data=visible)
            if for_done:
                yield StreamEvent(type="done", data={"finish": "stop"})
                return

        except Exception as e:
            yield StreamEvent(type="error", data={"message": str(e)})
            return

        # Parse all <tool_call> blocks from accumulated text
        calls = _parse_tool_calls(full_text)
        if calls:
            yield StreamEvent(type="tool_calls", data=calls)
        else:
            # Model responded with text but no valid tool call — treat as done
            # Emit whatever text wasn't streamed yet (the buffered part after the tag appeared)
            tag_pos = full_text.find(_TOOL_CALL_OPEN)
            if tag_pos == -1:
                pass  # all tokens already sent
            else:
                # Text before the tag was not emitted (we broke early but tag_pos > 0 is already sent)
                pass
            yield StreamEvent(type="done", data={"finish": "stop"})

    # ── Utility ───────────────────────────────────────────────────────────────

    async def get_models(self) -> list[str]:
        try:
            models = await self._client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            return [f"Error: {e}"]


# ─────────────────────────────────────────────────────────────────────────────
# Text tool-call parser
# ─────────────────────────────────────────────────────────────────────────────


class _ThinkStripper:
    """
    Splits a streaming token feed into (visible, think) text.
    `feed()` returns a tuple (visible: str, think: str) —
    callers can display think content separately (e.g. dimmed).

    Supports multiple model think-tag formats:
      - <think> / </think>          (DeepSeek-R1, MiniMax M2, Qwen-QwQ)
      - <|thinking|> / <|/thinking|>  (GLM-5, GLM-4 Thinking)
    """

    # Ordered list of (open_tag, close_tag) pairs — first match wins per stream.
    _TAG_PAIRS: list[tuple[str, str]] = [
        ("<think>", "</think>"),
        ("<|thinking|>", "<|/thinking|>"),
    ]
    # Combined regex for fast single-chunk complete blocks
    _RE = re.compile(
        r"<think>(.*?)</think>|<\|thinking\|>(.*?)<\|/thinking\|>",
        re.DOTALL,
    )

    def __init__(self) -> None:
        self._in_think = False
        self._held = ""  # buffered partial-tag content
        # Active tag pair — detected from first open tag seen in the stream.
        # Starts as None; set on first <think> or <|thinking|> occurrence.
        self._open_tag: str = "<think>"
        self._close_tag: str = "</think>"
        self._tag_detected: bool = False

    def _detect_tags(self, text: str) -> None:
        """Set active tag pair from the first recognised open tag in *text*."""
        for open_t, close_t in self._TAG_PAIRS:
            if open_t in text:
                self._open_tag = open_t
                self._close_tag = close_t
                self._tag_detected = True
                return

    def feed(self, token: str) -> tuple[str, str]:
        """
        Return (visible_text, think_text).
        Exactly one of the two will be non-empty per call (or both empty).
        """
        # Detect which tag format this model uses (once)
        if not self._tag_detected and not self._in_think:
            self._detect_tags(token)

        open_tag = self._open_tag
        close_tag = self._close_tag

        # Fast path: complete block arrives in one token (common with minimax/glm)
        if not self._in_think and open_tag in token and close_tag in token:
            think = ""
            visible = token
            for m in self._RE.finditer(token):
                # group(1) = <think> content, group(2) = <|thinking|> content
                think += m.group(1) or m.group(2) or ""
            visible = self._RE.sub("", token)
            return visible, think

        self._held += token
        visible = ""
        think = ""

        while True:
            if self._in_think:
                close = self._held.find(close_tag)
                if close == -1:
                    # Entirely inside think — emit everything as think content
                    think += self._held
                    self._held = ""
                    break
                think += self._held[:close]
                self._held = self._held[close + len(close_tag) :]
                self._in_think = False
                continue
            else:
                open_pos = self._held.find(open_tag)
                if open_pos == -1:
                    # No tag — hold back enough chars to detect partial opening tag
                    max_hold = max(len(open_tag), len(close_tag))
                    safe = max(0, len(self._held) - max_hold + 1)
                    visible += self._held[:safe]
                    self._held = self._held[safe:]
                    break
                visible += self._held[:open_pos]
                self._held = self._held[open_pos + len(open_tag) :]
                self._in_think = True

        return visible, think

    def flush(self) -> tuple[str, str]:
        """Flush any held partial content (call at end of stream)."""
        held, self._held = self._held, ""
        if self._in_think:
            return "", held  # held content is think text
        return held, ""  # held content is visible text


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract all <tool_call>{json}</tool_call> blocks from model output.

    Uses str.find() instead of a DOTALL regex so there is no backtracking:
    worst-case is a single O(n) linear scan per block, executed entirely in C.
    """
    calls: list[dict] = []
    pos = 0
    open_len = len(_TOOL_CALL_OPEN)  # len("<tool_call>")
    close_len = len(_TOOL_CALL_CLOSE)  # len("</tool_call>")
    while True:
        tag_start = text.find(_TOOL_CALL_OPEN, pos)
        if tag_start == -1:
            break
        json_start = tag_start + open_len
        tag_end = text.find(_TOOL_CALL_CLOSE, json_start)
        if tag_end == -1:
            break
        raw = text[json_start:tag_end].strip()
        pos = tag_end + close_len
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get("name", "")
        args = obj.get("args", obj.get("arguments", obj.get("parameters", {})))
        if not name:
            continue
        calls.append(
            {
                "id": str(uuid.uuid4()),
                "name": name,
                "args": args,
                "args_raw": json.dumps(args),
            }
        )
    return calls
