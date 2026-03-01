"""NVIDIA NIM API client — AsyncOpenAI wrapper with streaming.

Tool calling strategy (auto-detected):
  1. Native tools API (preferred, OpenAI-compatible)
  2. Prompt-based fallback (triggered on 404 "Function not found")
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

try:
    from openai import APIStatusError as _OAIStatus
    from openai import APITimeoutError as _OAITimeout
    from openai import RateLimitError as _OAIRateLimit
except ImportError:
    _OAIRateLimit = _OAIStatus = _OAITimeout = Exception  # type: ignore

try:
    import h2  # noqa: F401
    _HTTP2_ENABLED = True
except ImportError:
    _HTTP2_ENABLED = False

_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None or _SHARED_HTTP_CLIENT.is_closed:
        _SHARED_HTTP_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=60.0),
            timeout=httpx.Timeout(connect=10.0, read=45.0, write=30.0, pool=5.0),
            http2=_HTTP2_ENABLED,
        )
    return _SHARED_HTTP_CLIENT


@dataclass
class StreamEvent:
    type: str  # "token" | "think_token" | "tool_calls" | "usage" | "status" | "error" | "done"
    data: object


# ── Retry / error detection ────────────────────────────────────────────────────

_RETRY_MAX = 3
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _is_retryable(exc: Exception) -> tuple[bool, float]:
    if isinstance(exc, _OAIRateLimit):
        return True, 5.0
    if isinstance(exc, _OAITimeout):
        return False, 0.0
    if isinstance(exc, _OAIStatus):
        code = getattr(exc, "status_code", 0)
        if code in _RETRY_STATUSES:
            return True, 2.0 if code != 429 else 5.0
    msg = str(exc).lower()
    for status in _RETRY_STATUSES:
        if str(status) in msg:
            return True, 5.0 if status == 429 else 2.0
    if "connection" in msg:
        return True, 3.0
    return False, 0.0


def _is_tool_404(error: Exception) -> bool:
    """True when the API signals native tools are unsupported (not a transient routing 404)."""
    if isinstance(error, _OAIStatus):
        if getattr(error, "status_code", 0) != 404:
            return False
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            detail = str(body.get("detail", ""))
            # Exclude transient UUID-keyed routing errors
            if _UUID_RE.search(detail) and "not found for account" in detail.lower():
                return False
            return "function not found" in detail.lower() or (
                "tool" in detail.lower() and "not supported" in detail.lower()
            )
    return False


def _is_glm_model(model: str) -> bool:
    m = model.lower()
    return "glm" in m


# ── Prompt-based tool calling helpers ─────────────────────────────────────────

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _build_tool_system_addon(tools: list[dict]) -> str:
    lines = [
        "", "─────────────────────────────────────────────",
        "TOOLS", "─────────────────────────────────────────────",
        "You may call tools by outputting a JSON block wrapped in XML tags.",
        f"Format: {_TOOL_CALL_OPEN}{{\"name\": \"tool_name\", \"args\": {{\"param\": \"value\"}}}}{_TOOL_CALL_CLOSE}",
        "After emitting a tool_call block, STOP and wait for the result.",
        "Available tools:", "",
    ]
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        lines.append(f"  {fn['name']}: {fn.get('description', '')}")
        for p, meta in params.items():
            req = " (required)" if p in required else ""
            lines.append(f"    {p}: {meta.get('type', 'any')}{req} — {meta.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


def _normalize_for_text_tools(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            parts = [m.get("content") or ""]
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                    body = json.dumps({"name": fn.get("name", ""), "args": args})
                except Exception:
                    body = json.dumps({"name": fn.get("name", ""), "args": {}})
                parts.append(f"{_TOOL_CALL_OPEN}\n{body}\n{_TOOL_CALL_CLOSE}")
            out.append({"role": "assistant", "content": "\n".join(p for p in parts if p)})
        elif role == "tool":
            out.append({"role": "user", "content": f"<tool_result>\n{m.get('content', '')}\n</tool_result>"})
        else:
            out.append(m)
    return out


def _inject_tool_prompt(messages: list[dict], tools: list[dict]) -> list[dict]:
    addon = _build_tool_system_addon(tools)
    patched, found = [], False
    for m in messages:
        if m["role"] == "system" and not found:
            patched.append({**m, "content": m["content"] + "\n" + addon})
            found = True
        else:
            patched.append(m)
    if not found:
        patched.insert(0, {"role": "system", "content": addon.strip()})
    return patched


def _parse_tool_calls(text: str) -> list[dict]:
    calls, pos = [], 0
    open_len, close_len = len(_TOOL_CALL_OPEN), len(_TOOL_CALL_CLOSE)
    while True:
        start = text.find(_TOOL_CALL_OPEN, pos)
        if start == -1:
            break
        end = text.find(_TOOL_CALL_CLOSE, start + open_len)
        if end == -1:
            break
        raw = text[start + open_len:end].strip()
        pos = end + close_len
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get("name", "")
        args = obj.get("args", obj.get("arguments", {}))
        if name:
            calls.append({"id": str(uuid.uuid4()), "name": name, "args": args, "args_raw": json.dumps(args)})
    return calls


class _ThinkStripper:
    """Splits streaming tokens into (visible, think) text."""
    _TAG_PAIRS = [("<think>", "</think>"), ("<|thinking|>", "<|/thinking|>")]
    _RE = re.compile(r"<think>(.*?)</think>|<\|thinking\|>(.*?)<\|/thinking\|>", re.DOTALL)

    def __init__(self) -> None:
        self._in_think = False
        self._held = ""
        self._open_tag = "<think>"
        self._close_tag = "</think>"
        self._tag_detected = False

    def feed(self, token: str) -> tuple[str, str]:
        if not self._tag_detected and not self._in_think:
            for open_t, close_t in self._TAG_PAIRS:
                if open_t in token:
                    self._open_tag, self._close_tag = open_t, close_t
                    self._tag_detected = True
                    break

        open_tag, close_tag = self._open_tag, self._close_tag

        if not self._in_think and open_tag in token and close_tag in token:
            think = "".join(m.group(1) or m.group(2) or "" for m in self._RE.finditer(token))
            return self._RE.sub("", token), think

        self._held += token
        visible = think = ""

        while True:
            if self._in_think:
                close = self._held.find(close_tag)
                if close == -1:
                    think += self._held
                    self._held = ""
                    break
                think += self._held[:close]
                self._held = self._held[close + len(close_tag):]
                self._in_think = False
            else:
                open_pos = self._held.find(open_tag)
                if open_pos == -1:
                    max_hold = max(len(open_tag), len(close_tag))
                    safe = max(0, len(self._held) - max_hold + 1)
                    visible += self._held[:safe]
                    self._held = self._held[safe:]
                    break
                visible += self._held[:open_pos]
                self._held = self._held[open_pos + len(open_tag):]
                self._in_think = True

        return visible, think

    def flush(self) -> tuple[str, str]:
        held, self._held = self._held, ""
        return ("", held) if self._in_think else (held, "")


# ── NIMClient ──────────────────────────────────────────────────────────────────

class NIMClient:
    """Async NVIDIA NIM client with streaming and tool-calling support."""

    def __init__(self, config) -> None:
        self.config = config
        if not config.api_key:
            raise ValueError(
                "NVIDIA API key not set. Run: nvagent config set api_key nvapi-...\n"
                "Or: export NVIDIA_API_KEY=nvapi-..."
            )
        self._client = AsyncOpenAI(
            base_url=config.api.base_url,
            api_key=config.api_key,
            http_client=_get_http_client(),
        )
        self._tool_api_supported = True

    def get_model(self) -> str:
        return self.config.models.default

    async def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        force_tool_use: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        from nvagent.config import SUPPORTED_MODELS

        model = self.get_model()
        fallbacks = [model] + [m for m in SUPPORTED_MODELS if m != model]
        model_idx = 0
        attempt = 0

        while True:
            attempt += 1
            try:
                if tools and self._tool_api_supported:
                    try:
                        async for ev in self._stream_native(messages, tools, model, force_tool_use):
                            yield ev
                        return
                    except Exception as e:
                        if _is_tool_404(e):
                            self._tool_api_supported = False
                            yield StreamEvent(type="status", data="Native tool API unavailable — switching to prompt-based mode")
                        else:
                            raise

                if tools and not self._tool_api_supported:
                    async for ev in self._stream_text_tools(messages, tools, model):
                        yield ev
                    return

                async for ev in self._stream_native(messages, None, model):
                    yield ev
                return

            except Exception as exc:
                retryable, wait = _is_retryable(exc)
                if retryable and attempt < _RETRY_MAX:
                    yield StreamEvent(type="status", data=f"Retrying ({attempt}/{_RETRY_MAX - 1}): {exc}")
                    await asyncio.sleep(wait * attempt)
                    continue

                model_idx += 1
                if model_idx < len(fallbacks):
                    model = fallbacks[model_idx]
                    yield StreamEvent(type="status", data=f"Switching to {model.split('/')[-1]}")
                    attempt = 0
                    self._tool_api_supported = True
                    continue

                yield StreamEvent(type="error", data={"message": str(exc)})
                return

    async def _stream_native(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str,
        force_tool_use: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.agent.max_tokens,
            "temperature": self.config.agent.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "required" if force_tool_use else "auto"
        if _is_glm_model(model):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}

        tool_calls_acc: dict[int, dict] = {}
        finish_reason: str | None = None
        stripper = _ThinkStripper()
        input_tokens = output_tokens = 0

        t_req = time.monotonic()
        stream = await self._client.chat.completions.create(**kwargs)
        yield StreamEvent(type="status", data=f"⏱ connect {time.monotonic() - t_req:.2f}s")

        first = True
        async for chunk in stream:
            if first:
                first = False
                yield StreamEvent(type="status", data=f"⏱ first-chunk {time.monotonic() - t_req:.2f}s")

            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            finish_reason = choice.finish_reason or finish_reason

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield StreamEvent(type="think_token", data=reasoning)
            if delta.content is not None:
                visible, think = stripper.feed(delta.content)
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
                fv, ft = stripper.flush()
                if ft:
                    yield StreamEvent(type="think_token", data=ft)
                if fv:
                    yield StreamEvent(type="token", data=fv)
                yield StreamEvent(type="done", data={"finish": "stop", "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens})
                return

            if finish_reason == "length":
                yield StreamEvent(type="error", data={"message": "Response truncated (finish_reason=length). Try splitting the task into smaller steps."})
                return

            if finish_reason == "tool_calls":
                break

        fv, ft = stripper.flush()
        if ft:
            yield StreamEvent(type="think_token", data=ft)
        if fv:
            yield StreamEvent(type="token", data=fv)

        if tool_calls_acc:
            calls = []
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc["args"]}
                calls.append({"id": tc["id"] or str(uuid.uuid4()), "name": tc["name"], "args": args, "args_raw": tc["args"]})
            yield StreamEvent(type="tool_calls", data=calls)
            if input_tokens or output_tokens:
                yield StreamEvent(type="usage", data={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens})
        elif not first:
            yield StreamEvent(type="done", data={"finish": finish_reason or "unknown", "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens})
        else:
            yield StreamEvent(type="error", data={"message": "Empty response from API"})

    async def _stream_text_tools(self, messages: list[dict], tools: list[dict], model: str) -> AsyncIterator[StreamEvent]:
        patched = _inject_tool_prompt(_normalize_for_text_tools(messages), tools)
        kwargs: dict = {
            "model": model,
            "messages": patched,
            "max_tokens": self.config.agent.max_tokens,
            "temperature": self.config.agent.temperature,
            "stream": True,
        }
        if _is_glm_model(model):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}

        full_text = ""
        stripper = _ThinkStripper()
        buffering = False

        try:
            t_req = time.monotonic()
            stream = await self._client.chat.completions.create(**kwargs)
            yield StreamEvent(type="status", data=f"⏱ connect {time.monotonic() - t_req:.2f}s")

            for_done = False
            first = True
            async for chunk in stream:
                if first:
                    first = False
                    yield StreamEvent(type="status", data=f"⏱ first-chunk {time.monotonic() - t_req:.2f}s")
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                token = choice.delta.content or ""
                reasoning = getattr(choice.delta, "reasoning_content", None)
                if reasoning:
                    yield StreamEvent(type="think_token", data=reasoning)
                full_text += token

                if not buffering:
                    if _TOOL_CALL_OPEN in full_text:
                        buffering = True
                        yield StreamEvent(type="status", data="preparing tool call…")
                    else:
                        visible, think = stripper.feed(token)
                        if think:
                            yield StreamEvent(type="think_token", data=think)
                        if visible:
                            yield StreamEvent(type="token", data=visible)

                if buffering and _TOOL_CALL_CLOSE in full_text:
                    break
                if choice.finish_reason == "stop":
                    for_done = True
                    break

            fv, ft = stripper.flush()
            if ft:
                yield StreamEvent(type="think_token", data=ft)
            if fv:
                yield StreamEvent(type="token", data=fv)
            if for_done:
                yield StreamEvent(type="done", data={"finish": "stop"})
                return
        except Exception as e:
            yield StreamEvent(type="error", data={"message": str(e)})
            return

        calls = _parse_tool_calls(full_text)
        if calls:
            yield StreamEvent(type="tool_calls", data=calls)
        else:
            yield StreamEvent(type="done", data={"finish": "stop"})

    async def get_models(self) -> list[str]:
        try:
            models = await self._client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            return [f"Error: {e}"]
