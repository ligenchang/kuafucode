"""Helper functions for NIM API client."""

from __future__ import annotations

import json
import re
from enum import Enum

try:
    from openai import (
        APIStatusError as _OAIStatus,
    )
except ImportError:
    _OAIStatus = Exception  # type: ignore


class TaskType(Enum):
    """Task classification for model routing."""

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

# Pre-compiled single-pass matchers for classify_task  (faster than any(kw in s ...))
_RE_FAST = re.compile(
    "|".join(re.escape(kw) for kw in sorted(FAST_KEYWORDS, key=len, reverse=True))
)
_RE_CODE = re.compile(
    "|".join(re.escape(kw) for kw in sorted(CODE_KEYWORDS, key=len, reverse=True))
)

# Models that use the GLM thinking API (extra_body + delta.reasoning_content)
_GLM_MODEL_PREFIXES = ("z-ai/glm", "glm")

# UUID pattern: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (tool call IDs sent to NIM)
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

# Tool calling markers
_TOOL_CALL_TAG = "tool_call"
_TOOL_CALL_OPEN = f"<{_TOOL_CALL_TAG}>"
_TOOL_CALL_CLOSE = f"</{_TOOL_CALL_TAG}>"


def classify_task(prompt: str) -> TaskType:
    """Classify a prompt as FAST, CODE, or DEFAULT task type for model routing."""
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


def build_tool_system_addon(tools: list[dict]) -> str:
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


def normalize_for_text_tools(messages: list[dict]) -> list[dict]:
    """Convert OpenAI tool-call message format to plain-text equivalents.

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


def inject_tool_prompt(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Return messages with tool instructions appended to the system message."""
    addon = build_tool_system_addon(tools)
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


def is_glm_model(model: str) -> bool:
    """Return True for GLM-family models that require thinking params."""
    m = model.lower()
    return any(m.startswith(p) or f"/{p.split('/')[-1]}" in m for p in _GLM_MODEL_PREFIXES)


def is_nim_toolcall_routing_404(error: Exception) -> bool:
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


def is_tool_404(error: Exception) -> bool:
    """Return True when the NIM API signals the tools API is *unsupported* (404)."""
    if not isinstance(error, _OAIStatus):
        return False
    if getattr(error, "status_code", 0) != 404:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = str(body.get("detail", ""))
    # Must be about "chat/completions" endpoint not having "tools" param
    return "tools" in detail.lower() and "not supported" in detail.lower()


def is_retryable(exc: Exception) -> tuple[bool, float]:
    """Return (should_retry, backoff_seconds) for an exception."""
    # Rate limit: exponential backoff via the exception's retry_after attribute
    if isinstance(exc, _OAIStatus) and getattr(exc, "status_code", 0) == 429:
        retry_after = getattr(exc, "response", None)
        if retry_after and hasattr(retry_after, "headers"):
            try:
                secs = float(retry_after.headers.get("retry-after", "1.0"))
                return True, secs
            except ValueError:
                return True, 1.0
        return True, 1.0

    # Timeout, transient 5xx errors
    if isinstance(exc, _OAIStatus) and 500 <= getattr(exc, "status_code", 0) < 600:
        return True, 1.0

    return False, 0.0
