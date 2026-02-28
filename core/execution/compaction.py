"""Context compaction: summarise old conversation messages to keep context small.

Called automatically when session.messages exceeds _COMPACT_MSG_THRESHOLD,
and also available as Agent.compact() for manual /compact invocations.
"""

from __future__ import annotations

from nvagent.core.state import Session
from nvagent.core.client import NIMClient, TaskType

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

COMPACT_MSG_THRESHOLD = 40  # compact when history exceeds this many messages
COMPACT_KEEP_RECENT = 8  # always keep this many recent messages uncompacted


# ─────────────────────────────────────────────────────────────────────────────
# Auto-compaction (called by the agent loop)
# ─────────────────────────────────────────────────────────────────────────────


async def compact_history(
    client: NIMClient,
    session: Session,
    task_type: TaskType,
) -> None:
    """Summarise old messages in-place to keep context small.

    No-ops when the message count is below *COMPACT_MSG_THRESHOLD*.
    Skips raw tool-result messages (role=tool) because their format confuses
    the summariser and inflates the history.
    """
    msgs = session.messages
    if len(msgs) <= COMPACT_MSG_THRESHOLD:
        return

    keep_start = len(msgs) - COMPACT_KEEP_RECENT
    old_msgs = msgs[:keep_start]
    keep_msgs = msgs[keep_start:]

    history_text = "\n\n".join(
        f"{m['role'].upper()}: {m.get('content') or ''}"
        for m in old_msgs
        if m.get("content") and m.get("role") in ("user", "assistant")
    )
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "You are a concise summariser. Summarise the conversation history "
                "below into a tight bullet-point summary (≤200 words) preserving "
                "all important technical decisions, file paths, and context."
            ),
        },
        {"role": "user", "content": history_text[:12000]},
    ]

    summary_text = ""
    async for ev in client.stream_chat(messages=summary_prompt, tools=[], task=task_type):
        if ev.type == "token":
            summary_text += ev.data
        elif ev.type in ("done", "error"):
            break

    if summary_text.strip():
        summary_msg = {
            "role": "assistant",
            "content": (
                f"[Conversation summary — {len(old_msgs)} earlier messages compacted]\n"
                + summary_text.strip()
            ),
        }
        session.messages = [summary_msg] + keep_msgs
