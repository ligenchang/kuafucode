"""nvagent Agent — orchestrates the agentic coding session.

Simplified agent loop:
  1. Build system prompt (project context + memory)
  2. Optional git checkpoint
  3. Stream LLM response
  4. Execute tool calls
  5. Repeat until done or MAX_TURNS reached
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from nvagent.config import Config
from nvagent.core.client import NIMClient
from nvagent.core.context import build_system_prompt
from nvagent.core.mcp import McpClient
from nvagent.core.safety import GitCheckpointer, LoopDetector, ResourceGuard
from nvagent.core.session import Session, SessionStore, read_memory
from nvagent.tools import ToolExecutor

logger = logging.getLogger(__name__)


_TOOL_RESULT_HISTORY_CAP = 300  # chars to keep from old tool results in history


def _prune_old_tool_results(messages: list[dict], keep_last_n_full: int = 2) -> None:
    """Truncate tool result content in older turns to avoid context bloat.

    Keeps the most recent `keep_last_n_full` tool result messages at full length;
    older ones are capped at _TOOL_RESULT_HISTORY_CAP chars with a truncation note.
    This runs in-place on the live `messages` list (NOT session.messages).
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    to_truncate = tool_indices[:-keep_last_n_full] if len(tool_indices) > keep_last_n_full else []
    for idx in to_truncate:
        content = messages[idx].get("content", "")
        if isinstance(content, str) and len(content) > _TOOL_RESULT_HISTORY_CAP:
            messages[idx] = {
                **messages[idx],
                "content": content[:_TOOL_RESULT_HISTORY_CAP] + f"\n… [truncated {len(content) - _TOOL_RESULT_HISTORY_CAP} chars from history]",
            }


@dataclass
class AgentEvent:
    type: str  # "token" | "think_token" | "tool_start" | "tool_result" | "status" | "error" | "done" | "safety_violation" | "files_changed"
    data: object = None


# Tool aliases for common hallucinated names
_TOOL_ALIASES: dict[str, str] = {
    "execute_bash": "run_command", "bash": "run_command", "shell": "run_command",
    "run_bash": "run_command", "execute_command": "run_command",
    "str_replace": "str_replace_editor", "create_file": "write_file",
    "write_files": "write_file", "read_files": "read_file", "view_file": "read_file",
    "list_files": "list_dir", "ls": "list_dir",
}

# Words that indicate simple conversational queries — skip tool loop
_SIMPLE_QUERY_STARTS = frozenset({
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no",
    "what is", "what's", "explain", "describe", "help",
})


class Agent:
    """The nvagent coding agent.

    Create once per workspace session; call run() for each user message.
    run() is an async generator yielding AgentEvent objects.
    """

    MAX_TURNS = 20

    def __init__(
        self,
        config: Config,
        workspace: Path,
        session: Session,
        session_store: SessionStore,
        confirm_fn: Callable[[str, str], Awaitable[bool]] | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.session = session
        self.session_store = session_store

        self.client = NIMClient(config)
        self._mcp_client = McpClient(config.mcp.servers)
        self._mcp_started = False

        self._stream_fn: Callable[[str], None] | None = None

        self.tools = ToolExecutor(
            workspace,
            config.agent.max_file_bytes,
            confirm_fn=confirm_fn,
            safe_mode=config.agent.safe_mode,
            dry_run=config.agent.dry_run,
            mcp_client=self._mcp_client,
        )

        self._cancelled = False
        self.correction_queue: asyncio.Queue[str] = asyncio.Queue()

        # Accumulated token counts across turns
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Safety
        self.git = GitCheckpointer(workspace)
        self.loop_detector = LoopDetector(
            max_identical=config.safety.max_identical_calls,
            window=config.safety.loop_window,
        )
        self.resource_guard = ResourceGuard(config.safety)

        # Cached system prompt (rebuilt when memory changes)
        self._system_prompt: str | None = None

    def cancel(self) -> None:
        self._cancelled = True

    def mcp_server_status(self) -> list[dict]:
        """Return status of configured MCP servers (empty if none configured)."""
        if self._mcp_client and self._configs_have_mcp():
            return self._mcp_client.server_status()
        return []

    def _configs_have_mcp(self) -> bool:
        return bool(self.config.mcp.servers)

    def set_stream_fn(self, fn: Callable[[str], None] | None) -> None:
        """Set a callback for live tool output (e.g. run_command streaming)."""
        self._stream_fn = fn
        self.tools.stream_fn = fn

    def _save_session(self) -> None:
        threading.Thread(
            target=self.session_store.save_session,
            args=(self.session,),
            daemon=True,
        ).start()

    def _build_system_prompt(self) -> str:
        memory = read_memory(self.workspace)
        return build_system_prompt(self.workspace, self.config, memory)

    async def compact(self) -> str:
        """Summarise old messages to reduce context size."""
        msgs = self.session.messages
        if len(msgs) < 8:
            return "Nothing to compact — history is too short."
        keep_start = max(0, len(msgs) - 8)
        old_msgs = msgs[:keep_start]
        keep_msgs = msgs[keep_start:]
        history_text = "\n\n".join(
            f"{m['role'].upper()}: {m.get('content') or ''}"
            for m in old_msgs
            if m.get("content") and m.get("role") in ("user", "assistant")
        )
        summary_prompt = [
            {"role": "system", "content": "Summarise the conversation history below in ≤200 words, preserving key technical decisions, file paths, and context."},
            {"role": "user", "content": history_text[:12000]},
        ]
        summary = ""
        async for ev in self.client.stream_chat(messages=summary_prompt, tools=[]):
            if ev.type == "token":
                summary += ev.data
            elif ev.type in ("done", "error"):
                break

        if not summary.strip():
            return "Compaction failed — model returned no summary."

        original = len(msgs)
        self.session.messages = [
            {"role": "assistant", "content": f"[Summary — {len(old_msgs)} messages compacted]\n{summary.strip()}"}
        ] + keep_msgs
        self._save_session()
        return f"Compacted {original} → {len(self.session.messages)} messages."

    async def stop(self) -> None:
        if self._mcp_started and self._mcp_client:
            try:
                await self._mcp_client.stop()
            except Exception:
                pass
            self._mcp_started = False

    async def run(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """Process a user message, yielding AgentEvent objects."""
        self._cancelled = False
        self.resource_guard.start()
        self.loop_detector.reset()

        # Start MCP servers lazily
        if not self._mcp_started and self._mcp_client:
            try:
                await self._mcp_client.start()
            except Exception as e:
                logger.warning("MCP startup failed: %s", e)
            self._mcp_started = True

        active_schemas = self.tools.active_schemas
        schemas_by_name = {s["function"]["name"]: s for s in active_schemas}

        if self.config.agent.dry_run:
            yield AgentEvent(type="status", data="[DRY RUN] No files will be written or commands executed.")

        self.session.messages.append({"role": "user", "content": user_message})

        # Build system prompt
        yield AgentEvent(type="status", data="Building context...")
        try:
            system_prompt = self._build_system_prompt()
        except Exception as e:
            system_prompt = f"You are nvagent, a coding assistant. Workspace: {self.workspace}"
            logger.warning("Context build failed: %s", e)

        # Git checkpoint
        if self.config.safety.git_checkpoint:
            try:
                sha = await self.git.checkpoint(f"nvagent: pre-task [{user_message[:60].strip()}]")
                if sha:
                    yield AgentEvent(type="status", data=f"Git checkpoint: {sha[:8]}")
            except Exception:
                pass

        sys_msg = {"role": "system", "content": system_prompt}
        messages = [sys_msg] + self.session.messages

        # Auto-compact if history is long
        if len(self.session.messages) > 40:
            yield AgentEvent(type="status", data="Compacting context...")
            await self.compact()
            messages = [sys_msg] + self.session.messages

        turns = 0
        total_input = total_output = 0
        _turn_label = user_message[:60].strip()

        while turns < self.MAX_TURNS:
            if self._cancelled:
                yield AgentEvent(type="error", data={"message": "Cancelled by user."})
                return

            # Resource guard
            violation = self.resource_guard.check()
            if violation:
                yield AgentEvent(type="safety_violation", data={"kind": violation.kind, "message": violation.message, "fatal": violation.fatal})
                if violation.fatal:
                    yield AgentEvent(type="error", data={"message": str(violation)})
                    return

            turns += 1
            self.tools.begin_turn()
            model = self.client.get_model()
            yield AgentEvent(type="status", data=f"thinking [{model.split('/')[-1]}]")

            # Warn if context is getting large (rough estimate: 1 token ≈ 4 chars)
            _ctx_chars = sum(len(str(m.get("content") or "")) for m in messages)
            _ctx_tokens_est = _ctx_chars // 4
            _max_tokens = self.config.agent.max_tokens
            if _ctx_tokens_est > _max_tokens * 0.8:
                yield AgentEvent(type="status", data=(
                    f"⚠ Context approaching limit (~{_ctx_tokens_est:,} tokens estimated). "
                    f"Consider /compact to free space."
                ))

            # Stream LLM response
            assistant_text = ""
            tool_calls_data = None
            input_tokens = output_tokens = 0
            fatal_error = False

            async for ev in self.client.stream_chat(messages=messages, tools=active_schemas):
                if ev.type == "token":
                    assistant_text += ev.data
                    yield AgentEvent(type="token", data=ev.data)
                elif ev.type == "think_token":
                    yield AgentEvent(type="think_token", data=ev.data)
                elif ev.type == "tool_calls":
                    tool_calls_data = ev.data
                elif ev.type == "usage":
                    d = ev.data
                    input_tokens = d.get("input_tokens", 0)
                    output_tokens = d.get("output_tokens", 0)
                elif ev.type == "status":
                    yield AgentEvent(type="status", data=ev.data)
                elif ev.type == "done":
                    d = ev.data if isinstance(ev.data, dict) else {}
                    input_tokens = d.get("input_tokens", input_tokens)
                    output_tokens = d.get("output_tokens", output_tokens)
                elif ev.type == "error":
                    d = ev.data if isinstance(ev.data, dict) else {}
                    msg = d.get("message", str(ev.data))
                    if msg:
                        yield AgentEvent(type="error", data={"message": msg})
                        fatal_error = True
                    break

            if fatal_error:
                return

            total_input += input_tokens
            total_output += output_tokens
            self.resource_guard.update(tokens=input_tokens + output_tokens)

            # No tool calls — agent is done
            if not tool_calls_data:
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
                    self.session.messages.append({"role": "assistant", "content": assistant_text})

                self._total_input_tokens += total_input
                self._total_output_tokens += total_output
                self._save_session()

                yield AgentEvent(type="done", data={
                    "turns": turns,
                    "tokens_used": total_input + total_output,
                    "files_changed": list(self.tools.changed_files),
                })
                return

            # Execute tool calls
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text, "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["args_raw"]}}
                    for tc in tool_calls_data
                ]})

            tool_results = []
            for tc in tool_calls_data:
                name = tc["name"]
                args = tc["args"]
                call_id = tc["id"]

                # Resolve aliases
                if name not in schemas_by_name and name in _TOOL_ALIASES:
                    name = _TOOL_ALIASES[name]

                yield AgentEvent(type="tool_start", data={"name": name, "args": args, "id": call_id})

                # Install live-output streaming for this tool call
                streamed_lines: list[str] = []
                def _tool_stream_cb(line: str, _buf: list = streamed_lines) -> None:
                    _buf.append(line)
                    # We can't yield from a sync callback, so buffer and flush via events
                    # (REPL reads from tool_stream events emitted after execute returns)

                self.tools.stream_fn = _tool_stream_cb

                # Loop detection
                self.loop_detector.record(name, args)
                if self.loop_detector.is_looping():
                    yield AgentEvent(type="safety_violation", data={
                        "kind": "loop",
                        "message": self.loop_detector.description(),
                        "fatal": True,
                    })
                    yield AgentEvent(type="error", data={"message": "Loop detected — stopping."})
                    return

                self.resource_guard.update(tool_calls=1)

                # Execute
                try:
                    result = await self.tools.execute(name, args)
                except Exception as e:
                    result = f"Tool error [{name}]: {type(e).__name__}: {e}"

                # Emit buffered stream lines
                if streamed_lines:
                    yield AgentEvent(type="tool_stream", data={"lines": list(streamed_lines)})

                # Restore global stream fn (if any)
                self.tools.stream_fn = self._stream_fn

                yield AgentEvent(type="tool_result", data={"name": name, "result": result, "id": call_id})

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })

            messages.extend(tool_results)
            _prune_old_tool_results(messages)

            # Check for mid-turn user correction
            try:
                correction = self.correction_queue.get_nowait()
                messages.append({"role": "user", "content": f"[User correction]: {correction}"})
                self.session.messages.append({"role": "user", "content": f"[User correction]: {correction}"})
                yield AgentEvent(type="status", data=f"Correction injected: {correction[:60]}")
            except asyncio.QueueEmpty:
                pass

            if self.tools.changed_files:
                self.resource_guard.update(files_changed=len(self.tools.changed_files))
                self.tools.end_turn(label=_turn_label)
                yield AgentEvent(type="files_changed", data=list(self.tools.changed_files))

                # Rebuild system prompt so the file tree reflects new/deleted files
                try:
                    system_prompt = self._build_system_prompt()
                    sys_msg = {"role": "system", "content": system_prompt}
                    messages[0] = sys_msg
                except Exception:
                    pass

                # Per-turn git checkpoint after file writes
                if self.config.safety.git_checkpoint:
                    try:
                        changed_names = ", ".join(self.tools.changed_files[:3])
                        sha = await self.git.checkpoint(
                            f"nvagent: turn {turns} — {changed_names}"
                        )
                        if sha:
                            yield AgentEvent(type="status", data=f"Git checkpoint: {sha[:8]}")
                    except Exception:
                        pass
            else:
                self.tools.end_turn()

            # Save a compact turn summary to the session
            tool_summaries = []
            for tc, res_msg in zip(tool_calls_data, tool_results):
                preview = str(res_msg.get("content", ""))[:120]
                tool_summaries.append(f"{tc['name']}(...) → {preview}")
            self.session.messages.append({
                "role": "assistant",
                "content": f"[Turn {turns} — {len(tool_calls_data)} tool call(s)]\n" + "\n".join(tool_summaries),
            })
            self._save_session()

        yield AgentEvent(type="error", data={"message": f"Reached maximum turns ({self.MAX_TURNS})."})
