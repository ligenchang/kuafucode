"""nvagent Agent class — orchestrates a full agentic coding session.

This module houses just the :class:`Agent` class.  All heavy sub-tasks are
delegated to focused sibling modules:

  core/agent_constants.py — SKIP_PLAN_WORDS, PREMATURE_STOP_PHRASES, perf/log helpers
  core/agent_context.py   — context assembly phase (build_turn_context)
  core/agent_stream.py    — LLM stream processing (LLMStream, StreamResult)
  core/agent_toolexec.py  — tool batch execution (ToolBatchExecutor, ToolBatchResult)
  core/pricing.py         — cost_usd()
  core/compaction.py      — compact_history()
  core/feedback.py        — classify_tool_error(), RETRY_HINTS
  core/events.py          — AgentEvent dataclass
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Optional

from nvagent.config import Config
from nvagent.core.client import NIMClient, TaskType, classify_task
from nvagent.core.session import Session, SessionStore
from nvagent.core.planner import Planner, Plan, StepStatus
from nvagent.core.safety import GitCheckpointer, ChangeValidator, LoopDetector, ResourceGuard
from nvagent.core.mcp import McpClient
from nvagent.core.memory import get_memory
from nvagent.core.context import assemble_context
from nvagent.tools import ToolExecutor, TOOL_SCHEMAS

from nvagent.core.events import AgentEvent
from nvagent.core.pricing import cost_usd
from nvagent.core.compaction import compact_history, COMPACT_MSG_THRESHOLD, COMPACT_KEEP_RECENT

from nvagent.core.agent_constants import (
    SKIP_PLAN_WORDS,
    PREMATURE_STOP_PHRASES,
    make_session_logger,
    write_perf_log,
)
from nvagent.core.agent_context import build_turn_context
from nvagent.core.agent_stream import LLMStream
from nvagent.core.agent_toolexec import ToolBatchExecutor

logger = logging.getLogger(__name__)

# O(1) name -> schema lookup used by callers that need the full schema list
TOOL_SCHEMAS_BY_NAME: dict[str, dict] = {s["function"]["name"]: s for s in TOOL_SCHEMAS}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """The nvagent coding agent.

    Construct once per workspace session, then call :meth:`run` for each user
    message.  ``run`` is an async generator that yields :class:`AgentEvent`
    objects consumed by the TUI or a headless runner.
    """

    MAX_TURNS = 20  # safety limit on tool-call loops

    def __init__(
        self,
        config: Config,
        workspace: Path,
        session: Session,
        session_store: SessionStore,
        confirm_fn: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        plan_confirm_fn: Optional[Callable[["Plan", str], Awaitable[Optional[str]]]] = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.session = session
        self.session_store = session_store
        self.client = NIMClient(config)
        self._mcp_client = McpClient(config.mcp.servers)
        self._mcp_started = False
        self.tools = ToolExecutor(
            workspace,
            config.agent.max_file_bytes,
            confirm_fn=confirm_fn,
            safe_mode=config.agent.safe_mode,
            dry_run=config.agent.dry_run,
            mcp_client=self._mcp_client,
        )
        self.plan_confirm_fn = plan_confirm_fn
        self._pending_plan_correction: Optional[str] = None
        self._cancelled = False
        self._current_task: str = ""  # stored so ToolBatchExecutor.reflect() can read it

        # Running cost accumulators (persist across turns within this session)
        self._total_tokens: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        # TUI can push a correction message here to interrupt a tool loop
        self.correction_queue: asyncio.Queue[str] = asyncio.Queue()

        # Planning layer
        self.planner = Planner(self.client)
        self.plan: Optional[Plan] = None

        # Safety layer
        self.git = GitCheckpointer(workspace)
        self.loop_detector = LoopDetector(
            max_identical=config.safety.max_identical_calls,
            window=config.safety.loop_window,
        )
        self.resource_guard = ResourceGuard(config.safety)
        self.validator = ChangeValidator(
            workspace,
            run_linter=config.safety.lint_on_write,
        )

        # Per-tool missing-arg streak and plan nudge count (reset per task)
        self._missing_arg_streak: dict[str, int] = {}
        self._plan_nudge_count: int = 0
        # Structured error feedback: (tool_name, error_category) -> consecutive failures
        self._tool_error_streak: dict[tuple, int] = {}

        # Incremental active-file scanning
        self._active_paths_cache: list[Path] = []
        self._last_scanned_idx: int = 0

        # Context deduplication: abs_path -> mtime_ns already seen by the model
        self._context_shown_reads: dict[str, int] = {}

        # Context cache: (timestamp, prompt_str, active_paths_frozenset, undo_depth)
        self._ctx_cache: Optional[tuple] = None
        self._CTX_CACHE_TTL = 30.0  # seconds

        # Symbol context cache: (active_paths_frozenset, symbol_ctx_str)
        self._sym_ctx_cache: Optional[tuple] = None

        # Cached system-message dict — reused when prompt is unchanged
        self._sys_msg: Optional[dict] = None

        # Speculative context prefetch: (prompt, active_paths_key, undo_depth)
        self._ctx_prefetch_result: Optional[tuple] = None

        # Pre-warm retrieval index in background
        def _prewarm() -> None:
            try:
                from nvagent.core.retrieval import get_retrieval_index

                idx = get_retrieval_index(workspace)
                if not idx._built:
                    idx.build()
            except Exception:
                pass

        threading.Thread(target=_prewarm, daemon=True, name="nvagent-prewarm").start()

    # -------------------------------------------------------------------------
    # Lifecycle helpers
    # -------------------------------------------------------------------------

    def cancel(self) -> None:
        """Signal the agent to stop at the next safe checkpoint."""
        self._cancelled = True

    def _save_session_bg(self) -> None:
        """Fire-and-forget session save using a daemon thread."""
        try:
            t = threading.Thread(
                target=self.session_store.save_session,
                args=(self.session,),
                daemon=True,
                name="nvagent-session-save",
            )
            t.start()
        except Exception:
            try:
                self.session_store.save_session(self.session)
            except Exception:
                pass

    async def compact(self, keep_recent: int = COMPACT_KEEP_RECENT) -> str:
        """Force-compact conversation history; returns a human-readable summary."""
        msgs = self.session.messages
        if len(msgs) < 2:
            return "Nothing to compact — history is empty."
        keep_start = max(0, len(msgs) - keep_recent)
        if keep_start == 0:
            return (
                f"History is already minimal ({len(msgs)} messages"
                " — nothing older to summarise)."
            )

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
                    "below into a tight bullet-point summary (<=200 words) preserving "
                    "all important technical decisions, file paths, and context."
                ),
            },
            {"role": "user", "content": history_text[:12000]},
        ]
        task_type = classify_task("")
        summary_text = ""
        async for ev in self.client.stream_chat(messages=summary_prompt, tools=[], task=task_type):
            if ev.type == "token":
                summary_text += ev.data
            elif ev.type in ("done", "error"):
                break

        if not summary_text.strip():
            return "Compaction failed — model returned no summary."

        original_len = len(msgs)
        summary_msg = {
            "role": "assistant",
            "content": (
                f"[Conversation summary — {len(old_msgs)} earlier messages compacted]\n"
                + summary_text.strip()
            ),
        }
        self.session.messages = [summary_msg] + keep_msgs
        self._last_scanned_idx = len(self.session.messages)
        self._context_shown_reads.clear()
        self._sym_ctx_cache = None
        self._save_session_bg()
        return (
            f"Compacted {original_len} -> {len(self.session.messages)} messages "
            f"({len(old_msgs)} summarised, {len(keep_msgs)} kept verbatim)."
        )

    async def stop(self) -> None:
        """Shut down MCP servers and release resources. Safe to call multiple times."""
        if self._mcp_started and self._mcp_client:
            try:
                await self._mcp_client.stop()
            except Exception:
                pass
            self._mcp_started = False

    # -------------------------------------------------------------------------
    # Main agent loop
    # -------------------------------------------------------------------------

    async def run(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """Process a user message, yielding AgentEvent objects."""
        self._cancelled = False
        self._current_task = user_message
        self.resource_guard.start()
        self.loop_detector.reset()
        self._missing_arg_streak = {}
        self._plan_nudge_count = 0
        self._tool_error_streak = {}

        # -- Lazy MCP start ---------------------------------------------------
        if not self._mcp_started and self._mcp_client:
            try:
                await self._mcp_client.start()
            except Exception as _mcp_exc:
                logger.warning("MCP startup error (continuing without MCP): %s", _mcp_exc)
            self._mcp_started = True

        _active_schemas = self.tools.active_schemas
        _schemas_by_name: dict[str, dict] = {s["function"]["name"]: s for s in _active_schemas}

        # -- Session JSONL log ------------------------------------------------
        import datetime as _dt

        _log_dir = self.workspace / ".nvagent" / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_event = make_session_logger(_log_dir / f"session_{self.session.id}.jsonl")

        # -- Dry-run banner ---------------------------------------------------
        if self.config.agent.dry_run:
            yield AgentEvent(
                type="status",
                data="[DRY RUN] No files will be written or commands executed.",
            )
            yield AgentEvent(
                type="safety_violation",
                data={
                    "kind": "dry_run",
                    "message": (
                        "Dry-run mode active — all write/execute operations are simulated."
                    ),
                    "fatal": False,
                },
            )

        self.session.messages.append({"role": "user", "content": user_message})

        # -- Perf timing ------------------------------------------------------
        _t_run_start = time.monotonic()
        _perf: dict = {}

        def _t(stage: str, since: float) -> float:
            now = time.monotonic()
            _perf[stage] = now - since
            return now

        def _flush_perf() -> None:
            write_perf_log(self.workspace, user_message[:40], _perf)

        # -- Context assembly phase -------------------------------------------
        yield AgentEvent(type="status", data="Building project context...")
        ctx = await build_turn_context(self, user_message)
        for ev in ctx.events:
            yield ev
        _perf.update(ctx.perf)
        system_prompt = ctx.system_prompt

        # -- Planning phase ---------------------------------------------------
        _msg_lower = user_message.strip().lower()
        _skip_plan = any(
            _msg_lower == kw or _msg_lower.startswith(kw + " ") for kw in SKIP_PLAN_WORDS
        )

        plan: Optional[Plan] = None
        _plan_task: Optional[asyncio.Task] = None
        _perf["plan"] = 0.0

        if not _skip_plan:
            _ctx_hint = system_prompt[:400] if system_prompt else ""
            _t_plan_start = time.monotonic()
            _plan_task = asyncio.ensure_future(self.planner.decompose(user_message, _ctx_hint))

        self.plan = None
        self._pending_plan_correction = None

        # Wait for plan if TUI needs to confirm it before first LLM call
        if _plan_task is not None and self.plan_confirm_fn is not None:
            try:
                plan = await asyncio.wait_for(_plan_task, timeout=30)
                _plan_task = None
                _perf["plan"] = time.monotonic() - _t_plan_start
            except Exception:
                plan = None
                _plan_task = None

        if plan and plan.steps:
            system_prompt += plan.to_prompt_block()
            yield AgentEvent(type="plan", data={"task": plan.task, "steps": plan.to_list()})
            if self.plan_confirm_fn is not None:
                plan_response = await self.plan_confirm_fn(plan, user_message)
                if plan_response is not None:
                    self._pending_plan_correction = plan_response.strip()
                    yield AgentEvent(
                        type="status",
                        data=f"Plan modified by user: {plan_response.strip()[:60]}",
                    )
        elif plan is not None and not getattr(plan, "steps", None):
            plan = None

        self.plan = plan

        # -- Git checkpoint ---------------------------------------------------
        _t_stage = time.monotonic()
        if self.config.safety.git_checkpoint:
            try:
                sha = await self.git.checkpoint(f"nvagent: pre-task [{user_message[:60].strip()}]")
                if sha:
                    yield AgentEvent(
                        type="status",
                        data=f"Git checkpoint: {sha[:8]} — safe rollback point recorded.",
                    )
            except Exception:
                pass
        _t("git", _t_stage)

        _perf["pre_llm_total"] = time.monotonic() - _t_run_start
        _flush_perf()
        yield AgentEvent(
            type="status",
            data=(
                f"pre-LLM total {_perf['pre_llm_total']:.2f}s "
                f"[ret={_perf.get('retrieval',0):.2f} "
                f"ctx={_perf.get('context',0):.2f} "
                f"sym={_perf.get('symbols',0):.2f} "
                f"plan={_perf.get('plan',0):.2f} "
                f"git={_perf.get('git',0):.2f}]"
            ),
        )

        # -- Build initial message list ---------------------------------------
        task_type = classify_task(user_message)
        model = self.client.get_model(task_type)
        _perf["model"] = model
        _perf["task_type"] = task_type.name

        if self._sys_msg is None or self._sys_msg["content"] != system_prompt:
            self._sys_msg = {"role": "system", "content": system_prompt}
        messages = [self._sys_msg] + self.session.messages

        if self._pending_plan_correction:
            _corr = self._pending_plan_correction
            messages.append({"role": "user", "content": f"[Plan review feedback]: {_corr}"})
            self.session.messages.append(
                {"role": "user", "content": f"[Plan review feedback]: {_corr}"}
            )
            self._pending_plan_correction = None

        if len(self.session.messages) > COMPACT_MSG_THRESHOLD:
            yield AgentEvent(type="status", data="Compacting context...")
            await compact_history(self.client, self.session, task_type)
            self._save_session_bg()
            self._context_shown_reads.clear()
            yield AgentEvent(type="status", data="Context compacted.")

        # -- Turn-loop variables ----------------------------------------------
        turns = 0
        total_tokens = ctx.token_estimate
        _total_input_tokens = 0
        _total_output_tokens = 0
        _max_think_chars = self.config.agent.think_budget_tokens * 4
        consecutive_error_batches: int = 0
        _last_failure_detail: str = ""
        _prev_batch_wrote_files: bool = False
        _ephemeral_count: int = 0

        # =====================================================================
        # Main while loop
        # =====================================================================
        while turns < self.MAX_TURNS:

            # Strip previous turn's ephemeral messages
            if _ephemeral_count > 0 and len(messages) >= _ephemeral_count:
                messages = messages[:-_ephemeral_count]
            _ephemeral_count = 0

            # Force tool use after a write batch with remaining plan steps
            _force_tool_use = False
            if _prev_batch_wrote_files:
                _prev_batch_wrote_files = False
                _remaining_now = (
                    [
                        s
                        for s in plan.steps
                        if s.status not in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED)
                    ]
                    if plan and plan.steps
                    else []
                )
                if _remaining_now:
                    _force_tool_use = True
                    _rem_titles_now = ", ".join(f'"{{s.title}}"' for s in _remaining_now[:5])
                    if len(_remaining_now) > 5:
                        _rem_titles_now += f" and {len(_remaining_now) - 5} more"
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[CONTINUE NOW — {len(_remaining_now)} step(s) remain:"
                                f" {_rem_titles_now}]\n"
                                "You MUST call write_files (or another tool) IMMEDIATELY.\n"
                                "Do NOT write any text. Do NOT summarize what you just did.\n"
                                "Make your tool call right now."
                            ),
                        }
                    )
                    _ephemeral_count += 1

            if self._cancelled:
                yield AgentEvent(type="error", data={"message": "Cancelled by user."})
                return

            # Resource guard check
            self.resource_guard.update(files_changed=len(self.tools.changed_files))
            violation = self.resource_guard.check()
            if violation is not None:
                yield AgentEvent(
                    type="safety_violation",
                    data={
                        "kind": violation.kind,
                        "message": violation.message,
                        "fatal": violation.fatal,
                    },
                )
                if violation.fatal:
                    yield AgentEvent(type="error", data={"message": str(violation)})
                    return

            turns += 1
            yield AgentEvent(
                type="status",
                data=f"thinking  [{model}  {task_type.name}]",
            )

            # -- Stream LLM response ------------------------------------------
            stream = LLMStream(
                self.client,
                messages,
                _active_schemas,
                task_type,
                force_tool_use=_force_tool_use,
                max_think_chars=_max_think_chars,
                cancelled_fn=lambda: self._cancelled,
                resource_guard=self.resource_guard,
                turn_num=turns,
            )
            async for ev in stream:
                yield ev

            sr = stream.result
            _perf.update(sr.perf)
            _total_input_tokens += sr.input_tokens
            _total_output_tokens += sr.output_tokens
            if sr.total_tokens:
                total_tokens = sr.total_tokens

            if sr.fatal_error:
                return

            # Think budget exceeded -> inject concise directive and retry
            if sr.think_budget_exceeded:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System] Your previous response spent too long in the reasoning/"
                            "thinking phase and exceeded the allowed thinking budget.\n"
                            "Please respond IMMEDIATELY and CONCISELY:\n"
                            " * Skip lengthy internal analysis\n"
                            " * Make your tool call or give your answer directly\n"
                            " * If you need to reason, use at most 2-3 short sentences"
                        ),
                    }
                )
                _ephemeral_count += 1
                _flush_perf()
                continue

            # -- No tool calls: finish, nudge, or handle truncation -----------
            if not sr.tool_calls_data:
                if sr.truncated:
                    if sr.assistant_content:
                        messages.append({"role": "assistant", "content": sr.assistant_content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[Your previous response was truncated because it exceeded "
                                "the token limit.]\n"
                                "Please continue your work, but write FEWER files per tool call. "
                                "Split large files into separate write_file calls. "
                                "Pick up from where you left off and complete the remaining tasks."
                            ),
                        }
                    )
                    self.session.messages.append(messages[-1])
                    _ephemeral_count = 0
                    turns += 1
                    continue

                if sr.assistant_content:
                    messages.append({"role": "assistant", "content": sr.assistant_content})
                    self.session.messages.append(
                        {"role": "assistant", "content": sr.assistant_content}
                    )

                _remaining_steps = (
                    [
                        s
                        for s in plan.steps
                        if s.status not in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED)
                    ]
                    if plan and plan.steps
                    else []
                )
                _incomplete_plan = bool(_remaining_steps) and self._plan_nudge_count < 5
                _last_msg_text = (messages[-1].get("content") or "").lower() if messages else ""
                _talking_not_doing = (
                    plan is not None
                    and any(p in _last_msg_text for p in PREMATURE_STOP_PHRASES)
                    and self._plan_nudge_count < 5
                )

                if _incomplete_plan or _talking_not_doing:
                    self._plan_nudge_count += 1
                    if _incomplete_plan:
                        _rem_titles = ", ".join(f'"{{s.title}}"' for s in _remaining_steps[:5])
                        if len(_remaining_steps) > 5:
                            _rem_titles += f" and {len(_remaining_steps) - 5} more"
                        _plan_block = (
                            plan.to_prompt_block() if hasattr(plan, "to_prompt_block") else ""
                        )
                        _nudge = (
                            f"TOOL CALL REQUIRED -- {len(_remaining_steps)} step(s) not yet "
                            f"done: {_rem_titles}\n\n{_plan_block}\n"
                            "You MUST call `write_files` (or another tool) RIGHT NOW.\n"
                            "Do NOT write any explanatory text. Do NOT describe what you will do.\n"
                            "Make the tool call immediately -- write the files now."
                        )
                    else:
                        _nudge = (
                            "TOOL CALL REQUIRED -- you described work but did not do it.\n"
                            "You MUST call `write_files` (or another tool) RIGHT NOW.\n"
                            "Do NOT write any more text. Do NOT explain. "
                            "Make the tool call immediately."
                        )
                    messages.append({"role": "user", "content": _nudge})
                    _ephemeral_count += 1
                    yield AgentEvent(
                        type="status",
                        data=(
                            f"Continuing -- {len(_remaining_steps)} step(s) still remain..."
                            if _incomplete_plan
                            else "Nudging model to use tools instead of describing work..."
                        ),
                    )
                    continue

                # -- Normal completion ----------------------------------------
                self._save_session_bg()
                self._total_tokens += total_tokens
                self._total_input_tokens += _total_input_tokens
                self._total_output_tokens += _total_output_tokens
                cost = cost_usd(self._total_input_tokens, self._total_output_tokens, model)
                _log_event(
                    "done",
                    {
                        "tokens": total_tokens,
                        "input_tokens": _total_input_tokens,
                        "output_tokens": _total_output_tokens,
                        "cost_usd": cost,
                        "turns": turns,
                        "files_changed": list(self.tools.changed_files),
                    },
                )
                try:
                    mem = get_memory(self.workspace)
                    changed = list(self.tools.changed_files)
                    if changed:
                        mem.task_done(
                            summary=f"Task: {user_message[:120]}",
                            files=changed,
                        )
                    mem.save()
                except Exception:
                    pass

                yield AgentEvent(
                    type="done",
                    data={
                        "tokens_used": total_tokens,
                        "total_tokens": self._total_tokens,
                        "turns": turns,
                        "files_changed": list(self.tools.changed_files),
                        "cost_usd": cost,
                    },
                )
                _perf["total_run"] = time.monotonic() - _t_run_start
                _flush_perf()

                # Kick off speculative context prefetch for the next turn
                _prefetch_paths = list(ctx.active_paths)
                _prefetch_key = frozenset(ctx.active_paths)
                _prefetch_depth = len(self.tools.undo_stack)
                _prefetch_ws = self.workspace
                _prefetch_config = self.config

                async def _speculative_prefetch() -> None:
                    try:
                        _ex = asyncio.get_event_loop()
                        _result = await _ex.run_in_executor(
                            None,
                            lambda: assemble_context(
                                query="",
                                workspace=_prefetch_ws,
                                active_paths=_prefetch_paths,
                                config=_prefetch_config,
                                retrieved_paths=[],
                                max_tokens=8_000,
                            ),
                        )
                        self._ctx_prefetch_result = (_result, _prefetch_key, _prefetch_depth)
                    except Exception:
                        pass

                asyncio.ensure_future(_speculative_prefetch())
                return

            # -- Execute tool calls -------------------------------------------
            executor = ToolBatchExecutor(
                agent=self,
                tool_calls_data=sr.tool_calls_data,
                messages=messages,
                schemas_by_name=_schemas_by_name,
                plan=plan,
                consecutive_error_batches=consecutive_error_batches,
                last_failure_detail=_last_failure_detail,
                log_event=_log_event,
                assistant_content=sr.assistant_content,
            )
            async for ev in executor:
                yield ev

            consecutive_error_batches = executor.consecutive_error_batches
            _last_failure_detail = executor.last_failure_detail
            er = executor.result

            if er.fatal:
                return
            if er.wrote_files:
                _prev_batch_wrote_files = True

            # -- Mid-turn user correction -------------------------------------
            try:
                correction = self.correction_queue.get_nowait()
                messages.append({"role": "user", "content": f"[User correction]: {correction}"})
                self.session.messages.append(
                    {"role": "user", "content": f"[User correction]: {correction}"}
                )
                yield AgentEvent(type="status", data=f"Correction injected: {correction}")
            except asyncio.QueueEmpty:
                pass

            if self.tools.changed_files:
                yield AgentEvent(type="files_changed", data=list(self.tools.changed_files))

            # Compressed turn summary for session persistence
            _tool_summaries = []
            for tc in sr.tool_calls_data:
                _result_content = ""
                for _msg in reversed(messages):
                    if _msg.get("role") == "tool" and _msg.get("tool_call_id") == tc["id"]:
                        _result_content = str(_msg.get("content", ""))[:200]
                        break
                _tool_summaries.append(
                    f"{tc['name']}("
                    + ", ".join(f"{k}={str(v)[:40]}" for k, v in list(tc["args"].items())[:3])
                    + f") -> {_result_content}"
                )
            self.session.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"[Turn summary -- {len(sr.tool_calls_data)} tool call(s)]\n"
                        + "\n".join(_tool_summaries)
                    ),
                }
            )
            self._save_session_bg()

            # Proactive compaction advisory
            _msg_count = len(self.session.messages)
            if COMPACT_MSG_THRESHOLD - 10 <= _msg_count < COMPACT_MSG_THRESHOLD:
                yield AgentEvent(
                    type="status",
                    data=(
                        f"Context at {_msg_count} messages -- approaching compaction limit "
                        f"({COMPACT_MSG_THRESHOLD}). Type /compact to free space now."
                    ),
                )

        # Hit MAX_TURNS
        yield AgentEvent(
            type="error",
            data={
                "message": (f"Reached maximum turns ({self.MAX_TURNS}). Task may be incomplete.")
            },
        )
