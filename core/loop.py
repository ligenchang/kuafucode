"""
The agent loop — the heart of nvagent.

Async generator that:
1. Injects project context into system prompt
2. Streams LLM response
3. Executes tool calls
4. Feeds results back
5. Repeats until finish_reason == "stop"

Emits typed AgentEvent objects consumed by TUI or headless runner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Optional

from nvagent.config import Config
from nvagent.core.client import NIMClient, TaskType, classify_task
from nvagent.core.context import (
    build_system_prompt, estimate_tokens, extract_active_files,
    build_active_files_context, assemble_context,
)
from nvagent.core.session import Session, SessionStore
from nvagent.core.symbols import build_symbol_context
from nvagent.core.index import get_workspace_index
from nvagent.core.memory import get_memory
from nvagent.core.retrieval import retrieve_files
from nvagent.core.planner import Planner, Plan, StepStatus
from nvagent.core.safety import (
    GitCheckpointer, ChangeValidator, LoopDetector, ResourceGuard, Violation,
)
from nvagent.tools import ToolExecutor, TOOL_SCHEMAS
from nvagent.core.mcp import McpClient

logger = logging.getLogger(__name__)


# O(1) name → schema lookup — replaces O(32) linear scans on every tool call
TOOL_SCHEMAS_BY_NAME: dict[str, dict] = {
    s["function"]["name"]: s for s in TOOL_SCHEMAS
}


# ─────────────────────────────────────────────────────────────────────────────
# Agent events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    type: str
    # token           → data: str
    # think_token     → data: str
    # tool_start      → data: {name: str, args: dict}
    # tool_result     → data: {name: str, result: str}
    # error           → data: {message: str}
    # done            → data: {tokens_used: int, turns: int, files_changed: list[str], cost_usd: float}
    # files_changed   → data: list[str]
    # status          → data: str
    # plan            → data: {task: str, steps: list[dict]}
    # plan_update     → data: {steps: list[dict], progress: str}
    # reflection      → data: str
    # safety_violation → data: {kind: str, message: str, fatal: bool}
    # usage           → data: {input_tokens: int, output_tokens: int, total_tokens: int}
    data: object


# ─────────────────────────────────────────────────────────────────────────────
# Per-model pricing  (USD per 1M tokens — separate input/output rates)
# ─────────────────────────────────────────────────────────────────────────────

_PRICING: dict[str, dict[str, float]] = {
    # minimax
    "minimax-m2":     {"input": 0.40, "output": 1.60},
    "minimax-text":   {"input": 0.40, "output": 1.60},
    # kimi
    "kimi-k2":        {"input": 0.50, "output": 2.50},
    # qwen
    "qwen3.5":        {"input": 2.00, "output": 8.00},
    "qwq":            {"input": 2.00, "output": 8.00},
    # glm
    "glm5":           {"input": 0.50, "output": 2.00},
    "glm4":           {"input": 0.50, "output": 2.00},
    # llama nemotron
    "nemotron":               {"input": 1.00, "output": 4.00},
    "llama-3.1-nemotron":     {"input": 1.00, "output": 4.00},
    # default fallback (blended estimate)
    "default": {"input": 1.00, "output": 3.00},
}


def _cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost in USD using separate input/output rates for the model."""
    model_lower = model.lower()
    rates = _PRICING["default"]
    for key, price in _PRICING.items():
        if key != "default" and key in model_lower:
            rates = price
            break
    return (
        input_tokens  * rates["input"]  / 1_000_000
        + output_tokens * rates["output"] / 1_000_000
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context compaction
# ─────────────────────────────────────────────────────────────────────────────

_COMPACT_MSG_THRESHOLD = 40   # compact when history exceeds this many messages
_COMPACT_KEEP_RECENT   = 8    # always keep this many recent messages uncompacted


# ─────────────────────────────────────────────────────────────────────────────
# Structured tool-error feedback
# ─────────────────────────────────────────────────────────────────────────────

# Targeted corrective hints keyed by error category.
# Injected into the conversation after 2 consecutive same-category failures
# on the same tool so the model can self-correct without a full LLM round-trip.
_RETRY_HINTS: dict[str, str] = {
    "ambiguous_edit": (
        "The 'search' string matched multiple locations (or matched zero). "
        "Provide a LONGER, more unique search string that includes 3-5 lines of surrounding "
        "context — not just the single line you want to change."
    ),
    "edit_not_found": (
        "The 'search' string was not found in the file. "
        "Call read_file on that path first to get the EXACT current text (including all whitespace "
        "and indentation). Your search string must match the file content character-for-character."
    ),
    "file_not_found": (
        "The file path does not exist. "
        "Use list_dir or find_files to confirm the correct path before reading or writing."
    ),
    "timeout": (
        "The command timed out. "
        "If a longer runtime is expected, increase the 'timeout' argument. "
        "For test suites, pass a specific path to run_tests instead of the full suite."
    ),
    "command_error": (
        "The command exited with a non-zero code. "
        "Read the STDERR output carefully — it usually explains the failure. "
        "Fix the root cause (missing dependency, syntax error, wrong arguments) before retrying."
    ),
}


def _classify_tool_error(name: str, result: str) -> Optional[str]:
    """Return an error-category string for a failed tool result, or None."""
    import re as _re
    r_lower = result.lower()
    if "⏱" in result or "timed out" in r_lower:
        return "timeout"
    if name in ("edit_file", "str_replace_editor"):
        # ambiguous: matched N times / multiple matches
        if _re.search(r"found \d+ (times?|matches?)", r_lower) or "multiple" in r_lower:
            return "ambiguous_edit"
        # not found
        if "not found" in r_lower or "no match" in r_lower or "could not find" in r_lower:
            return "edit_not_found"
    if name in ("read_file", "write_file", "edit_file", "str_replace_editor", "find_files", "delete_file"):
        if "no such file" in r_lower or "does not exist" in r_lower or (
            "not found" in r_lower and name != "edit_file"
        ):
            return "file_not_found"
    if name == "run_command":
        if _re.search(r"exit code:\s*[1-9]", r_lower):
            return "command_error"
    return None


async def _compact_history(
    client: NIMClient,
    session: Session,
    task_type: TaskType,
) -> None:
    """Summarise old messages in-place to keep context small."""
    msgs = session.messages
    if len(msgs) <= _COMPACT_MSG_THRESHOLD:
        return

    keep_start = len(msgs) - _COMPACT_KEEP_RECENT
    old_msgs   = msgs[:keep_start]
    keep_msgs  = msgs[keep_start:]

    # Build a summarise request — exclude raw tool result messages (role=tool)
    # since their TOOL: <content> format confuses the summarizer and inflates the history
    history_text = "\n\n".join(
        f"{m['role'].upper()}: {m.get('content') or ''}"
        for m in old_msgs
        if m.get("content") and m.get("role") in ("user", "assistant")
    )
    summary_prompt = [
        {"role": "system", "content": "You are a concise summariser. Summarise the conversation history below into a tight bullet-point summary (≤200 words) preserving all important technical decisions, file paths, and context."},
        {"role": "user",   "content": history_text[:12000]},
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
            "content": f"[Conversation summary — {len(old_msgs)} earlier messages compacted]\n{summary_text.strip()}",
        }
        session.messages = [summary_msg] + keep_msgs


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """The nvagent coding agent."""

    MAX_TURNS = 20  # Safety limit on tool call loops

    def __init__(
        self,
        config: Config,
        workspace: Path,
        session: Session,
        session_store: SessionStore,
        confirm_fn: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        plan_confirm_fn: Optional[Callable[["Plan", str], Awaitable[Optional[str]]]] = None,
    ):
        self.config = config
        self.workspace = workspace
        self.session = session
        self.session_store = session_store
        self.client = NIMClient(config)
        # MCP client — starts lazily on first run() call
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
        # Plan hook: called after plan is built, before first LLM call.
        # Receives (plan, task_description) and returns:
        #   None  → proceed normally
        #   str   → inject as user correction message
        self.plan_confirm_fn = plan_confirm_fn
        self._pending_plan_correction: Optional[str] = None
        self._cancelled = False
        # Running cost accumulator (persists across turns within this session)
        self._total_tokens: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        # Correction queue: TUI pushes a message here to interrupt a tool loop
        self.correction_queue: asyncio.Queue[str] = asyncio.Queue()
        # Planning & Reasoning layer
        self.planner = Planner(self.client)
        self.plan: Optional[Plan] = None
        # Safety & Stability layer
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
        # Fix 9: Per-tool missing-arg streak (reset per task, not per run() call)
        # and plan nudge count (also reset per new task).
        # These must persist across run() calls within the same session because
        # self.plan persists and the nudge/streak logic references it.
        self._missing_arg_streak: dict[str, int] = {}
        self._plan_nudge_count: int = 0
        # Structured error feedback: (tool_name, error_category) → consecutive failure count.
        # Reset each task; per-category hints are injected after 2 consecutive failures.
        self._tool_error_streak: dict[tuple, int] = {}
        # Incremental active-file scanning — only scan new messages each turn
        # instead of rescanning the entire history (O(new) vs O(all)).
        self._active_paths_cache: list[Path] = []
        self._last_scanned_idx: int = 0
        # Context deduplication: tracks which abs_path → mtime_ns the model has
        # already received as a full read_file result in this session.
        # Re-reads of the same (path, mtime) are compressed to a short alias,
        # preventing repeated injection of thousands of tokens and reducing
        # the thinking time per turn.
        self._context_shown_reads: dict[str, int] = {}  # abs_path → mtime_ns
        # Context cache: skip expensive rebuilds when files haven't changed
        # between consecutive turns (git/tree/key-files remain stable).
        # (ts, prompt, active_paths, undo_stack_depth): invalidate on new writes or TTL.
        self._ctx_cache: Optional[tuple[float, str, list[Path], int]] = None
        self._CTX_CACHE_TTL = 30.0  # seconds
        # Symbol context cache: (active_paths_frozenset, symbol_ctx_str).
        # Avoids re-running build_symbol_context() on every turn when active
        # files haven't changed. Invalidated whenever a file is written.
        self._sym_ctx_cache: Optional[tuple[frozenset, str]] = None
        # Cached system-message dict — reused across turns when prompt is unchanged
        # to avoid allocating a new dict on every run() call (item #4).
        self._sys_msg: Optional[dict] = None
        # Speculative context prefetch — populated at end of each turn so the
        # next turn's assemble_context is already done before the user sends
        # their next message (item #1). Stores (prompt, active_paths_key, undo_depth).
        self._ctx_prefetch_result: Optional[tuple[str, frozenset, int]] = None
        # Pre-warm the retrieval index in a background thread so the FIRST
        # user query doesn't pay the cold-build cost.
        import threading as _threading
        def _prewarm() -> None:
            try:
                from nvagent.core.retrieval import get_retrieval_index
                idx = get_retrieval_index(workspace)
                if not idx._built:
                    idx.build()
            except Exception:
                pass
        _threading.Thread(target=_prewarm, daemon=True, name="nvagent-prewarm").start()

    def cancel(self) -> None:
        """Signal the agent to stop at the next safe point."""
        self._cancelled = True

    def _save_session_bg(self) -> None:
        """Fire-and-forget background session save.

        Uses a daemon thread so Python can exit cleanly on Ctrl+C without the
        ThreadPoolExecutor atexit handler (concurrent.futures.thread._python_exit)
        raising "Exception ignored on threading shutdown".
        Daemon threads are silently killed by the interpreter on exit — the
        worst case is the last save is skipped, which is acceptable for a
        background cache flush.
        """
        try:
            t = threading.Thread(
                target=self.session_store.save_session,
                args=(self.session,),
                daemon=True,
                name="nvagent-session-save",
            )
            t.start()
        except Exception:
            # Fallback: synchronous save (e.g. during interpreter shutdown)
            try:
                self.session_store.save_session(self.session)
            except Exception:
                pass

    async def compact(self, keep_recent: int = _COMPACT_KEEP_RECENT) -> str:
        """
        Force-compact conversation history regardless of length threshold.
        Returns a human-readable summary of what was done.
        """
        msgs = self.session.messages
        if len(msgs) < 2:
            return "Nothing to compact — history is empty."
        keep_start = max(0, len(msgs) - keep_recent)
        if keep_start == 0:
            return f"History is already minimal ({len(msgs)} messages — nothing older to summarise)."

        old_msgs  = msgs[:keep_start]
        keep_msgs = msgs[keep_start:]

        history_text = "\n\n".join(
            f"{m['role'].upper()}: {m.get('content') or ''}"
            for m in old_msgs
            if m.get("content") and m.get("role") in ("user", "assistant")
        )
        summary_prompt = [
            {"role": "system", "content": (
                "You are a concise summariser. Summarise the conversation history "
                "below into a tight bullet-point summary (≤200 words) preserving "
                "all important technical decisions, file paths, and context."
            )},
            {"role": "user", "content": history_text[:12000]},
        ]

        task_type = classify_task("")
        summary_text = ""
        async for ev in self.client.stream_chat(
            messages=summary_prompt, tools=[], task=task_type
        ):
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
        # Reset scan index since history was rewritten by compaction
        self._last_scanned_idx = len(self.session.messages)
        # Clear read-dedup registry — after compaction the model's context no
        # longer contains those previous read_file results, so the next read of
        # any file must inject full content again.
        self._context_shown_reads.clear()
        self._sym_ctx_cache = None   # force rebuild after history reset
        self._save_session_bg()
        return (
            f"Compacted {original_len} → {len(self.session.messages)} messages "
            f"({len(old_msgs)} summarised, {len(keep_msgs)} kept verbatim)."
        )

    async def stop(self) -> None:
        """Shut down MCP servers and release any resources.

        Call this when the agent is no longer needed (e.g. app exit or test teardown).
        Safe to call multiple times.
        """
        if self._mcp_started and self._mcp_client:
            try:
                await self._mcp_client.stop()
            except Exception:
                pass
            self._mcp_started = False

    async def run(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """
        Process a user message through the agent loop.
        Yields AgentEvent objects.
        """
        self._cancelled = False
        self.resource_guard.start()
        self.loop_detector.reset()
        # Reset per-task counters (new task = fresh plan = fresh nudge/streak state)
        self._missing_arg_streak = {}
        self._plan_nudge_count = 0
        self._tool_error_streak = {}

        # Lazy MCP server start (noop if no servers configured or already started)
        if not self._mcp_started and self._mcp_client:
            try:
                await self._mcp_client.start()
            except Exception as _mcp_exc:
                logger.warning("MCP startup error (continuing without MCP): %s", _mcp_exc)
            self._mcp_started = True

        # Schema lookup dict for this run — includes MCP tools if any are running
        _active_schemas = self.tools.active_schemas
        _schemas_by_name: dict[str, dict] = {
            s["function"]["name"]: s for s in _active_schemas
        }

        # Fix 20: Session logging — structured JSONL log for observability
        import datetime as _dt
        _log_dir = self.workspace / ".nvagent" / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _session_log_path = _log_dir / f"session_{self.session.id}.jsonl"

        def _log_event(kind: str, data: object) -> None:
            """Append a structured event to the session log."""
            try:
                entry = {
                    "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                    "kind": kind,
                    "data": data,
                }
                with open(_session_log_path, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(entry, default=str) + "\n")
            except Exception:
                pass  # logging must never abort the agent

        # Dry-run banner — makes it unmissable in both TUI and headless runner
        if self.config.agent.dry_run:
            yield AgentEvent(
                type="status",
                data="[DRY RUN] No files will be written or commands executed.",
            )
            yield AgentEvent(
                type="safety_violation",
                data={"kind": "dry_run", "message": "Dry-run mode active — all write/execute operations are simulated.", "fatal": False},
            )

        # Add user message to history
        self.session.messages.append({
            "role": "user",
            "content": user_message,
        })

        # ── Perf timing ── track each pre-LLM stage so we can see what's slow
        _t_run_start = time.monotonic()
        _perf: dict[str, float] = {}   # stage_name → wall seconds

        def _t(stage: str, since: float) -> float:
            """Record elapsed since *since*, return now."""
            now = time.monotonic()
            _perf[stage] = now - since
            return now

        def _write_perf_log() -> None:
            """Append a one-line perf summary to .nvagent/perf.log."""
            try:
                import datetime as _dt
                log_dir = self.workspace / ".nvagent"
                log_dir.mkdir(parents=True, exist_ok=True)
                parts = []
                for k, v in _perf.items():
                    if isinstance(v, float):
                        # think_chars_* stores character counts, not durations
                        suffix = "ch" if "_chars_" in k else "s"
                        parts.append(f"{k}={v:.0f}{suffix}" if suffix == "ch" else f"{k}={v:.2f}{suffix}")
                    else:
                        parts.append(f"{k}={v}")
                line = (
                    f"{_dt.datetime.now().strftime('%H:%M:%S')} "
                    f"msg={user_message[:40]!r}  "
                    + "  ".join(parts)
                    + "\n"
                )
                with open(log_dir / "perf.log", "a", encoding="utf-8") as _f:
                    _f.write(line)
            except Exception:
                pass

        # Build system prompt with budget-aware context assembly
        yield AgentEvent(type="status", data="Building project context...")
        # Incremental active-file scan: only inspect messages added since last
        # call to run() — avoids O(full_history) scan on every turn in long sessions.
        _new_msgs = self.session.messages[self._last_scanned_idx:]
        if _new_msgs:
            _seen_active = {str(p) for p in self._active_paths_cache}
            for _np in extract_active_files(_new_msgs, self.workspace):
                _s = str(_np)
                if _s not in _seen_active:
                    self._active_paths_cache.append(_np)
                    _seen_active.add(_s)
            self._last_scanned_idx = len(self.session.messages)
        active_paths = self._active_paths_cache
        mem = get_memory(self.workspace)
        _undo_depth = len(self.tools.undo_stack)

        # Run retrieval in executor (I/O + BM25 scoring)
        _t_stage = time.monotonic()
        try:
            loop_ex = asyncio.get_event_loop()
            retrieved_scored = await loop_ex.run_in_executor(
                None,
                lambda: retrieve_files(user_message, self.workspace, top_k=5),
            )
            retrieved = [sf.path for sf in retrieved_scored]
        except Exception:
            retrieved = []
        _t_stage = _t("retrieval", _t_stage)
        yield AgentEvent(type="status", data=f"⏱ retrieval {_perf['retrieval']:.2f}s")

        # Check context cache: reuse static base if nothing changed recently
        _now = asyncio.get_event_loop().time()
        _active_paths_key = frozenset(active_paths)
        _cache_hit = False
        if (
            self._ctx_cache is not None
            and not self.tools.changed_files          # no files written this session
            and _now - self._ctx_cache[0] < self._CTX_CACHE_TTL
            and self._ctx_cache[2] == _active_paths_key   # same active files
        ):
            system_prompt = self._ctx_cache[1]
            _cache_hit = True
            _perf["context"] = 0.0
            yield AgentEvent(type="status", data="⏱ context hit (0.00s)")
        elif (
            # Speculative prefetch hit (item #1): context was pre-built at end
            # of previous turn while the user was typing their next message.
            self._ctx_prefetch_result is not None
            and self._ctx_prefetch_result[1] == _active_paths_key
            and self._ctx_prefetch_result[2] == _undo_depth
        ):
            system_prompt = self._ctx_prefetch_result[0]
            self._ctx_prefetch_result = None
            _cache_hit = True
            _perf["context"] = 0.0
            self._ctx_cache = (_now, system_prompt, _active_paths_key, _undo_depth)
            yield AgentEvent(type="status", data="⏱ context prefetch (0.00s)")
        else:
            # Assemble context in a thread so file reads / git calls don't block
            # the asyncio event loop during the latency-sensitive startup phase.
            _t_stage = time.monotonic()
            system_prompt = await loop_ex.run_in_executor(
                None,
                lambda: assemble_context(
                    query=user_message,
                    workspace=self.workspace,
                    active_paths=active_paths,
                    config=self.config,
                    retrieved_paths=retrieved,
                    max_tokens=8_000,   # file tree + key files only; model reads files via tools
                ),
            )
            _t_stage = _t("context", _t_stage)
            yield AgentEvent(type="status", data=f"⏱ context {_perf['context']:.2f}s")
            # Cache for subsequent turns
            self._ctx_cache = (_now, system_prompt, _active_paths_key, _undo_depth)

        # Inject structured long-term memory block (BM25-ranked against query).
        # Skipped on cache hits — the cached prompt already has memory appended
        # from the previous turn's cache update below.
        if not _cache_hit:
            try:
                memory_ctx = mem.to_context_block(user_message)
                if memory_ctx:
                    system_prompt += memory_ctx
                    # Update cache with memory appended so next hit is correct.
                    self._ctx_cache = (self._ctx_cache[0], system_prompt, _active_paths_key, _undo_depth)
            except Exception:
                pass

        # Inject symbol index for active files + their import dependencies.
        # Cached per frozenset(active_paths); invalidated whenever a file is written.
        # build_symbol_context itself is O(1) on mtime-cache hits inside symbols_for,
        # but the string-building and dep-graph traversal add up across many turns.
        _t_stage = time.monotonic()
        if active_paths:
            try:
                _sym_key = frozenset(str(p) for p in active_paths)
                if (
                    not self.tools.changed_files
                    and self._sym_ctx_cache is not None
                    and self._sym_ctx_cache[0] == _sym_key
                ):
                    symbol_ctx = self._sym_ctx_cache[1]
                else:
                    symbol_ctx = build_symbol_context(
                        active_paths, self.workspace,
                        symbol_fetcher=get_workspace_index(self.workspace).symbols_for,
                    )
                    self._sym_ctx_cache = (_sym_key, symbol_ctx)
                if symbol_ctx:
                    system_prompt += symbol_ctx
            except Exception:
                pass
        _t_stage = _t("symbols", _t_stage)

        token_est = estimate_tokens(system_prompt)
        yield AgentEvent(
            type="status",
            data=f"⏱ symbols {_perf['symbols']:.2f}s  context ~{token_est:,} tok",
        )

        # ── Planning: decide whether to plan, and run it in background ───────
        # Skip entirely for:
        #   • Short conversational turns (greetings, yes/no, questions)
        #   • Read-only / analysis queries that don't require multi-step execution
        # For everything else: fire the planner as a background Task so the first
        # LLM call starts immediately.  The plan is injected into the system prompt
        # before turn 2 if it has arrived; otherwise planning is skipped silently.
        _SKIP_PLAN_WORDS = {
            # conversational
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
            "sure", "yes", "no", "bye", "goodbye",
            # question starters
            "what is", "what's", "what are", "what does", "what do",
            "who are", "who is", "how are", "how does", "how do",
            "why is", "why does", "when did", "where is",
            # analysis / read-only actions (no file writes needed)
            "help", "explain", "describe", "show", "list", "check",
            "analyze", "analyse", "review", "find", "search", "look",
            "summarize", "summarise", "tell me", "show me", "give me",
            "print", "display", "read", "open", "view", "inspect",
            "diagnose", "trace", "debug", "profile", "investigate",
        }
        _msg_lower = user_message.strip().lower()
        _skip_plan = any(
            _msg_lower == kw or _msg_lower.startswith(kw + " ")
            for kw in _SKIP_PLAN_WORDS
        )

        _plan_task: Optional["asyncio.Task[object]"] = None
        _perf["plan"] = 0.0
        if _skip_plan:
            plan = None
        else:
            # Fire planner as a background asyncio Task — does NOT block the loop.
            # The first LLM call starts immediately; plan arrives in time for turn 2.
            _ctx_hint = system_prompt[:400] if system_prompt else ""
            _t_plan_start = time.monotonic()
            _plan_task = asyncio.ensure_future(
                self.planner.decompose(user_message, _ctx_hint)
            )
            plan = None   # will be set before turn 2

        self.plan = plan
        self._pending_plan_correction = None

        # plan_confirm_fn requires plan before first LLM call — wait if registered
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
                    yield AgentEvent(type="status", data=f"Plan modified by user: {plan_response.strip()[:60]}")
        elif plan is not None and not getattr(plan, 'steps', None):
            plan = None

        # ── Git checkpoint — snapshot before first modification ───────────
        _t_stage = time.monotonic()
        if self.config.safety.git_checkpoint:
            try:
                sha = await self.git.checkpoint(
                    f"nvagent: pre-task [{user_message[:60].strip()}]"
                )
                if sha:
                    yield AgentEvent(
                        type="status",
                        data=f"Git checkpoint: {sha[:8]} — safe rollback point recorded.",
                    )
            except Exception:
                pass  # git unavailable — continue without checkpoint
        _t("git", _t_stage)
        yield AgentEvent(type="status", data=f"⏱ git {_perf.get('git', 0):.2f}s")

        # ── Write perf log and emit summary before first LLM call ─────────
        _perf["pre_llm_total"] = time.monotonic() - _t_run_start
        _write_perf_log()
        yield AgentEvent(
            type="status",
            data=(
                f"⏱ pre-LLM total {_perf['pre_llm_total']:.2f}s "
                f"[ret={_perf.get('retrieval',0):.2f} "
                f"ctx={_perf.get('context',0):.2f} "
                f"sym={_perf.get('symbols',0):.2f} "
                f"plan={_perf.get('plan',0):.2f} "
                f"git={_perf.get('git',0):.2f}]"
            ),
        )

        # Classify task for model routing
        task_type = classify_task(user_message)
        model = self.client.get_model(task_type)
        _perf["model"] = model
        _perf["task_type"] = task_type.name

        # Build messages for API — reuse the cached sys-msg dict to avoid
        # a fresh dict allocation (and potentially large string copy) every turn.
        if self._sys_msg is None or self._sys_msg["content"] != system_prompt:
            self._sys_msg = {"role": "system", "content": system_prompt}
        messages = [self._sys_msg] + self.session.messages

        # Inject plan correction if the user responded to the plan preview
        if self._pending_plan_correction:
            _corr = self._pending_plan_correction
            messages.append({"role": "user", "content": f"[Plan review feedback]: {_corr}"})
            self.session.messages.append({"role": "user", "content": f"[Plan review feedback]: {_corr}"})
            self._pending_plan_correction = None

        # Auto-compact context if message history is too long
        if len(self.session.messages) > _COMPACT_MSG_THRESHOLD:
            yield AgentEvent(type="status", data="Compacting context…")
            await _compact_history(self.client, self.session, task_type)
            self._save_session_bg()
            # After compaction the in-flight messages list no longer contains
            # previous read_file results — reset registry so next reads send full content.
            self._context_shown_reads.clear()
            yield AgentEvent(type="status", data="Context compacted.")

        turns = 0
        total_tokens = token_est
        _total_input_tokens = 0
        _total_output_tokens = 0
        assistant_content = ""
        # Think-budget: cut the stream and retry when reasoning chars exceed this.
        # 4 chars ≈ 1 token; 0 disables the cap.
        _max_think_chars = self.config.agent.think_budget_tokens * 4
        # Failure-recovery tracking
        consecutive_error_batches: int = 0
        _last_failure_detail: str = ""
        # Whether the previous batch wrote files — used to force tool use on next call
        _prev_batch_wrote_files: bool = False
        # ── Ephemeral message tracking ─────────────────────────────────────────
        # Some directive messages (CONTINUE NOW, tool-use nudges) must go into
        # `messages` for the current LLM call but must NOT pollute
        # `session.messages` (which is persisted and compacted).  We track how
        # many ephemeral messages were appended to `messages` each turn so we
        # can strip them at the start of the next turn, keeping both lists
        # consistent while avoiding context pollution across turns.
        _ephemeral_count: int = 0

        while turns < self.MAX_TURNS:
            _stream_truncated = False

            # ── Strip previous turn's ephemeral messages from live context ────
            # They were needed for exactly one LLM call — remove them now so
            # they don't accumulate and diverge from session.messages.
            if _ephemeral_count > 0:
                if len(messages) >= _ephemeral_count:
                    messages = messages[:-_ephemeral_count]
                _ephemeral_count = 0

            # After a file-write batch with remaining work, force the next LLM call to
            # produce a tool call (not prose) and inject a directive into messages.
            _force_tool_use = False
            if _prev_batch_wrote_files:
                _prev_batch_wrote_files = False
                _remaining_now = (
                    [s for s in plan.steps
                     if s.status not in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED)]
                    if plan and plan.steps else []
                )
                if _remaining_now:
                    _force_tool_use = True
                    _rem_titles_now = ", ".join(f'"{s.title}"' for s in _remaining_now[:5])
                    if len(_remaining_now) > 5:
                        _rem_titles_now += f" and {len(_remaining_now) - 5} more"
                    _continuation = (
                        f"[CONTINUE NOW — {len(_remaining_now)} step(s) remain: {_rem_titles_now}]\n"
                        "You MUST call write_files (or another tool) IMMEDIATELY.\n"
                        "Do NOT write any text. Do NOT summarize what you just did.\n"
                        "Make your tool call right now."
                    )
                    # EPHEMERAL: inject into live context only — not session history.
                    # Stripped at the start of the next iteration via _ephemeral_count.
                    messages.append({"role": "user", "content": _continuation})
                    _ephemeral_count += 1
            if self._cancelled:
                yield AgentEvent(type="error", data={"message": "Cancelled by user."})
                return

            # ── Resource guard check ──────────────────────────────────────
            self.resource_guard.update(
                files_changed=len(self.tools.changed_files),
            )
            violation = self.resource_guard.check()
            if violation is not None:
                yield AgentEvent(
                    type="safety_violation",
                    data={"kind": violation.kind, "message": violation.message, "fatal": violation.fatal},
                )
                if violation.fatal:
                    yield AgentEvent(type="error", data={"message": str(violation)})
                    return

            turns += 1
            assistant_content = ""
            tool_calls_data = None
            _think_budget_exceeded = False

            # Signal TUI to show spinner before blocking API call — include
            # model name and task type so the user can see what's running.
            yield AgentEvent(
                type="status",
                data=f"thinking  [{model}  {task_type.name}]",
            )

            # ── Stream LLM response ─────────────────────────────────────────
            _t_llm_call = time.monotonic()
            _first_token_logged = False
            # Think-phase tracking
            _t_think_start: float | None = None
            _think_char_count: int = 0
            _think_phase_done: bool = False
            async for event in self.client.stream_chat(
                messages=messages,
                tools=_active_schemas,
                task=task_type,                force_tool_use=_force_tool_use,            ):
                if self._cancelled:
                    yield AgentEvent(type="error", data={"message": "Cancelled."})
                    return

                if event.type == "token":
                    if not _first_token_logged:
                        _first_token_logged = True
                        _perf[f"llm_first_token_turn{turns}"] = time.monotonic() - _t_llm_call
                    # Think phase just ended — emit timing summary
                    if _t_think_start is not None and not _think_phase_done:
                        _think_phase_done = True
                        _think_dur = time.monotonic() - _t_think_start
                        _perf[f"think_turn{turns}"] = _think_dur
                        _perf[f"think_chars_turn{turns}"] = float(_think_char_count)
                        yield AgentEvent(
                            type="status",
                            data=f"⏱ think {_think_dur:.1f}s  {_think_char_count:,} chars (~{_think_char_count//4:,} tok)",
                        )
                        _write_perf_log()
                    assistant_content += event.data
                    yield AgentEvent(type="token", data=event.data)
                    total_tokens += 1
                    self.resource_guard.update(tokens=1)

                elif event.type == "think_token":
                    if not _first_token_logged:
                        _first_token_logged = True
                        _perf[f"llm_first_token_turn{turns}"] = time.monotonic() - _t_llm_call
                    if _t_think_start is None:
                         _t_think_start = time.monotonic()
                         yield AgentEvent(type="status", data="🧠 Thinking started")
                yield AgentEvent(type="status", data="🧠 Thinking started")
                    _think_char_count += len(event.data)
                yield AgentEvent(type="status", data=f"🧠 Still thinking: {_think_char_count:,} chars, {time.monotonic() - _t_think_start:.1f}s elapsed")
                    # Hard cap on think tokens — cut the stream when reasoning
                    # runs away before any output tokens have appeared.
                    if (
                        _max_think_chars > 0
                        and not assistant_content          # no output yet
                        and _think_char_count > _max_think_chars
                    ):
                        _think_budget_exceeded = True
                        yield AgentEvent(
                            type="status",
                            data=(
                                f"⚠ Think budget exceeded ({_think_char_count:,} chars / "
                                f"{_think_char_count // 4:,} tok) — retrying with concise directive"
                            ),
                        )
                        break  # abort stream; outer loop will retry
                    yield AgentEvent(type="think_token", data=event.data)

                elif event.type == "usage":
                    # Real token counts from the API — replace our estimates.
                    if isinstance(event.data, dict):
                        actual_total = event.data.get("total_tokens", 0)
                        actual_input = event.data.get("input_tokens", 0)
                        actual_output = event.data.get("output_tokens", 0)
                        if actual_total > 0:
                            total_tokens = actual_total
                            _total_input_tokens += actual_input
                            _total_output_tokens += actual_output
                            self.resource_guard.update(tokens=actual_total)
                            if actual_input > 25_600:  # 80% of 32K
                                yield AgentEvent(
                                    type="status",
                                    data=(
                                        f"⚠ Context pressure: {actual_input:,} input tokens "
                                        f"(+{actual_output:,} output). "
                                        f"Consider /compact if the session grows further."
                                    ),
                                )

                elif event.type == "tool_calls":
                    tool_calls_data = event.data

                elif event.type == "status":
                    # Forward status events from stream_chat (e.g. ⏱ api-connect,
                    # ⏱ api-first-chunk, "streaming") to the TUI.
                    # Also record api timing probes in the perf log.
                    _smsg = str(event.data)
                    if _smsg.startswith("⏱ api-connect "):
                        try:
                            _perf[f"api_connect_turn{turns}"] = float(_smsg.split()[2].rstrip("s"))
                        except Exception:
                            pass
                    elif _smsg.startswith("⏱ api-first-chunk "):
                        try:
                            _perf[f"api_first_chunk_turn{turns}"] = float(_smsg.split()[2].rstrip("s"))
                        except Exception:
                            pass
                    yield AgentEvent(type="status", data=event.data)

                elif event.type == "error":
                    msg = event.data.get("message", "") if isinstance(event.data, dict) else str(event.data)
                    # Truncation is recoverable — inject a hint and keep the loop going.
                    if "truncated" in msg or "finish_reason=length" in msg:
                        # Emit a visible truncation marker to the user so they know the response was cut
                        yield AgentEvent(type="status", data="[truncated] Response exceeded token limit — continuing…")
                        _trunc_hint = (
                            "[Your previous response was truncated because it exceeded the token limit.]\n"
                            "Please continue your work, but write FEWER files per tool call. "
                            "Split large files into separate write_file calls. "
                            "Pick up from where you left off and complete the remaining tasks."
                        )
                        if assistant_content:
                            messages.append({"role": "assistant", "content": assistant_content})
                            assistant_content = ""
                        messages.append({"role": "user", "content": _trunc_hint})
                        self.session.messages.append({"role": "user", "content": _trunc_hint})
                        # The ephemeral messages from the start of this turn have
                        # already been consumed by the LLM call — clear the counter
                        # so the next iteration doesn't double-strip.
                        _ephemeral_count = 0
                        _stream_truncated = True
                        turns += 1
                        break  # break inner stream loop, continue outer while loop
                    yield AgentEvent(type="error", data=event.data)
                    return

                elif event.type == "done":
                    # Normal stop — extract real token usage if the API included it
                    if isinstance(event.data, dict):
                        actual_total = event.data.get("total_tokens", 0)
                        if actual_total > 0:
                            total_tokens = actual_total
                            self.resource_guard.update(tokens=actual_total)
                    break

            # ── Think budget exceeded → retry with concise directive ──────────
            if _think_budget_exceeded:
                # Inject an ephemeral directive so the model skips extended reasoning.
                # Do NOT persist to session.messages — it's corrective noise.
                _concise_hint = (
                    "[System] Your previous response spent too long in the reasoning/thinking phase "
                    "and exceeded the allowed thinking budget.\n"
                    "Please respond IMMEDIATELY and CONCISELY:\n"
                    " • Skip lengthy internal analysis\n"
                    " • Make your tool call or give your answer directly\n"
                    " • If you need to reason, use at most 2–3 short sentences"
                )
                messages.append({"role": "user", "content": _concise_hint})
                _ephemeral_count += 1
                # Record in perf log so over-thinking turns are visible
                _perf[f"think_chars_turn{turns}"] = float(_think_char_count)
                _write_perf_log()
                continue

            # ── No tool calls → done ────────────────────────────────────────
            if not tool_calls_data:
                # If the response was truncated, continue the outer loop
                # (hint already injected into messages above).
                if _stream_truncated:
                    continue

                # Always save assistant text into the LLM message context so the
                # model can see its own previous response in the next turn.
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                    self.session.messages.append({"role": "assistant", "content": assistant_content})
                    assistant_content = ""

                # If the plan has unfinished steps the nudge fires; if plan.done
                # but the LLM response looks like it was about to do more work,
                # a secondary "stop talking, use tools" nudge fires.
                _remaining_steps = (
                    [s for s in plan.steps
                     if s.status not in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED)]
                    if plan and plan.steps else []
                )
                _incomplete_plan = bool(_remaining_steps) and self._plan_nudge_count < 5

                # Detect "I'll do X next" text responses that describe work but don't do it.
                # Only fire when there is an active plan — never on plain conversational turns
                # (greetings, question-answers, etc.) to avoid spurious thinking loops.
                _premature_stop_phrases = (
                    "i'll", "i will", "let me", "next i", "now i", "i need to",
                    "continuing", "i'll continue", "i should", "i'll now", "let's"
                )
                _last_msg = (messages[-1].get("content") or "").lower() if messages else ""
                _talking_not_doing = (
                    plan is not None  # only relevant when we have an active plan
                    and any(p in _last_msg for p in _premature_stop_phrases)
                    and self._plan_nudge_count < 5
                )

                if _incomplete_plan or _talking_not_doing:
                    self._plan_nudge_count += 1
                    if _incomplete_plan:
                        _rem_titles = ", ".join(f'"{s.title}"' for s in _remaining_steps[:5])
                        if len(_remaining_steps) > 5:
                            _rem_titles += f" and {len(_remaining_steps) - 5} more"
                        _plan_block = plan.to_prompt_block() if hasattr(plan, "to_prompt_block") else ""
                        _nudge = (
                            f"⚠️ TOOL CALL REQUIRED — {len(_remaining_steps)} step(s) not yet done: {_rem_titles}\n\n"
                            f"{_plan_block}\n"
                            "You MUST call `write_files` (or another tool) RIGHT NOW.\n"
                            "Do NOT write any explanatory text. Do NOT describe what you will do.\n"
                            "Make the tool call immediately — write the files now."
                        )
                    else:
                        _nudge = (
                            "⚠️ TOOL CALL REQUIRED — you described work but did not do it.\n"
                            "You MUST call `write_files` (or another tool) RIGHT NOW.\n"
                            "Do NOT write any more text. Do NOT explain. Make the tool call immediately."
                        )
                    messages.append({"role": "user", "content": _nudge})
                    # EPHEMERAL: this nudge is a one-shot directive for the
                    # next LLM call only.  Writing it to session.messages would
                    # mean it gets compacted into long-term history and could
                    # confuse future turns.  It is stripped from `messages` at
                    # the top of the next iteration via _ephemeral_count.
                    _ephemeral_count += 1
                    yield AgentEvent(
                        type="status",
                        data=(
                            f"Continuing — {len(_remaining_steps)} step(s) still remain…"
                            if _incomplete_plan
                            else "Nudging model to use tools instead of describing work…"
                        ),
                    )
                    # NOTE: do NOT increment turns here — the top of the while loop does it
                    continue

                # Save assistant response to history (non-blocking background write)
                self._save_session_bg()

                self._total_tokens += total_tokens
                self._total_input_tokens += _total_input_tokens
                self._total_output_tokens += _total_output_tokens
                cost = _cost_usd(self._total_input_tokens, self._total_output_tokens, model)

                # Fix 20: Log completion
                _log_event("done", {
                    "tokens": total_tokens,
                    "input_tokens": _total_input_tokens,
                    "output_tokens": _total_output_tokens,
                    "cost_usd": cost,
                    "turns": turns,
                    "files_changed": list(self.tools.changed_files),
                })

                # Record completed task in long-term memory
                try:
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
                    }
                )
                _perf["total_run"] = time.monotonic() - _t_run_start
                _write_perf_log()   # final write with all LLM timing included

                # ── Speculative context prefetch (item #1) ───────────────────
                # While the user is reading the response and typing their next
                # message, pre-build context for the current workspace state.
                # If active_paths and undo_depth haven't changed by then, the
                # next run() call uses the cached result with 0ms overhead.
                _prefetch_paths   = list(active_paths)  # snapshot
                _prefetch_key     = _active_paths_key
                _prefetch_depth   = _undo_depth
                _prefetch_ws      = self.workspace
                _prefetch_config  = self.config
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

            # ── Execute tool calls ───────────────────────────────────────────
            # Add assistant message with tool_calls to history
            openai_tool_calls = []
            for tc in tool_calls_data:
                openai_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["args_raw"],
                    }
                })

            messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": openai_tool_calls,
            })

            # Begin a new tool turn
            self.tools.begin_turn()

            # ── Pre-launch valid tool calls concurrently ──────────────────────
            # Skip when: approve-writes mode (confirm_fn is interactive per-call),
            # dry-run (instant, no I/O), or only a single call (no parallelism gain).
            _exec_tasks: dict[str, "asyncio.Task[str]"] = {}
            _can_parallelize = (
                not self.tools.confirm_fn
                and not self.config.agent.dry_run
                and len(tool_calls_data) > 1
            )
            if _can_parallelize:
                for _tc in tool_calls_data:
                    _n, _a = _tc["name"], _tc["args"]
                    _s = _schemas_by_name.get(_n)
                    if _s:
                        _req = _s["function"].get("parameters", {}).get("required", [])
                        if any(r not in _a or _a[r] is None or _a[r] == "" for r in _req):
                            continue  # invalid args — will error in main loop
                    _exec_tasks[_tc["id"]] = asyncio.create_task(
                        self.tools.execute(_n, _a)
                    )

            # Per-turn tool call counter — reset for each LLM response batch.
            # Used to enforce max_tool_calls_per_turn budget below.
            _turn_tool_calls = 0

            # Execute each tool
            for tc in tool_calls_data:
                if self._cancelled:
                    yield AgentEvent(type="error", data={"message": "Cancelled."})
                    return

                name = tc["name"]
                args = tc["args"]

                # Validate: reject calls with empty or missing required args
                schema = _schemas_by_name.get(name)
                if schema:
                    required = schema["function"].get("parameters", {}).get("required", [])
                    missing  = [r for r in required if r not in args or args[r] is None or args[r] == ""]
                    if missing:
                        err_msg = f"Missing required argument(s) for {name}: {', '.join(missing)}"
                        yield AgentEvent(type="tool_start", data={"name": name, "args": args})
                        yield AgentEvent(type="tool_result", data={"name": name, "result": f"Error: {err_msg}"})
                        messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": f"Error: {err_msg}"})
                        # Record in loop detector so fingerprint-based cycling is caught
                        if self.config.safety.loop_detection:
                            self.loop_detector.record(name, args)
                        # Hard cap: after 3 consecutive missing-arg calls for the same tool,
                        # inject a forceful correction so the model cannot spin indefinitely
                        self._missing_arg_streak[name] = self._missing_arg_streak.get(name, 0) + 1
                        if self._missing_arg_streak[name] >= 3:
                            _hint = (
                                f"[Error — tool call stuck] You have called '{name}' "
                                f"{self._missing_arg_streak[name]} consecutive times with missing "
                                f"required argument(s): {', '.join(missing)}.\n"
                                f"Required fields: {', '.join(required)}.\n"
                                "You MUST provide ALL required arguments with concrete, non-empty values. "
                                "If you do not yet know a required value, state that explicitly "
                                "instead of calling the tool with placeholder or empty arguments."
                            )
                            messages.append({"role": "user", "content": _hint})
                            self.session.messages.append({"role": "user", "content": _hint})
                            yield AgentEvent(
                                type="status",
                                data=f"[stuck] '{name}' called {self._missing_arg_streak[name]}x with missing args — hint injected",
                            )
                            self._missing_arg_streak[name] = 0  # reset after injecting hint
                        continue

                yield AgentEvent(type="tool_start", data={
                    "name": name,
                    "args": args,
                })
                # Fix 20: Log each tool call
                _log_event("tool_call", {"name": name, "args": {k: str(v)[:80] for k, v in args.items()}})

                # Check safe mode for destructive operations
                if self.config.agent.safe_mode and name in ("write_file", "delete_file"):
                    yield AgentEvent(type="status", data=f"[safe_mode] Executing {name}...")

                try:
                    if tc["id"] in _exec_tasks:
                        result = await _exec_tasks[tc["id"]]
                    else:
                        result = await self.tools.execute(name, args)
                except Exception as exc:
                    result = f"Tool error: {exc}"
                else:
                    # Successful call — reset missing-arg streak for this tool
                    self._missing_arg_streak.pop(name, None)

                # ── Post-write validation ─────────────────────────────────
                if (
                    self.config.safety.validate_writes
                    and name in ("write_file", "edit_file", "str_replace_editor", "apply_patch")
                    and not result.startswith("Error:")
                ):
                    try:
                        _written_path_str = args.get("path", "")
                        if _written_path_str:
                            _written_path = (
                                self.workspace / _written_path_str
                                if not Path(_written_path_str).is_absolute()
                                else Path(_written_path_str)
                            )
                            vr = await self.validator.validate_file_async(_written_path)
                            if not vr.ok:
                                _v_banner = f"[Validation] {vr.to_str()}"
                                result = result + "\n" + _v_banner
                                yield AgentEvent(
                                    type="safety_violation",
                                    data={"kind": "validation", "message": vr.to_str(), "fatal": False},
                                )
                    except Exception:
                        pass  # validation errors must never abort the agent

                # ── Loop detector — record this call ─────────────────────
                if self.config.safety.loop_detection:
                    self.loop_detector.record(name, args)

                # ── Resource guard — count tool call + output bytes ───────
                self.resource_guard.update(
                    tool_calls=1,
                    output_bytes=len(result.encode("utf-8", errors="replace")),
                )

                # ── Structured error feedback — targeted retry hints ──────
                _is_tool_error = (
                    result.startswith("Error:")
                    or result.startswith("Tool error:")
                    or "⏱" in result
                )
                if _is_tool_error:
                    _err_cat = _classify_tool_error(name, result)
                    if _err_cat:
                        _err_key = (name, _err_cat)
                        self._tool_error_streak[_err_key] = (
                            self._tool_error_streak.get(_err_key, 0) + 1
                        )
                        if self._tool_error_streak[_err_key] >= 2:
                            _hint_text = _RETRY_HINTS.get(_err_cat, "")
                            if _hint_text:
                                _retry_hint = (
                                    f"[Retry hint — '{name}' failed "
                                    f"{self._tool_error_streak[_err_key]}x "
                                    f"with error type '{_err_cat}']\n{_hint_text}"
                                )
                                messages.append({"role": "user", "content": _retry_hint})
                                self.session.messages.append({"role": "user", "content": _retry_hint})
                                yield AgentEvent(
                                    type="status",
                                    data=(
                                        f"[retry-hint] {_err_cat} on '{name}' "
                                        f"(x{self._tool_error_streak[_err_key]})"
                                    ),
                                )
                                self._tool_error_streak[_err_key] = 0  # reset after injecting hint
                else:
                    # Successful call — reset error streak entries for this tool
                    for _ek in [k for k in self._tool_error_streak if k[0] == name]:
                        del self._tool_error_streak[_ek]

                yield AgentEvent(type="tool_result", data={
                    "name": name,
                    "result": result,
                })
                # Fix 20: Log tool result (truncated)
                _log_event("tool_result", {"name": name, "result": result[:200], "ok": not result.startswith("Error")})

                # ── Per-turn tool budget enforcement ────────────────────
                _turn_tool_calls += 1
                _per_turn_limit = self.config.safety.max_tool_calls_per_turn
                if _turn_tool_calls >= _per_turn_limit:
                    _budget_msg = (
                        f"[Tool budget reached — {_turn_tool_calls} tool call(s) executed "
                        f"this turn (limit: {_per_turn_limit}). "
                        "Summarise what you have done so far and continue with the "
                        "remaining steps in your next response."
                    )
                    messages.append({"role": "user", "content": _budget_msg})
                    self.session.messages.append({"role": "user", "content": _budget_msg})
                    yield AgentEvent(
                        type="status",
                        data=(
                            f"[budget] Turn tool limit ({_per_turn_limit}) reached — "
                            "forcing new LLM turn"
                        ),
                    )
                    # Cancel any remaining pre-launched (but not yet iterated) tasks
                    for _rem_tc in tool_calls_data:
                        _t = _exec_tasks.get(_rem_tc["id"])
                        if _t and not _t.done():
                            _t.cancel()
                    break

                # ── Context deduplication for read_file ──────────────────
                # If the model has already seen the full content of this file
                # (same path + same mtime_ns), inject a compact alias instead
                # of the full content.  This prevents the context window from
                # growing with duplicate tokens on every re-read, which is the
                # primary cause of slow "thinking" between sequential reads.
                _msg_content = result
                if name == "read_file" and not result.startswith("Error"):
                    _rf_path_arg = args.get("path", "")
                    _rf_no_range = (
                        args.get("start_line") is None and args.get("end_line") is None
                    )
                    if _rf_path_arg and _rf_no_range:
                        _rf_abs = str(
                            (self.workspace / _rf_path_arg).resolve()
                            if not _rf_path_arg.startswith("/")
                            else _rf_path_arg
                        )
                        _rf_mtime = self.tools.last_read_mtime(_rf_abs)
                        if _rf_mtime is not None:
                            _prev_mtime = self._context_shown_reads.get(_rf_abs)
                            if _prev_mtime is not None and _prev_mtime == _rf_mtime:
                                # Model already has this exact file content in context
                                _msg_content = (
                                    f"[File unchanged since previous read — "
                                    f"content already in context. No re-injection needed.]"
                                )
                            else:
                                # First time or file changed — record so future reads can alias
                                self._context_shown_reads[_rf_abs] = _rf_mtime

                # Add tool result to messages (possibly compressed alias)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _msg_content,
                })

            # End turn — snapshot undo state for files changed this tool round
            self.tools.end_turn()

            # ── Loop detection — check after entire tool batch ────────────
            if self.config.safety.loop_detection and self.loop_detector.is_looping():
                desc = self.loop_detector.description()
                yield AgentEvent(
                    type="safety_violation",
                    data={"kind": "loop", "message": desc, "fatal": False},
                )
                # Inject a strong hint so the model breaks the pattern
                _hint = (
                    f"[Safety — loop detected] {desc}\n"
                    "You must try a completely different approach. "
                    "Do NOT repeat the same tool calls with the same arguments."
                )
                messages.append({"role": "user", "content": _hint})
                self.session.messages.append({"role": "user", "content": _hint})
                self.loop_detector.reset()

            # ── Plan step tracking & failure reflection ──────────────────────
            if plan and plan.steps:
                batch_had_error = any(
                    r.startswith("Error:") or r.startswith("Tool error:")
                    for r in [
                        msg["content"]
                        for msg in messages
                        if msg.get("role") == "tool"
                    ][-len(tool_calls_data):]
                    if isinstance(r, str)
                )

                if batch_had_error:
                    consecutive_error_batches += 1
                    # Capture the last error detail for reflection
                    for msg in reversed(messages):
                        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                            if msg["content"].startswith("Error:") or msg["content"].startswith("Tool error:"):
                                _last_failure_detail = msg["content"]
                                break

                    if plan.current_step:
                        plan.mark_failed(plan.current_step.id)
                        yield AgentEvent(
                            type="plan_update",
                            data={"steps": plan.to_list(), "progress": plan.progress_summary()},
                        )

                    # After 2 consecutive error batches, trigger self-reflection
                    if consecutive_error_batches >= 2:
                        consecutive_error_batches = 0
                        _step_title = (
                            plan.current_step.title
                            if plan.current_step else "(unknown step)"
                        )
                        _done_titles = [s.title for s in plan.completed_steps]
                        yield AgentEvent(type="status", data="Reflecting on failures…")
                        try:
                            reflection_text = await self.planner.reflect(
                                step_title=_step_title,
                                failure_detail=_last_failure_detail,
                                task=user_message,
                                completed_step_titles=_done_titles,
                            )
                        except Exception:
                            reflection_text = (
                                "Previous approach failed repeatedly. "
                                "Try a different tool or verify paths."
                            )
                        yield AgentEvent(type="reflection", data=reflection_text)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[Self-reflection — recovery hint]\n{reflection_text}"
                            ),
                        })
                        self.session.messages.append({
                            "role": "user",
                            "content": f"[Self-reflection — recovery hint]\n{reflection_text}",
                        })
                else:
                    # Fix 6: Don't advance the plan on ANY successful tool batch.
                    # Only advance when the tools called actually match what the
                    # current step requires (based on tool_hint or write/run ops).
                    consecutive_error_batches = 0
                    if plan.current_step:
                        step = plan.current_step
                        _batch_names = {tc["name"] for tc in tool_calls_data}
                        # Determine if this batch plausibly completed the current step:
                        # 1. Step has a tool_hint — at least one call matches it
                        # 2. No tool_hint — any write/run/edit tool counts as progress
                        _write_tools = {"write_file", "write_files", "edit_file", "str_replace_editor", "apply_patch"}
                        _run_tools   = {"run_command", "run_tests", "run_formatter"}
                        _progress_tools = _write_tools | _run_tools
                        hint = getattr(step, "tool_hint", None) or ""
                        _step_done = (
                            (hint and hint in _batch_names)
                            or (not hint and bool(_batch_names & _progress_tools))
                        )
                        if _step_done:
                            plan.advance()
                            yield AgentEvent(
                                type="plan_update",
                                data={"steps": plan.to_list(), "progress": plan.progress_summary()},
                            )

            # Mark that this batch wrote files so next call can be forced
            _batch_tool_names = {tc["name"] for tc in tool_calls_data}
            if _batch_tool_names & {"write_file", "write_files", "edit_file", "str_replace_editor", "apply_patch"}:
                _prev_batch_wrote_files = True

            # Check for a mid-turn user correction
            try:
                correction = self.correction_queue.get_nowait()
                # Inject as a new user turn so the model sees it immediately
                messages.append({"role": "user", "content": f"[User correction]: {correction}"})
                self.session.messages.append({"role": "user", "content": f"[User correction]: {correction}"})
                yield AgentEvent(type="status", data=f"Correction injected: {correction}")
            except asyncio.QueueEmpty:
                pass

            # Emit files-changed summary if any files were written
            if self.tools.changed_files:
                yield AgentEvent(type="files_changed", data=list(self.tools.changed_files))

            # Continue the loop — LLM will see tool results
            # Fix 8: Store compressed but accurate tool result history so the model
            # can see what tools returned in prior turns after session reload/compact.
            # We keep the first 200 chars of each result — enough for context without
            # bloating the history uncontrollably.
            _tool_summaries = []
            for tc in tool_calls_data:
                # Find the tool result message for this call
                _result_content = ""
                for _msg in reversed(messages):
                    if (
                        _msg.get("role") == "tool"
                        and _msg.get("tool_call_id") == tc["id"]
                    ):
                        _result_content = str(_msg.get("content", ""))[:200]
                        break
                _tool_summaries.append(
                    f"{tc['name']}({', '.join(f'{k}={str(v)[:40]}' for k, v in list(tc['args'].items())[:3])})"
                    f" → {_result_content}"
                )
            self.session.messages.append({
                "role": "assistant",
                "content": (
                    f"[Turn summary — {len(tool_calls_data)} tool call(s)]\n"
                    + "\n".join(_tool_summaries)
                ),
            })
            self._save_session_bg()

            # Proactive compaction advisory: warn when history is 75% of threshold
            _msg_count = len(self.session.messages)
            if _msg_count >= _COMPACT_MSG_THRESHOLD - 10 and _msg_count < _COMPACT_MSG_THRESHOLD:
                yield AgentEvent(
                    type="status",
                    data=(
                        f"⏳ Context at {_msg_count} messages — approaching compaction limit "
                        f"({_COMPACT_MSG_THRESHOLD}). Type /compact to free space now."
                    ),
                )

            # Fix 20: Log tool batch to session log
            _log_event("tool_batch", {
                "tools": [tc["name"] for tc in tool_calls_data],
                "turn": turns,
            })

        # Hit MAX_TURNS
        yield AgentEvent(
            type="error",
            data={"message": f"Reached maximum turns ({self.MAX_TURNS}). Task may be incomplete."}
        )
