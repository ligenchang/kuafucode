"""Context-assembly phase for each agent turn.

Handles:
- Active-file tracking (incremental scan of new messages)
- Retrieval-augmented context (vector index)
- System-prompt assembly with caching and TTL
- Memory-block injection
- Symbol-context overlay

Returns a :class:`ContextResult` that the :class:`~nvagent.core.agent.Agent`
can inspect and then emit the bundled status events to the TUI.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List

from nvagent.core.context import estimate_tokens, extract_active_files, assemble_context
from nvagent.core.symbols import build_symbol_context
from nvagent.core.index import get_workspace_index
from nvagent.core.memory import get_memory
from nvagent.core.retrieval import retrieve_files
from nvagent.core.events import AgentEvent

if TYPE_CHECKING:
    from nvagent.core.agent import Agent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextResult:
    """All outputs produced during the context-assembly phase."""

    system_prompt: str = ""
    """Fully assembled system prompt (context + memory + symbols)."""

    active_paths: List[Path] = field(default_factory=list)
    """Deduplicated list of files referenced so far in the session."""

    retrieved: List[Path] = field(default_factory=list)
    """Files selected by the retrieval index for this turn."""

    token_estimate: int = 0
    """Rough token count of *system_prompt* (char/4 heuristic)."""

    perf: dict = field(default_factory=dict)
    """Timing measurements: 'retrieval', 'context', 'symbols' (seconds)."""

    events: List[AgentEvent] = field(default_factory=list)
    """Status :class:`~nvagent.core.events.AgentEvent` objects to forward to the TUI."""


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

async def build_turn_context(agent: "Agent", user_message: str) -> ContextResult:
    """Assemble the system prompt and gather context for a new turn.

    Mutates the agent's internal caches (``_active_paths_cache``,
    ``_last_scanned_idx``, ``_ctx_cache``, ``_ctx_prefetch_result``,
    ``_sym_ctx_cache``) as a side-effect so subsequent turns benefit from the
    cached state.

    All TUI-readable events are collected in :attr:`ContextResult.events`
    rather than being yielded directly, so the caller controls when to emit
    them.
    """
    result = ContextResult()
    loop_ex = asyncio.get_event_loop()
    perf = result.perf
    events = result.events

    # ── Incremental active-file scan ─────────────────────────────────────────
    _new_msgs = agent.session.messages[agent._last_scanned_idx :]
    if _new_msgs:
        _seen = {str(p) for p in agent._active_paths_cache}
        for _np in extract_active_files(_new_msgs, agent.workspace):
            _s = str(_np)
            if _s not in _seen:
                agent._active_paths_cache.append(_np)
                _seen.add(_s)
        agent._last_scanned_idx = len(agent.session.messages)

    result.active_paths = agent._active_paths_cache
    mem = get_memory(agent.workspace)
    _undo_depth = len(agent.tools.undo_stack)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    _t_stage = time.monotonic()
    try:
        retrieved_scored = await loop_ex.run_in_executor(
            None,
            lambda: retrieve_files(user_message, agent.workspace, top_k=5),
        )
        result.retrieved = [sf.path for sf in retrieved_scored]
    except Exception:
        result.retrieved = []
    perf["retrieval"] = time.monotonic() - _t_stage
    events.append(AgentEvent(type="status", data=f"⏱ retrieval {perf['retrieval']:.2f}s"))

    # ── Context assembly (with TTL cache + speculative prefetch) ─────────────
    _now = loop_ex.time()
    _active_paths_key = frozenset(result.active_paths)
    _cache_hit = False

    if (
        agent._ctx_cache is not None
        and not agent.tools.changed_files
        and _now - agent._ctx_cache[0] < agent._CTX_CACHE_TTL
        and agent._ctx_cache[2] == _active_paths_key
    ):
        result.system_prompt = agent._ctx_cache[1]
        _cache_hit = True
        perf["context"] = 0.0
        events.append(AgentEvent(type="status", data="⏱ context hit (0.00s)"))
    elif (
        agent._ctx_prefetch_result is not None
        and agent._ctx_prefetch_result[1] == _active_paths_key
        and agent._ctx_prefetch_result[2] == _undo_depth
    ):
        result.system_prompt = agent._ctx_prefetch_result[0]
        agent._ctx_prefetch_result = None
        _cache_hit = True
        perf["context"] = 0.0
        agent._ctx_cache = (_now, result.system_prompt, _active_paths_key, _undo_depth)
        events.append(AgentEvent(type="status", data="⏱ context prefetch (0.00s)"))
    else:
        _t_stage = time.monotonic()
        result.system_prompt = await loop_ex.run_in_executor(
            None,
            lambda: assemble_context(
                query=user_message,
                workspace=agent.workspace,
                active_paths=result.active_paths,
                config=agent.config,
                retrieved_paths=result.retrieved,
                max_tokens=8_000,
            ),
        )
        perf["context"] = time.monotonic() - _t_stage
        events.append(AgentEvent(type="status", data=f"⏱ context {perf['context']:.2f}s"))
        agent._ctx_cache = (_now, result.system_prompt, _active_paths_key, _undo_depth)

    # ── Memory block injection ────────────────────────────────────────────────
    if not _cache_hit:
        try:
            memory_ctx = mem.to_context_block(user_message)
            if memory_ctx:
                result.system_prompt += memory_ctx
                agent._ctx_cache = (
                    agent._ctx_cache[0],
                    result.system_prompt,
                    _active_paths_key,
                    _undo_depth,
                )
        except Exception:
            pass

    # ── Symbol context ────────────────────────────────────────────────────────
    _t_stage = time.monotonic()
    if result.active_paths:
        try:
            _sym_key = frozenset(str(p) for p in result.active_paths)
            if (
                not agent.tools.changed_files
                and agent._sym_ctx_cache is not None
                and agent._sym_ctx_cache[0] == _sym_key
            ):
                symbol_ctx = agent._sym_ctx_cache[1]
            else:
                symbol_ctx = build_symbol_context(
                    result.active_paths,
                    agent.workspace,
                    symbol_fetcher=get_workspace_index(agent.workspace).symbols_for,
                )
                agent._sym_ctx_cache = (_sym_key, symbol_ctx)
            if symbol_ctx:
                result.system_prompt += symbol_ctx
        except Exception:
            pass
    perf["symbols"] = time.monotonic() - _t_stage

    result.token_estimate = estimate_tokens(result.system_prompt)
    events.append(
        AgentEvent(
            type="status",
            data=(
                f"⏱ symbols {perf['symbols']:.2f}s"
                f"  context ~{result.token_estimate:,} tok"
            ),
        )
    )

    return result
