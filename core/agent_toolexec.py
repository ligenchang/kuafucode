"""Tool-batch execution for a single agent turn.

:class:`ToolBatchExecutor` is an async iterator that:

1. Optionally pre-launches valid tool calls concurrently.
2. Validates required arguments for each tool call.
3. Executes each tool (or awaits its pre-launched task).
4. Applies post-write validation, loop detection, and structured error hints.
5. Appends ``role=tool`` messages to the live message list.
6. Enforces per-turn tool-call budgets.
7. Deduplicates ``read_file`` results already in context.
8. Updates plan progress and triggers failure reflection when needed.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from nvagent.core.events import AgentEvent
from nvagent.core.feedback import classify_tool_error, RETRY_HINTS
from nvagent.core.planner import StepStatus

if TYPE_CHECKING:
    from nvagent.core.agent import Agent
    from nvagent.core.planner import Plan

logger = logging.getLogger(__name__)

# Tool names that count as "write" operations (used for plan-step tracking and
# post-batch ``_prev_batch_wrote_files`` flag).
_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file", "write_files", "edit_file", "str_replace_editor", "apply_patch",
})
_RUN_TOOLS: frozenset[str] = frozenset({"run_command", "run_tests", "run_formatter"})
_PROGRESS_TOOLS = _WRITE_TOOLS | _RUN_TOOLS

# Common hallucinated names → correct equivalents (populated from training data bleed)
_TOOL_ALIASES: dict[str, str] = {
    "execute_bash": "run_command",
    "bash": "run_command",
    "shell": "run_command",
    "run_bash": "run_command",
    "execute_command": "run_command",
    "computer": "run_command",
    "str_replace": "str_replace_editor",
    "create_file": "write_file",
    "write_files": "write_file",
    "read_files": "read_file",
    "view_file": "read_file",
    "cat_file": "read_file",
    "list_files": "list_directory",
    "ls": "list_directory",
}


def _closest_tool(name: str, schemas_by_name: dict) -> str:
    """Return the most likely correct tool name for a hallucinated *name*.

    Checks the hard-coded alias table first, then falls back to a simple
    substring/prefix match against available tool names.
    """
    if name in _TOOL_ALIASES and _TOOL_ALIASES[name] in schemas_by_name:
        return _TOOL_ALIASES[name]
    name_lower = name.lower().replace("_", "").replace("-", "")
    for available in schemas_by_name:
        av_lower = available.lower().replace("_", "")
        if name_lower in av_lower or av_lower in name_lower:
            return available
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolBatchResult:
    """Data produced by :class:`ToolBatchExecutor` after iteration completes."""

    wrote_files: bool = False
    """True if any write/edit/patch tool was successfully called."""

    per_turn_limit_hit: bool = False
    """True when the per-turn tool-call cap was reached mid-batch."""

    fatal: bool = False
    """True when execution was cancelled mid-batch."""

    perf: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Async iterator
# ──────────────────────────────────────────────────────────────────────────────

class ToolBatchExecutor:
    """Async-iterable executor for one batch of tool calls from the LLM.

    Usage::

        executor = ToolBatchExecutor(agent, tool_calls_data, messages,
                                     schemas_by_name, plan, consecutive_errors,
                                     last_failure_detail, log_event)
        async for event in executor:
            yield event
        result = executor.result
        # also read executor.consecutive_error_batches / executor.last_failure_detail
    """

    def __init__(
        self,
        agent: "Agent",
        tool_calls_data: list,
        messages: list,
        schemas_by_name: dict,
        plan: Optional["Plan"],
        consecutive_error_batches: int,
        last_failure_detail: str,
        log_event,  # callable(kind, data)
        assistant_content: str = "",
    ) -> None:
        self._agent = agent
        self._tool_calls = tool_calls_data
        self._messages = messages
        self._schemas = schemas_by_name
        self._plan = plan
        self._log_event = log_event
        self._assistant_content = assistant_content

        # Mutable state that caller reads back after iteration
        self.consecutive_error_batches = consecutive_error_batches
        self.last_failure_detail = last_failure_detail
        self.result = ToolBatchResult()

    def __aiter__(self):
        return self._process()

    async def _process(self):  # noqa: C901
        agent = self._agent
        tool_calls_data = self._tool_calls
        messages = self._messages
        schemas_by_name = self._schemas
        plan = self._plan
        result = self.result

        # ── Append assistant message with tool_calls ──────────────────────────
        openai_tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["args_raw"]},
            }
            for tc in tool_calls_data
        ]
        messages.append({
            "role": "assistant",
            "content": self._assistant_content or None,
            "tool_calls": openai_tool_calls,
        })

        agent.tools.begin_turn()

        # ── Optional concurrent pre-launch ────────────────────────────────────
        _exec_tasks: dict[str, asyncio.Task] = {}
        _can_parallelize = (
            not agent.tools.confirm_fn
            and not agent.config.agent.dry_run
            and len(tool_calls_data) > 1
        )
        if _can_parallelize:
            for _tc in tool_calls_data:
                _n, _a = _tc["name"], _tc["args"]
                _s = schemas_by_name.get(_n)
                # Skip unknown tools — they get an error message in the main loop
                if not _s:
                    continue
                _req = _s["function"].get("parameters", {}).get("required", [])
                if any(r not in _a or _a[r] is None or _a[r] == "" for r in _req):
                    continue
                _exec_tasks[_tc["id"]] = asyncio.create_task(
                    agent.tools.execute(_n, _a)
                )

        _turn_tool_calls = 0
        _per_turn_limit = agent.config.safety.max_tool_calls_per_turn

        # ── Per-tool execution ────────────────────────────────────────────────
        for tc in tool_calls_data:
            if agent._cancelled:
                yield AgentEvent(type="error", data={"message": "Cancelled."})
                result.fatal = True
                return

            name = tc["name"]
            args = tc["args"]

            # Reject calls to tools not in the active schema
            schema = schemas_by_name.get(name)
            if schema is None:
                _suggestion = _closest_tool(name, schemas_by_name)
                _suggest_str = (
                    f" Did you mean `{_suggestion}`?" if _suggestion else ""
                )
                _available = ", ".join(sorted(schemas_by_name)[:20])
                _err_msg = (
                    f"Unknown tool '{name}'. This tool is not available."
                    f"{_suggest_str}\n"
                    f"Available tools (first 20): {_available}\n"
                    "Use only the tools listed above."
                )
                yield AgentEvent(type="tool_start", data={"name": name, "args": args})
                yield AgentEvent(
                    type="tool_result",
                    data={"name": name, "result": f"Error: {_err_msg}"},
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: {_err_msg}",
                })
                # Inject a one-shot system correction so the model stops retrying
                _correction = (
                    f"[System] You called the non-existent tool `{name}`."
                    f"{_suggest_str}\n"
                    "Only call tools from the provided schema. "
                    "Do not invent tool names."
                )
                messages.append({"role": "user", "content": _correction})
                agent.session.messages.append({"role": "user", "content": _correction})
                yield AgentEvent(
                    type="status",
                    data=f"[unknown-tool] '{name}' not in schema — corrective hint injected",
                )
                continue

            # schema is guaranteed non-None here
            required = schema["function"].get("parameters", {}).get("required", [])
            missing = [
                r for r in required
                if r not in args or args[r] is None or args[r] == ""
            ]
            if missing:
                err_msg = (
                    f"Missing required argument(s) for {name}: {', '.join(missing)}"
                )
                yield AgentEvent(type="tool_start", data={"name": name, "args": args})
                yield AgentEvent(
                    type="tool_result",
                    data={"name": name, "result": f"Error: {err_msg}"},
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: {err_msg}",
                })
                if agent.config.safety.loop_detection:
                    agent.loop_detector.record(name, args)
                agent._missing_arg_streak[name] = (
                    agent._missing_arg_streak.get(name, 0) + 1
                )
                if agent._missing_arg_streak[name] >= 3:
                    _hint = (
                        f"[Error — tool call stuck] You have called '{name}' "
                        f"{agent._missing_arg_streak[name]} consecutive times with "
                        f"missing required argument(s): {', '.join(missing)}.\n"
                        f"Required fields: {', '.join(required)}.\n"
                        "You MUST provide ALL required arguments with concrete, "
                        "non-empty values. If you do not yet know a required value, "
                        "state that explicitly instead of calling the tool with "
                        "placeholder or empty arguments."
                    )
                    messages.append({"role": "user", "content": _hint})
                    agent.session.messages.append(
                        {"role": "user", "content": _hint}
                    )
                    yield AgentEvent(
                        type="status",
                        data=(
                            f"[stuck] '{name}' called "
                            f"{agent._missing_arg_streak[name]}x "
                            "with missing args — hint injected"
                        ),
                    )
                    agent._missing_arg_streak[name] = 0
                continue

            yield AgentEvent(type="tool_start", data={"name": name, "args": args})
            self._log_event(
                "tool_call",
                {"name": name, "args": {k: str(v)[:80] for k, v in args.items()}},
            )

            if agent.config.agent.safe_mode and name in ("write_file", "delete_file"):
                yield AgentEvent(
                    type="status", data=f"[safe_mode] Executing {name}..."
                )

            # Execute (use pre-launched task if available)
            try:
                if tc["id"] in _exec_tasks:
                    tool_result = await _exec_tasks[tc["id"]]
                else:
                    tool_result = await agent.tools.execute(name, args)
            except Exception as exc:
                tool_result = f"Tool error: {exc}"
            else:
                agent._missing_arg_streak.pop(name, None)

            # Post-write validation
            if (
                agent.config.safety.validate_writes
                and name in ("write_file", "edit_file", "str_replace_editor", "apply_patch")
                and not tool_result.startswith("Error:")
            ):
                try:
                    _wp_str = args.get("path", "")
                    if _wp_str:
                        _wp = (
                            (agent.workspace / _wp_str)
                            if not Path(_wp_str).is_absolute()
                            else Path(_wp_str)
                        )
                        vr = await agent.validator.validate_file_async(_wp)
                        if not vr.ok:
                            tool_result += "\n" + f"[Validation] {vr.to_str()}"
                            yield AgentEvent(
                                type="safety_violation",
                                data={
                                    "kind": "validation",
                                    "message": vr.to_str(),
                                    "fatal": False,
                                },
                            )
                except Exception:
                    pass

            if agent.config.safety.loop_detection:
                agent.loop_detector.record(name, args)

            agent.resource_guard.update(
                tool_calls=1,
                output_bytes=len(tool_result.encode("utf-8", errors="replace")),
            )

            # Structured error feedback
            _is_tool_error = (
                tool_result.startswith("Error:")
                or tool_result.startswith("Tool error:")
            )
            if _is_tool_error:
                _err_cat = classify_tool_error(name, tool_result)
                if _err_cat:
                    _err_key = (name, _err_cat)
                    agent._tool_error_streak[_err_key] = (
                        agent._tool_error_streak.get(_err_key, 0) + 1
                    )
                    if agent._tool_error_streak[_err_key] >= 2:
                        _hint_text = RETRY_HINTS.get(_err_cat, "")
                        if _hint_text:
                            _retry_hint = (
                                f"[Retry hint — '{name}' failed "
                                f"{agent._tool_error_streak[_err_key]}x "
                                f"with error type '{_err_cat}']\n{_hint_text}"
                            )
                            messages.append({"role": "user", "content": _retry_hint})
                            agent.session.messages.append(
                                {"role": "user", "content": _retry_hint}
                            )
                            yield AgentEvent(
                                type="status",
                                data=(
                                    f"[retry-hint] {_err_cat} on '{name}' "
                                    f"(x{agent._tool_error_streak[_err_key]})"
                                ),
                            )
                            agent._tool_error_streak[_err_key] = 0
            else:
                for _ek in [k for k in agent._tool_error_streak if k[0] == name]:
                    del agent._tool_error_streak[_ek]

            yield AgentEvent(
                type="tool_result", data={"name": name, "result": tool_result}
            )
            self._log_event(
                "tool_result",
                {
                    "name": name,
                    "result": tool_result[:200],
                    "ok": not tool_result.startswith("Error"),
                },
            )

            # Per-turn tool budget
            _turn_tool_calls += 1
            if _turn_tool_calls >= _per_turn_limit:
                _budget_msg = (
                    f"[Tool budget reached — {_turn_tool_calls} tool call(s) executed "
                    f"this turn (limit: {_per_turn_limit}). "
                    "Summarise what you have done so far and continue with the "
                    "remaining steps in your next response."
                )
                messages.append({"role": "user", "content": _budget_msg})
                agent.session.messages.append({"role": "user", "content": _budget_msg})
                yield AgentEvent(
                    type="status",
                    data=f"[budget] Turn tool limit ({_per_turn_limit}) reached — forcing new LLM turn",
                )
                for _rem_tc in tool_calls_data:
                    _t2 = _exec_tasks.get(_rem_tc["id"])
                    if _t2 and not _t2.done():
                        _t2.cancel()
                result.per_turn_limit_hit = True
                # Still need to append the current tool result before breaking
                _msg_content = self._dedup_read(name, args, tool_result, agent)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _msg_content,
                })
                break

            # Context deduplication for read_file
            _msg_content = self._dedup_read(name, args, tool_result, agent)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": _msg_content,
            })

        agent.tools.end_turn()

        # ── Loop detection ────────────────────────────────────────────────────
        if agent.config.safety.loop_detection and agent.loop_detector.is_looping():
            desc = agent.loop_detector.description()
            yield AgentEvent(
                type="safety_violation",
                data={"kind": "loop", "message": desc, "fatal": False},
            )
            _hint = (
                f"[Safety — loop detected] {desc}\n"
                "You must try a completely different approach. "
                "Do NOT repeat the same tool calls with the same arguments."
            )
            messages.append({"role": "user", "content": _hint})
            agent.session.messages.append({"role": "user", "content": _hint})
            agent.loop_detector.reset()

        # ── Plan step tracking and failure reflection ─────────────────────────
        if plan and plan.steps:
            _batch_tool_results: List[str] = [
                msg["content"]
                for msg in messages
                if msg.get("role") == "tool"
            ][-len(tool_calls_data):]
            batch_had_error = any(
                isinstance(r, str)
                and (r.startswith("Error:") or r.startswith("Tool error:"))
                for r in _batch_tool_results
            )

            if batch_had_error:
                self.consecutive_error_batches += 1
                for msg in reversed(messages):
                    if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                        if msg["content"].startswith(("Error:", "Tool error:")):
                            self.last_failure_detail = msg["content"]
                            break
                if plan.current_step:
                    plan.mark_failed(plan.current_step.id)
                    yield AgentEvent(
                        type="plan_update",
                        data={"steps": plan.to_list(), "progress": plan.progress_summary()},
                    )

                if self.consecutive_error_batches >= 2:
                    self.consecutive_error_batches = 0
                    _step_title = (
                        plan.current_step.title
                        if plan.current_step
                        else "(unknown step)"
                    )
                    _done_titles = [s.title for s in plan.completed_steps]
                    yield AgentEvent(type="status", data="Reflecting on failures…")
                    try:
                        reflection_text = await agent.planner.reflect(
                            step_title=_step_title,
                            failure_detail=self.last_failure_detail,
                            task=agent._current_task,  # set by agent.run()
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
                        "content": f"[Self-reflection — recovery hint]\n{reflection_text}",
                    })
                    agent.session.messages.append({
                        "role": "user",
                        "content": f"[Self-reflection — recovery hint]\n{reflection_text}",
                    })
            else:
                self.consecutive_error_batches = 0
                if plan.current_step:
                    step = plan.current_step
                    _batch_names = {tc["name"] for tc in tool_calls_data}
                    hint = getattr(step, "tool_hint", None) or ""
                    _step_done = (
                        (hint and hint in _batch_names)
                        or (not hint and bool(_batch_names & _PROGRESS_TOOLS))
                    )
                    if _step_done:
                        plan.advance()
                        yield AgentEvent(
                            type="plan_update",
                            data={
                                "steps": plan.to_list(),
                                "progress": plan.progress_summary(),
                            },
                        )

        # Mark wrote_files
        _batch_names_final = {tc["name"] for tc in tool_calls_data}
        if _batch_names_final & _WRITE_TOOLS:
            result.wrote_files = True

        self._log_event(
            "tool_batch",
            {"tools": [tc["name"] for tc in tool_calls_data]},
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_read(name: str, args: dict, raw_result: str, agent: "Agent") -> str:
        """Return a stub replacement when *read_file* re-reads an unchanged file."""
        if name != "read_file" or raw_result.startswith("Error"):
            return raw_result
        _rf_path_arg = args.get("path", "")
        _rf_no_range = (
            args.get("start_line") is None and args.get("end_line") is None
        )
        if not (_rf_path_arg and _rf_no_range):
            return raw_result
        _rf_abs = str(
            (agent.workspace / _rf_path_arg).resolve()
            if not _rf_path_arg.startswith("/")
            else _rf_path_arg
        )
        _rf_mtime = agent.tools.last_read_mtime(_rf_abs)
        if _rf_mtime is None:
            return raw_result
        _prev_mtime = agent._context_shown_reads.get(_rf_abs)
        if _prev_mtime is not None and _prev_mtime == _rf_mtime:
            return (
                "[File unchanged since previous read — "
                "content already in context. No re-injection needed.]"
            )
        agent._context_shown_reads[_rf_abs] = _rf_mtime
        return raw_result
