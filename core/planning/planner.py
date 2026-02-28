"""
Planning & Reasoning layer for nvagent.

Provides:
  - Task decomposition into ordered, concrete steps (via LLM)
  - Per-step status tracking: pending → active → done / failed / skipped
  - Tool-selection hints embedded in each step
  - Self-reflection on repeated failures with recovery suggestions
  - Format helpers that inject the plan into the system prompt

Architecture note
─────────────────
The Planner makes two lightweight LLM calls (non-streaming still uses
stream_chat under the hood so the same client/retry logic applies):

  1. decompose(task)  →  Plan            (called once before the agent loop)
  2. reflect(…)       →  str             (called after ≥2 consecutive error batches)

The Plan's to_prompt_block() is appended to the main system prompt so the
agent LLM always has the full task structure in context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nvagent.core.client import NIMClient, TaskType


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


_STATUS_ICON: dict[StepStatus, str] = {
    StepStatus.PENDING: "○",
    StepStatus.ACTIVE: "◉",
    StepStatus.DONE: "✓",
    StepStatus.FAILED: "✗",
    StepStatus.SKIPPED: "⊘",
}


@dataclass
class PlanStep:
    id: int
    title: str
    description: str = ""
    tool_hint: Optional[str] = None  # suggested tool name from TOOL_SCHEMAS
    status: StepStatus = StepStatus.PENDING

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tool_hint": self.tool_hint,
            "status": self.status.value,
        }


@dataclass
class Plan:
    task: str
    steps: list[PlanStep] = field(default_factory=list)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def current_step(self) -> Optional[PlanStep]:
        """First non-terminal step (pending or active)."""
        for s in self.steps:
            if s.status not in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED):
                return s
        return None

    @property
    def done(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)

    @property
    def completed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.DONE]

    def step_by_id(self, step_id: int) -> Optional[PlanStep]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    # ── Mutations ─────────────────────────────────────────────────────────────

    def activate(self, step_id: int) -> Optional[PlanStep]:
        """Mark *step_id* as active (clears any previous active step first)."""
        for s in self.steps:
            if s.status == StepStatus.ACTIVE:
                s.status = StepStatus.PENDING
        step = self.step_by_id(step_id)
        if step:
            step.status = StepStatus.ACTIVE
        return step

    def mark_done(self, step_id: int) -> Optional[PlanStep]:
        step = self.step_by_id(step_id)
        if step:
            step.status = StepStatus.DONE
        return step

    def mark_failed(self, step_id: int) -> Optional[PlanStep]:
        step = self.step_by_id(step_id)
        if step:
            step.status = StepStatus.FAILED
        return step

    def mark_skipped(self, step_id: int) -> Optional[PlanStep]:
        step = self.step_by_id(step_id)
        if step:
            step.status = StepStatus.SKIPPED
        return step

    def advance(self) -> Optional[PlanStep]:
        """
        Mark the current active/pending step done and activate the next one.
        Returns the newly activated step, or None if the plan is complete.
        """
        # Mark current active step done
        for s in self.steps:
            if s.status == StepStatus.ACTIVE:
                s.status = StepStatus.DONE
                break
        # Activate the next pending step
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                s.status = StepStatus.ACTIVE
                return s
        return None

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.steps]

    def to_prompt_block(self) -> str:
        """
        Render the plan as a structured block for injection into the system
        prompt.  The agent LLM reads this to understand what it should do next
        and which tools to favour for each step.
        """
        if not self.steps:
            return ""

        lines = [
            "",
            "═════════════════════════════════════════════",
            "TASK PLAN  (follow these steps in order)",
            "═════════════════════════════════════════════",
            f"Goal: {self.task}",
            "",
            "Instructions:",
            "  • Work through each step in sequence.",
            "  • Before each step, mentally confirm: 'What do I need to do?'",
            "  • If a step fails, reflect briefly and try an alternative approach",
            "    before moving on.  Do NOT silently skip failures.",
            "  • Use the suggested tool where provided — it is your best starting",
            "    point, but use your judgment.",
            "",
        ]
        for s in self.steps:
            icon = _STATUS_ICON[s.status]
            hint = f"  [→ {s.tool_hint}]" if s.tool_hint else ""
            lines.append(f"  {icon} Step {s.id}: {s.title}{hint}")
            if s.description:
                lines.append(f"       {s.description}")
        lines += ["", "═════════════════════════════════════════════", ""]
        return "\n".join(lines)

    def progress_summary(self) -> str:
        """One-line progress string, e.g. '3/6 steps done'."""
        done = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        total = len(self.steps)
        return f"{done}/{total} steps done"


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
You are a senior software-engineering task planner.
Decompose the given task into a short, ordered list of concrete steps.

Rules
─────
• 3–8 steps max; merge trivial sub-steps.
• Each step is a single actionable unit (e.g. "Read API entry-point file",
  "Add cache abstraction", "Run tests", "Fix failures").
• Assign each step the single most useful tool from this list:
    read_file, write_file, list_dir, run_command, search_code,
    git_status, git_diff, read_url, get_symbols, update_memory
• Keep titles short (≤10 words).  Use the description field for detail.
• Output ONLY a valid JSON object — no markdown fences, no commentary.

JSON format (strict):
{
  "steps": [
    {"id": 1, "title": "...", "description": "...", "tool_hint": "..."},
    ...
  ]
}
"""

_REFLECT_SYSTEM = """\
You are an expert debugging assistant helping a coding agent recover from a failure.

Analyse the failure and provide a concise recovery strategy (2–5 bullet points,
≤150 words total).  Be concrete:
  • What likely went wrong?
  • What alternative tool or approach should the agent try?
  • What should the agent verify or check first?

Respond in plain text — no JSON, no markdown fences.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────


class Planner:
    """
    Generates task plans and reflection notes via LLM.

    Uses TaskType.FAST model for both calls (speed > quality is acceptable
    for planning; the main agent handles the heavy lifting).
    """

    def __init__(self, client: "NIMClient") -> None:
        self.client = client

    # ── Public API ────────────────────────────────────────────────────────────

    async def decompose(
        self,
        task: str,
        context_hint: str = "",
    ) -> Plan:
        """
        Ask the LLM to break *task* into an ordered Plan.

        *context_hint* is a brief excerpt of project context to help the model
        give relevant, specific steps (capped at 600 chars to stay fast).
        """
        user_content = f"Task: {task}"
        if context_hint:
            user_content += f"\n\nProject context (brief):\n{context_hint[:600]}"

        messages = [
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        raw = await self._collect(messages)
        return self._parse_plan(task, raw)

    async def reflect(
        self,
        step_title: str,
        failure_detail: str,
        task: str,
        completed_step_titles: list[str] | None = None,
    ) -> str:
        """
        Ask the LLM to reflect on why *step_title* failed and suggest recovery.
        Returns plain-text advice suitable for injection as a user message.
        """
        completed_so_far = ""
        if completed_step_titles:
            completed_so_far = "\n".join(f"  ✓ {t}" for t in completed_step_titles)

        user_content = (
            f"Overall task: {task}\n"
            f"Failed step: {step_title}\n"
            f"Failure detail:\n{failure_detail[:800]}\n"
        )
        if completed_so_far:
            user_content += f"\nSteps completed so far:\n{completed_so_far}"

        messages = [
            {"role": "system", "content": _REFLECT_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        reflection = await self._collect(messages)
        return reflection.strip() or (
            "The previous approach failed. Try a different tool or verify "
            "file paths before retrying."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _collect(self, messages: list[dict]) -> str:
        """Drain the stream into a single string."""
        # Import here to avoid circular imports at module level
        from nvagent.core.client import TaskType

        raw = ""
        try:
            async for ev in self.client.stream_chat(
                messages=messages,
                tools=[],
                task=TaskType.FAST,
            ):
                if ev.type == "token":
                    raw += ev.data
                elif ev.type in ("done", "error"):
                    break
        except Exception:
            pass
        return raw

    def _parse_plan(self, task: str, raw: str) -> Plan:
        """Parse LLM JSON into a Plan; returns an empty plan on failure."""
        plan = Plan(task=task)
        try:
            # Be tolerant: extract first {...} block even if model adds prose
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return plan
            data = json.loads(m.group())
            for i, raw_step in enumerate(data.get("steps", []), start=1):
                plan.steps.append(
                    PlanStep(
                        id=raw_step.get("id", i),
                        title=raw_step.get("title", f"Step {i}"),
                        description=raw_step.get("description", ""),
                        tool_hint=raw_step.get("tool_hint") or None,
                    )
                )
            # Activate the first step
            if plan.steps:
                plan.steps[0].status = StepStatus.ACTIVE
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass
        return plan
