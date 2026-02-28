"""
Todo list handlers:
  todo_write, todo_read
"""

from __future__ import annotations

from nvagent.tools.handlers import BaseHandler


class TodoHandler(BaseHandler):
    """Handles todo_write and todo_read."""

    _VALID_STATUS = frozenset({"pending", "in_progress", "completed", "cancelled"})
    _VALID_PRIORITY = frozenset({"high", "medium", "low"})

    _STATUS_ICON = {
        "pending": "○",
        "in_progress": "◉",
        "completed": "✓",
        "cancelled": "✗",
    }
    _PRIORITY_LABEL = {"high": "[H]", "medium": "[M]", "low": "[L]"}
    _STATUS_ORDER = {"in_progress": 0, "pending": 1, "completed": 2, "cancelled": 3}
    _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

    # ── todo_write ────────────────────────────────────────────────────────────

    async def todo_write(self, todos: list) -> str:
        validated: list[dict] = []
        errors: list[str] = []
        in_progress_count = 0

        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                errors.append(f"Item {i}: expected object, got {type(item).__name__}")
                continue
            tid = str(item.get("id", "")).strip()
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).strip()
            priority = str(item.get("priority", "medium")).strip()
            if not tid:
                errors.append(f"Item {i}: 'id' is required")
                continue
            if not content:
                errors.append(f"Item {i}: 'content' is required")
                continue
            if status not in self._VALID_STATUS:
                errors.append(f"Item {i}: invalid status {status!r}")
                continue
            if priority not in self._VALID_PRIORITY:
                errors.append(f"Item {i}: invalid priority {priority!r}")
                continue
            if status == "in_progress":
                in_progress_count += 1
            validated.append(
                {"id": tid, "content": content, "status": status, "priority": priority}
            )

        if errors:
            return "Error in todo list:\n" + "\n".join(f"  • {e}" for e in errors)

        if in_progress_count > 1:
            return (
                f"Error: {in_progress_count} todos have status 'in_progress'. "
                "Only 1 task should be in progress at a time."
            )

        self.ctx._todos = validated

        counts: dict[str, int] = {s: 0 for s in self._VALID_STATUS}
        for t in validated:
            counts[t["status"]] += 1
        summary = (
            f"✓ Todo list updated: {len(validated)} task(s) — "
            f"{counts['pending']} pending, {counts['in_progress']} in progress, "
            f"{counts['completed']} completed, {counts['cancelled']} cancelled."
        )
        return summary

    # ── todo_read ─────────────────────────────────────────────────────────────

    async def todo_read(self) -> str:
        if not self.ctx._todos:
            return "Todo list is empty. Use todo_write to create tasks."

        sorted_todos = sorted(
            self.ctx._todos,
            key=lambda t: (
                self._STATUS_ORDER.get(t["status"], 9),
                self._PRIORITY_ORDER.get(t["priority"], 9),
            ),
        )

        lines = ["## Task List\n"]
        for t in sorted_todos:
            icon = self._STATUS_ICON.get(t["status"], "?")
            pri = self._PRIORITY_LABEL.get(t["priority"], "")
            content = t["content"]
            tid = t["id"]
            lines.append(f"  {icon} {pri} [{tid}] {content}")

        counts: dict[str, int] = {}
        for t in self.ctx._todos:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        pending = counts.get("pending", 0)
        active = counts.get("in_progress", 0)
        completed = counts.get("completed", 0)
        cancelled = counts.get("cancelled", 0)
        lines.append(
            f"\n  {len(self.ctx._todos)} total — "
            f"{active} active, {pending} pending, {completed} done, {cancelled} cancelled"
        )
        return "\n".join(lines)
