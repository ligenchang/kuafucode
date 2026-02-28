"""
Memory handlers:
  update_memory, memory_learn, memory_recall, memory_forget, memory_note
"""

from __future__ import annotations

import asyncio
from typing import Optional

from nvagent.core.state import get_memory
from nvagent.tools.handlers import BaseHandler


class MemoryHandler(BaseHandler):
    """Handles update_memory, memory_learn, memory_recall, memory_forget, memory_note."""

    # ── update_memory ─────────────────────────────────────────────────────────

    async def update_memory(self, content: str, mode: str = "append") -> str:
        memory_file = self.ctx.workspace / ".nvagent" / "memory.md"
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()

        if mode == "replace":
            await loop.run_in_executor(
                None, lambda: memory_file.write_text(content, encoding="utf-8")
            )
            return "✓ Memory replaced."
        else:

            def _read_and_write() -> None:
                existing = memory_file.read_text(encoding="utf-8") if memory_file.exists() else ""
                new_content = existing.rstrip() + "\n\n" + content.strip() + "\n"
                memory_file.write_text(new_content, encoding="utf-8")

            await loop.run_in_executor(None, _read_and_write)
            return "✓ Memory updated."

    # ── memory_learn ──────────────────────────────────────────────────────────

    async def memory_learn(
        self,
        content: str,
        tags: Optional[list] = None,
        file: Optional[str] = None,
    ) -> str:
        mem = get_memory(self.ctx.workspace)
        key = mem.learn(
            content=content,
            tags=[str(t) for t in (tags or [])],
            file=file,
        )
        mem.save()
        return f"✓ Memory stored (key={key}): {content[:80]}"

    # ── memory_recall ─────────────────────────────────────────────────────────

    async def memory_recall(
        self,
        query: str,
        max_results: int = 8,
        tags: Optional[list] = None,
    ) -> str:
        mem = get_memory(self.ctx.workspace)
        entries = mem.recall(
            query=query,
            max_k=min(max(1, max_results), 20),
            tags=[str(t) for t in (tags or [])] or None,
        )
        if not entries:
            return f"No memories found for query: {query!r}"
        lines = [f"## Memory recall: {query!r} ({len(entries)} result(s))\n"]
        for e in entries:
            tag_str = f"  [{', '.join(e.tags)}]" if e.tags else ""
            file_str = f"  ({e.file})" if e.file else ""
            lines.append(f"**{e.key}**{tag_str}{file_str}\n{e.content}\n")
        return "\n".join(lines)

    # ── memory_forget ─────────────────────────────────────────────────────────

    async def memory_forget(self, key: str) -> str:
        mem = get_memory(self.ctx.workspace)
        ok = mem.forget(key)
        if ok:
            mem.save()
            return f"✓ Removed memory entry: {key}"
        return f"No memory entry found with key: {key}"

    # ── memory_note ───────────────────────────────────────────────────────────

    async def memory_note(
        self,
        path: str,
        summary: Optional[str] = None,
        note: Optional[str] = None,
    ) -> str:
        mem = get_memory(self.ctx.workspace)
        mem.file_note(path=path, summary=summary or "", note=note or "")
        mem.save()
        return f"✓ File note saved for {path}"
