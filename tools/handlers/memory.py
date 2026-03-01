"""Memory handlers: update_memory (simple read/write to memory.md)."""

from __future__ import annotations

import asyncio

from nvagent.tools.handlers import BaseHandler


class MemoryHandler(BaseHandler):
    """Handles update_memory for persistent project memory."""

    async def update_memory(self, content: str, mode: str = "append") -> str:
        """Update the .nvagent/memory.md file."""
        memory_file = self.ctx.workspace / ".nvagent" / "memory.md"
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()

        if mode == "replace":
            await loop.run_in_executor(None, lambda: memory_file.write_text(content, encoding="utf-8"))
            return "✓ Memory replaced."

        def _append():
            existing = memory_file.read_text(encoding="utf-8") if memory_file.exists() else ""
            memory_file.write_text(existing.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")

        await loop.run_in_executor(None, _append)
        return "✓ Memory updated."

    # Stubs for backward compatibility with schemas
    async def memory_learn(self, content: str, tags=None, file=None) -> str:
        return await self.update_memory(content, mode="append")

    async def memory_recall(self, query: str, max_results: int = 8, tags=None) -> str:
        memory_file = self.ctx.workspace / ".nvagent" / "memory.md"
        if not memory_file.exists():
            return "No project memory found."
        content = memory_file.read_text(encoding="utf-8")
        return f"## Project Memory\n{content}"

    async def memory_forget(self, key: str) -> str:
        return "Memory forget not supported in simplified mode. Edit .nvagent/memory.md manually."

    async def memory_note(self, path: str, note: str) -> str:
        return await self.update_memory(f"## File note: {path}\n{note}", mode="append")
