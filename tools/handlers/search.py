"""
Search handlers:
  search_code, find_symbol, find_definition, find_references
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from pathlib import Path
from typing import Optional

from nvagent.core.symbols import (
    find_definition,
    find_references,
    ReferenceSite,
)
from nvagent.core.index import get_workspace_index
from nvagent.tools.handlers import BaseHandler


class SearchHandler(BaseHandler):
    """Handles search_code, find_symbol, find_definition, find_references."""

    # ── search_code ───────────────────────────────────────────────────────────

    async def search_code(
        self,
        query: str,
        path: Optional[str] = None,
        file_pattern: Optional[str] = None,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> str:
        search_path = self.ctx._resolve_path(path) if path else self.ctx.workspace
        if self.ctx._rg_path:
            return await self._search_rg(query, search_path, file_pattern, regex, case_sensitive)
        else:
            return await self._search_python(query, search_path, file_pattern, regex, case_sensitive)

    async def _search_rg(self, query, search_path, file_pattern, regex, case_sensitive) -> str:
        cmd = [self.ctx._rg_path, "--line-number", "--no-heading", "--color=never"]
        if not case_sensitive:
            cmd.append("-i")
        if not regex:
            cmd.append("-F")
        if file_pattern:
            cmd.extend(["-g", file_pattern])
        cmd.extend(["--max-count", "5"])
        cmd.extend(["--", query, str(search_path)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace").strip()

            if not output:
                return f"No matches found for: {query!r}"

            lines = output.splitlines()
            if len(lines) > 100:
                lines = lines[:100]
                output = "\n".join(lines) + f"\n... (showing first 100 of more results)"
            return output
        except Exception:
            return await self._search_python(query, search_path, file_pattern, regex, case_sensitive)

    async def _search_python(self, query, search_path, file_pattern, regex, case_sensitive) -> str:
        pattern = None
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(query, flags)
            except re.error as e:
                return f"Invalid regex: {e}"

        ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
        max_results = 100
        needle = query if case_sensitive else query.lower()
        workspace = self.ctx.workspace

        def _walk_and_search() -> list[str]:
            results: list[str] = []
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if d not in ignore_dirs]
                for fname in files:
                    if file_pattern and not fnmatch.fnmatch(fname, file_pattern):
                        continue
                    fpath = Path(root) / fname
                    if fpath.suffix in {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
                                        ".jpg", ".png", ".gif", ".pdf"}:
                        continue
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    for lineno, line in enumerate(content.splitlines(), 1):
                        if pattern:
                            match = pattern.search(line)
                        else:
                            haystack = line if case_sensitive else line.lower()
                            match = needle in haystack
                        if match:
                            rel_path = fpath.relative_to(workspace)
                            results.append(f"{rel_path}:{lineno}: {line.strip()}")
                            if len(results) >= max_results:
                                return results
            return results

        loop_sp = asyncio.get_event_loop()
        results = await loop_sp.run_in_executor(None, _walk_and_search)

        if not results:
            return f"No matches found for: {query!r}"
        output = "\n".join(results)
        if len(results) >= max_results:
            output += f"\n... (showing first {max_results} results)"
        return output

    # ── find_symbol ───────────────────────────────────────────────────────────

    async def find_symbol(
        self,
        query: str,
        exact: bool = False,
        kinds: Optional[list] = None,
        max_results: int = 30,
    ) -> str:
        idx = get_workspace_index(self.ctx.workspace)
        with idx._lock:
            n_cached = len(idx._cache)
        if n_cached < 5:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: idx.build(max_files=2000))

        if exact:
            matches = idx.find_symbol(query, kinds=kinds)
        else:
            matches = idx.search_symbols(query, kinds=kinds, max_results=max_results)

        if not matches:
            return f"No symbols found matching {query!r}."

        lines = [f"## Symbols matching {query!r} ({len(matches)} result{'s' if len(matches) != 1 else ''}):"]
        for m in matches[:max_results]:
            lines.append(m.render(self.ctx.workspace))
        return "\n".join(lines)

    # ── find_definition ───────────────────────────────────────────────────────

    async def find_definition(
        self,
        name: str,
        hint_file: Optional[str] = None,
    ) -> str:
        hint_paths: list[Path] = []
        if hint_file:
            hp = self.ctx._resolve_path(hint_file)
            if hp.exists():
                hint_paths.append(hp)

        sites = find_definition(name, self.ctx.workspace, hint_paths=hint_paths or None)

        if not sites:
            return f"No definition found for '{name}' in workspace."

        lines = [f"## Definition sites for `{name}` ({len(sites)} found)\n"]
        for s in sites:
            lines.append(s.render(self.ctx.workspace))
        return "\n".join(lines)

    # ── find_references ───────────────────────────────────────────────────────

    async def find_references(
        self,
        name: str,
        hint_file: Optional[str] = None,
        include_definitions: bool = False,
    ) -> str:
        hint_paths: list[Path] = []
        if hint_file:
            hp = self.ctx._resolve_path(hint_file)
            if hp.exists():
                hint_paths.append(hp)

        refs = find_references(
            name, self.ctx.workspace,
            hint_paths=hint_paths or None,
            include_definitions=include_definitions,
        )

        if not refs:
            return f"No references found for '{name}' in workspace."

        by_kind: dict[str, list[ReferenceSite]] = {}
        for r in refs:
            by_kind.setdefault(r.ref_kind, []).append(r)

        kind_order = ["call", "import", "type_hint", "assign", "definition", "unknown"]
        lines = [f"## References to `{name}` ({len(refs)} found)\n"]
        for kind in kind_order:
            group = by_kind.get(kind, [])
            if not group:
                continue
            lines.append(f"### {kind.replace('_', ' ').title()} ({len(group)})")
            for r in group:
                lines.append(r.render(self.ctx.workspace))
            lines.append("")
        return "\n".join(lines)
