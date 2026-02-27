"""
Code intelligence handlers:
  get_symbols, get_dep_graph, run_analysis
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from nvagent.core.symbols import (
    extract_symbols, build_symbol_context, resolve_imports,
)
from nvagent.core.symbols import get_dependency_graph
from nvagent.core.analysis import (
    run_analysis, run_all_linters, detect_linters,
    format_issues, AnalysisIssue,
)
from nvagent.tools.handlers import BaseHandler


class CodeHandler(BaseHandler):
    """Handles get_symbols, get_dep_graph, run_analysis."""

    _SOURCE_EXTS = {
        ".py", ".ts", ".tsx", ".js", ".jsx",
        ".go", ".rs", ".java", ".cs", ".c", ".cpp", ".h", ".hpp",
    }

    # ── get_symbols ───────────────────────────────────────────────────────────

    async def get_symbols(
        self,
        path: str,
        include_imports: bool = False,
        follow_imports: bool = False,
    ) -> str:
        target = self.ctx._resolve_path(path)

        if not target.exists():
            return f"Error: Path not found: {target}"

        if target.is_dir():
            files = sorted(
                [f for f in target.iterdir() if f.is_file() and f.suffix in self._SOURCE_EXTS],
                key=lambda p: p.name,
            )
            if not files:
                return f"No source files found in {target}"
            ctx = build_symbol_context(files, self.ctx.workspace, include_imports=include_imports)
            return ctx or f"No symbols extracted from {target}"

        idx = extract_symbols(target)
        if idx.is_empty():
            return f"No symbols found in {target} (unsupported language or empty file)"

        try:
            rel = target.relative_to(self.ctx.workspace)
        except ValueError:
            rel = target

        lines: list[str] = [f"## Symbols: {rel}  [{idx.language}]", ""]

        if include_imports and idx.imports:
            lines.append("### Imports")
            for imp in idx.imports[:20]:
                lines.append(f"  {imp}")
            if len(idx.imports) > 20:
                lines.append(f"  ... ({len(idx.imports) - 20} more)")
            lines.append("")

        if idx.symbols:
            lines.append("### Definitions")
            for sym in idx.symbols[:80]:
                lines.append(f"  {sym}")
            if len(idx.symbols) > 80:
                lines.append(f"  ... ({len(idx.symbols) - 80} more symbols)")

        if follow_imports:
            loop_gs = asyncio.get_event_loop()
            deps = await loop_gs.run_in_executor(
                None, lambda: resolve_imports(target, self.ctx.workspace, max_depth=1)
            )
            if deps:
                lines.append("")
                lines.append("### Imported workspace files")
                dep_idxs = await asyncio.gather(*[
                    loop_gs.run_in_executor(None, extract_symbols, dep)
                    for dep in deps[:10]
                ])
                for dep, dep_idx in zip(deps[:10], dep_idxs):
                    if not dep_idx.is_empty():
                        try:
                            dep_rel = dep.relative_to(self.ctx.workspace)
                        except ValueError:
                            dep_rel = dep
                        lines.append(f"  ── {dep_rel}  [{dep_idx.language}]")
                        for sym in dep_idx.symbols[:15]:
                            lines.append(f"    {sym}")
                        if len(dep_idx.symbols) > 15:
                            lines.append(f"    ... ({len(dep_idx.symbols) - 15} more)")
                if len(deps) > 10:
                    lines.append(f"  ... ({len(deps) - 10} more imported files)")

        return "\n".join(lines)

    # ── get_dep_graph ─────────────────────────────────────────────────────────

    async def get_dep_graph(
        self,
        path: str,
        show_dependents:  bool = True,
        show_transitive:  bool = False,
        show_external:    bool = True,
        detect_cycles:    bool = False,
        max_depth:        int  = 3,
    ) -> str:
        target = self.ctx._resolve_path(path)
        if not target.exists():
            return f"Error: Path not found: {target}"

        graph = get_dependency_graph(self.ctx.workspace)

        if target.is_dir():
            files = sorted(f for f in target.iterdir() if f.is_file() and f.suffix in self._SOURCE_EXTS)
            if not files:
                return f"No source files found in {target}"
        else:
            files = [target]

        lines: list[str] = []

        for fpath in files:
            try:
                rel = fpath.relative_to(self.ctx.workspace)
            except ValueError:
                rel = fpath

            node = graph.build_file(fpath)
            lines.append(f"## Dependency graph: {rel}  [{node.language}]")

            if node.dep_files:
                lines.append(f"\n### Direct imports ({len(node.dep_files)})")
                for dep in node.dep_files:
                    try:
                        dr = dep.relative_to(self.ctx.workspace)
                    except ValueError:
                        dr = dep
                    lines.append(f"  {dr}")
            else:
                lines.append("\n### Direct imports\n  (none resolved in workspace)")

            if show_dependents:
                deps_on_me = graph.dependents(fpath)
                if deps_on_me:
                    lines.append(f"\n### Imported by ({len(deps_on_me)})")
                    for dep in deps_on_me:
                        try:
                            dr = dep.relative_to(self.ctx.workspace)
                        except ValueError:
                            dr = dep
                        lines.append(f"  {dr}")
                else:
                    loop_dg = asyncio.get_event_loop()
                    await loop_dg.run_in_executor(None, lambda: graph.build_workspace(max_files=500))
                    deps_on_me = graph.dependents(fpath)
                    if deps_on_me:
                        lines.append(f"\n### Imported by ({len(deps_on_me)})")
                        for dep in deps_on_me:
                            try:
                                dr = dep.relative_to(self.ctx.workspace)
                            except ValueError:
                                dr = dep
                            lines.append(f"  {dr}")
                    else:
                        lines.append("\n### Imported by\n  (not imported by any workspace file)")

            if show_transitive:
                lines.append(f"\n### Transitive dependency tree (depth ≤ {max_depth})")
                tree = graph.render_tree(fpath, max_depth=max_depth)
                for tl in tree.splitlines()[1:]:
                    lines.append(tl)

            if show_external and node.external_pkgs:
                lines.append(f"\n### External packages ({len(node.external_pkgs)})")
                for pkg in sorted(set(node.external_pkgs)):
                    lines.append(f"  {pkg}")

            lines.append("")

        if detect_cycles:
            loop_dg = asyncio.get_event_loop()
            await loop_dg.run_in_executor(None, lambda: graph.build_workspace(max_files=1000))
            cycles = graph.detect_cycles()
            if cycles:
                lines.append(f"### ⚠ Circular import cycles ({len(cycles)} detected)")
                for cycle in cycles:
                    lines.append("  " + " → ".join(cycle))
            else:
                lines.append("### ✓ No circular imports detected")

        lines.append(f"\n---\nGraph cache: {graph.stats()}")
        return "\n".join(lines)

    # ── run_analysis ──────────────────────────────────────────────────────────

    async def run_analysis(
        self,
        tool: str,
        path: Optional[str] = None,
        fix: bool = False,
        max_issues: int = 50,
    ) -> str:
        tool_lower = tool.lower()
        target_path = Path(path) if path else None
        if target_path and not target_path.is_absolute():
            target_path = (self.ctx.workspace / target_path).resolve()

        if tool_lower == "detect":
            available = detect_linters(self.ctx.workspace)
            if not available:
                return "No supported linters found in PATH (ruff, mypy, pyright, tsc, eslint)."
            return "Available linters: " + ", ".join(available)

        if tool_lower == "all":
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: run_all_linters(self.ctx.workspace, target_path, fix)
            )
            all_issues: list[AnalysisIssue] = []
            for issues in results.values():
                all_issues.extend(issues)
            if not all_issues:
                return "All linters passed — no issues found."
            _sev_order = {"error": 0, "warning": 1, "note": 2, "info": 3}
            all_issues.sort(key=lambda i: (_sev_order.get(i.severity, 9), i.file, i.line))
            summary = f"{len(all_issues)} issue(s) across {', '.join(results)} — "
            summary += ", ".join(f"{t}: {len(v)}" for t, v in results.items() if v)
            return summary + "\n\n" + format_issues(all_issues, self.ctx.workspace, max_issues=max_issues)

        loop = asyncio.get_event_loop()
        try:
            issues = await loop.run_in_executor(
                None, lambda: run_analysis(tool_lower, self.ctx.workspace, target_path, fix)
            )
        except ValueError as exc:
            return f"Error: {exc}"

        if not issues:
            return f"{tool} — no issues found." + (" (auto-fix applied)" if fix else "")

        _sev_order2 = {"error": 0, "warning": 1, "note": 2, "info": 3}
        issues.sort(key=lambda i: (_sev_order2.get(i.severity, 9), i.file, i.line))
        summary2 = f"{tool}: {len(issues)} issue(s)" + (" (auto-fix applied)" if fix else "")
        return summary2 + "\n\n" + format_issues(issues, self.ctx.workspace, max_issues=max_issues)
