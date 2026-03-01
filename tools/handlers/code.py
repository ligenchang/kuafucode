"""Code intelligence handlers: get_symbols, get_dep_graph, run_analysis."""

from __future__ import annotations

import ast
import asyncio
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from nvagent.tools.handlers import BaseHandler

_SOURCE_EXTS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h"}


def _extract_symbols_python(path: Path) -> list[str]:
    """Extract function/class definitions from a Python file using ast."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                symbols.append(f"  def {node.name}({', '.join(args)})  [line {node.lineno}]")
            elif isinstance(node, ast.ClassDef):
                symbols.append(f"  class {node.name}  [line {node.lineno}]")
        return symbols
    except Exception:
        return []


def _extract_symbols_generic(path: Path) -> list[str]:
    """Extract function/class definitions using regex for non-Python files."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        patterns = [
            (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
            (r"^\s*(?:pub\s+)?fn\s+(\w+)", "fn"),
            (r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+)\s+(\w+)\s*\(", "method"),
            (r"^\s*(?:func|def)\s+(\w+)", "func"),
            (r"^\s*type\s+(\w+)\s*(?:struct|interface)", "type"),
        ]
        symbols = []
        for i, line in enumerate(source.splitlines(), 1):
            for pattern, kind in patterns:
                m = re.match(pattern, line)
                if m:
                    symbols.append(f"  {kind} {m.group(1)}  [line {i}]")
                    break
        return symbols
    except Exception:
        return []


class CodeHandler(BaseHandler):
    """Handles get_symbols, get_dep_graph, run_analysis."""

    async def get_symbols(self, path: str, include_imports: bool = False, follow_imports: bool = False) -> str:
        """Extract symbol definitions from a file or directory."""
        target = self.ctx._resolve_path(path)
        if not target.exists():
            return f"Error: Path not found: {target}"

        loop = asyncio.get_event_loop()

        if target.is_dir():
            files = sorted(f for f in target.iterdir() if f.is_file() and f.suffix in _SOURCE_EXTS)
            if not files:
                return f"No source files found in {target}"
            results = []
            for f in files[:20]:
                syms = await loop.run_in_executor(None, _extract_symbols_python if f.suffix == ".py" else _extract_symbols_generic, f)
                if syms:
                    rel = f.relative_to(self.ctx.workspace) if f.is_relative_to(self.ctx.workspace) else f
                    results.append(f"### {rel}\n" + "\n".join(syms))
            return "\n\n".join(results) or f"No symbols found in {target}"

        rel = target.relative_to(self.ctx.workspace) if target.is_relative_to(self.ctx.workspace) else target
        if target.suffix == ".py":
            symbols = await loop.run_in_executor(None, _extract_symbols_python, target)
        else:
            symbols = await loop.run_in_executor(None, _extract_symbols_generic, target)

        if not symbols:
            return f"No symbols found in {rel}"
        return f"## Symbols: {rel}\n" + "\n".join(symbols)

    async def get_dep_graph(self, path: str, show_dependents: bool = True, show_transitive: bool = False, show_external: bool = True) -> str:
        """Show import dependencies using grep."""
        target = self.ctx._resolve_path(path)
        if not target.exists():
            return f"Error: Path not found: {target}"

        try:
            source = target.read_text(encoding="utf-8", errors="replace")
            imports = []
            for line in source.splitlines():
                line = line.strip()
                if line.startswith(("import ", "from ")) or "require(" in line:
                    imports.append(f"  {line}")
                if len(imports) >= 50:
                    break
            rel = target.relative_to(self.ctx.workspace) if target.is_relative_to(self.ctx.workspace) else target
            if not imports:
                return f"No imports found in {rel}"
            return f"## Dependencies: {rel}\n" + "\n".join(imports)
        except Exception as e:
            return f"Error reading {path}: {e}"

    async def run_analysis(self, path: str = ".", checks: Optional[list] = None) -> str:
        """Run available linters on the specified path."""
        target = self.ctx._resolve_path(path)
        results = []
        loop = asyncio.get_event_loop()

        def _run_linter(cmd: list[str]) -> str:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=self.ctx.workspace, timeout=30)
                return (r.stdout + r.stderr).strip()
            except Exception as e:
                return f"Error: {e}"

        if shutil.which("ruff") and (not checks or "ruff" in checks):
            out = await loop.run_in_executor(None, _run_linter, ["ruff", "check", str(target)])
            if out:
                results.append(f"## Ruff\n{out[:3000]}")

        if shutil.which("mypy") and (not checks or "mypy" in checks):
            out = await loop.run_in_executor(None, _run_linter, ["mypy", str(target), "--no-error-summary"])
            if out:
                results.append(f"## Mypy\n{out[:3000]}")

        if not results:
            return "No linters available. Install ruff or mypy for code analysis."
        return "\n\n".join(results)
