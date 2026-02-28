"""
Jupyter notebook handlers:
  read_notebook, edit_notebook
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from nvagent.tools.handlers import BaseHandler


class NotebookHandler(BaseHandler):
    """Handles read_notebook and edit_notebook."""

    # ── read_notebook ─────────────────────────────────────────────────────────

    async def read_notebook(
        self,
        path: str,
        cell_index: Optional[int] = None,
    ) -> str:
        fpath = self.ctx._resolve_path(path)
        ok, reason = self.ctx.sandbox.validate_path(fpath)
        if not ok:
            return f"Error: {reason}"
        if not fpath.exists():
            return f"Error: File not found: {fpath}"
        if fpath.suffix != ".ipynb":
            return f"Error: Not a Jupyter notebook (.ipynb): {fpath}"

        loop = asyncio.get_event_loop()
        try:
            nb = await loop.run_in_executor(
                None, lambda: json.loads(fpath.read_text(encoding="utf-8"))
            )
        except Exception as e:
            return f"Error reading notebook: {e}"

        cells = nb.get("cells", [])
        if not cells:
            return f"Notebook has no cells: {fpath}"

        def _fmt_output(output: dict) -> str:
            otype = output.get("output_type", "")
            if otype == "stream":
                text = output.get("text", [])
                raw = "".join(text) if isinstance(text, list) else text
                return f"[stream/{output.get('name','stdout')}] {raw[:200].rstrip()}"
            if otype in ("display_data", "execute_result"):
                data = output.get("data", {})
                if "text/plain" in data:
                    txt = data["text/plain"]
                    raw = "".join(txt) if isinstance(txt, list) else txt
                    return f"[result] {raw[:200].rstrip()}"
                if "image/png" in data:
                    return "[image/png output]"
                return f"[{otype}]"
            if otype == "error":
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                return f"[error] {ename}: {evalue}"
            return f"[{otype}]"

        # ── Single cell full read ─────────────────────────────────────────────
        if cell_index is not None:
            if not (0 <= cell_index < len(cells)):
                return f"Error: cell_index {cell_index} out of range (0–{len(cells)-1})"
            cell = cells[cell_index]
            ctype = cell.get("cell_type", "code")
            src_raw = cell.get("source", [])
            source = "".join(src_raw) if isinstance(src_raw, list) else src_raw
            outputs = cell.get("outputs", [])
            parts = [
                f"## Cell {cell_index} [{ctype}]",
                "### Source",
                source if source else "(empty)",
            ]
            if outputs:
                parts.append(f"### Outputs ({len(outputs)})")
                for o in outputs:
                    parts.append("  " + _fmt_output(o))
            else:
                parts.append("### Outputs\n  (none)")
            return "\n".join(parts)

        # ── Summary of all cells ──────────────────────────────────────────────
        try:
            kernel = (
                nb.get("metadata", {}).get("kernelspec", {}).get("display_name", "unknown kernel")
            )
            lang = nb.get("metadata", {}).get("language_info", {}).get("name", "")
        except Exception:
            kernel, lang = "unknown", ""

        lines = [
            f"## Notebook: {fpath.name}  [{kernel}{' / ' + lang if lang else ''}]",
            f"   {len(cells)} cell(s)\n",
            f"{'#':>3}  {'type':<8}  {'exec':>5}  preview",
            "─" * 72,
        ]
        for i, cell in enumerate(cells):
            ctype = cell.get("cell_type", "code")
            src_raw = cell.get("source", [])
            source = "".join(src_raw) if isinstance(src_raw, list) else src_raw
            preview_lines = source.splitlines()
            preview = preview_lines[0][:65] if preview_lines else "(empty)"
            if len(preview_lines) > 1:
                preview += " …"
            exec_count = cell.get("execution_count") or "-"
            outputs = cell.get("outputs", [])
            out_hint = ""
            if outputs:
                out_hint = f"  → {len(outputs)} output(s)"
                if any(o.get("output_type") == "error" for o in outputs):
                    out_hint += " ⚠ error"
            lines.append(f"{i:>3}  {ctype:<8}  {str(exec_count):>5}  {preview}{out_hint}")
        return "\n".join(lines)

    # ── edit_notebook ─────────────────────────────────────────────────────────

    async def edit_notebook(
        self,
        path: str,
        operation: str,
        cell_index: Optional[int] = None,
        source: Optional[str] = None,
        cell_type: str = "code",
    ) -> str:
        fpath = self.ctx._resolve_path(path)
        ok, reason = self.ctx.sandbox.validate_path(fpath)
        if not ok:
            return f"Error: {reason}"
        if not fpath.exists():
            return f"Error: File not found: {fpath}"
        if fpath.suffix != ".ipynb":
            return f"Error: Not a Jupyter notebook (.ipynb): {fpath}"

        loop = asyncio.get_event_loop()
        try:
            nb = await loop.run_in_executor(
                None, lambda: json.loads(fpath.read_text(encoding="utf-8"))
            )
        except Exception as e:
            return f"Error reading notebook: {e}"

        cells = nb.setdefault("cells", [])
        n = len(cells)

        if operation == "update":
            if cell_index is None:
                return "Error: cell_index is required for 'update'."
            if not (0 <= cell_index < n):
                return f"Error: cell_index {cell_index} out of range (0–{n-1})."
            if source is None:
                return "Error: source is required for 'update'."
            cells[cell_index]["source"] = source
            cells[cell_index]["outputs"] = []
            cells[cell_index]["execution_count"] = None
            action = f"updated cell {cell_index}"

        elif operation == "insert":
            if source is None:
                return "Error: source is required for 'insert'."
            if cell_type not in ("code", "markdown"):
                return f"Error: invalid cell_type {cell_type!r}, must be 'code' or 'markdown'."
            new_cell: dict = {
                "cell_type": cell_type,
                "source": source,
                "metadata": {},
                "outputs": [],
            }
            if cell_type == "code":
                new_cell["execution_count"] = None
            idx = cell_index if cell_index is not None else n
            idx = max(0, min(idx, n))
            cells.insert(idx, new_cell)
            action = f"inserted {cell_type} cell at index {idx}"

        elif operation == "delete":
            if cell_index is None:
                return "Error: cell_index is required for 'delete'."
            if not (0 <= cell_index < n):
                return f"Error: cell_index {cell_index} out of range (0–{n-1})."
            deleted_type = cells[cell_index].get("cell_type", "code")
            cells.pop(cell_index)
            action = f"deleted {deleted_type} cell {cell_index}"

        else:
            return f"Error: unknown operation {operation!r}. Must be update, insert, or delete."

        if self.ctx.dry_run:
            return f"[DRY RUN] Would {action} in {fpath.name}."

        if str(fpath) not in self.ctx._current_turn_backups:
            self.ctx._current_turn_backups[str(fpath)] = None

        await loop.run_in_executor(
            None,
            lambda: fpath.write_text(
                json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
            ),
        )
        rel = (
            str(fpath.relative_to(self.ctx.workspace))
            if fpath.is_relative_to(self.ctx.workspace)
            else str(fpath)
        )
        if rel not in self.ctx.changed_files:
            self.ctx.changed_files.append(rel)
        return f"✓ {action} in {fpath.name} ({len(cells)} cells total)."
