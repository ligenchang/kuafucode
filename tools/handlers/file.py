"""
File operation handlers:
  read_file, write_file, write_files, edit_file, delete_file, list_dir
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
from pathlib import Path
from typing import Optional

from nvagent.core.symbols import extract_symbols
from nvagent.tools.handlers import BaseHandler


class FileHandler(BaseHandler):
    """Handles read_file, write_file, write_files, edit_file, delete_file, list_dir."""

    # ── read_file ─────────────────────────────────────────────────────────────

    async def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        fpath = self.ctx._resolve_path(path)
        lock = self.ctx._get_path_lock(fpath)
        async with lock:
            return await self._read_file_locked(path, fpath, start_line, end_line)

    async def _read_file_locked(
        self,
        path: str,
        fpath: Path,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        if not fpath.exists():
            return f"Error: File not found: {fpath}"
        if fpath.is_dir():
            return f"Error: {fpath} is a directory, use list_dir"
        if fpath.stat().st_size > self.ctx.max_file_bytes:
            size_kb = fpath.stat().st_size // 1024
            if start_line is None:
                return (
                    f"File is large ({size_kb}KB). Use start_line/end_line to read sections. "
                    f"Or use search_code to find specific parts."
                )

        try:
            stat = fpath.stat()
            mtime_ns = stat.st_mtime_ns
        except Exception as e:
            return f"Error reading file: {e}"

        _cache_key_path = str(fpath)

        # ── Large-file guard: redirect to symbol map for full-file reads ───────
        _PRECISE_READ_THRESHOLD = 150
        if start_line is None and end_line is None:
            _cached = self.ctx._read_result_cache.get(_cache_key_path, mtime_ns)
            if _cached is not None:
                self.ctx._read_mtimes[_cache_key_path] = mtime_ns
                return _cached

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                self.ctx._read_mtimes[_cache_key_path] = mtime_ns
            except Exception as e:
                return f"Error reading file: {e}"

            lines_list = content.splitlines(keepends=True)
            total = len(lines_list)

            if total > _PRECISE_READ_THRESHOLD:
                _sym_lines: list[str] = [
                    f"[Symbol map: {fpath.name} — {total} lines total]",
                    "Use read_file(path, start_line=N, end_line=M) to read a specific section.",
                    "Use search_code(query) to find a specific keyword/pattern and its line number.",
                    "",
                ]
                try:
                    _sym_idx = extract_symbols(fpath)
                    if not _sym_idx.is_empty() and _sym_idx.symbols:
                        _sym_lines.append("## Symbols (name  [line]):")
                        for sym in _sym_idx.symbols[:120]:
                            _sym_lines.append(f"  {sym}")
                        if len(_sym_idx.symbols) > 120:
                            _sym_lines.append(f"  ... ({len(_sym_idx.symbols) - 120} more — use search_code)")
                    else:
                        _sym_lines.append("## File preview (first 30 lines):")
                        for i, ln in enumerate(lines_list[:30], 1):
                            _sym_lines.append(f"  {i:4d} | {ln.rstrip()}")
                        if total > 30:
                            _sym_lines.append(f"  ... {total - 30} more lines")
                except Exception:
                    _sym_lines.append("## File preview (first 30 lines):")
                    for i, ln in enumerate(lines_list[:30], 1):
                        _sym_lines.append(f"  {i:4d} | {ln.rstrip()}")

                result = "\n".join(_sym_lines)
                self.ctx._read_result_cache.put(_cache_key_path, mtime_ns, result)
                return result

            numbered = [f"{i+1:4d} | {ln}" for i, ln in enumerate(lines_list)]
            result = f"[{fpath.name} — {total} lines]\n" + "".join(numbered)
            self.ctx._read_result_cache.put(_cache_key_path, mtime_ns, result)
            return result

        # ── Targeted line-range read ───────────────────────────────────────────
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines_list = content.splitlines(keepends=True)
        total = len(lines_list)
        s = max(0, (start_line or 1) - 1)
        e_line = min(total, end_line or total)
        section = lines_list[s:e_line]
        header = f"[{fpath.name} lines {s+1}-{e_line} of {total}]\n"
        numbered = [f"{s + 1 + i:4d} | {ln}" for i, ln in enumerate(section)]
        return header + "".join(numbered)

    # ── write_file ────────────────────────────────────────────────────────────

    async def write_file(self, path: str, content: str) -> str:
        fpath = self.ctx._resolve_path(path)
        lock = self.ctx._get_path_lock(fpath)
        async with lock:
            return await self._write_file_locked(path, fpath, content)

    async def _write_file_locked(self, path: str, fpath: Path, content: str) -> str:
        ok, reason = self.ctx.sandbox.validate_path(fpath)
        if not ok:
            return f"Error: {reason}"

        stale_warning = self.ctx._check_stale(fpath)

        new_sl    = content.splitlines()
        new_sl_kw = content.splitlines(keepends=True)

        diff_str = ""
        diff_info = ""
        if fpath.exists():
            loop = asyncio.get_event_loop()
            def _read_and_diff() -> tuple[str, list[str]]:
                old = fpath.read_text(encoding="utf-8", errors="replace")
                d = list(difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new_sl_kw,
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    n=2,
                ))
                return old, d
            old_content, diff = await loop.run_in_executor(None, _read_and_diff)
            if str(fpath) not in self.ctx._current_turn_backups:
                self.ctx._current_turn_backups[str(fpath)] = old_content
            if diff:
                diff_lines = diff[:60]
                if len(diff) > 60:
                    diff_lines.append(f"\n... [{len(diff) - 60} more diff lines]\n")
                diff_str = "".join(diff_lines)
                diff_info = f"\nDiff:\n```diff\n{diff_str}\n```"
            else:
                return f"(No changes) {fpath}"
        else:
            if str(fpath) not in self.ctx._current_turn_backups:
                self.ctx._current_turn_backups[str(fpath)] = None

        if self.ctx.confirm_fn and diff_str:
            approved = await self.ctx.confirm_fn(path, diff_str)
            if not approved:
                return f"✗ Skipped (user declined): {path}"

        if self.ctx.dry_run:
            lines_count = len(new_sl)
            preview = "\n".join(new_sl[:20])
            more = f"\n... [{lines_count - 20} more lines]" if lines_count > 20 else ""
            action = "overwrite" if fpath.exists() else "create"
            return (
                f"[DRY RUN] Would {action} '{path}' ({lines_count} lines).\n"
                + (f"Diff:\n```diff\n{diff_str}\n```" if diff_str else f"Content preview:\n{preview}{more}")
            )

        fpath.parent.mkdir(parents=True, exist_ok=True)

        lines_count_pre = len(new_sl)
        _large_file_note = ""
        if lines_count_pre > 200:
            _large_file_note = f"[Writing {lines_count_pre} lines to {path}…]\n"

        fpath.write_text(content, encoding="utf-8")

        rel = str(fpath.relative_to(self.ctx.workspace)) if fpath.is_relative_to(self.ctx.workspace) else str(fpath)
        if rel not in self.ctx.changed_files:
            self.ctx.changed_files.append(rel)

        try:
            self.ctx._read_mtimes[str(fpath)] = fpath.stat().st_mtime_ns
        except OSError:
            pass

        lines = len(new_sl)
        result = f"{_large_file_note}✓ Written: {fpath} ({lines} lines){diff_info}"
        if stale_warning:
            result = stale_warning + "\n" + result
        return result

    # ── write_files ───────────────────────────────────────────────────────────

    async def write_files(self, files: list) -> str:
        if not files:
            return "Error: No files provided to write_files."
        _seen: dict[str, dict] = {}
        for _f in files:
            _seen[_f.get("path", "")] = _f
        deduped = list(_seen.values())
        tasks = [self.write_file(f.get("path", ""), f.get("content", "")) for f in deduped]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parts = []
        for f, r in zip(deduped, results):
            path = f.get("path", "(unknown)")
            parts.append(str(r) if not isinstance(r, Exception) else f"Error [{path}]: {r}")
        return "\n".join(parts)

    # ── edit_file ─────────────────────────────────────────────────────────────

    async def edit_file(
        self,
        path: str,
        edits: list,
        create_if_missing: bool = False,
    ) -> str:
        fpath = self.ctx._resolve_path(path)
        lock = self.ctx._get_path_lock(fpath)
        async with lock:
            return await self._edit_file_locked(path, fpath, edits, create_if_missing)

    async def _edit_file_locked(
        self,
        path: str,
        fpath: Path,
        edits: list,
        create_if_missing: bool = False,
    ) -> str:
        fpath = self.ctx._resolve_path(path)
        ok, reason = self.ctx.sandbox.validate_path(fpath)
        if not ok:
            return f"Error: {reason}"

        stale_warning = self.ctx._check_stale(fpath)

        if not fpath.exists():
            if create_if_missing and edits:
                content = edits[0].get("replace", "") if edits else ""
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content, encoding="utf-8")
                if str(fpath) not in self.ctx._current_turn_backups:
                    self.ctx._current_turn_backups[str(fpath)] = None
                rel = str(fpath.relative_to(self.ctx.workspace)) if fpath.is_relative_to(self.ctx.workspace) else str(fpath)
                if rel not in self.ctx.changed_files:
                    self.ctx.changed_files.append(rel)
                return f"✓ Created: {fpath} ({len(content.splitlines())} lines)"
            return f"Error: File not found: {fpath}. Use create_if_missing=true to create it."

        loop_ed = asyncio.get_event_loop()
        original = await loop_ed.run_in_executor(
            None, lambda: fpath.read_text(encoding="utf-8", errors="replace")
        )
        if str(fpath) not in self.ctx._current_turn_backups:
            self.ctx._current_turn_backups[str(fpath)] = original

        current = original
        applied: list[str] = []
        errors:  list[str] = []

        for i, edit in enumerate(edits):
            search  = edit.get("search", "")
            replace = edit.get("replace", "")
            if not search:
                errors.append(f"Edit #{i+1}: empty 'search' string — skipped")
                continue

            occurrence_count = current.count(search)
            if occurrence_count > 1:
                match_lines: list[int] = []
                pos = 0
                for _ in range(min(occurrence_count, 5)):
                    idx_found = current.find(search, pos)
                    if idx_found == -1:
                        break
                    line_no = current[:idx_found].count("\n") + 1
                    match_lines.append(line_no)
                    pos = idx_found + 1
                more_msg = (
                    f" (and {occurrence_count - len(match_lines)} more)"
                    if occurrence_count > len(match_lines)
                    else ""
                )
                errors.append(
                    f"Edit #{i+1}: search string matches {occurrence_count} locations "
                    f"(lines {', '.join(str(l) for l in match_lines)}{more_msg}) — "
                    f"ambiguous edit rejected. "
                    f"Please include more surrounding context in 'search' to uniquely "
                    f"identify the target location (e.g. add the preceding or following line)."
                )
                continue

            if search in current:
                current = current.replace(search, replace, 1)
                applied.append(f"Edit #{i+1}: replaced {len(search)} chars")
            else:
                space_re = re.compile(re.escape(re.sub(r"[ \t]+", " ", search)).replace("\\ ", r"[ \t]+"))
                ws_matches = list(space_re.finditer(current))
                if len(ws_matches) > 1:
                    match_lines = [current[:m.start()].count("\n") + 1 for m in ws_matches[:5]]
                    more_msg = (
                        f" (and {len(ws_matches) - 5} more)" if len(ws_matches) > 5 else ""
                    )
                    errors.append(
                        f"Edit #{i+1}: search string (whitespace-normalised) matches "
                        f"{len(ws_matches)} locations "
                        f"(lines {', '.join(str(l) for l in match_lines)}{more_msg}) — "
                        f"ambiguous edit rejected. Add more context to make it unique."
                    )
                elif len(ws_matches) == 1:
                    m = ws_matches[0]
                    current = current[:m.start()] + replace + current[m.end():]
                    applied.append(f"Edit #{i+1}: replaced (whitespace-normalised match)")
                else:
                    errors.append(f"Edit #{i+1}: search string not found: {search[:80]!r}")

        if not applied and errors:
            return "\n".join(errors)

        loop = asyncio.get_event_loop()
        orig_sl = original.splitlines(keepends=True)
        curr_sl = current.splitlines(keepends=True)
        diff = await loop.run_in_executor(
            None,
            lambda: list(difflib.unified_diff(
                orig_sl, curr_sl,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=2,
            ))
        )
        diff_str = ""
        if diff:
            diff_lines = diff[:60]
            if len(diff) > 60:
                diff_lines.append(f"\n... [{len(diff)-60} more diff lines]\n")
            diff_str = "".join(diff_lines)

        if self.ctx.confirm_fn and diff_str:
            approved = await self.ctx.confirm_fn(path, diff_str)
            if not approved:
                return f"✗ Skipped (user declined): {path}"

        if self.ctx.dry_run:
            add_lines = sum(1 for l in diff_str.splitlines() if l.startswith("+") and not l.startswith("+++ "))
            del_lines = sum(1 for l in diff_str.splitlines() if l.startswith("-") and not l.startswith("--- "))
            return (
                f"[DRY RUN] Would edit '{path}' — {len(applied)} edit(s), +{add_lines}/-{del_lines} lines.\n"
                + (f"Diff:\n```diff\n{diff_str}\n```" if diff_str else "")
            )

        fpath.write_text(current, encoding="utf-8")
        rel = str(fpath.relative_to(self.ctx.workspace)) if fpath.is_relative_to(self.ctx.workspace) else str(fpath)
        if rel not in self.ctx.changed_files:
            self.ctx.changed_files.append(rel)

        try:
            self.ctx._read_mtimes[str(fpath)] = fpath.stat().st_mtime_ns
        except OSError:
            pass

        summary = f"✓ Edited: {path} — {len(applied)} edit(s) applied"
        if errors:
            summary += "\nWarnings:\n" + "\n".join(errors)
        if diff_str:
            summary += f"\nDiff:\n```diff\n{diff_str}\n```"
        if stale_warning:
            summary = stale_warning + "\n" + summary
        return summary

    # ── delete_file ───────────────────────────────────────────────────────────

    async def delete_file(self, path: str) -> str:
        fpath = self.ctx._resolve_path(path)
        ok, reason = self.ctx.sandbox.validate_path(fpath)
        if not ok:
            return f"Error: {reason}"
        if not fpath.exists():
            return f"Error: {fpath} does not exist."
        if self.ctx.dry_run:
            return f"[DRY RUN] Would delete '{path}'."
        if fpath.is_dir():
            try:
                fpath.rmdir()
                return f"✓ Deleted directory: {fpath}"
            except OSError as e:
                return f"Error: {e}. Directory must be empty."
        else:
            fpath.unlink()
            return f"✓ Deleted: {fpath}"

    # ── list_dir ──────────────────────────────────────────────────────────────

    async def list_dir(self, path: str, recursive: bool = False) -> str:
        dpath = self.ctx._resolve_path(path)

        if not dpath.exists():
            return f"Error: Directory not found: {dpath}"
        if not dpath.is_dir():
            return f"Error: {dpath} is not a directory"

        lines: list[str] = []
        if recursive:
            loop_ld = asyncio.get_event_loop()
            def _walk_recursive() -> list[str]:
                result: list[str] = []
                for root, dirs, files in os.walk(dpath):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".venv")]
                    rel_root = Path(root).relative_to(dpath)
                    depth = len(rel_root.parts)
                    indent = "  " * depth
                    if depth > 0:
                        result.append(f"{indent}{Path(root).name}/")
                    for f in sorted(files):
                        result.append(f"{'  ' * (depth + 1)}{f}")
                return result
            lines = await loop_ld.run_in_executor(None, _walk_recursive)
        else:
            for entry in sorted(dpath.iterdir(), key=lambda p: (p.is_file(), p.name)):
                if entry.is_dir():
                    lines.append(f"  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    size_str = f" ({size:,} bytes)" if size < 10240 else f" ({size // 1024}KB)"
                    lines.append(f"  {entry.name}{size_str}")

        return f"{dpath}/\n" + "\n".join(lines) if lines else f"{dpath}/ (empty)"

    # ── str_replace_editor (Anthropic compatibility shim) ─────────────────────

    async def str_replace_editor(
        self,
        command: str = "str_replace",
        path: str = "",
        old_str: str = "",
        new_str: str = "",
        view_range: Optional[list] = None,
        insert_line: Optional[int] = None,
    ) -> str:
        """Compatibility shim for models that emit Anthropic-style str_replace_editor calls.

        Maps to native nvagent tools:
          view          → read_file (with optional start/end from view_range)
          create/write  → write_file
          str_replace   → edit_file with a single search/replace edit
          insert        → edit_file: insert new_str after the line at insert_line
          undo_edit     → undo_last_turn (not wired here, returns guidance)
        """
        cmd = command.lower()

        if cmd in ("view",):
            if not path:
                return "Error: str_replace_editor(view) requires 'path'."
            if view_range and len(view_range) == 2:
                return await self.read_file(path, start_line=view_range[0], end_line=view_range[1])
            return await self.read_file(path)

        if cmd in ("create", "write"):
            if not path:
                return "Error: str_replace_editor(create) requires 'path'."
            return await self.write_file(path, new_str)

        if cmd in ("str_replace",):
            if not path:
                return "Error: str_replace_editor(str_replace) requires 'path'."
            if not old_str:
                # No old text → treat as full overwrite
                return await self.write_file(path, new_str)
            return await self.edit_file(path, [{"search": old_str, "replace": new_str}])

        if cmd in ("insert",):
            if not path:
                return "Error: str_replace_editor(insert) requires 'path'."
            if insert_line is None:
                return "Error: str_replace_editor(insert) requires 'insert_line'."
            fpath = self.ctx._resolve_path(path)
            if not fpath.exists():
                return f"Error: File not found: {fpath}"
            loop = asyncio.get_event_loop()
            lines = await loop.run_in_executor(
                None, lambda: fpath.read_text(encoding="utf-8").splitlines(keepends=True)
            )
            idx = min(insert_line, len(lines))
            insert_text = new_str if new_str.endswith("\n") else new_str + "\n"
            lines.insert(idx, insert_text)
            new_content = "".join(lines)
            return await self.write_file(path, new_content)

        if cmd in ("undo_edit",):
            return "Use the 'undo_last_turn' tool to undo recent file edits."

        return f"str_replace_editor: unknown command '{command}'. Supported: view, create, str_replace, insert."
