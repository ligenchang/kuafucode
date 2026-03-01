"""
Version-control / patch handlers:
  apply_patch, checkpoint, rollback
"""

from __future__ import annotations

import asyncio
import difflib
import re
import tempfile
from pathlib import Path

from nvagent.tools.handlers import BaseHandler


class VcHandler(BaseHandler):
    """Handles apply_patch, checkpoint, rollback."""

    # ── apply_patch ───────────────────────────────────────────────────────────

    async def apply_patch(self, patch: str, dry_run: bool = False) -> str:
        if not patch.strip():
            return "Error: empty patch string."
        if self.ctx._patch_bin:
            return await self._apply_patch_binary(patch, dry_run, self.ctx._patch_bin)
        else:
            return await self._apply_patch_python(patch, dry_run)

    async def _apply_patch_binary(self, patch: str, dry_run: bool, patch_bin: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False, encoding="utf-8"
        ) as f:
            f.write(patch)
            patch_file = f.name

        try:
            base_cmd = [patch_bin, "-p1", "--batch", "--reject-file=-"]

            if dry_run:
                dry_proc = await asyncio.create_subprocess_exec(
                    *base_cmd,
                    "--dry-run",
                    "-i",
                    patch_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.ctx.workspace,
                )
                dry_out, dry_err = await asyncio.wait_for(dry_proc.communicate(), timeout=30)
                if dry_proc.returncode != 0:
                    err_text = (dry_err or dry_out).decode("utf-8", errors="replace").strip()
                    return f"✗ Patch validation failed:\n{err_text}"
                out_text = dry_out.decode("utf-8", errors="replace").strip()
                return f"✓ Patch is valid (dry run — not applied):\n{out_text}"

            apply_proc = await asyncio.create_subprocess_exec(
                *base_cmd,
                "-i",
                patch_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.ctx.workspace,
            )
            app_out, app_err = await asyncio.wait_for(apply_proc.communicate(), timeout=60)
            if apply_proc.returncode == 0:
                out_text = app_out.decode("utf-8", errors="replace").strip()
                for line in out_text.splitlines():
                    if line.startswith("patching file "):
                        touched = line[len("patching file ") :].strip()
                        if touched not in self.ctx.changed_files:
                            self.ctx.changed_files.append(touched)
                        abs_p = self.ctx.workspace / touched
                        if str(abs_p) not in self.ctx._current_turn_backups:
                            self.ctx._current_turn_backups[str(abs_p)] = None
                return f"✓ Patch applied:\n{out_text}"
            else:
                err_text = (app_err or app_out).decode("utf-8", errors="replace").strip()
                return f"✗ Patch failed:\n{err_text}"
        finally:
            Path(patch_file).unlink(missing_ok=True)

    async def _apply_patch_python(self, patch: str, dry_run: bool) -> str:
        """Pure-Python unified diff patch application."""
        file_patches: list[tuple[str, list[tuple[int, list[str], list[str]]]]] = []
        current_file: str | None = None
        current_hunks: list[tuple[int, list[str], list[str]]] = []
        hunk_old: list[str] = []
        hunk_new: list[str] = []
        hunk_start: int = 0

        def _flush_hunk() -> None:
            if hunk_old or hunk_new:
                current_hunks.append((hunk_start, hunk_old[:], hunk_new[:]))
                hunk_old.clear()
                hunk_new.clear()

        for line in patch.splitlines(keepends=True):
            stripped = line.rstrip("\n")
            if stripped.startswith("--- "):
                if current_file is not None:
                    _flush_hunk()
                    file_patches.append((current_file, current_hunks))
                current_file = None
                current_hunks = []
                hunk_old.clear()
                hunk_new.clear()
            elif stripped.startswith("+++ "):
                raw = stripped[4:].split("\t")[0].strip()
                current_file = re.sub(r"^[ab]/", "", raw)
            elif stripped.startswith("@@"):
                _flush_hunk()
                m = re.match(r"@@ -(\d+)", stripped)
                hunk_start = int(m.group(1)) if m else 1
            elif current_file is not None and stripped.startswith("-"):
                hunk_old.append(stripped[1:])
            elif current_file is not None and stripped.startswith("+"):
                hunk_new.append(stripped[1:])

        if current_file is not None:
            _flush_hunk()
            file_patches.append((current_file, current_hunks))

        if not file_patches:
            return "Error: Could not parse any file patches from the unified diff."

        results: list[str] = []
        errors: list[str] = []

        for rel_path, hunks in file_patches:
            fpath = self.ctx.workspace / rel_path
            if fpath.exists():
                original = fpath.read_text(encoding="utf-8", errors="replace")
                lines = original.splitlines(keepends=False)
            else:
                original = ""
                lines = []

            new_lines = lines[:]
            offset = 0

            for hunk_start_1based, old_lines, new_lines_hunk in hunks:
                start = hunk_start_1based - 1 + offset
                end = start + len(old_lines)

                if old_lines:
                    actual = new_lines[start:end]
                    if actual != old_lines:
                        errors.append(
                            f"Hunk mismatch in {rel_path} at line {hunk_start_1based}: "
                            f"expected {old_lines[:1]!r}, got {actual[:1]!r}"
                        )
                        continue

                new_lines[start:end] = new_lines_hunk
                offset += len(new_lines_hunk) - len(old_lines)

            new_content = "\n".join(new_lines) + ("\n" if new_lines else "")

            if dry_run:
                diff = list(
                    difflib.unified_diff(
                        lines,
                        new_lines,
                        fromfile=f"a/{rel_path}",
                        tofile=f"b/{rel_path}",
                    )
                )
                results.append(f"✓ {rel_path} (dry run): {len(diff)} diff lines")
                continue

            if not self.ctx.dry_run:
                if str(fpath) not in self.ctx._current_turn_backups:
                    self.ctx._current_turn_backups[str(fpath)] = original if original else None
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(new_content, encoding="utf-8")
                if rel_path not in self.ctx.changed_files:
                    self.ctx.changed_files.append(rel_path)
            results.append(f"✓ Patched: {rel_path}")

        summary = "\n".join(results)
        if errors:
            summary += "\nWarnings:\n" + "\n".join(errors)
        return summary or "Nothing to patch."

    # ── checkpoint ────────────────────────────────────────────────────────────

    async def checkpoint(
        self,
        name: str | None = None,
        include_paths: list | None = None,
    ) -> str:
        label = name or f"cp{len(self.ctx._checkpoints) + 1}"
        loop_cp = asyncio.get_event_loop()

        def _safe_read(p: Path) -> tuple[str, str | None]:
            try:
                return (str(p), p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                return (str(p), None)

        candidate_paths: list[Path] = []
        for rel_path in self.ctx.changed_files:
            abs_path = self.ctx.workspace / rel_path
            if abs_path.exists():
                candidate_paths.append(abs_path)
        for p in include_paths or []:
            fpath = self.ctx._resolve_path(str(p))
            if fpath.exists() and fpath.is_file():
                ok, _ = self.ctx.sandbox.validate_path(fpath)
                if ok:
                    candidate_paths.append(fpath)

        read_tasks = [loop_cp.run_in_executor(None, _safe_read, p) for p in candidate_paths]
        results = await asyncio.gather(*read_tasks)
        snapshot: dict[str, str | None] = {k: v for k, v in results if v is not None}

        self.ctx._checkpoints[label] = snapshot
        return (
            f"✓ Checkpoint '{label}' saved "
            f"({len(snapshot)} file{'s' if len(snapshot) != 1 else ''})."
            + (
                f"\n  Files: {', '.join(Path(k).name for k in list(snapshot)[:10])}"
                if snapshot
                else ""
            )
        )

    # ── rollback ──────────────────────────────────────────────────────────────

    async def rollback(self, name: str | None = None) -> str:
        if not self.ctx._checkpoints:
            return (
                "No checkpoints available. "
                "Use checkpoint() before making changes to create a rollback point."
            )

        if name:
            if name not in self.ctx._checkpoints:
                available = ", ".join(f"'{k}'" for k in self.ctx._checkpoints)
                return f"Checkpoint '{name}' not found. Available: {available}"
            label = name
        else:
            label = list(self.ctx._checkpoints.keys())[-1]

        snapshot = self.ctx._checkpoints[label]
        restored: list[str] = []

        for abs_path_str, content in snapshot.items():
            fpath = Path(abs_path_str)
            ok, reason = self.ctx.sandbox.validate_path(fpath)
            if not ok:
                restored.append(f"  (skipped {fpath.name}: {reason})")
                continue
            try:
                if content is None:
                    if fpath.exists():
                        fpath.unlink()
                        restored.append(f"deleted {fpath.name}")
                else:
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(content, encoding="utf-8")
                    restored.append(fpath.name)
            except Exception as exc:
                restored.append(f"  (error restoring {fpath.name}: {exc})")

        if not restored:
            return f"Checkpoint '{label}' exists but contained no file snapshots."
        return (
            f"✓ Rolled back to checkpoint '{label}': "
            + ", ".join(restored[:10])
            + (f" … and {len(restored) - 10} more" if len(restored) > 10 else "")
        )
