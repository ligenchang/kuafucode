"""
Execution handlers:
  run_command, run_tests, run_formatter, find_files
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import sys
from pathlib import Path
from typing import Optional

from nvagent.core.execution import (
    CommandResult,
    parse_test_output,
    detect_test_framework,
    build_test_command,
    detect_formatters,
    build_formatter_command,
)
from nvagent.tools.handlers import BaseHandler
from nvagent.core.execution import _kill_proc_group


class ExecHandler(BaseHandler):
    """Handles run_command, run_tests, run_formatter, find_files."""

    # ── run_command ───────────────────────────────────────────────────────────

    async def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
        max_output_chars: int = 8000,
        filter: Optional[str] = None,
    ) -> str:
        ok, reason = self.ctx.sandbox.validate_command(command)
        if not ok:
            return f"Error (safe_mode): {reason}"
        if self.ctx.dry_run:
            return f"[DRY RUN] Would execute: {command}"
        work_dir = self.ctx._resolve_path(cwd) if cwd else self.ctx.workspace

        per_stream_cap = max(512, max_output_chars // 2)
        stream_fn = self.ctx.stream_fn

        try:
            _popen_kwargs: dict = {}
            if sys.platform != "win32":
                _popen_kwargs["start_new_session"] = True
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                **_popen_kwargs,
            )
            self.ctx.active_proc = proc

            # If we have a stream_fn, read stdout/stderr concurrently and emit lines live
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            async def _read_stream(stream: asyncio.StreamReader, buf: list[str], label: str) -> None:
                while True:
                    try:
                        line = await asyncio.wait_for(stream.readline(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    buf.append(decoded)
                    if stream_fn:
                        stream_fn(f"{label}{decoded.rstrip()}")

            try:
                if stream_fn:
                    await asyncio.wait_for(
                        asyncio.gather(
                            _read_stream(proc.stdout, stdout_chunks, ""),
                            _read_stream(proc.stderr, stderr_chunks, "[stderr] "),
                        ),
                        timeout=timeout,
                    )
                    await proc.wait()
                else:
                    raw_stdout, raw_stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    stdout_chunks = [raw_stdout.decode("utf-8", errors="replace")]
                    stderr_chunks = [raw_stderr.decode("utf-8", errors="replace")]
            except asyncio.TimeoutError:
                _kill_proc_group(proc)
                await proc.wait()
                return f"⏱ Command timed out after {timeout}s: {command}"
            except asyncio.CancelledError:
                _kill_proc_group(proc)
                await proc.wait()
                raise
            finally:
                self.ctx.active_proc = None

            stdout_str = "".join(stdout_chunks).strip()
            stderr_str = "".join(stderr_chunks).strip()
            exit_code = proc.returncode

            # Apply filter if specified (grep-style: keep lines matching pattern)
            filter_note = ""
            if filter:
                import re as _re
                try:
                    pat = _re.compile(filter, _re.IGNORECASE)
                    def _filter_text(text: str) -> str:
                        matched = [l for l in text.splitlines() if pat.search(l)]
                        return "\n".join(matched)
                    orig_stdout_lines = len(stdout_str.splitlines())
                    orig_stderr_lines = len(stderr_str.splitlines())
                    stdout_str = _filter_text(stdout_str)
                    stderr_str = _filter_text(stderr_str)
                    kept = len(stdout_str.splitlines()) + len(stderr_str.splitlines())
                    total_orig = orig_stdout_lines + orig_stderr_lines
                    filter_note = f"[filter={filter!r}: {kept}/{total_orig} lines matched]"
                except _re.error as e:
                    filter_note = f"[filter error: {e}]"

            parts = [f"$ {command}", f"Exit code: {exit_code}"]
            if filter_note:
                parts.append(filter_note)

            if stdout_str:
                stdout_total = len(stdout_str)
                stdout_truncated = stdout_total > per_stream_cap
                if stdout_truncated:
                    stdout_str = stdout_str[:per_stream_cap]
                parts.append(f"STDOUT:\n{stdout_str}")
                if stdout_truncated:
                    remaining = stdout_total - per_stream_cap
                    parts.append(
                        f"[TRUNCATED: {remaining:,} more chars not shown "
                        f"(total stdout: {stdout_total:,} chars). "
                        f"To see more: re-run with `max_output_chars={min(stdout_total + 2000, 40000)}`, "
                        f"or pipe: `{command} | tail -50`, "
                        f"or filter: `{command} 2>&1 | grep <pattern>`]"
                    )

            if stderr_str:
                stderr_total = len(stderr_str)
                stderr_truncated = stderr_total > per_stream_cap
                if stderr_truncated:
                    stderr_str = stderr_str[:per_stream_cap]
                parts.append(f"STDERR:\n{stderr_str}")
                if stderr_truncated:
                    remaining = stderr_total - per_stream_cap
                    parts.append(
                        f"[TRUNCATED: {remaining:,} more stderr chars not shown "
                        f"(total stderr: {stderr_total:,} chars). "
                        f"To see more: re-run with `max_output_chars={min(stderr_total * 2 + 2000, 40000)}`, "
                        f"or capture: `{command} 2>/tmp/err.txt && cat /tmp/err.txt`]"
                    )

            if not stdout_str and not stderr_str:
                parts.append("(no output)")

            return "\n".join(parts)

        except FileNotFoundError:
            return f"Error: Command not found. Make sure it's installed."
        except Exception as e:
            return f"Error running command: {e}"

    # ── run_tests ─────────────────────────────────────────────────────────────

    async def run_tests(
        self,
        path: Optional[str] = None,
        framework: Optional[str] = None,
        extra_args: Optional[list] = None,
        retry_on_fail: bool = False,
        filter: Optional[str] = None,
    ) -> str:
        fw = framework or detect_test_framework(self.ctx.workspace)
        if not fw:
            return (
                "Could not detect a test framework. "
                "Specify framework= (pytest|jest|vitest|cargo|go|unittest) or "
                "ensure a recognized config file exists."
            )

        cmd = build_test_command(fw, path, [str(a) for a in (extra_args or [])])
        max_attempts = 3 if retry_on_fail else 1

        last_result: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            t0 = __import__("time").monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.ctx.workspace,
                )
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=300)
                duration = __import__("time").monotonic() - t0
            except asyncio.TimeoutError:
                return f"⏱ Test run timed out (300s). Command: {' '.join(cmd)}"
            except FileNotFoundError:
                return f"Error: test runner not found: {cmd[0]!r}. Is it installed?"

            raw = (
                stdout_b.decode("utf-8", errors="replace")
                + "\n"
                + stderr_b.decode("utf-8", errors="replace")
            )
            exit_code = proc.returncode

            suite = parse_test_output(raw, fw)
            suite.duration_s = suite.duration_s or duration

            agent_str = suite.to_agent_str()

            # Apply output filter if requested
            if filter:
                import re as _re
                try:
                    pat = _re.compile(filter, _re.IGNORECASE)
                    filtered_lines = [l for l in agent_str.splitlines() if pat.search(l)]
                    total = len(agent_str.splitlines())
                    kept = len(filtered_lines)
                    agent_str = "\n".join(filtered_lines) + f"\n[filter={filter!r}: {kept}/{total} lines]"
                except _re.error:
                    pass

            if suite.success or not retry_on_fail:
                return agent_str

            cr = CommandResult(
                command=" ".join(cmd),
                exit_code=exit_code,
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                duration_s=duration,
            )
            if not self.ctx._retry_policy.should_retry(cr, attempt):
                return agent_str

            last_result = agent_str

        return last_result or "No test output."

    # ── run_formatter ─────────────────────────────────────────────────────────

    async def run_formatter(
        self,
        path: Optional[str] = None,
        formatter: Optional[str] = None,
        check_only: bool = False,
    ) -> str:
        if formatter:
            fmt = formatter.lower()
        else:
            available = detect_formatters(self.ctx.workspace)
            if not available:
                return "No formatter found. Install black, ruff, prettier, gofmt, or rustfmt."
            fmt = available[0]

        target = path or None
        cmd = build_formatter_command(fmt, target, check_only=check_only)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.ctx.workspace,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            return f"⏱ Formatter timed out: {' '.join(cmd)}"
        except FileNotFoundError:
            return f"Error: formatter not found: {cmd[0]!r}"

        out = stdout_b.decode("utf-8", errors="replace").strip()
        err = stderr_b.decode("utf-8", errors="replace").strip()
        rc = proc.returncode

        if check_only:
            if rc == 0:
                return f"✓ {fmt}: code is already formatted — no changes needed."
            diff_text = out or err
            if len(diff_text) > 6000:
                diff_text = diff_text[:6000] + "\n… [truncated]"
            return f"⚠ {fmt} (check only — not applied):\n```diff\n{diff_text}\n```"

        if rc == 0:
            changed_info = out or err or "(no output)"
            return f"✓ {fmt}: formatting applied.\n{changed_info[:2000]}"
        return f"✗ {fmt} failed (exit {rc}):\n{(err or out)[:2000]}"

    # ── find_files ────────────────────────────────────────────────────────────

    async def find_files(
        self,
        pattern: str,
        path: Optional[str] = None,
        max_results: int = 100,
    ) -> str:
        search_root = self.ctx._resolve_path(path) if path else self.ctx.workspace
        if not search_root.exists():
            return f"Error: Path not found: {search_root}"

        ignore_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            ".mypy_cache",
        }
        matches: list[Path] = []

        if "**" in pattern:
            for fpath in search_root.rglob(pattern.replace("**/", "")):
                if any(part in ignore_dirs for part in fpath.parts):
                    continue
                matches.append(fpath)
                if len(matches) >= max_results:
                    break
        else:
            loop_ff = asyncio.get_event_loop()

            def _ff_walk() -> list[Path]:
                found: list[Path] = []
                for root, dirs, files in os.walk(search_root):
                    dirs[:] = [d for d in dirs if d not in ignore_dirs]
                    for fname in files:
                        if fnmatch.fnmatch(fname, pattern):
                            found.append(Path(root) / fname)
                            if len(found) >= max_results:
                                return found
                return found

            matches = await loop_ff.run_in_executor(None, _ff_walk)

        if not matches:
            return f"No files found matching '{pattern}' in {search_root}"

        lines = [f"Found {len(matches)} file(s) matching '{pattern}':"]
        for m in matches:
            try:
                rel = m.relative_to(self.ctx.workspace)
            except ValueError:
                rel = m
            lines.append(f"  {rel}")
        if len(matches) >= max_results:
            lines.append(f"  ... (limit {max_results} reached)")
        return "\n".join(lines)
