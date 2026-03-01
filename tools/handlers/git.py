"""
Git operation handlers:
  git_status, git_diff, git_add, git_commit, git_log
"""

from __future__ import annotations

import asyncio

from nvagent.tools.handlers import BaseHandler


class GitHandler(BaseHandler):
    """Handles git_status, git_diff, git_add, git_commit, git_log."""

    # ── git_status ────────────────────────────────────────────────────────────

    async def git_status(self) -> str:
        async def _git_run(cmd: list[str]) -> tuple[int, str]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.ctx.workspace,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                return proc.returncode, stdout.decode("utf-8", errors="replace").strip()
            except Exception:
                return -1, ""

        cmds = [
            ("Branch", ["git", "branch", "--show-current"]),
            ("Status", ["git", "status", "--short"]),
            ("Stashes", ["git", "stash", "list", "--oneline"]),
            ("Recent", ["git", "log", "--oneline", "-10"]),
        ]
        outcomes = await asyncio.gather(*[_git_run(cmd) for _, cmd in cmds])
        lines = [
            f"### {label}\n{text}"
            for (label, _), (rc, text) in zip(cmds, outcomes)
            if rc == 0 and text
        ]
        return "\n\n".join(lines) if lines else "Not a git repository or no git info available."

    # ── git_diff ──────────────────────────────────────────────────────────────

    async def git_diff(
        self,
        staged: bool = False,
        file: str | None = None,
        commit: str | None = None,
    ) -> str:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if commit:
            cmd.append(commit)
        if file:
            cmd.extend(["--", file])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.ctx.workspace,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                return f"git diff error: {stderr.decode('utf-8', errors='replace').strip()}"
            diff = stdout.decode("utf-8", errors="replace").strip()
            if not diff:
                return "No differences found."
            if len(diff) > 16000:
                diff = diff[:16000] + "\n... [diff truncated, use file= to narrow scope]"
            return diff
        except TimeoutError:
            return "git diff timed out."
        except Exception as e:
            return f"Error: {e}"

    # ── git_add ───────────────────────────────────────────────────────────────

    async def git_add(self, paths: list) -> str:
        if not paths:
            return "Error: No paths provided to git_add."
        str_paths = [str(p) for p in paths]
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "add",
                "--",
                *str_paths,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.ctx.workspace,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                return f"git add failed: {stderr.decode('utf-8', errors='replace').strip()}"
            status_proc = await asyncio.create_subprocess_exec(
                "git",
                "status",
                "--short",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.ctx.workspace,
            )
            status_out, _ = await asyncio.wait_for(status_proc.communicate(), timeout=10)
            return f"✓ Staged: {', '.join(str_paths)}\n{status_out.decode('utf-8', errors='replace').strip()}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # ── git_commit ────────────────────────────────────────────────────────────

    async def git_commit(self, message: str, add_all: bool = False) -> str:
        if not message.strip():
            return "Error: Commit message cannot be empty."
        cmd = ["git", "commit"]
        if add_all:
            cmd.append("-a")
        cmd.extend(["-m", message])
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.ctx.workspace,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                err = (
                    stderr.decode("utf-8", errors="replace")
                    or stdout.decode("utf-8", errors="replace")
                ).strip()
                return f"git commit failed: {err}"
            return f"✓ Committed:\n{stdout.decode('utf-8', errors='replace').strip()}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # ── git_log ───────────────────────────────────────────────────────────────

    async def git_log(
        self,
        limit: int = 10,
        file: str | None = None,
        oneline: bool = True,
    ) -> str:
        cmd = ["git", "log", f"-{max(1, limit)}"]
        if oneline:
            cmd.append("--oneline")
        else:
            cmd.extend(["--format=%h %an <%ae> %ar%n    %s%n"])
        if file:
            cmd.extend(["--", file])
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.ctx.workspace,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                return f"git log failed: {stderr.decode('utf-8', errors='replace').strip()}"
            output = stdout.decode("utf-8", errors="replace").strip()
            return output if output else "No commits found."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
