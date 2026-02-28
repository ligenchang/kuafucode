"""Git information retrieval utilities."""

from __future__ import annotations

import concurrent.futures
import subprocess
from pathlib import Path


def git_branch(workspace: Path) -> str:
    """Return 'branch' string or empty string if not in a git repo.
    
    Shows the current branch name, or detached@<hash> if in detached HEAD state.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=2,
        )
        branch = result.stdout.strip()
        if branch and branch != "HEAD":
            return branch
        # Detached HEAD — show short hash
        result2 = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=2,
        )
        h = result2.stdout.strip()
        return f"detached@{h}" if h else ""
    except Exception:
        return ""


def git_status_summary(workspace: Path) -> str:
    """Return short summary like '+2 ~1 -0' or 'clean' if no changes.
    
    Counts added, modified, and deleted files from git status --porcelain.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=2,
        )
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        if not lines:
            return "clean"
        added = sum(1 for l in lines if l[:2] in ("A ", "??", "AM"))
        modified = sum(1 for l in lines if l[:2] in (" M", "M ", "MM"))
        deleted = sum(1 for l in lines if l[:2] in (" D", "D "))
        parts_s = []
        if added:
            parts_s.append(f"+{added}")
        if modified:
            parts_s.append(f"~{modified}")
        if deleted:
            parts_s.append(f"-{deleted}")
        return " ".join(parts_s) if parts_s else "clean"
    except Exception:
        return ""


def git_info(workspace: Path) -> tuple[str, str]:
    """Return (branch, status_summary) running both git calls in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
        _bf = _pool.submit(git_branch, workspace)
        _sf = _pool.submit(git_status_summary, workspace)
        return _bf.result(), _sf.result()
