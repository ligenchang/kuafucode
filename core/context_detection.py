"""Project detection helpers for context building."""

from __future__ import annotations

import fnmatch
import os
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Optional


def is_ignored(path: Path, patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern."""
    name = path.name
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        if pattern in str(path):
            return True
    return False


def build_file_tree(
    root: Path,
    ignore_patterns: list[str],
    max_depth: int = 4,
    max_files: int = 200,
    _depth: int = 0,
    _count: list = None,
) -> str:
    """Build a compact file tree string using os.scandir (DirEntry caches stat)."""
    if _count is None:
        _count = [0]

    if _depth > max_depth or _count[0] >= max_files:
        return ""

    lines = []
    try:
        with os.scandir(root) as _it:
            entries = sorted(_it, key=lambda e: (e.is_file(follow_symlinks=False), e.name.lower()))
    except (PermissionError, OSError):
        return ""

    for entry in entries:
        if _count[0] >= max_files:
            lines.append(f"{'  ' * _depth}... (truncated)")
            break
        p = Path(entry.path)
        if is_ignored(p, ignore_patterns):
            continue
        indent = "  " * _depth
        if entry.is_dir(follow_symlinks=False):
            lines.append(f"{indent}{entry.name}/")
            sub = build_file_tree(p, ignore_patterns, max_depth, max_files, _depth + 1, _count)
            if sub:
                lines.append(sub)
        else:
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                size = 0
            if size > 1024 * 1024:
                size_str = f" ({size // 1024 // 1024}MB)"
            elif size > 1024:
                size_str = f" ({size // 1024}KB)"
            else:
                size_str = ""
            lines.append(f"{indent}{entry.name}{size_str}")
            _count[0] += 1

    return "\n".join(lines)


# Module-level project-type cache: workspace path → detected type string
_project_type_cache: dict[str, str] = {}


def detect_project_type(workspace: Path) -> str:
    """Detect project type from files present using a single os.scandir pass."""
    _key = str(workspace)
    if _key in _project_type_cache:
        return _project_type_cache[_key]

    try:
        _ws_files = {e.name for e in os.scandir(workspace) if e.is_file(follow_symlinks=False)}
    except OSError:
        _ws_files = set()

    checks = [
        ("pyproject.toml", "Python (pyproject)"),
        ("setup.py", "Python (setup.py)"),
        ("requirements.txt", "Python"),
        ("package.json", "Node.js/JavaScript"),
        ("tsconfig.json", "TypeScript"),
        ("Cargo.toml", "Rust"),
        ("go.mod", "Go"),
        ("pom.xml", "Java (Maven)"),
        ("build.gradle", "Java/Kotlin (Gradle)"),
        ("CMakeLists.txt", "C/C++ (CMake)"),
        ("Makefile", "Make-based"),
        ("docker-compose.yml", "Docker Compose"),
        ("docker-compose.yaml", "Docker Compose"),
        ("Dockerfile", "Docker"),
    ]

    indicators = [label for filename, label in checks if filename in _ws_files]
    result = ", ".join(indicators) if indicators else "Unknown"
    _project_type_cache[_key] = result
    return result


def read_key_file(path: Path, max_bytes: int = 4096) -> Optional[str]:
    """Read a key project file, truncating if too large."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_bytes:
            content = (
                content[:max_bytes] + f"\n... [truncated, {len(content) - max_bytes} more bytes]"
            )
        return content
    except Exception:
        return None


def get_git_summary(workspace: Path) -> str:
    """Get a brief git summary for context."""
    try:
        # Check if git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            cwd=workspace,
            timeout=5,
        )
        if result.returncode != 0:
            return "Not a git repository."

        # Run branch, status, and log in parallel
        def _run(cmd: list[str]) -> str:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=workspace,
                timeout=5,
            ).stdout.strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f_branch = ex.submit(_run, ["git", "branch", "--show-current"])
            f_status = ex.submit(_run, ["git", "status", "--short"])
            f_log = ex.submit(_run, ["git", "log", "--oneline", "-5"])
        branch = f_branch.result()
        status = f_status.result()
        log = f_log.result()

        parts = [f"Branch: {branch}"]
        if status:
            parts.append(f"Changes:\n{status}")
        else:
            parts.append("Working tree clean.")
        if log:
            parts.append(f"Recent commits:\n{log}")

        return "\n".join(parts)
    except Exception:
        return "Git info unavailable."


def read_memory(workspace: Path) -> str:
    """Read .nvagent/memory.md project memory."""
    memory_file = workspace / ".nvagent" / "memory.md"
    if memory_file.exists():
        content = memory_file.read_text(encoding="utf-8", errors="replace")
        if content.strip() and content.strip() != "# Project Memory":
            return content
    return ""
