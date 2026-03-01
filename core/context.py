"""Project context assembly — builds the system prompt with project info."""

from __future__ import annotations

import concurrent.futures
import fnmatch
import os
import subprocess
from pathlib import Path
from typing import Optional

from nvagent.config import Config

KEY_FILES = [
    "README.md", "README.rst", "README.txt",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "tsconfig.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "Makefile", "CMakeLists.txt",
    "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
    ".env.example", "ARCHITECTURE.md",
]

SYSTEM_PROMPT_BASE = """\
You are nvagent, an expert coding assistant powered by NVIDIA NIM models.
You operate directly on the user's codebase with full file system access.

## Your capabilities:
- Read, write, create, and delete files
- Run shell commands and capture output
- Search code across the entire codebase
- Understand git history and diffs
- Build, test, and debug projects end-to-end

## How you work:
- You ALWAYS read relevant files before making changes
- You write complete, production-quality code — never placeholders or TODOs unless asked
- When you encounter errors, you debug systematically
- You follow the project's existing code style and conventions

## Tool call discipline — CRITICAL:
- ALWAYS finish your current sentence BEFORE issuing a tool call.
- Pattern: write a complete thought ending with ".\\n", THEN call the tool.

## File reading strategy:
- For small files (<150 lines): read_file(path) is fine
- For larger files: use search_code to find the section you need, then read_file with start_line/end_line
- Use edit_file for targeted changes; write_file for new files or complete rewrites
- When creating multiple files, use write_files (concurrent, single round-trip)
"""


def is_ignored(path: Path, patterns: list[str]) -> bool:
    name = path.name
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern) or pattern in str(path):
            return True
    return False


def build_file_tree(root: Path, ignore_patterns: list[str], max_depth: int = 4, max_files: int = 200) -> str:
    def _walk(path: Path, depth: int, count: list) -> list[str]:
        if depth > max_depth or count[0] >= max_files:
            return []
        lines = []
        try:
            entries = sorted(os.scandir(path), key=lambda e: (e.is_file(follow_symlinks=False), e.name.lower()))
        except (PermissionError, OSError):
            return []
        indent = "  " * depth
        for entry in entries:
            if count[0] >= max_files:
                lines.append(f"{indent}... (truncated)")
                break
            p = Path(entry.path)
            if is_ignored(p, ignore_patterns):
                continue
            if entry.is_dir(follow_symlinks=False):
                lines.append(f"{indent}{entry.name}/")
                lines.extend(_walk(p, depth + 1, count))
            else:
                try:
                    size = entry.stat(follow_symlinks=False).st_size
                    size_str = f" ({size // 1024}KB)" if size > 1024 else ""
                except OSError:
                    size_str = ""
                lines.append(f"{indent}{entry.name}{size_str}")
                count[0] += 1
        return lines

    return "\n".join(_walk(root, 0, [0]))


def detect_project_type(workspace: Path) -> str:
    try:
        ws_files = {e.name for e in os.scandir(workspace) if e.is_file(follow_symlinks=False)}
    except OSError:
        return "Unknown"
    checks = [
        ("pyproject.toml", "Python"), ("setup.py", "Python"), ("requirements.txt", "Python"),
        ("package.json", "Node.js"), ("tsconfig.json", "TypeScript"),
        ("Cargo.toml", "Rust"), ("go.mod", "Go"),
        ("pom.xml", "Java (Maven)"), ("build.gradle", "Java/Kotlin (Gradle)"),
        ("CMakeLists.txt", "C/C++"), ("Makefile", "Make"),
        ("Dockerfile", "Docker"),
    ]
    detected = [label for filename, label in checks if filename in ws_files]
    return ", ".join(detected) if detected else "Unknown"


def _read_key_file(path: Path, max_bytes: int = 3000) -> Optional[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_bytes:
            content = content[:max_bytes] + f"\n... [truncated]"
        return content
    except Exception:
        return None


def get_git_summary(workspace: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, cwd=workspace, timeout=5)
        if r.returncode != 0:
            return "Not a git repository."

        def _run(cmd: list[str]) -> str:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=workspace, timeout=5).stdout.strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f_branch = ex.submit(_run, ["git", "branch", "--show-current"])
            f_status = ex.submit(_run, ["git", "status", "--short"])
            f_log = ex.submit(_run, ["git", "log", "--oneline", "-5"])

        parts = [f"Branch: {f_branch.result()}"]
        status = f_status.result()
        parts.append(f"Changes:\n{status}" if status else "Working tree clean.")
        log = f_log.result()
        if log:
            parts.append(f"Recent commits:\n{log}")
        return "\n".join(parts)
    except Exception:
        return "Git info unavailable."


def load_nvagent_ignore(workspace: Path) -> list[str]:
    """Load patterns from .nvagent/ignore, one pattern per line."""
    ignore_file = workspace / ".nvagent" / "ignore"
    if not ignore_file.exists():
        return []
    try:
        lines = ignore_file.read_text(encoding="utf-8").splitlines()
        return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    except Exception:
        return []


def build_system_prompt(workspace: Path, config: Config, memory: str = "") -> str:
    """Build the full system prompt with project context."""
    parts = [SYSTEM_PROMPT_BASE]
    parts.append(f"\n## Current Workspace\n`{workspace}`\n")
    parts.append(f"**Project type:** {detect_project_type(workspace)}\n")

    # Merge config patterns + .nvagent/ignore patterns
    extra_ignore = load_nvagent_ignore(workspace)
    all_ignore = config.context.ignore_patterns + extra_ignore

    tree = build_file_tree(workspace, all_ignore, max_files=config.agent.max_context_files)
    if tree:
        parts.append(f"\n## Project File Tree\n```\n{tree}\n```\n")

    key_file_contents = []
    for fname in KEY_FILES:
        fpath = workspace / fname
        if fpath.exists():
            content = _read_key_file(fpath)
            if content:
                key_file_contents.append(f"### {fname}\n```\n{content}\n```")
    if key_file_contents:
        parts.append("\n## Key Project Files\n" + "\n\n".join(key_file_contents))

    parts.append(f"\n## Git Status\n```\n{get_git_summary(workspace)}\n```\n")

    if memory and memory.strip():
        parts.append(f"\n## Project Memory (.nvagent/memory.md)\n{memory}\n")

    return "\n".join(parts)
