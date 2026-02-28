"""
Project context manager — builds rich context for the LLM without RAG.

Strategy:
1. Read .nvagent/memory.md (persistent project notes)
2. Build file tree (respecting ignore patterns)
3. Detect project type from files present
4. Read key files (package.json, pyproject.toml, README, etc.)
5. Inject all as system message prefix
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Optional

from nvagent.config import Config

# Files that are always read for project understanding
KEY_FILES = [
    "README.md",
    "README.rst",
    "README.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Makefile",
    "CMakeLists.txt",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    ".env.example",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
]

SYSTEM_PROMPT_BASE = """You are nvagent, an expert coding assistant powered by NVIDIA NIM models.
You operate directly on the user's codebase with full file system access.

## Your capabilities:
- Read, write, create, and delete files
- Run shell commands and capture output
- Search code across the entire codebase
- Understand git history and diffs
- Build, test, and debug projects end-to-end

## How you work:
- You ALWAYS read relevant files before making changes
- You ALWAYS show what you're about to do before doing it
- You write complete, production-quality code — never placeholders or TODOs unless asked
- You explain your reasoning concisely
- When you encounter errors, you debug systematically
- You prefer making targeted changes over rewriting everything
- You follow the project's existing code style and conventions

## Tool call discipline — CRITICAL:
- ALWAYS finish your current sentence and end with a newline BEFORE issuing a tool call.
- NEVER stop mid-word or mid-sentence to call a tool.
- Pattern: write a complete thought ending with ".\n", THEN call the tool.
- Correct:   "Let me read the file to understand the structure.\n" → read_file(...)
- Incorrect: "Let me read the file to understand the str" → read_file(...)

## Precise file reading — CRITICAL (Claude Code style):
You must read code precisely, like a skilled engineer — NOT by dumping entire files.

**The workflow for any unfamiliar file:**
1. `get_symbols(path)` — get all function/class names with their exact line numbers (fast, no content)
2. `search_code(query, path)` — find which lines contain what you need
3. `read_file(path, start_line=N, end_line=M)` — read ONLY the relevant section

**Rules:**
- Files under ~150 lines: `read_file(path)` is fine (no line range needed)
- Files 150-500 lines: use `get_symbols` first, then read only the function/class you need
- Files 500+ lines: ALWAYS use `get_symbols` + targeted read. NEVER read the whole file.
- To read one function: start ~5 lines before its `def`/`class`, end ~5 lines after its closing line
- To understand imports/top-level: read lines 1-50
- Reading the same file twice means your first read was too broad
- `find_definition(name)` jumps directly to any symbol's definition — use it instead of reading whole files to locate something

**Shell equivalent:** You use `grep -n` to find a line, then `sed -n '100,150p'` to read it.
That's exactly what `search_code` + `read_file(start_line, end_line)` does.

## File editing rules:
- Always read the specific section you're about to edit before editing it
- **Prefer `edit_file` over `write_file`** — use `edit_file` for any change touching less than ~40% of the file
- Use `write_file` only for new files or complete rewrites
- Show the diff or describe the change before writing
- Create parent directories if needed

## Code awareness:
- `get_symbols(path)` returns every function/class/type with its **line number** — use this as your map
- `find_definition(name)` returns the exact file + line where any symbol is defined
- `find_references(name)` shows every call site — essential before any rename or refactor
- A `## Symbol Index` block may be injected for files you've worked with; it shows signatures
- Use these tools to navigate — not blind whole-file reads

## Shell execution rules:
- Prefer non-destructive commands first
- Always capture and interpret output
- If a command might be destructive, explain what it does first

## Performance — batch file creation:
- When creating or overwriting **2 or more files**, use `write_files` (plural) instead of
  multiple `write_file` calls. `write_files` writes all files concurrently in a single
  round-trip, so a scaffold of 10 files takes the same time as writing 1.
- Group related new files (components, pages, lib modules, tests) into one `write_files` call.
"""


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


def build_system_prompt(workspace: Path, config: Config) -> str:
    """Build the full system prompt with project context."""
    parts = [SYSTEM_PROMPT_BASE]

    # ── Workspace info ─────────────────────────────────────────────────────
    parts.append(f"\n## Current Workspace\n`{workspace}`\n")

    # ── Project type detection ─────────────────────────────────────────────
    project_type = detect_project_type(workspace)
    parts.append(f"**Project type:** {project_type}\n")

    # ── File tree ──────────────────────────────────────────────────────────
    tree = build_file_tree(
        workspace,
        config.context.ignore_patterns,
        max_depth=4,
        max_files=config.agent.max_context_files,
    )
    if tree:
        parts.append(f"\n## Project File Tree\n```\n{tree}\n```\n")

    # ── Key files ──────────────────────────────────────────────────────────
    key_file_contents = []
    for fname in KEY_FILES:
        fpath = workspace / fname
        if fpath.exists():
            content = read_key_file(fpath, max_bytes=3000)
            if content:
                key_file_contents.append(f"### {fname}\n```\n{content}\n```")

    if key_file_contents:
        parts.append("\n## Key Project Files\n" + "\n\n".join(key_file_contents))

    # ── Git status ─────────────────────────────────────────────────────────
    git_summary = get_git_summary(workspace)
    parts.append(f"\n## Git Status\n```\n{git_summary}\n```\n")

    # ── Project memory ─────────────────────────────────────────────────────
    memory = read_memory(workspace)
    if memory:
        parts.append(f"\n## Project Memory (.nvagent/memory.md)\n{memory}\n")

    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


# ─────────────────────────────────────────────────────────────────────────────
# Budget-aware context assembly
# ─────────────────────────────────────────────────────────────────────────────


class ContextBudget:
    """
    Token budget tracker for context assembly.
    Uses chars-per-token=4 as a conservative estimate.
    """

    _CPT = 4  # chars per token

    def __init__(self, total_tokens: int = 32_000) -> None:
        self.total_tokens = total_tokens
        self.used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_tokens - self.used)

    def fits(self, text: str) -> bool:
        return (self.used + len(text) // self._CPT) <= self.total_tokens

    def consume(self, text: str) -> bool:
        """Consume budget for *text*. Returns True if it fit, False if over budget."""
        tokens = len(text) // self._CPT
        if self.used + tokens > self.total_tokens:
            return False
        self.used += tokens
        return True

    def consume_partial(self, text: str, max_chars: Optional[int] = None) -> str:
        """Return as much of *text* as the remaining budget allows (truncated if needed)."""
        remaining_chars = self.remaining * self._CPT
        if max_chars is not None:
            remaining_chars = min(remaining_chars, max_chars)
        if len(text) <= remaining_chars:
            self.used += len(text) // self._CPT
            return text
        truncated = text[:remaining_chars] + "\n... [truncated to fit context budget]"
        self.used += remaining_chars // self._CPT
        return truncated

    def __repr__(self) -> str:
        pct = 100 * self.used // max(self.total_tokens, 1)
        return f"ContextBudget({self.used}/{self.total_tokens} tokens, {pct}% used)"


def assemble_context(
    query: str,
    workspace: Path,
    active_paths: list[Path],
    config: "Config",
    retrieved_paths: Optional[list[Path]] = None,
    max_tokens: int = 32_000,
) -> str:
    """
    Budget-aware system-prompt assembly.

    Priority (descending) — higher-priority sections are inserted first and
    are guaranteed space; lower-priority sections fill whatever is left:

    1. Base system prompt
    2. Workspace info + project type
    3. File tree (capped)
    4. Key project files
    5. Git status
    6. Project memory (memory.md)
    7. Active files — full content
    8. Retrieved files — symbols-only fallback
    """
    # Lazy import to avoid circular at module load time
    from nvagent.core.symbols import extract_symbols

    budget = ContextBudget(max_tokens)
    parts: list[str] = []

    def _add(text: str) -> bool:
        if budget.consume(text):
            parts.append(text)
            return True
        return False

    # ── 1-2. Base + workspace info ────────────────────────────────────────────
    _add(SYSTEM_PROMPT_BASE)
    _add(f"\n## Current Workspace\n`{workspace}`\n")

    # Run project-type detection, file-tree build, and key-file reads in one
    # parallel thread pool so the 3 independent I/O-bound ops fully overlap.
    def _try_read_key(fname: str) -> tuple[str, Optional[str]]:
        fpath = workspace / fname
        if not fpath.exists():
            return fname, None
        return fname, read_key_file(fpath, max_bytes=3000)

    _n_workers = min(8, len(KEY_FILES)) + 2  # +2 for tree and type futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers) as _ex:
        _f_type = _ex.submit(detect_project_type, workspace)
        _f_tree = _ex.submit(
            build_file_tree,
            workspace,
            config.context.ignore_patterns,
            4,
            config.agent.max_context_files,
        )
        _kf_results = list(_ex.map(_try_read_key, KEY_FILES))

    project_type = _f_type.result()
    tree = _f_tree.result()
    _add(f"**Project type:** {project_type}\n")

    if tree:
        tree_block = f"\n## Project File Tree\n```\n{tree}\n```\n"
        if not budget.fits(tree_block):
            cap = budget.remaining * budget._CPT - 80
            tree_block = f"\n## Project File Tree\n```\n{tree[:cap]}\n... [truncated]\n```\n"
        _add(tree_block)

    key_parts: list[str] = []
    for fname, c in _kf_results:
        if c:
            key_parts.append(f"### {fname}\n```\n{c}\n```")
    if key_parts:
        key_block = "\n## Key Project Files\n" + "\n\n".join(key_parts)
        if not budget.fits(key_block):
            # Include only as many key files as fit
            fitted: list[str] = []
            sub = ContextBudget(budget.remaining)
            for kp in key_parts:
                if sub.consume(kp):
                    fitted.append(kp)
            if fitted:
                key_block = "\n## Key Project Files\n" + "\n\n".join(fitted)
                for _ in fitted:
                    pass  # budget already tracked in sub; sync
                budget.used += sub.used
                parts.append(key_block)
        else:
            _add(key_block)

    # ── 5. Git status ─────────────────────────────────────────────────────────
    git_summary = get_git_summary(workspace)
    _add(f"\n## Git Status\n```\n{git_summary}\n```\n")

    # ── 6. Project memory.md ──────────────────────────────────────────────────
    memory_md = read_memory(workspace)
    if memory_md:
        _add(f"\n## Project Memory (.nvagent/memory.md)\n{memory_md}\n")

    # ── 7-8. Active + retrieved file content (parallel reads) ─────────────────
    all_paths: list[Path] = list(active_paths)
    if retrieved_paths:
        seen_s = {str(p.resolve()) for p in all_paths}
        for rp in retrieved_paths:
            rs = str(rp.resolve())
            if rs not in seen_s:
                all_paths.append(rp)
                seen_s.add(rs)

    active_file_parts: list[str] = []
    symbol_only_parts: list[str] = []

    # Read all active/retrieved files concurrently in a thread pool.
    def _read_af(fpath: Path) -> tuple[Path, str | None]:
        try:
            return fpath, fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return fpath, None

    if all_paths:
        _n_af = min(8, len(all_paths))
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n_af) as _af_ex:
            _af_results: list[tuple[Path, str | None]] = list(_af_ex.map(_read_af, all_paths))
    else:
        _af_results = []

    for fpath, content in _af_results:
        try:
            rel = fpath.relative_to(workspace)
        except ValueError:
            rel = fpath

        file_block = f"### {rel}\n```\n{content[:6000]}\n```"
        if budget.fits(file_block):
            budget.consume(file_block)
            active_file_parts.append(file_block)
        else:
            # Symbols-only fallback
            try:
                sym_idx = extract_symbols(fpath)
                if not sym_idx.is_empty():
                    sym_lines = [f"  {s}" for s in sym_idx.symbols[:25]]
                    sym_block = f"### {rel} (symbols only)\n" + "\n".join(sym_lines)
                    if budget.consume(sym_block):
                        symbol_only_parts.append(sym_block)
            except Exception:
                pass

    if active_file_parts:
        parts.append(
            "\n## Active Files (recently read/modified)\n" + "\n\n".join(active_file_parts) + "\n"
        )
    if symbol_only_parts:
        parts.append(
            "\n## Files (symbols only — context budget reached)\n"
            + "\n\n".join(symbol_only_parts)
            + "\n"
        )

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Active-file context injection (multi-file awareness)
# ─────────────────────────────────────────────────────────────────────────────

_PATH_LIKE = re.compile(
    r"[\w./\-]+\.(?:py|ts|js|tsx|jsx|go|rs|java|c|cpp|h|hpp|rb|sh|yaml|yml|toml|json|md|txt|cfg|ini|env)"
)


def extract_active_files(messages: list[dict], workspace: Path) -> list[Path]:
    """
    Scan message history for file paths that were read or written via tools.
    Returns a deduplicated list of existing workspace-relative Paths.
    """
    seen: dict[str, Path] = {}
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            continue
        for match in _PATH_LIKE.finditer(content):
            raw = match.group(0)
            # Try as relative to workspace first, then absolute
            for candidate in (workspace / raw, Path(raw)):
                try:
                    if candidate.exists() and candidate.is_file():
                        key = str(candidate.resolve())
                        if key not in seen:
                            seen[key] = candidate.resolve()
                        break
                except Exception:
                    pass
    return list(seen.values())


def build_active_files_context(
    active_paths: list[Path],
    workspace: Path,
    max_bytes_each: int = 6000,
    max_total_bytes: int = 24000,
) -> str:
    """
    Build a '## Active Files' context block from the given paths.
    Skips files that are already injected as key files to avoid duplication.
    """
    if not active_paths:
        return ""
    parts: list[str] = []
    total = 0
    for fpath in active_paths:
        if total >= max_total_bytes:
            parts.append(
                f"... ({len(active_paths) - len(parts)} more active files — context limit reached)"
            )
            break
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(content) > max_bytes_each:
            content = (
                content[:max_bytes_each]
                + f"\n... [truncated, {len(content) - max_bytes_each} more bytes]"
            )
        rel = fpath.relative_to(workspace) if fpath.is_relative_to(workspace) else fpath
        parts.append(f"### {rel}\n```\n{content}\n```")
        total += len(content)
    if not parts:
        return ""
    return "\n## Active Files (recently read/modified)\n" + "\n\n".join(parts) + "\n"
