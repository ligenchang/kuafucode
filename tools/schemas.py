"""
Tool JSON schemas (OpenAI tool_use format) and process-group utilities.
"""

from __future__ import annotations

import os
import signal
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kill_proc_group(proc: "asyncio.subprocess.Process") -> None:  # type: ignore[name-defined]
    """Kill the process and its entire process group on Unix (SIGKILL).

    Uses os.killpg so child processes spawned by the shell command are also
    terminated, avoiding zombie accumulation across long sessions.
    The subprocess must have been started with start_new_session=True.
    Falls back to proc.kill() on Windows or if the process is already dead.
    """
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Tool JSON schemas (OpenAI tool_use format)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file or a specific line range within it. "
                "SYMBOL MAP FAST-PATH: For any file larger than ~150 lines, calling read_file WITHOUT "
                "start_line/end_line does NOT return the raw content — it returns a SYMBOL MAP listing "
                "every function/class/method with its exact line number. "
                "Use that line number directly in the NEXT read_file call with start_line/end_line. "
                "TWO-STEP WORKFLOW for large files: "
                "(1) read_file(path) → get symbol map with line numbers, "
                "(2) read_file(path, start_line=N, end_line=M) → read only the function you need. "
                "NEVER page through a large file in sequential chunks (L1-300, L300-600 …) — "
                "that wastes tokens and ignores the symbol map. "
                "NEVER read 400+ lines just to find one function; use the symbol map line numbers instead. "
                "For files under ~150 lines, omit start_line/end_line to get the full content directly. "
                "For a targeted read: include ~5 lines before the def line and ~5 lines past the function end."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file (relative to workspace or absolute)"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed, inclusive). Use with end_line to read a targeted section."},
                    "end_line": {"type": "integer", "description": "Last line to read (1-indexed, inclusive). Use with start_line to read a targeted section."},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a file with the given content. Creates parent directories if needed. Always provide the COMPLETE file content. For 2+ files, prefer write_files (plural) instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Complete file content to write"},
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_files",
            "description": (
                "Write multiple files simultaneously in a single call. "
                "PREFER this over multiple write_file calls whenever you need to create or overwrite 2 or more files — "
                "all writes happen concurrently so the batch completes in roughly the time of one write. "
                "Creates parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "description": "List of files to write. Each item must have 'path' and 'content'.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path":    {"type": "string", "description": "File path (relative to workspace or absolute)"},
                                "content": {"type": "string", "description": "Complete file content to write"},
                            },
                            "required": ["path", "content"]
                        }
                    }
                },
                "required": ["files"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories. Use to explore project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command. Use for building, testing, installing deps, running scripts.\n"
                "When output is truncated, a [TRUNCATED] notice appears with the total byte count. "
                "To see more: increase max_output_chars, re-run with `| head -N` / `| tail -N`, "
                "pipe to a file with `> /tmp/out.txt` then read it, or filter with `| grep pattern`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "cwd": {"type": "string", "description": "Working directory (optional, defaults to workspace)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
                    "max_output_chars": {
                        "type": "integer",
                        "description": (
                            "Max total chars of output to return (default: 8000). "
                            "Set higher (e.g. 20000) when you need full output. "
                            "stdout and stderr are each capped at max_output_chars/2."
                        ),
                    },
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for text or patterns across the codebase. Returns file paths, line numbers, and matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in (optional, defaults to workspace)"},
                    "file_pattern": {"type": "string", "description": "File pattern to filter, e.g. '*.py' (optional)"},
                    "regex": {"type": "boolean", "description": "Treat query as regex (default: false)"},
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search (default: false)"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Get git repository status — modified files, staged changes, branch info.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Get git diff to see what has changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "staged": {"type": "boolean", "description": "Show staged changes (default: false = unstaged)"},
                    "file": {"type": "string", "description": "Specific file to diff (optional)"},
                    "commit": {"type": "string", "description": "Compare with a specific commit hash (optional)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Fetch content from a URL — documentation, APIs, GitHub raw files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default: 8000)"},
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Update the project memory file (.nvagent/memory.md) with important information to remember across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to append or the full new memory content"},
                    "mode": {"type": "string", "enum": ["append", "replace"], "description": "append (default) or replace"},
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file or empty directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to delete"},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbols",
            "description": (
                "Extract function, class, and type signatures from a file or directory "
                "WITHOUT reading full file contents. Use this to understand a file's API surface, "
                "find where a symbol is defined, or explore a module before deciding which parts to read. "
                "Much faster and cheaper than read_file for large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory to analyse"},
                    "include_imports": {"type": "boolean", "description": "Also show import statements (default: false)"},
                    "follow_imports": {"type": "boolean", "description": "Also show signatures of locally-imported files (default: false)"},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dep_graph",
            "description": (
                "Analyse the dependency graph for one or more files. "
                "Shows direct imports, transitive dependencies, and which files import THIS file (dependents — "
                "critical before renaming or deleting). Also reports external packages used and detects "
                "circular imports. Use before any cross-file refactor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":               {"type": "string",  "description": "File or directory to analyse"},
                    "show_dependents":    {"type": "boolean", "description": "Show which files import this file (default: true)"},
                    "show_transitive":    {"type": "boolean", "description": "Show full transitive dependency tree (default: false)"},
                    "show_external":      {"type": "boolean", "description": "List external (third-party) packages used (default: true)"},
                    "detect_cycles":      {"type": "boolean", "description": "Check for circular import cycles (default: false)"},
                    "max_depth":          {"type": "integer", "description": "Maximum traversal depth for transitive analysis (default: 3)"},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": (
                "Find where a symbol (function, class, variable, type) is DEFINED in the workspace. "
                "Use before renaming a symbol, before editing a function, or when you see a name you "
                "don't recognise. Returns file path, line number, and signature. "
                "Much more precise than search_code for locating definitions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The exact symbol name to find (e.g. 'Agent', 'build_context', 'MAX_RETRIES')"},
                    "hint_file": {"type": "string", "description": "Optional file path to search first (speeds things up when you know the likely file)"},
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": (
                "Find all places in the workspace where a symbol is USED (called, imported, assigned, "
                "or used as a type hint). Essential before renaming a symbol — shows all call sites "
                "that would need updating. Returns file, line, context snippet, and usage kind "
                "(call/import/assign/type_hint/unknown)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The exact symbol name to find usages of"},
                    "hint_file": {"type": "string", "description": "Optional file to search first"},
                    "include_definitions": {"type": "boolean", "description": "Also include definition sites (default: false)"},
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make targeted search-and-replace edits to an existing file. "
                "PREFER this over write_file when changing less than ~40% of the file. "
                "Each edit finds the FIRST occurrence of 'search' text and replaces it with 'replace'. "
                "Always read the file first so your 'search' strings match exactly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "edits": {
                        "type": "array",
                        "description": "List of search/replace operations applied in order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "search":  {"type": "string", "description": "Exact text to find (must be unique enough to locate the right spot)"},
                                "replace": {"type": "string", "description": "Text to substitute in place of 'search'"},
                            },
                            "required": ["search", "replace"]
                        }
                    },
                    "create_if_missing": {"type": "boolean", "description": "Create the file with 'replace' content if it doesn't exist (default: false)"},
                },
                "required": ["path", "edits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": (
                "Search the workspace-wide symbol index for symbols matching a name or query. "
                "Returns file paths and line numbers where matching symbols are defined. "
                "Faster than grep — uses the pre-built mtime-cached index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string",  "description": "Symbol name or substring to search for"},
                    "exact":  {"type": "boolean", "description": "If true, require exact name match (default false = substring)"},
                    "kinds":  {"type": "array", "items": {"type": "string"}, "description": "Optional list of kinds to filter: function, class, method, field, const, type, import"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 30)"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": (
                "Run a static analysis linter (ruff, mypy, pyright, tsc, eslint) on a file or the whole workspace. "
                "Returns structured issues with file, line, column, code, and message. "
                "Use 'detect' as tool name to list available linters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool":  {"type": "string", "description": "Linter to run: ruff | mypy | pyright | tsc | eslint | all | detect"},
                    "path":  {"type": "string", "description": "Optional: path to a specific file or directory (default: whole workspace)"},
                    "fix":   {"type": "boolean", "description": "Auto-fix issues where supported (ruff --fix, eslint --fix)"},
                    "max_issues": {"type": "integer", "description": "Max issues to return (default 50)"},
                },
                "required": ["tool"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_learn",
            "description": (
                "Store a fact, insight, convention, or project note in long-term memory. "
                "Persists across sessions. Use for important findings about architecture, "
                "conventions, known issues, or anything worth remembering for future tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact or insight to remember (be specific and concise)"},
                    "tags":    {"type": "array",  "items": {"type": "string"}, "description": "Optional classification tags, e.g. ['architecture', 'bug', 'convention']"},
                    "file":    {"type": "string", "description": "Optional: file path this memory relates to"},
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": (
                "Retrieve relevant facts from long-term memory by natural-language query. "
                "Returns the most relevant stored memories ranked by relevance. "
                "Use before starting a task to check for prior context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string", "description": "Natural-language search query"},
                    "max_results": {"type": "integer", "description": "Maximum memories to return (default 8)"},
                    "tags":        {"type": "array", "items": {"type": "string"}, "description": "Optional: filter to memories with these tags"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget",
            "description": "Remove a specific memory entry by its key (obtained from memory_recall results).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The memory entry key to delete (12-char hex id)"},
                },
                "required": ["key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_note",
            "description": (
                "Attach a structured summary or note to a specific file in long-term memory. "
                "Useful for recording what a file does, known issues, or conventions specific to that file. "
                "Notes persist across sessions and are injected into context when the file is active."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Workspace-relative path of the file to annotate"},
                    "summary": {"type": "string", "description": "One-line summary of what this file does"},
                    "note":    {"type": "string", "description": "Additional detail, conventions, or known issues for this file"},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the project test suite (auto-detects pytest/jest/vitest/cargo/go). "
                "Parses output into structured pass/fail/skip counts and failing test names. "
                "Always run tests before considering a task done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string", "description": "Optional: file or directory to limit the test run"},
                    "framework":  {"type": "string", "description": "Override framework detection: pytest|jest|vitest|cargo|go|unittest"},
                    "extra_args": {"type": "array",  "items": {"type": "string"}, "description": "Extra args forwarded to the test runner"},
                    "retry_on_fail": {"type": "boolean", "description": "Re-run on transient failures (default: false)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_formatter",
            "description": (
                "Run a code formatter (auto-detects black/ruff-format/prettier/gofmt/rustfmt). "
                "Formats in-place by default; use check_only=true for a dry-run diff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string",  "description": "File or directory to format (optional, defaults to whole workspace)"},
                    "formatter":  {"type": "string",  "description": "Override formatter: black|ruff-format|prettier|gofmt|rustfmt"},
                    "check_only": {"type": "boolean", "description": "Dry-run, show diff without writing (default: false)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "undo_last_turn",
            "description": (
                "Undo all file changes made in the previous agent turn. "
                "Restores files to their state before the last batch of tool calls. "
                "Use when the previous changes were incorrect or need to be retried."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "Find files matching a glob pattern in the workspace. "
                "Much faster than list_dir for pattern-based searches. "
                "Examples: '*.test.ts', 'src/**/*.py', '**/__init__.py'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match, e.g. '*.py', 'src/**/*.ts'"},
                    "path": {"type": "string", "description": "Directory to search in (optional, defaults to workspace)"},
                    "max_results": {"type": "integer", "description": "Max files to return (default: 100)"},
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stage files for git commit. Use before git_commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to stage. Use ['.'] to stage all changes."
                    },
                },
                "required": ["paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Create a git commit with a message. Stage files first with git_add.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                    "add_all": {"type": "boolean", "description": "Auto-stage all tracked modified files before committing (git commit -a). Default: false."},
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show git commit history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of commits to show (default: 10)"},
                    "file": {"type": "string", "description": "Show only commits affecting this file (optional)"},
                    "oneline": {"type": "boolean", "description": "Compact one-line format (default: true)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff patch (git diff / diff -u format) to the workspace. "
                "Validates the patch before applying. Uses pure-Python fallback when the "
                "'patch' binary is unavailable (e.g. in minimal Docker containers)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch":   {"type": "string",  "description": "Unified diff patch text (--- a/ ... +++ b/ ... format)"},
                    "dry_run": {"type": "boolean", "description": "Validate only, do not apply (default: false)"},
                },
                "required": ["patch"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_notebook",
            "description": (
                "Read a Jupyter notebook (.ipynb). "
                "Without cell_index, returns a summary of all cells: index, type (code/markdown), "
                "first ~3 lines of source, and a compact output summary. "
                "With cell_index, returns the full source + all outputs of that one cell. "
                "Use this instead of read_file for .ipynb files — read_file returns raw JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string",  "description": "Path to the .ipynb file"},
                    "cell_index": {"type": "integer", "description": "0-based cell index to read in full (omit for summary of all cells)"},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_notebook",
            "description": (
                "Edit a Jupyter notebook (.ipynb) cell-by-cell. "
                "Operations: "
                "'update' — replace the source of an existing cell (cell_index required); "
                "'insert' — insert a new cell before cell_index (or append if cell_index omitted); "
                "'delete' — remove the cell at cell_index. "
                "Use read_notebook first to see cell indexes and current content. "
                "All edits clear the cell's outputs automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string",  "description": "Path to the .ipynb file"},
                    "operation":  {"type": "string",  "enum": ["update", "insert", "delete"], "description": "Edit operation"},
                    "cell_index": {"type": "integer", "description": "0-based cell index to update/delete, or insertion point for insert"},
                    "source":     {"type": "string",  "description": "New cell source (required for update and insert)"},
                    "cell_type":  {"type": "string",  "enum": ["code", "markdown"], "description": "Cell type for insert (default: code)"},
                },
                "required": ["path", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "checkpoint",
            "description": (
                "Save a named rollback point for all currently modified files. "
                "Call before risky changes so you can restore with rollback()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name":          {"type": "string", "description": "Checkpoint name, e.g. 'before-refactor' (auto-assigned if omitted)"},
                    "include_paths": {"type": "array",  "items": {"type": "string"}, "description": "Additional file paths to include in the snapshot"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rollback",
            "description": (
                "Restore workspace files to a previously saved checkpoint. "
                "Defaults to the most recent checkpoint when name is omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Checkpoint to restore (optional, defaults to most recent)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Write or update the in-session task list. "
                "ALWAYS replace the full list — include ALL existing todos plus any new ones. "
                "Use this to track multi-step plans so task state is always visible. "
                "Call todo_read first to get the current list before making updates. "
                "Status values: 'pending' (not started), 'in_progress' (currently working — max 1 at a time), "
                "'completed' (done), 'cancelled' (dropped). "
                "Priority: 'high', 'medium', 'low'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Complete replacement list of all todos (existing + new). Reuse IDs across updates.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":       {"type": "string", "description": "Stable unique ID (e.g. '1', 'setup-db'). Keep the same ID when updating status."},
                                "content":  {"type": "string", "description": "What needs to be done — concise, action-oriented"},
                                "status":   {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            },
                            "required": ["id", "content", "status", "priority"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_read",
            "description": (
                "Read the current in-session task list. "
                "Call at the start of complex work to see outstanding todos, "
                "and before todo_write to avoid overwriting the existing list."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]
