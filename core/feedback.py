"""Structured tool-error feedback: targeted retry hints for the model.

When the same tool fails with the same error category two times in a row,
a concise corrective hint is injected into the conversation so the model
can self-correct without a full extra LLM round-trip.
"""

from __future__ import annotations

import re as _re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Retry hint messages (keyed by error category)
# ─────────────────────────────────────────────────────────────────────────────

RETRY_HINTS: dict[str, str] = {
    "ambiguous_edit": (
        "The 'search' string matched multiple locations (or matched zero). "
        "Provide a LONGER, more unique search string that includes 3-5 lines of surrounding "
        "context — not just the single line you want to change."
    ),
    "edit_not_found": (
        "The 'search' string was not found in the file. "
        "Call read_file on that path first to get the EXACT current text (including all whitespace "
        "and indentation). Your search string must match the file content character-for-character."
    ),
    "file_not_found": (
        "The file path does not exist. "
        "Use list_dir or find_files to confirm the correct path before reading or writing."
    ),
    "timeout": (
        "The command timed out. "
        "If a longer runtime is expected, increase the 'timeout' argument. "
        "For test suites, pass a specific path to run_tests instead of the full suite."
    ),
    "command_error": (
        "The command exited with a non-zero code. "
        "Read the STDERR output carefully — it usually explains the failure. "
        "Fix the root cause (missing dependency, syntax error, wrong arguments) before retrying."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Error classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_tool_error(name: str, result: str) -> Optional[str]:
    """Return an error-category key for a failed tool result, or None.

    The returned key can be looked up in RETRY_HINTS to get a corrective message
    to inject into the conversation.
    """
    r_lower = result.lower()
    if "⏱" in result or "timed out" in r_lower:
        return "timeout"
    if name in ("edit_file", "str_replace_editor"):
        if _re.search(r"found \d+ (times?|matches?)", r_lower) or "multiple" in r_lower:
            return "ambiguous_edit"
        if "not found" in r_lower or "no match" in r_lower or "could not find" in r_lower:
            return "edit_not_found"
    if name in ("read_file", "write_file", "edit_file", "str_replace_editor", "find_files", "delete_file"):
        if "no such file" in r_lower or "does not exist" in r_lower or (
            "not found" in r_lower and name != "edit_file"
        ):
            return "file_not_found"
    if name == "run_command":
        if _re.search(r"exit code:\s*[1-9]", r_lower):
            return "command_error"
    return None
