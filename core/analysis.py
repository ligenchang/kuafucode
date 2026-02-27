"""
Static analysis integration: ruff, mypy, pyright, tsc, eslint.

Unified output via AnalysisIssue dataclass.  Zero new dependencies —
all tools are shelled out; presence is detected at runtime via shutil.which.

Public API
----------
  detect_linters(workspace)                   → list[str]
  run_analysis(tool, workspace, path, fix)    → list[AnalysisIssue]
  run_all_linters(workspace, path)            → dict[str, list[AnalysisIssue]]
  format_issues(issues, workspace)            → str  (human-readable block)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisIssue
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisIssue:
    file:     str
    line:     int
    col:      int
    code:     str          # e.g. "E501", "TS2345", "no-unused-vars"
    message:  str
    severity: str          # "error" | "warning" | "info" | "note"
    tool:     str = ""     # "ruff" | "mypy" | "pyright" | "tsc" | "eslint"

    def render(self, workspace: Optional[Path] = None) -> str:
        try:
            ws_r = workspace.resolve() if workspace else None
            rel  = Path(self.file).relative_to(ws_r) if ws_r else Path(self.file)
        except ValueError:
            rel = Path(self.file)
        code_part = f" [{self.code}]" if self.code else ""
        return (
            f"{rel}:{self.line}:{self.col}  {self.severity.upper()}{code_part}  {self.message}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Linter detection
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_LINTERS = [
    "ruff",    # Python — fast, covers flake8 + isort + pyupgrade rules
    "mypy",    # Python — type checking
    "pyright", # Python — type checking (alternative to mypy)
    "tsc",     # TypeScript
    "eslint",  # JavaScript / TypeScript
]


def detect_linters(workspace: Optional[Path] = None) -> list[str]:
    """Return which supported linters are available in PATH."""
    available = [t for t in _SUPPORTED_LINTERS if shutil.which(t)]

    # Also check workspace-local node_modules/.bin for tsc / eslint
    if workspace:
        bin_dir = workspace / "node_modules" / ".bin"
        for tool in ("tsc", "eslint"):
            if tool not in available and (bin_dir / tool).exists():
                available.append(tool)

    return available


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 60) -> tuple[str, str, int]:
    """Run *cmd* and return (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except FileNotFoundError:
        return "", f"command not found: {cmd[0]}", -1


# ─────────────────────────────────────────────────────────────────────────────
# Per-tool runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_ruff(
    workspace: Path,
    path: Optional[Path] = None,
    fix: bool = False,
) -> list[AnalysisIssue]:
    """Run `ruff check --output-format=json`."""
    target = str(path) if path else "."
    cmd = [shutil.which("ruff") or "ruff", "check", "--output-format=json", "--quiet"]
    if fix:
        cmd.append("--fix")
    cmd.append(target)

    stdout, stderr, _ = _run(cmd, cwd=workspace)
    issues: list[AnalysisIssue] = []
    if not stdout.strip():
        return issues
    try:
        data: list = json.loads(stdout)
    except json.JSONDecodeError:
        return issues

    for item in data:
        location = item.get("location", {})
        end_loc  = item.get("end_location", {})
        issues.append(AnalysisIssue(
            file     = item.get("filename", ""),
            line     = location.get("row", 0),
            col      = location.get("column", 0),
            code     = item.get("code", ""),
            message  = item.get("message", ""),
            severity = "warning" if item.get("code", "").startswith("W") else "error",
            tool     = "ruff",
        ))
    return issues


def _run_mypy(
    workspace: Path,
    path: Optional[Path] = None,
    fix: bool = False,
) -> list[AnalysisIssue]:
    """Run `mypy --show-column-numbers --show-error-codes`."""
    target = str(path) if path else "."
    cmd = [
        shutil.which("mypy") or "mypy",
        "--show-column-numbers",
        "--show-error-codes",
        "--no-error-summary",
        "--no-pretty",
        target,
    ]
    stdout, stderr, _ = _run(cmd, cwd=workspace)
    issues: list[AnalysisIssue] = []

    # Pattern: path/to/file.py:12:34: error: msg  [error-code]
    pattern = re.compile(
        r"^(.+?):(\d+):(\d+):\s+(error|warning|note):\s+(.+?)(?:\s+\[([^\]]+)\])?$"
    )
    for line in (stdout + "\n" + stderr).splitlines():
        m = pattern.match(line.strip())
        if m:
            issues.append(AnalysisIssue(
                file     = m.group(1),
                line     = int(m.group(2)),
                col      = int(m.group(3)),
                code     = m.group(6) or "",
                message  = m.group(5),
                severity = m.group(4),
                tool     = "mypy",
            ))
    return issues


def _run_pyright(
    workspace: Path,
    path: Optional[Path] = None,
    fix: bool = False,
) -> list[AnalysisIssue]:
    """Run `pyright --outputjson`."""
    pyright = shutil.which("pyright")
    if not pyright:
        return []

    cmd = [pyright, "--outputjson"]
    if path:
        cmd.append(str(path))
    stdout, stderr, _ = _run(cmd, cwd=workspace)

    issues: list[AnalysisIssue] = []
    if not stdout.strip():
        return issues

    # pyright sometimes prints non-JSON lines before the JSON blob
    json_start = stdout.find("{")
    if json_start < 0:
        return issues
    try:
        data = json.loads(stdout[json_start:])
    except json.JSONDecodeError:
        return issues

    for diag in data.get("generalDiagnostics", []):
        rng  = diag.get("range", {})
        start = rng.get("start", {})
        sev   = diag.get("severity", "error")
        issues.append(AnalysisIssue(
            file     = diag.get("file", ""),
            line     = start.get("line", 0) + 1,   # pyright is 0-indexed
            col      = start.get("character", 0) + 1,
            code     = diag.get("rule", ""),
            message  = diag.get("message", ""),
            severity = sev,
            tool     = "pyright",
        ))
    return issues


def _run_tsc(
    workspace: Path,
    path: Optional[Path] = None,
    fix: bool = False,
) -> list[AnalysisIssue]:
    """Run `tsc --noEmit`."""
    # Prefer workspace-local tsc
    local_tsc = workspace / "node_modules" / ".bin" / "tsc"
    tsc = str(local_tsc) if local_tsc.exists() else (shutil.which("tsc") or "tsc")

    cmd = [tsc, "--noEmit"]
    stdout, stderr, _ = _run(cmd, cwd=workspace)
    issues: list[AnalysisIssue] = []

    # Pattern: file(line,col): error TSnnn: message
    pattern = re.compile(r"^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)$")
    for line in (stdout + "\n" + stderr).splitlines():
        m = pattern.match(line.strip())
        if m:
            issues.append(AnalysisIssue(
                file     = m.group(1),
                line     = int(m.group(2)),
                col      = int(m.group(3)),
                code     = m.group(5),
                message  = m.group(6),
                severity = m.group(4),
                tool     = "tsc",
            ))
    return issues


def _run_eslint(
    workspace: Path,
    path: Optional[Path] = None,
    fix: bool = False,
) -> list[AnalysisIssue]:
    """Run `eslint --format json`."""
    local_eslint = workspace / "node_modules" / ".bin" / "eslint"
    eslint = str(local_eslint) if local_eslint.exists() else (shutil.which("eslint") or "eslint")

    target = str(path) if path else "."
    cmd = [eslint, "--format", "json"]
    if fix:
        cmd.append("--fix")
    # Limit to common source extensions
    cmd += ["--ext", ".js,.jsx,.ts,.tsx,.mjs,.cjs", target]

    stdout, stderr, _ = _run(cmd, cwd=workspace)
    issues: list[AnalysisIssue] = []
    if not stdout.strip():
        return issues

    try:
        data: list = json.loads(stdout)
    except json.JSONDecodeError:
        return issues

    _sev = {0: "info", 1: "warning", 2: "error"}
    for file_result in data:
        fpath = file_result.get("filePath", "")
        for msg in file_result.get("messages", []):
            issues.append(AnalysisIssue(
                file     = fpath,
                line     = msg.get("line", 0),
                col      = msg.get("column", 0),
                code     = msg.get("ruleId") or "",
                message  = msg.get("message", ""),
                severity = _sev.get(msg.get("severity", 2), "error"),
                tool     = "eslint",
            ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoints
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_RUNNERS = {
    "ruff":    _run_ruff,
    "mypy":    _run_mypy,
    "pyright": _run_pyright,
    "tsc":     _run_tsc,
    "eslint":  _run_eslint,
}


def run_analysis(
    tool: str,
    workspace: Path,
    path: Optional[Path] = None,
    fix: bool = False,
) -> list[AnalysisIssue]:
    """
    Run *tool* against *path* (or the whole workspace if path is None).
    Returns a list of AnalysisIssue objects.
    Raises ValueError for unknown tools.
    """
    tool = tool.lower()
    if tool not in _TOOL_RUNNERS:
        raise ValueError(
            f"Unknown tool {tool!r}. Supported: {', '.join(_TOOL_RUNNERS)}"
        )
    runner = _TOOL_RUNNERS[tool]
    return runner(workspace, path, fix)


def run_all_linters(
    workspace: Path,
    path: Optional[Path] = None,
    fix: bool = False,
    tools: Optional[list[str]] = None,
) -> dict[str, list[AnalysisIssue]]:
    """
    Run all available linters (or *tools* subset) and return
    a mapping of tool → [AnalysisIssue, ...].
    Skips tools that are not installed.
    """
    available = detect_linters(workspace)
    run_tools = tools or available

    results: dict[str, list[AnalysisIssue]] = {}
    for tool in run_tools:
        if tool not in available:
            continue
        try:
            results[tool] = run_analysis(tool, workspace, path, fix)
        except Exception as exc:
            results[tool] = []  # Silently skip broken tools

    return results


def format_issues(
    issues: list[AnalysisIssue],
    workspace: Optional[Path] = None,
    max_issues: int = 100,
    group_by_file: bool = True,
) -> str:
    """
    Render AnalysisIssue list as a human-readable (and LLM-friendly) string.
    """
    if not issues:
        return "No issues found."

    shown = issues[:max_issues]

    if group_by_file:
        # Group by file path
        from collections import defaultdict
        grouped: dict[str, list[AnalysisIssue]] = defaultdict(list)
        for issue in shown:
            grouped[issue.file].append(issue)

        lines = []
        for fpath in sorted(grouped):
            try:
                ws_r = workspace.resolve() if workspace else None
                rel  = Path(fpath).relative_to(ws_r) if ws_r else Path(fpath)
            except ValueError:
                rel = Path(fpath)
            lines.append(f"\n{rel}")
            for iss in sorted(grouped[fpath], key=lambda i: (i.line, i.col)):
                code_part = f" [{iss.code}]" if iss.code else ""
                sev_icon  = {"error": "✗", "warning": "⚠", "note": "·", "info": "·"}.get(iss.severity, "·")
                lines.append(
                    f"  {sev_icon} {iss.line}:{iss.col}{code_part}  {iss.message}"
                    + (f"  ({iss.tool})" if iss.tool else "")
                )
        result = "\n".join(lines)
    else:
        result = "\n".join(i.render(workspace) for i in shown)

    if len(issues) > max_issues:
        result += f"\n\n… {len(issues) - max_issues} more issues not shown."

    return result.strip()
