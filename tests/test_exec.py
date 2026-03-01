"""Tests for execution handlers — run_command filter, truncation, active_proc tracking."""

import asyncio
import tempfile
from pathlib import Path

try:
    try:
        import pytest
    except ImportError:
        pytest = None
except ImportError:
    pytest = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # /home/claude -> nvagent symlink

from nvagent.tools import ToolExecutor


def _make_executor() -> tuple[ToolExecutor, Path]:
    tmp = Path(tempfile.mkdtemp())
    ex = ToolExecutor(workspace=tmp, safe_mode=False)
    return ex, tmp


def test_run_command_basic():
    ex, _ = _make_executor()
    result = asyncio.run(ex.execute("run_command", {"command": "echo hello_world"}))
    assert "hello_world" in result
    assert "Exit code: 0" in result


def test_run_command_exit_code_nonzero():
    ex, _ = _make_executor()
    result = asyncio.run(ex.execute("run_command", {"command": "exit 42"})    )
    assert "42" in result


def test_run_command_filter_keeps_matching_lines():
    ex, _ = _make_executor()
    # Use a Python one-liner to avoid shell escaping issues
    cmd = 'python3 -c "print(\'PASSED test_a\'); print(\'FAILED test_b\'); print(\'PASSED test_c\')"'
    result = asyncio.run(ex.execute("run_command", {"command": cmd, "filter": "FAILED"}))
    # The STDOUT section should only contain FAILED lines (after filtering)
    stdout_section = result.split("STDOUT:")[-1] if "STDOUT:" in result else ""
    assert "FAILED" in stdout_section
    assert "PASSED" not in stdout_section
    assert "filter=" in result


def test_run_command_filter_case_insensitive():
    ex, _ = _make_executor()
    cmd = 'python3 -c "print(\'Error: something went wrong\'); print(\'all good here\')"'
    result = asyncio.run(ex.execute("run_command", {"command": cmd, "filter": "error"}))
    stdout_section = result.split("STDOUT:")[-1] if "STDOUT:" in result else ""
    assert "Error" in stdout_section
    assert "all good" not in stdout_section


def test_run_command_filter_no_matches():
    ex, _ = _make_executor()
    result = asyncio.run(ex.execute("run_command", {
        "command": 'echo "all is well"',
        "filter": "CRITICAL_FAILURE",
    }))
    assert "filter=" in result
    assert "0/" in result  # 0 matched


def test_run_command_filter_invalid_regex():
    ex, _ = _make_executor()
    result = asyncio.run(ex.execute("run_command", {
        "command": "echo hello",
        "filter": "[invalid",  # bad regex
    }))
    # Should not crash — should report filter error
    assert "filter error" in result.lower() or "Exit code" in result


def test_run_command_timeout():
    ex, _ = _make_executor()
    result = asyncio.run(ex.execute("run_command", {
        "command": "sleep 10",
        "timeout": 1,
    }))
    assert "timed out" in result.lower() or "timeout" in result.lower()


def test_run_command_stdout_stderr_captured():
    ex, _ = _make_executor()
    result = asyncio.run(ex.execute("run_command", {
        "command": 'echo "stdout_line" && echo "stderr_line" >&2'
    }))
    assert "stdout_line" in result
    assert "stderr_line" in result


def test_run_command_streaming_callback():
    ex, ws = _make_executor()
    streamed: list[str] = []
    ex._ctx.stream_fn = lambda line: streamed.append(line)

    asyncio.run(ex.execute("run_command", {
        "command": 'printf "line1\\nline2\\nline3\\n"',
    }))

    # Streaming should have captured lines
    combined = "\n".join(streamed)
    assert "line1" in combined
    assert "line2" in combined


def test_run_command_dry_run():
    ex, ws = _make_executor()
    ex._ctx.dry_run = True
    result = asyncio.run(ex.execute("run_command", {"command": "rm -rf /important"}))
    assert "DRY RUN" in result
    # Should not have executed


def test_find_files_basic():
    ex, ws = _make_executor()
    (ws / "foo.py").write_text("")
    (ws / "bar.txt").write_text("")
    result = asyncio.run(ex.execute("find_files", {"pattern": "*.py"}))
    assert "foo.py" in result
    assert "bar.txt" not in result


def test_find_files_no_match():
    ex, ws = _make_executor()
    result = asyncio.run(ex.execute("find_files", {"pattern": "*.nonexistent"}))
    assert "No files found" in result


def test_active_proc_cleared_after_command():
    ex, _ = _make_executor()
    asyncio.run(ex.execute("run_command", {"command": "echo done"}))
    assert ex._ctx.active_proc is None


def test_kill_active_proc_no_proc():
    ex, _ = _make_executor()
    result = ex.kill_active_proc()
    assert result is False
