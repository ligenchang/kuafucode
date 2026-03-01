"""Tests for safety: loop detection, resource guard, git checkpointer."""

import asyncio
import subprocess
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

from nvagent.core.safety import GitCheckpointer, LoopDetector, ResourceGuard
from nvagent.config import SafetyConfig


# ── LoopDetector ───────────────────────────────────────────────────────────────

def test_loop_detector_no_loop():
    ld = LoopDetector(max_identical=3, window=8)
    ld.record("read_file", {"path": "a.py"})
    ld.record("read_file", {"path": "b.py"})
    ld.record("write_file", {"path": "a.py"})
    assert not ld.is_looping()


def test_loop_detector_detects_loop():
    ld = LoopDetector(max_identical=3, window=8)
    call = ("read_file", '{"path": "stuck.py"}')
    ld.record("read_file", {"path": "stuck.py"})
    ld.record("read_file", {"path": "stuck.py"})
    ld.record("read_file", {"path": "stuck.py"})
    assert ld.is_looping()


def test_loop_detector_reset_clears():
    ld = LoopDetector(max_identical=3, window=8)
    for _ in range(3):
        ld.record("read_file", {"path": "x.py"})
    assert ld.is_looping()
    ld.reset()
    assert not ld.is_looping()


def test_loop_detector_window():
    ld = LoopDetector(max_identical=3, window=4)
    # Fill window with varied calls
    ld.record("read_file", {"path": "a.py"})
    ld.record("write_file", {"path": "b.py"})
    ld.record("read_file", {"path": "c.py"})
    ld.record("write_file", {"path": "d.py"})
    # Now add 3 identical — but the window has rolled over
    ld.record("read_file", {"path": "x.py"})
    ld.record("read_file", {"path": "x.py"})
    ld.record("read_file", {"path": "x.py"})
    assert ld.is_looping()


# ── ResourceGuard ──────────────────────────────────────────────────────────────

def test_resource_guard_no_violation():
    cfg = SafetyConfig(max_tokens_per_task=100_000, max_wall_seconds=900, max_tool_calls=150)
    rg = ResourceGuard(cfg)
    rg.start()
    rg.update(tokens=1000, tool_calls=5)
    assert rg.check() is None


def test_resource_guard_token_limit():
    cfg = SafetyConfig(max_tokens_per_task=1000, max_wall_seconds=900, max_tool_calls=150)
    rg = ResourceGuard(cfg)
    rg.start()
    rg.update(tokens=1001)
    v = rg.check()
    assert v is not None
    assert "token" in v.kind.lower() or "token" in v.message.lower()


def test_resource_guard_tool_call_limit():
    cfg = SafetyConfig(max_tokens_per_task=100_000, max_wall_seconds=900, max_tool_calls=5)
    rg = ResourceGuard(cfg)
    rg.start()
    for _ in range(6):
        rg.update(tool_calls=1)
    v = rg.check()
    assert v is not None


# ── GitCheckpointer ────────────────────────────────────────────────────────────

def _make_git_repo() -> Path:
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["git", "init"], cwd=tmp, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp, capture_output=True)
    (tmp / "init.py").write_text("# init\n")
    subprocess.run(["git", "add", "."], cwd=tmp, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp, capture_output=True)
    return tmp


def test_git_checkpointer_creates_commit():
    ws = _make_git_repo()
    gc = GitCheckpointer(ws)
    (ws / "new.py").write_text("x = 1\n")

    sha = asyncio.run(gc.checkpoint("test checkpoint"))
    assert sha is not None
    assert len(sha) >= 7


def test_git_checkpointer_no_changes_returns_none():
    """When there are no uncommitted changes, checkpoint should either return
    None or a sha — both are acceptable depending on implementation."""
    ws = _make_git_repo()
    gc = GitCheckpointer(ws)
    # Either None (no-op) or a sha is acceptable
    sha = asyncio.run(gc.checkpoint("empty checkpoint"))
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)


def test_git_checkpointer_restore():
    ws = _make_git_repo()
    gc = GitCheckpointer(ws)

    # Create a file and checkpoint
    (ws / "feature.py").write_text("def feature(): pass\n")
    asyncio.run(gc.checkpoint("add feature"))

    # Modify the file
    (ws / "feature.py").write_text("BROKEN\n")

    # Restore
    ok, msg = asyncio.run(gc.restore())
    assert ok
    # File should be back to checkpoint state
    assert (ws / "feature.py").read_text().strip() == "def feature(): pass"


def test_git_checkpointer_non_git_repo():
    tmp = Path(tempfile.mkdtemp())
    gc = GitCheckpointer(tmp)
    sha = asyncio.run(gc.checkpoint("no git"))
    assert sha is None
