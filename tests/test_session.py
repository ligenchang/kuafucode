"""Tests for session persistence (SQLite store)."""

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

from nvagent.core.session import SessionStore, read_memory


def _make_store() -> tuple[SessionStore, Path]:
    tmp = Path(tempfile.mkdtemp())
    store = SessionStore(tmp / "sessions.db")
    return store, tmp


def test_create_session():
    store, _ = _make_store()
    session = store.create_session("/workspace/test")
    assert session.id is not None
    assert session.workspace == "/workspace/test"
    assert session.messages == []
    assert session.summary == ""


def test_save_and_get_session():
    store, _ = _make_store()
    session = store.create_session("/ws")
    session.messages.append({"role": "user", "content": "hello world"})
    store.save_session(session)

    last = store.get_last_session("/ws")
    assert last is not None
    assert last.id == session.id
    assert len(last.messages) == 1
    assert last.messages[0]["content"] == "hello world"


def test_summary_auto_populated():
    store, _ = _make_store()
    session = store.create_session("/ws")
    session.messages.append({"role": "user", "content": "add type hints to everything"})
    store.save_session(session)

    last = store.get_last_session("/ws")
    assert "type hints" in last.summary


def test_summary_truncated_at_80():
    store, _ = _make_store()
    session = store.create_session("/ws")
    long_msg = "x" * 200
    session.messages.append({"role": "user", "content": long_msg})
    store.save_session(session)

    last = store.get_last_session("/ws")
    assert len(last.summary) <= 80


def test_list_sessions():
    store, _ = _make_store()
    s1 = store.create_session("/ws")
    s1.messages.append({"role": "user", "content": "first task"})
    store.save_session(s1)

    s2 = store.create_session("/ws")
    s2.messages.append({"role": "user", "content": "second task"})
    store.save_session(s2)

    sessions = store.list_sessions("/ws", limit=10)
    assert len(sessions) == 2


def test_list_sessions_workspace_isolation():
    store, _ = _make_store()
    s1 = store.create_session("/ws/project-a")
    store.save_session(s1)
    s2 = store.create_session("/ws/project-b")
    store.save_session(s2)

    a_sessions = store.list_sessions("/ws/project-a")
    b_sessions = store.list_sessions("/ws/project-b")
    assert len(a_sessions) == 1
    assert len(b_sessions) == 1


def test_get_last_session_returns_none_for_empty():
    store, _ = _make_store()
    result = store.get_last_session("/nonexistent")
    assert result is None


def test_read_memory_missing_file():
    tmp = Path(tempfile.mkdtemp())
    result = read_memory(tmp)
    assert result == ""


def test_read_memory_reads_file():
    tmp = Path(tempfile.mkdtemp())
    mem_dir = tmp / ".nvagent"
    mem_dir.mkdir()
    (mem_dir / "memory.md").write_text("# Notes\n\nDon't touch the legacy code.")
    result = read_memory(tmp)
    assert "legacy code" in result


def test_multiple_saves_update_timestamp():
    import time
    store, _ = _make_store()
    session = store.create_session("/ws")
    session.messages.append({"role": "user", "content": "task"})
    store.save_session(session)
    t1 = session.updated_at

    time.sleep(0.01)
    session.messages.append({"role": "assistant", "content": "done"})
    store.save_session(session)
    t2 = session.updated_at

    assert t2 >= t1
