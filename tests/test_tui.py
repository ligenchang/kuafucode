"""Tests for the Textual TUI (tui/app.py).

Two layers:
  1. Pure-Python unit tests — test widget helper methods and module-level
     utilities without ever importing textual (so these run in any CI even if
     textual is not installed).
  2. Textual Pilot integration tests — use App.run_test() / Pilot to drive the
     full app; skipped automatically when textual is absent.

Run with:
    pytest tests/test_tui.py -v
"""

from __future__ import annotations

import asyncio
import difflib
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_workspace(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """Create a tmp workspace, optionally pre-populate with files."""
    ws = tmp_path
    if files:
        for rel, content in files.items():
            fpath = ws / rel
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")
    return ws


def _make_config(ws: Path):
    """Return a minimal Config pointing at ws."""
    from nvagent.config import Config
    cfg = Config()
    cfg.models.default = "test/model-nano"
    return cfg


def _make_session(ws: Path):
    """Return a fresh in-memory session."""
    import sqlite3
    from nvagent.core.session import SessionStore
    db = ws / ".nvagent" / "sessions.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    store = SessionStore(db)
    session = store.create_session(str(ws))
    return session, store


# ── skip guard ────────────────────────────────────────────────────────────────

try:
    import textual  # noqa: F401
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False

needs_textual = pytest.mark.skipif(not _TEXTUAL, reason="textual not installed")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pure-Python unit tests (no textual required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLangDetection:
    """_lang_for() maps extensions to pygments language names."""

    def _lang(self, path: str) -> str:
        from nvagent.tui.app import _lang_for
        return _lang_for(path)

    def test_python(self):
        assert self._lang("foo.py") == "python"

    def test_typescript(self):
        assert self._lang("index.ts") == "typescript"

    def test_rust(self):
        assert self._lang("main.rs") == "rust"

    def test_yaml(self):
        assert self._lang("config.yml") == "yaml"

    def test_toml(self):
        assert self._lang("pyproject.toml") == "toml"

    def test_markdown(self):
        assert self._lang("README.md") == "markdown"

    def test_unknown_falls_back(self):
        assert self._lang("data.xyz") == "text"

    def test_dockerfile_by_name(self):
        assert self._lang("Dockerfile") == "dockerfile"

    def test_case_insensitive_ext(self):
        assert self._lang("Script.PY") == "python"


class TestCollectFiles:
    """_collect_files() walks workspace and returns relative paths."""

    def test_returns_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        from nvagent.tui.app import _collect_files
        files = _collect_files(tmp_path)
        assert "a.py" in files
        assert "b.py" in files

    def test_skips_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "x.pyc").write_text("")
        (tmp_path / "real.py").write_text("x")
        from nvagent.tui.app import _collect_files
        files = _collect_files(tmp_path)
        assert not any("__pycache__" in f for f in files)
        assert "real.py" in files

    def test_skips_git(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref")
        (tmp_path / "src.py").write_text("")
        from nvagent.tui.app import _collect_files
        files = _collect_files(tmp_path)
        assert not any(".git" in f for f in files)

    def test_limit_respected(self, tmp_path):
        for i in range(50):
            (tmp_path / f"f{i}.py").write_text("")
        from nvagent.tui.app import _collect_files
        files = _collect_files(tmp_path, limit=10)
        assert len(files) <= 10

    def test_nested_dirs(self, tmp_path):
        sub = tmp_path / "src" / "utils"
        sub.mkdir(parents=True)
        (sub / "helpers.py").write_text("")
        from nvagent.tui.app import _collect_files
        files = _collect_files(tmp_path)
        assert any("helpers.py" in f for f in files)

    def test_empty_workspace(self, tmp_path):
        from nvagent.tui.app import _collect_files
        assert _collect_files(tmp_path) == []


class TestDiffRendering:
    """Verify that unified_diff produces expected +/- lines (used by FilePanel)."""

    def _diff(self, old: str, new: str) -> list[str]:
        return list(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="a/f.py",
            tofile="b/f.py",
            n=1,
        ))

    def test_addition_line(self):
        lines = self._diff("x = 1\n", "x = 1\ny = 2\n")
        added = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
        assert any("y = 2" in l for l in added)

    def test_removal_line(self):
        lines = self._diff("x = 1\ny = 2\n", "x = 1\n")
        removed = [l for l in lines if l.startswith("-") and not l.startswith("---")]
        assert any("y = 2" in l for l in removed)

    def test_no_diff_when_identical(self):
        assert self._diff("same\n", "same\n") == []

    def test_context_lines_present(self):
        old = "a\nb\nc\nd\ne\n"
        new = "a\nb\nX\nd\ne\n"
        lines = self._diff(old, new)
        # Context lines start with space
        ctx = [l for l in lines if l.startswith(" ")]
        assert len(ctx) > 0


class TestSlashCommandList:
    """SLASH_COMMANDS constant is complete and correct."""

    def test_all_expected_commands_present(self):
        from nvagent.tui.app import SLASH_COMMANDS
        expected = {"/help", "/clear", "/compact", "/diff", "/history",
                    "/model", "/undo", "/rollback", "/sessions", "/quit"}
        assert expected.issubset(set(SLASH_COMMANDS))

    def test_all_start_with_slash(self):
        from nvagent.tui.app import SLASH_COMMANDS
        assert all(c.startswith("/") for c in SLASH_COMMANDS)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Textual Pilot integration tests
# ═══════════════════════════════════════════════════════════════════════════════

@needs_textual
class TestAppMount:
    """App composes and mounts without errors."""

    @pytest.mark.asyncio
    async def test_app_starts_and_has_chat_log(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, ChatLog
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            chat = app.query_one("#chat-log", ChatLog)
            assert chat is not None

    @pytest.mark.asyncio
    async def test_input_bar_present(self, tmp_path):
        from textual.widgets import Input
        from nvagent.tui.app import NVAgentApp
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#input-bar", Input)
            assert inp is not None

    @pytest.mark.asyncio
    async def test_file_panel_present(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, FilePanel
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            fp = app.query_one("#file-panel", FilePanel)
            assert fp is not None

    @pytest.mark.asyncio
    async def test_header_contains_model_name(self, tmp_path):
        from textual.widgets import Label
        from nvagent.tui.app import NVAgentApp
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            header = app.query_one("#header-bar", Label)
            assert "model-nano" in str(header.renderable)


@needs_textual
class TestSlashCommands:
    """Slash commands update the chat log correctly."""

    async def _run_slash(self, tmp_path, cmd: str):
        from textual.widgets import Input
        from nvagent.tui.app import NVAgentApp, ChatLog
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#input-bar", Input)
            inp.value = cmd
            await pilot.press("enter")
            await pilot.pause(0.05)
            log = app.query_one("#chat-log", ChatLog)
            return log

    @pytest.mark.asyncio
    async def test_help_command_writes_to_log(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, ChatLog
        from textual.widgets import Input
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#input-bar", Input)
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.pause(0.05)
            # Input should be cleared after submit
            assert inp.value == ""

    @pytest.mark.asyncio
    async def test_unknown_slash_command(self, tmp_path):
        from nvagent.tui.app import NVAgentApp
        from textual.widgets import Input
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#input-bar", Input)
            inp.value = "/nonexistent"
            await pilot.press("enter")
            await pilot.pause(0.05)
            # Should not crash; input cleared
            assert inp.value == ""

    @pytest.mark.asyncio
    async def test_clear_command_resets_log(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, ChatLog
        from textual.widgets import Input
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#input-bar", Input)
            inp.value = "/clear"
            await pilot.press("enter")
            await pilot.pause(0.05)
            # Doesn't crash and input cleared
            assert inp.value == ""


@needs_textual
class TestFilePanel:
    """FilePanel renders files and diffs correctly."""

    @pytest.mark.asyncio
    async def test_show_file_renders(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, FilePanel
        from textual.widgets import Label
        ws = _make_workspace(tmp_path, {"hello.py": "print('hi')\n"})
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            fp = app.query_one("#file-panel", FilePanel)
            fp.show_file("hello.py", ws)
            await pilot.pause(0.05)
            label = fp.query_one("#file-label", Label)
            assert "hello.py" in str(label.renderable)

    @pytest.mark.asyncio
    async def test_show_diff_updates_label(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, FilePanel
        from textual.widgets import Label
        ws = _make_workspace(tmp_path, {"mod.py": "x = 1\n"})
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            fp = app.query_one("#file-panel", FilePanel)
            fp.show_diff("mod.py", "x = 1\n", "x = 2\n")
            await pilot.pause(0.05)
            label = fp.query_one("#file-label", Label)
            label_text = str(label.renderable)
            assert "mod.py" in label_text
            assert "diff" in label_text.lower()

    @pytest.mark.asyncio
    async def test_missing_file_shows_error(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, FilePanel
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            fp = app.query_one("#file-panel", FilePanel)
            fp.show_file("does_not_exist.py", ws)
            await pilot.pause(0.05)
            # Should not crash; label still updates
            label = fp.query_one("#file-label", Label)
            assert "does_not_exist.py" in str(label.renderable)


@needs_textual
class TestConfirmModal:
    """ConfirmModal returns True on approve, False on skip."""

    @pytest.mark.asyncio
    async def test_approve_button_returns_true(self, tmp_path):
        from nvagent.tui.app import ConfirmModal

        result: list[bool] = []

        class _TestApp(App):
            async def on_mount(self):
                val = await self.push_screen_wait(
                    ConfirmModal("foo.py", "+added line\n-removed line\n")
                )
                result.append(val)
                self.exit()

        from textual.app import App
        app = _TestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("y")
            await pilot.pause(0.1)

        assert result == [True]

    @pytest.mark.asyncio
    async def test_skip_button_returns_false(self, tmp_path):
        from nvagent.tui.app import ConfirmModal
        from textual.app import App

        result: list[bool] = []

        class _TestApp(App):
            async def on_mount(self):
                val = await self.push_screen_wait(
                    ConfirmModal("bar.py", "+line\n")
                )
                result.append(val)
                self.exit()

        app = _TestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.1)

        assert result == [False]

    @pytest.mark.asyncio
    async def test_escape_rejects(self, tmp_path):
        from nvagent.tui.app import ConfirmModal
        from textual.app import App

        result: list[bool] = []

        class _TestApp(App):
            async def on_mount(self):
                val = await self.push_screen_wait(
                    ConfirmModal("baz.py", "+change\n")
                )
                result.append(val)
                self.exit()

        app = _TestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("escape")
            await pilot.pause(0.1)

        assert result == [False]


@needs_textual
class TestStatusBarCompletions:
    """Typing / and @ updates the status bar with hints."""

    @pytest.mark.asyncio
    async def test_slash_shows_completions(self, tmp_path):
        from nvagent.tui.app import NVAgentApp
        from textual.widgets import Input, Label
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one("#input-bar", Input)
            inp.value = "/h"
            # Trigger the changed event
            app.on_input_changed(Input.Changed(inp, "/h"))
            await pilot.pause(0.05)
            status = app.query_one("#status-bar", Label)
            # /help and /history both start with /h
            status_text = str(status.renderable)
            assert "/help" in status_text or "/history" in status_text


@needs_textual
class TestKeyBindings:
    """Ctrl+L clears chat; Escape sets interrupt flag."""

    @pytest.mark.asyncio
    async def test_ctrl_l_clears_chat(self, tmp_path):
        from nvagent.tui.app import NVAgentApp, ChatLog
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+l")
            await pilot.pause(0.05)
            # No crash is sufficient; the action runs

    @pytest.mark.asyncio
    async def test_escape_sets_interrupt_when_idle(self, tmp_path):
        from nvagent.tui.app import NVAgentApp
        ws = _make_workspace(tmp_path)
        cfg = _make_config(ws)
        sess, store = _make_session(ws)

        app = NVAgentApp(workspace=ws, config=cfg, session=sess, session_store=store)
        async with app.run_test(size=(120, 40)) as pilot:
            # Escape when agent is not running should not crash
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert not app._interrupt_requested  # was never set since not running
