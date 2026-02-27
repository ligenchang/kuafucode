"""
nvagent test suite.

Run with: python -m pytest tests/ -v
Or: python tests/test_nvagent.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent to path for import
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.workspace = Path(self.tmpdir)

    def test_init_workspace_creates_files(self):
        from nvagent.config import init_workspace
        config_dir = init_workspace(self.workspace)
        self.assertTrue((config_dir / "config.toml").exists())
        self.assertTrue((config_dir / "memory.md").exists())

    def test_load_default_config(self):
        from nvagent.config import init_workspace, load_config
        init_workspace(self.workspace)
        config = load_config(self.workspace)
        self.assertEqual(config.api.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertIsInstance(config.agent.max_tokens, int)

    def test_save_and_load_config(self):
        from nvagent.config import init_workspace, load_config, save_config
        init_workspace(self.workspace)
        config = load_config(self.workspace)
        config.api.api_key = "test-key-123"
        config.models.default = "test-model"
        save_config(config, self.workspace)

        loaded = load_config(self.workspace)
        self.assertEqual(loaded.api.api_key, "test-key-123")
        self.assertEqual(loaded.models.default, "test-model")

    def test_api_key_from_env(self):
        from nvagent.config import init_workspace, load_config
        init_workspace(self.workspace)
        config = load_config(self.workspace)
        config.api.api_key = ""
        os.environ["NVIDIA_API_KEY"] = "env-key-456"
        self.assertEqual(config.api_key, "env-key-456")
        del os.environ["NVIDIA_API_KEY"]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Context tests
# ─────────────────────────────────────────────────────────────────────────────

class TestContext(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.workspace = Path(self.tmpdir)

    def _create_files(self):
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "main.py").write_text("def hello():\n    return 'world'\n")
        (self.workspace / "src" / "utils.py").write_text("import os\n")
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_main.py").write_text("def test_hello(): pass\n")
        (self.workspace / "README.md").write_text("# Test Project\n\nA test project.\n")
        (self.workspace / "pyproject.toml").write_text('[project]\nname = "test"\n')

    def test_build_file_tree(self):
        from nvagent.core.context import build_file_tree
        self._create_files()
        tree = build_file_tree(self.workspace, [".git", "__pycache__"])
        self.assertIn("src", tree)
        self.assertIn("main.py", tree)
        self.assertIn("README.md", tree)

    def test_detect_project_type_python(self):
        from nvagent.core.context import detect_project_type
        (self.workspace / "pyproject.toml").write_text("")
        result = detect_project_type(self.workspace)
        self.assertIn("Python", result)

    def test_detect_project_type_node(self):
        from nvagent.core.context import detect_project_type
        (self.workspace / "package.json").write_text("{}")
        result = detect_project_type(self.workspace)
        self.assertIn("Node", result)

    def test_build_system_prompt_contains_workspace(self):
        from nvagent.config import init_workspace, load_config
        from nvagent.core.context import build_system_prompt
        self._create_files()
        init_workspace(self.workspace)
        config = load_config(self.workspace)
        prompt = build_system_prompt(self.workspace, config)
        self.assertIn(str(self.workspace), prompt)
        self.assertIn("README.md", prompt)

    def test_ignore_patterns(self):
        from nvagent.core.context import build_file_tree, is_ignored
        # Create ignored dir
        (self.workspace / "__pycache__").mkdir()
        (self.workspace / "__pycache__" / "main.cpython-312.pyc").write_text("")
        (self.workspace / "main.py").write_text("")
        tree = build_file_tree(self.workspace, ["__pycache__", "*.pyc"])
        self.assertNotIn("__pycache__", tree)
        self.assertIn("main.py", tree)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSession(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "sessions.db"

    def test_create_and_load_session(self):
        from nvagent.core.session import SessionStore
        store = SessionStore(self.db_path)
        session = store.create_session("/test/workspace")
        self.assertIsNotNone(session.id)

        loaded = store.load_session(session.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.workspace, "/test/workspace")

    def test_save_messages(self):
        from nvagent.core.session import SessionStore
        store = SessionStore(self.db_path)
        session = store.create_session("/test/workspace")
        session.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        store.save_session(session)

        loaded = store.load_session(session.id)
        self.assertEqual(len(loaded.messages), 2)
        self.assertEqual(loaded.messages[0]["content"], "hello")

    def test_list_sessions(self):
        from nvagent.core.session import SessionStore
        store = SessionStore(self.db_path)
        s1 = store.create_session("/workspace/a")
        s2 = store.create_session("/workspace/a")
        s3 = store.create_session("/workspace/b")

        sessions = store.list_sessions("/workspace/a")
        self.assertEqual(len(sessions), 2)

    def test_get_last_session(self):
        from nvagent.core.session import SessionStore
        store = SessionStore(self.db_path)
        s1 = store.create_session("/test/ws")
        s2 = store.create_session("/test/ws")

        last = store.get_last_session("/test/ws")
        self.assertEqual(last.id, s2.id)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestToolExecutor(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.workspace = Path(self.tmpdir)
        from nvagent.tools import ToolExecutor
        self.executor = ToolExecutor(self.workspace)

    async def test_read_file(self):
        test_file = self.workspace / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")
        result = await self.executor.execute("read_file", {"path": "test.py"})
        self.assertIn("hello", result)
        self.assertIn("world", result)

    async def test_read_file_with_line_range(self):
        test_file = self.workspace / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = await self.executor.execute("read_file", {
            "path": "test.py", "start_line": 2, "end_line": 3
        })
        self.assertIn("line2", result)
        self.assertIn("line3", result)
        self.assertNotIn("line1", result)
        self.assertNotIn("line5", result)

    async def test_write_file_creates_file(self):
        result = await self.executor.execute("write_file", {
            "path": "new_file.py",
            "content": "# new file\nprint('hello')\n"
        })
        self.assertIn("✓", result)
        self.assertTrue((self.workspace / "new_file.py").exists())

    async def test_write_file_creates_dirs(self):
        result = await self.executor.execute("write_file", {
            "path": "src/utils/helpers.py",
            "content": "def helper(): pass\n"
        })
        self.assertTrue((self.workspace / "src" / "utils" / "helpers.py").exists())

    async def test_write_file_shows_diff(self):
        test_file = self.workspace / "test.py"
        test_file.write_text("old content\n")
        result = await self.executor.execute("write_file", {
            "path": "test.py",
            "content": "new content\n"
        })
        self.assertIn("diff", result.lower())

    async def test_list_dir(self):
        (self.workspace / "file1.py").write_text("")
        (self.workspace / "file2.txt").write_text("")
        (self.workspace / "subdir").mkdir()
        result = await self.executor.execute("list_dir", {"path": "."})
        self.assertIn("file1.py", result)
        self.assertIn("subdir", result)

    async def test_run_command(self):
        result = await self.executor.execute("run_command", {"command": "echo hello_test"})
        self.assertIn("hello_test", result)
        self.assertIn("Exit code: 0", result)

    async def test_run_command_captures_stderr(self):
        result = await self.executor.execute("run_command", {
            "command": "python3 -c \"import sys; sys.stderr.write('err_output\\n')\""
        })
        self.assertIn("err_output", result)

    async def test_run_command_timeout(self):
        result = await self.executor.execute("run_command", {
            "command": "sleep 100",
            "timeout": 1,
        })
        self.assertIn("timed out", result.lower())

    async def test_search_code_finds_match(self):
        (self.workspace / "main.py").write_text("def authenticate(user):\n    pass\n")
        (self.workspace / "utils.py").write_text("# authentication utilities\n")
        result = await self.executor.execute("search_code", {"query": "authenticate"})
        self.assertIn("authenticate", result)

    async def test_search_code_no_match(self):
        (self.workspace / "main.py").write_text("def hello(): pass\n")
        result = await self.executor.execute("search_code", {"query": "xyz_not_found_123"})
        self.assertIn("No matches", result)

    async def test_git_status_no_repo(self):
        result = await self.executor.execute("git_status", {})
        # Should not crash, just report
        self.assertIsInstance(result, str)

    async def test_update_memory_append(self):
        from nvagent.config import init_workspace
        init_workspace(self.workspace)
        result = await self.executor.execute("update_memory", {
            "content": "## New Note\nThis is important.",
        })
        self.assertIn("✓", result)
        memory_file = self.workspace / ".nvagent" / "memory.md"
        self.assertTrue(memory_file.exists())
        content = memory_file.read_text()
        self.assertIn("This is important.", content)

    async def test_delete_file(self):
        test_file = self.workspace / "delete_me.txt"
        test_file.write_text("bye")
        result = await self.executor.execute("delete_file", {"path": "delete_me.txt"})
        self.assertIn("✓", result)
        self.assertFalse(test_file.exists())

    async def test_read_nonexistent_file(self):
        result = await self.executor.execute("read_file", {"path": "does_not_exist.py"})
        self.assertIn("Error", result)

    async def test_unknown_tool(self):
        result = await self.executor.execute("unknown_tool", {})
        self.assertIn("Unknown tool", result)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Client / model routing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRouting(unittest.TestCase):
    def test_classify_fast_task(self):
        from nvagent.core.client import classify_task, TaskType
        self.assertEqual(classify_task("explain what this function does"), TaskType.FAST)
        self.assertEqual(classify_task("what is the difference between X and Y"), TaskType.FAST)
        self.assertEqual(classify_task("summarize the codebase"), TaskType.FAST)

    def test_classify_code_task(self):
        from nvagent.core.client import classify_task, TaskType
        self.assertEqual(classify_task("implement a full authentication system"), TaskType.CODE)
        self.assertEqual(classify_task("build a REST API from scratch"), TaskType.CODE)

    def test_classify_default_task(self):
        from nvagent.core.client import classify_task, TaskType
        self.assertEqual(classify_task("add type hints to auth.py"), TaskType.DEFAULT)
        self.assertEqual(classify_task("fix the bug in the login function"), TaskType.DEFAULT)

    def test_nim_client_no_key_raises(self):
        from nvagent.config import Config
        from nvagent.core.client import NIMClient
        config = Config()
        config.api.api_key = ""
        with self.assertRaises(ValueError):
            NIMClient(config)

    def test_get_model_routing(self):
        from nvagent.config import Config
        from nvagent.core.client import NIMClient, TaskType
        config = Config()
        config.api.api_key = "test-key"
        client = NIMClient(config)
        self.assertEqual(client.get_model(TaskType.FAST), config.models.fast)
        self.assertEqual(client.get_model(TaskType.CODE), config.models.code)
        self.assertEqual(client.get_model(TaskType.DEFAULT), config.models.default)


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop tests (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.workspace = Path(self.tmpdir)
        from nvagent.config import init_workspace, load_config
        init_workspace(self.workspace)
        self.config = load_config(self.workspace)
        self.config.api.api_key = "test-key"
        from nvagent.core.session import SessionStore
        self.store = SessionStore(self.workspace / ".nvagent" / "sessions.db")
        self.session = self.store.create_session(str(self.workspace))

    def _make_agent(self):
        from nvagent.core.loop import Agent
        return Agent(
            config=self.config,
            workspace=self.workspace,
            session=self.session,
            session_store=self.store,
        )

    async def test_agent_collects_events(self):
        """Agent should emit events including token and done."""
        from nvagent.core.client import StreamEvent
        from nvagent.core.loop import Agent

        call_count = [0]

        async def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is from the planner — return an empty done
                yield StreamEvent(type="done", data={})
            else:
                yield StreamEvent(type="token", data="Hello ")
                yield StreamEvent(type="token", data="world!")
                yield StreamEvent(type="done", data={})

        agent = self._make_agent()
        with patch.object(agent.client, "stream_chat", side_effect=mock_stream):
            events = []
            async for event in agent.run("say hello"):
                events.append(event)

        types = [e.type for e in events]
        self.assertIn("token", types)
        self.assertIn("done", types)

    async def test_agent_handles_tool_call(self):
        """Agent should execute tool calls and continue."""
        from nvagent.core.client import StreamEvent
        from nvagent.core.loop import Agent

        call_count = [0]

        async def mock_stream_with_tool(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is from the planner — return an empty done
                yield StreamEvent(type="done", data={})
            elif call_count[0] == 2:
                # Second call: emit tool call
                yield StreamEvent(type="tool_calls", data=[{
                    "id": "call_123",
                    "name": "list_dir",
                    "args": {"path": "."},
                    "args_raw": '{"path": "."}',
                }])
            else:
                # Third call: normal response
                yield StreamEvent(type="token", data="I listed the directory.")
                yield StreamEvent(type="done", data={})

        agent = self._make_agent()
        with patch.object(agent.client, "stream_chat", side_effect=mock_stream_with_tool):
            events = []
            async for event in agent.run("list directory"):
                events.append(event)

        types = [e.type for e in events]
        self.assertIn("tool_start", types)
        self.assertIn("tool_result", types)
        self.assertIn("done", types)

    async def test_agent_cancellation(self):
        """Agent should stop cleanly when cancelled."""
        from nvagent.core.client import StreamEvent
        from nvagent.core.loop import Agent
        import asyncio

        async def slow_stream(*args, **kwargs):
            for i in range(100):
                await asyncio.sleep(0.01)
                yield StreamEvent(type="token", data=f"token{i}")

        agent = self._make_agent()

        async def cancel_after_start():
            events = []
            async for event in agent.run("long task"):
                events.append(event)
                if len(events) >= 3:
                    agent.cancel()
            return events

        with patch.object(agent.client, "stream_chat", return_value=slow_stream()):
            events = await cancel_after_start()

        # Should have gotten some events then error/cancel
        self.assertTrue(len(events) > 0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Tool schemas
# ─────────────────────────────────────────────────────────────────────────────

class TestToolSchemas(unittest.TestCase):
    def test_all_tools_have_required_fields(self):
        from nvagent.tools import TOOL_SCHEMAS
        for tool in TOOL_SCHEMAS:
            self.assertEqual(tool["type"], "function")
            fn = tool["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertGreater(len(fn["description"]), 10)

    def test_tool_count(self):
        from nvagent.tools import TOOL_SCHEMAS
        self.assertGreaterEqual(len(TOOL_SCHEMAS), 8)

    def test_tool_names_are_unique(self):
        from nvagent.tools import TOOL_SCHEMAS
        names = [t["function"]["name"] for t in TOOL_SCHEMAS]
        self.assertEqual(len(names), len(set(names)))


# ─────────────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n⬛ nvagent test suite\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestConfig,
        TestContext,
        TestSession,
        TestToolExecutor,
        TestModelRouting,
        TestAgentLoop,
        TestToolSchemas,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print(f"\n✓ All {result.testsRun} tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)
