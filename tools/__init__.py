"""
nvagent tool executor — thin dispatch layer.

Public API (unchanged from the original monolithic tools.py):
  ToolExecutor   — executes tool calls dispatched by the agent loop
  TOOL_SCHEMAS   — list of OpenAI tool_use JSON schemas

Internal structure:
  tools/context.py         — shared ToolContext (state + path utilities)
  tools/schemas.py         — TOOL_SCHEMAS definitions
  tools/handlers/
      file.py              — read_file, write_file, write_files, edit_file, delete_file, list_dir
      search.py            — search_code, find_symbol, find_definition, find_references
      git.py               — git_status, git_diff, git_add, git_commit, git_log
      code.py              — get_symbols, get_dep_graph, run_analysis
      memory.py            — update_memory, memory_learn/recall/forget/note
      exec.py              — run_command, run_tests, run_formatter, find_files
      vc.py                — apply_patch, checkpoint, rollback
      notebook.py          — read_notebook, edit_notebook
      url.py               — read_url
      todo.py              — todo_write, todo_read
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Awaitable, Callable, Optional

from nvagent.tools.context import ToolContext
from nvagent.tools.schemas import TOOL_SCHEMAS
from nvagent.tools.handlers.file import FileHandler
from nvagent.tools.handlers.search import SearchHandler
from nvagent.tools.handlers.git import GitHandler
from nvagent.tools.handlers.code import CodeHandler
from nvagent.tools.handlers.memory import MemoryHandler
from nvagent.tools.handlers.exec import ExecHandler
from nvagent.tools.handlers.vc import VcHandler
from nvagent.tools.handlers.notebook import NotebookHandler
from nvagent.tools.handlers.url import UrlHandler
from nvagent.tools.handlers.todo import TodoHandler
from nvagent.core.mcp import McpClient

__all__ = ["ToolExecutor", "TOOL_SCHEMAS"]


class ToolExecutor:
    """Executes tool calls with workspace context.

    Thin dispatch layer: all business logic lives in the handler modules under
    tools/handlers/.  ToolExecutor creates a shared ToolContext, instantiates
    each handler, and routes incoming tool calls to the right method.
    """

    def __init__(
        self,
        workspace: Path,
        max_file_bytes: int = 102400,
        confirm_fn: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        safe_mode: bool = True,
        dry_run: bool = False,
        mcp_client: Optional[McpClient] = None,
        stream_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        # ── Shared context (all handlers share this object) ───────────────────
        self._ctx = ToolContext(
            workspace=workspace,
            max_file_bytes=max_file_bytes,
            confirm_fn=confirm_fn,
            safe_mode=safe_mode,
            dry_run=dry_run,
            stream_fn=stream_fn,
        )

        # ── Optional MCP client ───────────────────────────────────────────────
        self._mcp_client: Optional[McpClient] = mcp_client

        # ── Convenience aliases kept for backward-compat ─────────────────────
        self.workspace = self._ctx.workspace

        # ── Handler instances ─────────────────────────────────────────────────
        self._file = FileHandler(self._ctx)
        self._search = SearchHandler(self._ctx)
        self._git = GitHandler(self._ctx)
        self._code = CodeHandler(self._ctx)
        self._memory = MemoryHandler(self._ctx)
        self._exec = ExecHandler(self._ctx)
        self._vc = VcHandler(self._ctx)
        self._notebook = NotebookHandler(self._ctx)
        self._url = UrlHandler(self._ctx)
        self._todo = TodoHandler(self._ctx)

        # ── Dispatch table ────────────────────────────────────────────────────
        self._dispatch: dict[str, object] = {
            # file
            "read_file": self._file.read_file,
            "write_file": self._file.write_file,
            "write_files": self._file.write_files,
            "edit_file": self._file.edit_file,
            "delete_file": self._file.delete_file,
            "list_dir": self._file.list_dir,
            "str_replace_editor": self._file.str_replace_editor,
            # search
            "search_code": self._search.search_code,
            "find_symbol": self._search.find_symbol,
            "find_definition": self._search.find_definition,
            "find_references": self._search.find_references,
            # git
            "git_status": self._git.git_status,
            "git_diff": self._git.git_diff,
            "git_add": self._git.git_add,
            "git_commit": self._git.git_commit,
            "git_log": self._git.git_log,
            # code intelligence
            "get_symbols": self._code.get_symbols,
            "get_dep_graph": self._code.get_dep_graph,
            "run_analysis": self._code.run_analysis,
            # memory
            "update_memory": self._memory.update_memory,
            "memory_learn": self._memory.memory_learn,
            "memory_recall": self._memory.memory_recall,
            "memory_forget": self._memory.memory_forget,
            "memory_note": self._memory.memory_note,
            # execution
            "run_command": self._exec.run_command,
            "run_tests": self._exec.run_tests,
            "run_formatter": self._exec.run_formatter,
            "find_files": self._exec.find_files,
            # version control / patches
            "apply_patch": self._vc.apply_patch,
            "checkpoint": self._vc.checkpoint,
            "rollback": self._vc.rollback,
            # notebook
            "read_notebook": self._notebook.read_notebook,
            "edit_notebook": self._notebook.edit_notebook,
            # url
            "read_url": self._url.read_url,
            # todo
            "todo_write": self._todo.todo_write,
            "todo_read": self._todo.todo_read,
            # undo (implemented directly — touches undo_stack on ctx)
            "undo_last_turn": self.undo_last_turn,
        }

        # Pre-cache valid parameter names per tool to avoid inspect overhead per call.
        self._dispatch_params: dict[str, frozenset | None] = {}
        for _tname, _tfn in self._dispatch.items():
            try:
                _sig = inspect.signature(_tfn)
                _has_varkw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
                )
                self._dispatch_params[_tname] = None if _has_varkw else frozenset(_sig.parameters)
            except (ValueError, TypeError):
                self._dispatch_params[_tname] = None

        # Load user plugins from .nvagent/tools/*.py
        self._plugin_schemas: list[dict] = []
        self._load_plugins(workspace)

    def _load_plugins(self, workspace: Path) -> None:
        """Load custom tool plugins from .nvagent/tools/*.py.

        Each plugin file should define one or more async functions decorated
        with @nvagent_tool(description="...", parameters={...}).  The decorator
        is injected into the module's namespace automatically so plugins don't
        need to import it.

        Example plugin (.nvagent/tools/my_tool.py):

            @nvagent_tool(
                description="Fetch a URL and return its body",
                parameters={"type": "object", "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                }, "required": ["url"]}
            )
            async def fetch_url(url: str) -> str:
                import urllib.request
                return urllib.request.urlopen(url).read(4096).decode()
        """
        import importlib.util

        plugin_dir = workspace / ".nvagent" / "tools"
        if not plugin_dir.is_dir():
            return

        # Decorator injected into plugin namespace
        _registered: list[tuple[str, object, dict]] = []

        def nvagent_tool(description: str = "", parameters: dict | None = None):
            def decorator(fn):
                _registered.append((fn.__name__, fn, {
                    "type": "function",
                    "function": {
                        "name": fn.__name__,
                        "description": description or fn.__doc__ or "",
                        "parameters": parameters or {"type": "object", "properties": {}, "required": []},
                    }
                }))
                return fn
            return decorator

        for plugin_path in sorted(plugin_dir.glob("*.py")):
            if plugin_path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"nvagent_plugin_{plugin_path.stem}", plugin_path
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                # Inject decorator and workspace into plugin namespace
                module.nvagent_tool = nvagent_tool  # type: ignore[attr-defined]
                module.workspace = workspace  # type: ignore[attr-defined]
                _registered.clear()
                spec.loader.exec_module(module)
                for tool_name, fn, schema in _registered:
                    if tool_name in self._dispatch:
                        import logging
                        logging.getLogger(__name__).warning(
                            "Plugin tool %r overrides built-in — skipping. "
                            "Rename the function to avoid conflicts.", tool_name
                        )
                        continue
                    self._dispatch[tool_name] = fn
                    self._plugin_schemas.append(schema)
                    # Cache params for new tool
                    try:
                        _sig = inspect.signature(fn)
                        _has_varkw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in _sig.parameters.values()
                        )
                        self._dispatch_params[tool_name] = (
                            None if _has_varkw else frozenset(_sig.parameters)
                        )
                    except (ValueError, TypeError):
                        self._dispatch_params[tool_name] = None
                if _registered:
                    import logging
                    logging.getLogger(__name__).info(
                        "Loaded %d tool(s) from plugin %s: %s",
                        len(_registered),
                        plugin_path.name,
                        ", ".join(r[0] for r in _registered),
                    )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to load plugin %s: %s", plugin_path.name, exc
                )

    # ── Context property passthrough (backward-compat) ────────────────────────

    @property
    def changed_files(self) -> list[str]:
        return self._ctx.changed_files

    @property
    def dry_run(self) -> bool:
        return self._ctx.dry_run

    @property
    def sandbox(self):
        return self._ctx.sandbox

    @property
    def undo_stack(self) -> list:
        return self._ctx.undo_stack

    @property
    def confirm_fn(self):
        return self._ctx.confirm_fn

    @confirm_fn.setter
    def confirm_fn(self, value):
        self._ctx.confirm_fn = value

    @property
    def stream_fn(self):
        return self._ctx.stream_fn

    @stream_fn.setter
    def stream_fn(self, value):
        self._ctx.stream_fn = value

    # ── Active schemas ────────────────────────────────────────────────────────

    @property
    def active_schemas(self) -> list[dict]:
        """Built-in tool schemas + plugin schemas + MCP server schemas."""
        schemas = list(TOOL_SCHEMAS)
        if self._plugin_schemas:
            schemas = schemas + self._plugin_schemas
        if self._mcp_client:
            mcp_schemas = self._mcp_client.tool_schemas
            if mcp_schemas:
                schemas = schemas + mcp_schemas
        return schemas

    # ── Turn lifecycle ────────────────────────────────────────────────────────

    def kill_active_proc(self) -> bool:
        """Kill the currently running subprocess, if any. Returns True if something was killed."""
        proc = self._ctx.active_proc
        if proc is not None:
            from nvagent.core.execution import _kill_proc_group
            _kill_proc_group(proc)
            self._ctx.active_proc = None
            return True
        return False

    def begin_turn(self) -> None:
        """Reset per-turn tracking at the start of each agent turn."""
        self._ctx.changed_files = []
        self._ctx._current_turn_backups = {}

    def end_turn(self, label: str = "") -> None:
        """Save undo snapshot at end of a turn that modified files.

        label is shown in /undo output so the user knows what's being rolled back.
        """
        if self._ctx._current_turn_backups:
            entry = {"_label": label, **self._ctx._current_turn_backups.copy()}
            self._ctx.undo_stack.append(entry)
        self._ctx._current_turn_backups = {}

    async def undo_last_turn(self) -> str:
        """Restore files to state before the last agent turn."""
        if not self._ctx.undo_stack:
            return "Nothing to undo — no file changes recorded this session."
        backups = self._ctx.undo_stack.pop()
        label = backups.pop("_label", "")
        restored: list[str] = []
        for abs_path_str, old_content in backups.items():
            fpath = Path(abs_path_str)
            if old_content is None:
                if fpath.exists():
                    fpath.unlink()
                restored.append(f"deleted {fpath.name}")
            else:
                fpath.write_text(old_content, encoding="utf-8")
                restored.append(f"restored {fpath.name}")
        suffix = f' — "{label}"' if label else ""
        return f"↩ Undone{suffix}: {', '.join(restored)}" if restored else "Nothing to restore."

    # ── last_read_mtime (public API used by core/loop.py) ─────────────────────

    def last_read_mtime(self, abs_path: str) -> int | None:
        """Return the mtime_ns recorded when this file was last read, or None."""
        return self._ctx.last_read_mtime(abs_path)

    # ── Main dispatch ─────────────────────────────────────────────────────────

    async def execute(self, name: str, args: dict) -> str:
        """Dispatch a tool call by name to the appropriate handler method."""
        # Route MCP-namespaced tools to the MCP client
        if self._mcp_client is not None and self._mcp_client.is_mcp_tool(name):
            return await self._mcp_client.call_tool(name, args)

        fn = self._dispatch.get(name)
        if not fn:
            return f"Unknown tool: {name}"

        # Strip extra model-generated fields (e.g. 'description') using cached param sets.
        allowed = self._dispatch_params.get(name)
        if allowed is not None:
            args = {k: v for k, v in args.items() if k in allowed}

        try:
            return await fn(**args)
        except TypeError as e:
            return f"Tool argument error for {name}: {e}"
        except Exception as e:
            return f"Tool error [{name}]: {type(e).__name__}: {e}"
