"""nvagent Textual TUI — two-panel layout (chat + file viewer).

Layout:
  ┌──────────────────────────────┬───────────────────────────────┐
  │  HEADER (model / session)    │                               │
  ├──────────────────────────────┤  FILE PANEL                   │
  │                              │  (syntax-highlighted or diff) │
  │  CHAT LOG                    │                               │
  │  (scrollable, rich markdown) │                               │
  │                              │                               │
  ├──────────────────────────────┴───────────────────────────────┤
  │  INPUT BAR  (slash-complete, @-file complete)                │
  └──────────────────────────────────────────────────────────────┘

Falls back gracefully to the ANSI REPL if textual is unavailable.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import subprocess
from pathlib import Path

# ── Guard: fall back if textual not installed ──────────────────────────────────
try:
    from rich.syntax import Syntax
    from rich.text import Text
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.css.query import NoMatches
    from textual.reactive import reactive
    from textual.screen import ModalScreen
    from textual.widget import Widget
    from textual.widgets import (
        Button,
        Input,
        Label,
        RichLog,
        Static,
    )
    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False


# ── Language detection ─────────────────────────────────────────────────────────

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".rs": "rust", ".go": "go",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".java": "java", ".kt": "kotlin", ".rb": "ruby",
    ".sh": "bash", ".zsh": "bash", ".fish": "fish",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".html": "html", ".css": "css",
    ".scss": "scss", ".md": "markdown", ".sql": "sql",
    ".dockerfile": "dockerfile", ".tf": "hcl",
}


def _lang_for(path: str) -> str:
    ext = Path(path).suffix.lower()
    if Path(path).name.lower() in ("dockerfile", "makefile"):
        return Path(path).name.lower()
    return _EXT_TO_LANG.get(ext, "text")


# ── Slash commands + @-file completion helpers ─────────────────────────────────

SLASH_COMMANDS = [
    "/help", "/clear", "/compact", "/diff",
    "/history", "/model", "/undo", "/rollback",
    "/sessions", "/quit",
]


def _collect_files(workspace: Path, limit: int = 2000) -> list[str]:
    """Return relative file paths for @-mention completion."""
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
    result: list[str] = []
    try:
        for root, dirs, files in os.walk(workspace):
            dirs[:] = [d for d in dirs if d not in skip]
            rp = Path(root)
            for f in files:
                try:
                    result.append(str((rp / f).relative_to(workspace)))
                except ValueError:
                    pass
                if len(result) >= limit:
                    return result
    except Exception:
        pass
    return sorted(result)


if _TEXTUAL_AVAILABLE:

    # ── Confirm Modal ──────────────────────────────────────────────────────────

    class ConfirmModal(ModalScreen[bool]):
        """Pause the agent and ask the user to approve or reject a file write.

        Shows the unified diff in a scrollable panel with [Apply] / [Skip] buttons.
        Pressing 'y' / Enter approves; 'n' / Escape rejects.
        """

        BINDINGS = [
            Binding("y", "approve", "Apply", show=True),
            Binding("n", "reject", "Skip", show=True),
            Binding("enter", "approve", "Apply", show=False),
            Binding("escape", "reject", "Skip", show=False),
        ]

        CSS = """
        ConfirmModal {
            align: center middle;
        }
        #confirm-dialog {
            width: 80%;
            max-height: 70%;
            background: #161b22;
            border: tall #30363d;
            padding: 0;
        }
        #confirm-title {
            background: #f0883e;
            color: #0d1117;
            text-style: bold;
            padding: 0 2;
            height: 1;
        }
        #confirm-diff {
            height: 1fr;
            background: #0d1117;
            padding: 0 1;
            border-bottom: tall #30363d;
        }
        #confirm-buttons {
            height: 3;
            align: center middle;
            background: #161b22;
        }
        #btn-apply {
            background: #238636;
            color: #ffffff;
            border: tall #2ea043;
            margin: 0 2;
            min-width: 12;
        }
        #btn-apply:hover {
            background: #2ea043;
        }
        #btn-skip {
            background: #21262d;
            color: #8b949e;
            border: tall #30363d;
            margin: 0 2;
            min-width: 12;
        }
        #btn-skip:hover {
            background: #30363d;
            color: #e6edf3;
        }
        """

        def __init__(self, path: str, diff_str: str) -> None:
            super().__init__()
            self._path = path
            self._diff_str = diff_str

        def compose(self) -> ComposeResult:
            with Vertical(id="confirm-dialog"):
                yield Label(f"  ⚠  Apply changes to {self._path}?", id="confirm-title")
                diff_log = RichLog(id="confirm-diff", highlight=False, markup=False, wrap=False)
                yield diff_log
                with Horizontal(id="confirm-buttons"):
                    yield Button("✓ Apply  [y]", id="btn-apply", variant="success")
                    yield Button("✗ Skip   [n]", id="btn-skip", variant="default")

        def on_mount(self) -> None:
            diff_log = self.query_one("#confirm-diff", RichLog)
            for line in self._diff_str.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    diff_log.write(Text(line, style="bold #8b949e"))
                elif line.startswith("+"):
                    diff_log.write(Text(line, style="#76b900"))
                elif line.startswith("-"):
                    diff_log.write(Text(line, style="#f85149"))
                elif line.startswith("@@"):
                    diff_log.write(Text(line, style="bold #a78bfa"))
                else:
                    diff_log.write(Text(line, style="dim #e6edf3"))
            # Focus the Apply button so Enter works immediately
            self.query_one("#btn-apply").focus()

        @on(Button.Pressed, "#btn-apply")
        def on_apply(self) -> None:
            self.dismiss(True)

        @on(Button.Pressed, "#btn-skip")
        def on_skip(self) -> None:
            self.dismiss(False)

        def action_approve(self) -> None:
            self.dismiss(True)

        def action_reject(self) -> None:
            self.dismiss(False)


    # ── Custom Widgets ─────────────────────────────────────────────────────────

    class ChatLog(RichLog):
        """Scrollable chat log with rich rendering."""

        DEFAULT_CSS = """
        ChatLog {
            width: 1fr;
            height: 1fr;
            background: $surface;
            padding: 0 1;
            border-right: tall $primary-darken-3;
            scrollbar-color: $primary-darken-2;
            scrollbar-background: $surface;
        }
        """

        def add_user(self, text: str) -> None:
            self.write(Text(""))
            self.write(Text(f"  ❯ {text}", style="bold #79c0ff"))
            self.write(Text(""))

        def add_tool_start(self, name: str, args: dict) -> None:
            path = args.get("path", args.get("command", args.get("query", "")))
            if path:
                label = f"{name}({str(path)[:55]!r})"
            elif args:
                k, v = next(iter(args.items()))
                label = f"{name}({k}={str(v)[:40]!r})"
            else:
                label = f"{name}()"
            self.write(Text(f"  ⚙  {label}", style="bold #f0883e"))

        def add_tool_result(self, result: str, max_lines: int = 6) -> None:
            lines = result.strip().splitlines()
            if not lines:
                return
            for line in lines[:max_lines]:
                self.write(Text(f"     {line}", style="dim #8b949e"))
            if len(lines) > max_lines:
                self.write(Text(f"     … {len(lines) - max_lines} more lines", style="dim #8b949e"))

        def add_status(self, msg: str) -> None:
            self.write(Text(f"  ◌  {msg}", style="dim #a78bfa"))

        def add_done(self, turns: int, tokens: int, files: list[str]) -> None:
            self.write(Text(""))
            parts = [f"{turns} turn{'s' if turns != 1 else ''}"]
            if tokens:
                parts.append(f"~{tokens:,} tokens")
            if files:
                parts.append(f"{len(files)} file{'s' if len(files) != 1 else ''} changed")
            self.write(Text(f"  ✓  Done  ({', '.join(parts)})", style="bold #76b900"))
            self.write(Text(""))

        def add_error(self, msg: str) -> None:
            self.write(Text(f"  ✗  {msg}", style="bold red"))

        def add_safety(self, msg: str, fatal: bool = False) -> None:
            icon = "✗" if fatal else "⚠"
            style = "bold red" if fatal else "bold yellow"
            self.write(Text(f"  {icon}  {msg}", style=style))

        def add_files_changed(self, files: list[str]) -> None:
            summary = ", ".join(str(f) for f in files[:4])
            if len(files) > 4:
                summary += f"… +{len(files)-4}"
            self.write(Text(f"  ✎  Modified: {summary}", style="dim #76b900"))

        def add_confirm_result(self, path: str, approved: bool) -> None:
            if approved:
                self.write(Text(f"  ✓  Applied: {path}", style="bold #76b900"))
            else:
                self.write(Text(f"  ✗  Skipped: {path}", style="dim #8b949e"))

        def add_divider(self) -> None:
            self.write(Text("─" * 60, style="dim #30363d"))

        def add_help(self) -> None:
            self.write(Text(""))
            self.write(Text("  Commands:", style="bold #76b900"))
            cmds = [
                ("/help",       "Show this help"),
                ("/clear",      "Clear chat log"),
                ("/compact",    "Summarize & compress history"),
                ("/diff",       "Show uncommitted git diff"),
                ("/history",    "Session activity log"),
                ("/model <n>",  "Switch model  e.g. /model 1"),
                ("/undo",       "Undo last file changes"),
                ("/rollback",   "Git rollback to checkpoint"),
                ("/sessions",   "List recent sessions"),
                ("/quit",       "Exit nvagent"),
            ]
            for cmd, desc in cmds:
                self.write(Text(f"  {cmd:<16}  {desc}", style="#8b949e"))
            self.write(Text(""))
            self.write(Text("  Keybindings:", style="bold #76b900"))
            bindings = [
                ("Ctrl+D",  "Toggle git diff in file panel"),
                ("Ctrl+L",  "Clear chat"),
                ("Ctrl+C",  "Quit"),
                ("Esc",     "Interrupt running agent"),
                ("Tab",     "Cycle focus between chat / input"),
            ]
            for key, desc in bindings:
                self.write(Text(f"  {key:<12}  {desc}", style="#8b949e"))
            self.write(Text(""))


    class FilePanel(Widget):
        """Right panel: shows syntax-highlighted file or diff."""

        DEFAULT_CSS = """
        FilePanel {
            width: 1fr;
            height: 1fr;
            background: $surface-darken-1;
            padding: 0;
        }
        FilePanel > RichLog {
            width: 1fr;
            height: 1fr;
            background: $surface-darken-1;
            padding: 0 1;
            scrollbar-color: $primary-darken-2;
            scrollbar-background: $surface-darken-1;
        }
        FilePanel > #file-label {
            background: $primary-darken-3;
            color: $text-muted;
            padding: 0 2;
            height: 1;
            text-style: bold;
        }
        """

        _current_path: reactive[str] = reactive("")

        def compose(self) -> ComposeResult:
            yield Label("  No file selected", id="file-label")
            yield RichLog(id="file-log", highlight=False, markup=False, wrap=False)

        def show_file(self, path: str, workspace: Path) -> None:
            """Load and syntax-highlight a file."""
            self._current_path = path
            log = self.query_one("#file-log", RichLog)
            label = self.query_one("#file-label", Label)
            log.clear()
            full = workspace / path
            label.update(f"  📄 {path}")
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                lang = _lang_for(path)
                syntax = Syntax(
                    content, lang,
                    theme="github-dark",
                    line_numbers=True,
                    word_wrap=False,
                )
                log.write(syntax)
            except FileNotFoundError:
                log.write(Text(f"  File not found: {path}", style="red"))
            except Exception as e:
                log.write(Text(f"  Error reading file: {e}", style="red"))

        def show_diff(self, path: str, old_content: str, new_content: str) -> None:
            """Show unified diff between old and new content."""
            self._current_path = path
            log = self.query_one("#file-log", RichLog)
            label = self.query_one("#file-label", Label)
            log.clear()
            label.update(f"  ⟳ {path}  [diff]")

            diff = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            ))
            if not diff:
                log.write(Text("  No changes.", style="dim"))
                return
            for line in diff:
                line = line.rstrip("\n")
                if line.startswith("+++") or line.startswith("---"):
                    log.write(Text(line, style="bold #8b949e"))
                elif line.startswith("+"):
                    log.write(Text(line, style="#76b900"))
                elif line.startswith("-"):
                    log.write(Text(line, style="#f85149"))
                elif line.startswith("@@"):
                    log.write(Text(line, style="bold #a78bfa"))
                else:
                    log.write(Text(line, style="dim #e6edf3"))

        def show_git_diff(self, workspace: Path) -> None:
            """Run git diff HEAD and display it."""
            log = self.query_one("#file-log", RichLog)
            label = self.query_one("#file-label", Label)
            log.clear()
            label.update("  ⟳ git diff HEAD")
            try:
                result = subprocess.run(
                    ["git", "diff", "HEAD"],
                    capture_output=True, text=True,
                    cwd=workspace, timeout=10,
                )
                diff_text = result.stdout.strip()
                if not diff_text:
                    log.write(Text("  No uncommitted changes.", style="dim"))
                    return
                for line in diff_text.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        log.write(Text(line, style="#76b900"))
                    elif line.startswith("-") and not line.startswith("---"):
                        log.write(Text(line, style="#f85149"))
                    elif line.startswith("@@"):
                        log.write(Text(line, style="bold #a78bfa"))
                    elif line.startswith("+++") or line.startswith("---"):
                        log.write(Text(line, style="bold #8b949e"))
                    else:
                        log.write(Text(line, style="dim #e6edf3"))
            except Exception as e:
                log.write(Text(f"  git diff failed: {e}", style="red"))

        def clear_panel(self) -> None:
            self.query_one("#file-log", RichLog).clear()
            self.query_one("#file-label", Label).update("  No file selected")
            self._current_path = ""


    # ── Main App ───────────────────────────────────────────────────────────────

    class NVAgentApp(App):
        """nvagent two-panel Textual TUI."""

        TITLE = "nvagent"
        SUB_TITLE = "NVIDIA NIM coding agent"

        CSS = """
        Screen {
            background: #0d1117;
        }
        #header-bar {
            dock: top;
            height: 1;
            background: #161b22;
            padding: 0 2;
            color: #8b949e;
        }
        #main-split {
            width: 1fr;
            height: 1fr;
            layout: horizontal;
        }
        #left-pane {
            width: 55%;
            height: 1fr;
            layout: vertical;
        }
        #right-pane {
            width: 45%;
            height: 1fr;
            border-left: tall #30363d;
        }
        InputBar {
            height: 3;
            dock: bottom;
            background: #161b22;
            border-top: tall #30363d;
            border-bottom: none;
            border-left: none;
            border-right: none;
            color: #e6edf3;
            padding: 0 2;
        }
        InputBar:focus {
            border-top: tall #76b900;
            background: #0d1117;
        }
        #status-bar {
            dock: bottom;
            height: 1;
            background: #010409;
            color: #8b949e;
            padding: 0 2;
        }
        ChatLog {
            border: none;
            padding: 0 1;
        }
        FilePanel {
            border: none;
        }
        """

        BINDINGS = [
            Binding("ctrl+c", "quit", "Quit", priority=True),
            Binding("ctrl+l", "clear_chat", "Clear"),
            Binding("ctrl+d", "toggle_diff", "Git Diff"),
            Binding("escape", "interrupt_agent", "Interrupt", show=False),
            Binding("tab", "cycle_focus", "Focus", show=False),
        ]

        def __init__(
            self,
            workspace: Path,
            config,
            session,
            session_store,
            no_confirm: bool = False,
        ) -> None:
            super().__init__()
            self.workspace = workspace
            self.config = config
            self.session = session
            self.session_store = session_store
            self.no_confirm = no_confirm

            self._agent = None
            self._agent_running = False
            self._session_tokens = 0
            self._history_log: list[dict] = []
            self._file_cache: dict[str, str] = {}
            self._workspace_files: list[str] = []
            self._interrupt_requested = False

        # ── Compose ────────────────────────────────────────────────────────────

        def compose(self) -> ComposeResult:
            model_short = self.config.models.default.split("/")[-1]
            branch = self._get_git_branch()
            branch_str = f"  {branch}" if branch else ""
            header_text = (
                f"⬛ nvagent  │  {model_short}  │  "
                f"session #{self.session.id}  │  {self.workspace.name}{branch_str}"
            )
            yield Label(header_text, id="header-bar")

            with Horizontal(id="main-split"):
                with Vertical(id="left-pane"):
                    yield ChatLog(id="chat-log", highlight=False, markup=False, wrap=True)
                with Vertical(id="right-pane"):
                    yield FilePanel(id="file-panel")

            yield Input(
                placeholder="  Message nvagent…  (/ for commands, @ for files)",
                id="input-bar",
            )
            yield Label("  Ready", id="status-bar")

        def on_mount(self) -> None:
            self.query_one("#input-bar").focus()
            log = self.query_one("#chat-log", ChatLog)
            log.add_divider()
            n = sum(1 for m in self.session.messages if m.get("role") == "user")
            if n > 0:
                log.add_status(f"Resumed session #{self.session.id}  ·  {n} exchange{'s' if n != 1 else ''}")
            else:
                log.add_status("New session. Type a message or /help for commands.")
            log.add_divider()
            self._workspace_files = _collect_files(self.workspace)

        # ── Input handling ─────────────────────────────────────────────────────

        @on(Input.Submitted, "#input-bar")
        async def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if not text:
                return
            self.query_one("#input-bar").value = ""

            if text.startswith("/"):
                await self._handle_slash(text)
                return

            text, mentioned = self._expand_at_mentions(text)
            if mentioned:
                self.query_one("#chat-log", ChatLog).add_status(
                    f"Included: {', '.join(mentioned)}"
                )

            self._stream_message(text)

        def on_input_changed(self, event: Input.Changed) -> None:
            """Show completion hints in status bar while typing."""
            val = event.value
            status = self.query_one("#status-bar", Label)
            if val.startswith("/"):
                matches = [c for c in SLASH_COMMANDS if c.startswith(val)]
                if matches:
                    status.update("  " + "  ".join(matches[:6]))
                    return
            at_idx = val.rfind("@")
            if at_idx != -1:
                prefix = val[at_idx + 1:].lower()
                matches = [f for f in self._workspace_files if prefix in f.lower()][:6]
                if matches:
                    status.update("  @  " + "  ".join(m.split("/")[-1] for m in matches))
                    return
            if self._agent_running:
                status.update("  Agent running…  (Esc to interrupt)")
            else:
                status.update("  Ready")

        # ── Slash commands ─────────────────────────────────────────────────────

        async def _handle_slash(self, cmd: str) -> None:
            parts = cmd.strip().split(maxsplit=1)
            verb = parts[0].lower()
            log = self.query_one("#chat-log", ChatLog)

            if verb in ("/quit", "/exit", "/q"):
                if self._agent:
                    self._agent.cancel()
                self.exit()
                return

            if verb == "/help":
                log.add_help()
                return

            if verb == "/clear":
                log.clear()
                log.add_divider()
                log.add_status("Chat cleared.")
                log.add_divider()
                return

            if verb == "/diff":
                self.query_one("#file-panel", FilePanel).show_git_diff(self.workspace)
                log.add_status("Showing git diff in file panel →")
                return

            if verb == "/compact":
                if self._agent:
                    log.add_status("Compacting…")
                    result = await self._agent.compact()
                    log.add_status(f"Compacted: {result}")
                else:
                    log.add_status("No active agent session to compact.")
                return

            if verb == "/undo":
                if self._agent:
                    result = await self._agent.tools.undo_last_turn()
                    log.add_status(result)
                else:
                    log.add_status("No active agent session.")
                return

            if verb == "/rollback":
                if self._agent:
                    ok, msg = await self._agent.git.restore()
                    log.add_status(f"{'✓' if ok else '✗'} {msg}")
                else:
                    log.add_status("No active agent session.")
                return

            if verb == "/model":
                from nvagent.config import SUPPORTED_MODELS
                log.add_status("Available models:")
                for i, m in enumerate(SUPPORTED_MODELS):
                    current = " ← current" if m == self.config.models.default else ""
                    log.write(Text(f"  {i + 1}. {m}{current}", style="#8b949e"))
                if len(parts) > 1 and parts[1].strip().isdigit():
                    idx = int(parts[1].strip()) - 1
                    if 0 <= idx < len(SUPPORTED_MODELS):
                        self.config.models.default = SUPPORTED_MODELS[idx]
                        log.add_status(f"Model set to {self.config.models.default}")
                        self._refresh_header()
                else:
                    log.add_status("Use: /model <number>  e.g. /model 1")
                return

            if verb == "/sessions":
                sessions = self.session_store.list_sessions(str(self.workspace), limit=10)
                log.add_status("Recent sessions:")
                for s in sessions:
                    n = sum(1 for m in s.messages if m.get("role") == "user")
                    current = " ← current" if s.id == self.session.id else ""
                    log.write(Text(
                        f"  #{s.id:<4}  {s.updated_at[:16]}  {n} exchanges{current}",
                        style="#8b949e",
                    ))
                return

            if verb == "/history":
                if not self._history_log:
                    log.add_status("No activity this session yet.")
                    return
                log.add_divider()
                log.write(Text("  Session History", style="bold #76b900"))
                for entry in self._history_log:
                    log.write(Text(
                        f"  Turn {entry['turn']}  {entry['message'][:50]}",
                        style="bold #79c0ff",
                    ))
                    if entry.get("tokens"):
                        log.write(Text(f"    tokens: ~{entry['tokens']:,}", style="dim"))
                    for f in entry.get("files", [])[:5]:
                        log.write(Text(f"    ✎ {f}", style="dim #76b900"))
                    for c in entry.get("commands", [])[:3]:
                        log.write(Text(f"    $ {c[:70]}", style="dim #f0883e"))
                log.add_divider()
                return

            log.add_status(f"Unknown command: {verb}  (type /help)")

        # ── Agent streaming ────────────────────────────────────────────────────

        def _stream_message(self, message: str) -> None:
            if self._agent_running:
                self.query_one("#chat-log", ChatLog).add_status(
                    "Agent is busy. Press Esc to interrupt."
                )
                return

            log = self.query_one("#chat-log", ChatLog)
            log.add_user(message)
            self._agent_running = True
            self._interrupt_requested = False
            self.query_one("#status-bar", Label).update("  Agent running…  (Esc to interrupt)")
            self.query_one("#input-bar").disabled = True

            self._agent_worker(message)

        @work(exclusive=True)
        async def _agent_worker(self, message: str) -> None:
            """Drive the agent loop; post events back to the UI thread."""
            from nvagent.core.agent import Agent

            log = self.query_one("#chat-log", ChatLog)
            fp = self.query_one("#file-panel", FilePanel)

            # ── Real confirm_fn: pushes ConfirmModal, awaits user answer ───────
            async def confirm_write(path: str, diff_str: str) -> bool:
                """Pause the agent, show a modal diff, return user's decision."""
                approved: bool = await self.app.push_screen_wait(
                    ConfirmModal(path, diff_str)
                )
                # Log the decision back into the chat so it's visible in history
                log.add_confirm_result(path, approved)
                return approved

            if self._agent is None:
                self._agent = Agent(
                    config=self.config,
                    workspace=self.workspace,
                    session=self.session,
                    session_store=self.session_store,
                    confirm_fn=None if self.no_confirm else confirm_write,
                )
            else:
                self._agent.tools.confirm_fn = None if self.no_confirm else confirm_write

            _turn_files: list[str] = []
            _turn_commands: list[str] = []
            _turn_tokens: int = 0
            _in_think = False
            _token_buf: list[str] = []

            def flush_tokens() -> None:
                nonlocal _token_buf
                if _token_buf:
                    chunk = "".join(_token_buf)
                    for line in chunk.split("\n"):
                        log.write(Text(f"  {line}", style="#e6edf3"))
                    _token_buf = []

            try:
                async for event in self._agent.run(message):
                    if self._interrupt_requested:
                        self._agent.cancel()
                        log.add_status("Interrupted.")
                        break

                    etype = event.type

                    if etype == "token":
                        if _in_think:
                            _in_think = False
                            log.write(Text(""))
                        _token_buf.append(event.data)
                        if "\n" in event.data:
                            flush_tokens()

                    elif etype == "think_token":
                        if not _in_think:
                            flush_tokens()
                            _in_think = True
                            log.write(Text("  ◌ thinking…", style="dim italic #a78bfa"))

                    elif etype == "tool_start":
                        flush_tokens()
                        d = event.data or {}
                        log.add_tool_start(d.get("name", "?"), d.get("args", {}))
                        # Snapshot file before write so we can show a diff after
                        if d.get("name") in ("write_file", "edit_file", "str_replace_editor"):
                            path = d.get("args", {}).get("path", "")
                            if path:
                                full = self.workspace / path
                                try:
                                    self._file_cache[path] = full.read_text(
                                        encoding="utf-8", errors="replace"
                                    )
                                except Exception:
                                    self._file_cache[path] = ""
                        if d.get("name") == "run_command":
                            cmd = d.get("args", {}).get("command", "")
                            if cmd:
                                _turn_commands.append(cmd[:80])

                    elif etype == "tool_result":
                        d = event.data or {}
                        result = d.get("result", "")
                        name = d.get("name", "")
                        args = d.get("args", {})
                        log.add_tool_result(result)
                        path = args.get("path", "")
                        # Show the file on read
                        if path and name in ("read_file", "view_file"):
                            fp.show_file(path, self.workspace)
                        # Show the resulting diff on write
                        if path and name in ("write_file", "edit_file", "str_replace_editor"):
                            old = self._file_cache.pop(path, "")
                            try:
                                new = (self.workspace / path).read_text(
                                    encoding="utf-8", errors="replace"
                                )
                            except Exception:
                                new = ""
                            if old != new:
                                fp.show_diff(path, old, new)

                    elif etype == "files_changed":
                        files = event.data if isinstance(event.data, list) else []
                        if files:
                            log.add_files_changed(files)
                            _turn_files.extend(f for f in files if f not in _turn_files)
                            fp.show_file(str(files[-1]), self.workspace)
                            self._workspace_files = _collect_files(self.workspace)

                    elif etype == "status":
                        flush_tokens()
                        log.add_status(str(event.data))

                    elif etype == "error":
                        flush_tokens()
                        d = event.data if isinstance(event.data, dict) else {}
                        msg = d.get("message", str(event.data))
                        if msg:
                            log.add_error(msg)

                    elif etype == "safety_violation":
                        flush_tokens()
                        d = event.data if isinstance(event.data, dict) else {}
                        log.add_safety(
                            d.get("message", str(event.data)),
                            d.get("fatal", False),
                        )

                    elif etype == "done":
                        flush_tokens()
                        d = event.data if isinstance(event.data, dict) else {}
                        turns = d.get("turns", 0)
                        tokens = d.get("tokens_used", 0)
                        files = d.get("files_changed", [])
                        self._session_tokens += tokens
                        _turn_tokens = tokens
                        log.add_done(turns, tokens, files)
                        self._history_log.append({
                            "turn": len(self._history_log) + 1,
                            "message": message[:60],
                            "files": list(_turn_files),
                            "commands": list(_turn_commands),
                            "tokens": _turn_tokens,
                        })

            except asyncio.CancelledError:
                flush_tokens()
                log.add_status("Cancelled.")
            except Exception as e:
                flush_tokens()
                log.add_error(f"Agent error: {e}")
            finally:
                self._agent_running = False
                self.query_one("#status-bar", Label).update(
                    f"  Ready  ·  session #{self.session.id}  ·  "
                    f"~{self._session_tokens:,} tokens total"
                )
                self.query_one("#input-bar").disabled = False
                self.query_one("#input-bar").focus()

        # ── Actions ────────────────────────────────────────────────────────────

        def action_quit(self) -> None:
            if self._agent:
                self._agent.cancel()
            self.exit()

        def action_clear_chat(self) -> None:
            log = self.query_one("#chat-log", ChatLog)
            log.clear()
            log.add_divider()
            log.add_status("Chat cleared.")
            log.add_divider()

        def action_toggle_diff(self) -> None:
            self.query_one("#file-panel", FilePanel).show_git_diff(self.workspace)

        def action_interrupt_agent(self) -> None:
            if self._agent_running and self._agent:
                self._interrupt_requested = True
                self._agent.cancel()

        def action_cycle_focus(self) -> None:
            if self.focused and self.focused.id == "input-bar":
                try:
                    self.query_one("#chat-log").focus()
                except Exception:
                    pass
            else:
                self.query_one("#input-bar").focus()

        # ── Helpers ────────────────────────────────────────────────────────────

        def _get_git_branch(self) -> str:
            try:
                r = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True, text=True,
                    cwd=self.workspace, timeout=2,
                )
                return r.stdout.strip()
            except Exception:
                return ""

        def _refresh_header(self) -> None:
            model_short = self.config.models.default.split("/")[-1]
            branch = self._get_git_branch()
            branch_str = f"  {branch}" if branch else ""
            self.query_one("#header-bar", Label).update(
                f"⬛ nvagent  │  {model_short}  │  "
                f"session #{self.session.id}  │  {self.workspace.name}{branch_str}"
            )

        def _expand_at_mentions(self, message: str) -> tuple[str, list[str]]:
            mentioned: list[str] = []

            def _replace(m: re.Match) -> str:
                ref = m.group(1)
                candidates = list(self.workspace.rglob(ref))
                if not candidates:
                    p = self.workspace / ref
                    candidates = [p] if p.exists() else []
                if not candidates:
                    return m.group(0)
                fpath = candidates[0]
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    if len(content) > 8000:
                        content = content[:8000] + "\n… [truncated]"
                    rel = (
                        fpath.relative_to(self.workspace)
                        if fpath.is_relative_to(self.workspace)
                        else fpath
                    )
                    mentioned.append(str(rel))
                    return f"\n\n[Contents of {rel}]\n```\n{content}\n```\n"
                except Exception:
                    return m.group(0)

            expanded = re.sub(r"@([\w./\-]+)", _replace, message)
            return expanded, mentioned


# ── Public entry point ─────────────────────────────────────────────────────────

def launch_textual_tui(
    workspace: Path,
    config,
    session,
    session_store,
    no_confirm: bool = False,
) -> None:
    """Launch the Textual TUI. Raises ImportError if textual is not installed."""
    if not _TEXTUAL_AVAILABLE:
        raise ImportError(
            "textual is not installed. Install it with:\n"
            "  pip install 'textual>=0.70'\n"
            "Or use the classic ANSI interface with --no-tui"
        )
    app = NVAgentApp(
        workspace=workspace,
        config=config,
        session=session,
        session_store=session_store,
        no_confirm=no_confirm,
    )
    app.run()
