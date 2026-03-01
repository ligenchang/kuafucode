"""nvagent — terminal REPL.

Writes directly to stdout (ANSI colors). Fully scrollable, selectable, copyable.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
from pathlib import Path

from nvagent.config import Config
from nvagent.core.agent import Agent
from nvagent.core.session import Session, SessionStore
from nvagent.tui.ansi import (
    BLUE,
    BOLD,
    DIM,
    GRAY,
    GREEN,
    HIDE_CURSOR,
    ORANGE,
    RED,
    RESET,
    SHOW_CURSOR,
    VIOLET,
    WHITE,
    YELLOW,
    c,
    out,
    rule,
    strip_ansi,
    ts,
)

BRED = "\033[1;31m"

# ── Optional prompt_toolkit for multi-line input ───────────────────────────────

try:
    from prompt_toolkit import PromptSession as _PTSession
    from prompt_toolkit.completion import Completer as _PTCompleter
    from prompt_toolkit.completion import Completion as _PTCompletion
    from prompt_toolkit.history import InMemoryHistory as _PTInMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings as _PTKeyBindings
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False

_pt_session: _PTSession | None = None

_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".mypy_cache"}


class _FileIndex:
    """Lightweight cached file list for @-mention completion.

    Built once on first access; invalidated when files_changed is called.
    Rebuilds lazily on next Tab press after invalidation.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._files: list[str] = []
        self._dirty = True

    def invalidate(self) -> None:
        self._dirty = True

    def _rebuild(self) -> None:
        files: list[str] = []
        try:
            for root, dirs, filenames in os.walk(self.workspace):
                dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
                root_path = Path(root)
                for fname in filenames:
                    try:
                        rel = str((root_path / fname).relative_to(self.workspace))
                        files.append(rel)
                    except ValueError:
                        pass
                if len(files) > 5000:
                    break
        except Exception:
            pass
        self._files = sorted(files)
        self._dirty = False

    def completions(self, prefix: str) -> list[str]:
        if self._dirty:
            self._rebuild()
        pl = prefix.lower()
        return [f for f in self._files if pl in f.lower()][:40]


class _AtMentionCompleter:
    """Completes @filename mentions using a cached file index (Tab only, not while typing)."""

    def __init__(self, index: _FileIndex) -> None:
        self._index = index

    def get_completions(self, document, complete_event):
        if not _PT_AVAILABLE:
            return
        from prompt_toolkit.completion import Completion
        text = document.text_before_cursor
        at_idx = text.rfind("@")
        if at_idx == -1:
            return
        prefix = text[at_idx + 1:]
        for rel in self._index.completions(prefix):
            yield Completion(rel, start_position=-len(prefix))


_file_index: _FileIndex | None = None


def _get_file_index(workspace: Path) -> _FileIndex:
    global _file_index
    if _file_index is None or _file_index.workspace != workspace:
        _file_index = _FileIndex(workspace)
    return _file_index


def _reset_pt_session() -> None:
    global _pt_session
    _pt_session = None


def _get_pt_session(workspace: Path | None = None) -> _PTSession:
    global _pt_session
    if _pt_session is None:
        kb = _PTKeyBindings()

        @kb.add("escape", "enter")
        def _insert_newline(event):
            event.app.current_buffer.newline()

        completer = None
        if workspace:
            idx = _get_file_index(workspace)
            completer = _AtMentionCompleter(idx)

        _pt_session = _PTSession(
            key_bindings=kb,
            history=_PTInMemoryHistory(),
            completer=completer,
            complete_while_typing=False,  # only on Tab
        )
    return _pt_session


async def _ainput(prompt: str, workspace: Path | None = None) -> str:
    """Read a line of user input, supporting multi-line with Alt+Enter and @-file completion."""
    if _PT_AVAILABLE:
        try:
            from prompt_toolkit.formatted_text import ANSI as _PTANSI
            session = _get_pt_session(workspace)
            result = await session.prompt_async(_PTANSI(prompt))
            return result
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception:
            pass
    # Fallback to readline input in executor
    loop = asyncio.get_event_loop()
    try:
        sys.stdout.write(strip_ansi(prompt))
        sys.stdout.flush()
        return await loop.run_in_executor(None, input)
    except EOFError:
        raise


# ── Spinner ────────────────────────────────────────────────────────────────────

class Spinner:
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _INTERVAL = 0.08

    def __init__(self) -> None:
        self._active = False
        self._msg = ""
        self._task: asyncio.Task | None = None
        self._col = 0

    def start(self, msg: str = "") -> None:
        self._msg = msg
        self._active = True
        self._task = asyncio.ensure_future(self._run())

    def update(self, msg: str) -> None:
        self._msg = msg

    async def stop(self) -> None:
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._clear()

    def _clear(self) -> None:
        if self._col > 0:
            sys.stdout.write(f"\r{' ' * self._col}\r")
            sys.stdout.flush()
            self._col = 0

    async def _run(self) -> None:
        i = 0
        while self._active:
            frame = self._FRAMES[i % len(self._FRAMES)]
            line = f"  {VIOLET}{frame}{RESET} {DIM}{self._msg}{RESET}"
            visible = strip_ansi(line)
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()
            self._col = len(visible) + 2
            i += 1
            await asyncio.sleep(self._INTERVAL)


# ── Markdown rendering ─────────────────────────────────────────────────────────

def _render_md_line(line: str, in_fence: list[bool]) -> str:
    """Minimal inline markdown rendering."""
    # Code fences
    if line.startswith("```"):
        in_fence[0] = not in_fence[0]
        return f"{DIM}{line}{RESET}"
    if in_fence[0]:
        return f"{DIM}{line}{RESET}"

    # Headers
    if line.startswith("### "):
        return c(BOLD, WHITE, line)
    if line.startswith("## "):
        return c(BOLD, WHITE, line)
    if line.startswith("# "):
        return c(BOLD, GREEN, line)

    # Inline code `...`
    line = re.sub(r"`([^`]+)`", lambda m: f"{ORANGE}{m.group(1)}{RESET}", line)
    # Bold **...**
    line = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", line)

    return line


def _fmt_tool_args(name: str, args: dict) -> str:
    """Format tool call arguments for display."""
    path = args.get("path", args.get("command", args.get("query", "")))
    if path:
        return f"{name}({str(path)[:60]!r})"
    if args:
        first_k, first_v = next(iter(args.items()))
        return f"{name}({first_k}={str(first_v)[:40]!r})"
    return f"{name}()"


def _preview_result(result: str, max_lines: int = 5) -> str:
    """Compact preview of a tool result."""
    lines = result.strip().splitlines()
    if not lines:
        return ""
    preview = lines[:max_lines]
    suffix = f"\n  {DIM}... {len(lines) - max_lines} more lines{RESET}" if len(lines) > max_lines else ""
    return "\n".join(f"  {DIM}{l}{RESET}" for l in preview) + suffix


# ── @-mention file expansion ───────────────────────────────────────────────────

def _expand_at_mentions(message: str, workspace: Path) -> tuple[str, list[str]]:
    """Replace @filename mentions with file contents. Returns (expanded_msg, file_list)."""
    mentioned = []
    def _replace(m: re.Match) -> str:
        ref = m.group(1)
        # Try to find the file
        candidates = list(workspace.rglob(ref))
        if not candidates:
            p = workspace / ref
            candidates = [p] if p.exists() else []
        if not candidates:
            return m.group(0)
        fpath = candidates[0]
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n... [truncated]"
            rel = fpath.relative_to(workspace) if fpath.is_relative_to(workspace) else fpath
            mentioned.append(str(rel))
            return f"\n\n[Contents of {rel}]\n```\n{content}\n```\n"
        except Exception:
            return m.group(0)

    expanded = re.sub(r"@([\w./\-]+)", _replace, message)
    return expanded, mentioned


# ── SLASH COMMANDS ─────────────────────────────────────────────────────────────

SLASH_COMMANDS = {
    "/help": "Show this help message",
    "/clear": "Clear the screen",
    "/compact": "Compact conversation history",
    "/diff": "Show diff of all changes since session start",
    "/history": "Show what happened in this session (files changed, commands run)",
    "/paste": "Enter multi-line paste mode (Ctrl+D to submit)",
    "/model": "Switch model",
    "/undo": "Undo last file changes",
    "/rollback": "Git rollback to checkpoint",
    "/sessions": "List recent sessions",
    "/quit": "Exit nvagent",
}


# ── Main REPL class ────────────────────────────────────────────────────────────

class NVAgentREPL:
    def __init__(
        self,
        workspace: Path,
        config: Config,
        session: Session,
        session_store: SessionStore,
        no_confirm: bool = False,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.session = session
        self.session_store = session_store
        self.no_confirm = no_confirm
        self.agent: Agent | None = None
        self._interrupt_event = asyncio.Event()
        self._session_tokens = 0  # cumulative tokens this session
        self._history_log: list[dict] = []  # {turn, files, commands, git_sha, tokens}

    def _print_header(self) -> None:
        out()
        out(rule())
        out(f"  {BOLD}{GREEN}⬛ nvagent{RESET}  {DIM}NVIDIA NIM coding agent{RESET}")
        try:
            import subprocess
            branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=self.workspace, timeout=3).stdout.strip()
            if branch:
                out(f"  {DIM}workspace:{RESET} {self.workspace}  {DIM}branch:{RESET} {GREEN}{branch}{RESET}")
            else:
                out(f"  {DIM}workspace:{RESET} {self.workspace}")
        except Exception:
            out(f"  {DIM}workspace:{RESET} {self.workspace}")
        out(f"  {DIM}model:{RESET} {self.config.models.default}")
        out(f"  {DIM}session:{RESET} #{self.session.id}")
        # MCP server status (only shown if servers are configured)
        if self.config.mcp.servers:
            if self.agent is not None:
                mcp_status = self.agent.mcp_server_status()
                for srv in mcp_status:
                    icon = f"{GREEN}✓{RESET}" if srv["running"] else f"{RED}✗{RESET}"
                    tools_note = f"  {DIM}({srv['tool_count']} tools){RESET}" if srv["running"] else f"  {DIM}(offline){RESET}"
                    out(f"  {DIM}mcp:{RESET} {icon} {srv['name']}{tools_note}")
            else:
                for srv in self.config.mcp.servers:
                    out(f"  {DIM}mcp:{RESET} {DIM}○ {srv.name} (not started){RESET}")
        out(rule())
        out(f"  {DIM}Type your message. /help for commands. Ctrl+C to interrupt.{RESET}")
        out()

    async def _handle_slash_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True if handled."""
        parts = cmd.strip().split(maxsplit=1)
        verb = parts[0].lower()

        if verb in ("/quit", "/exit", "/q"):
            out(f"\n  {DIM}Goodbye.{RESET}\n")
            raise SystemExit(0)

        if verb == "/help":
            out(f"\n{rule()}")
            out(f"  {BOLD}Commands:{RESET}")
            for cmd_name, desc in SLASH_COMMANDS.items():
                out(f"  {GREEN}{cmd_name:<12}{RESET} {DIM}{desc}{RESET}")
            out(rule())
            return True

        if verb == "/clear":
            os.system("clear" if os.name == "posix" else "cls")
            self._print_header()
            return True

        if verb == "/compact":
            if self.agent:
                out(f"  {DIM}Compacting…{RESET}")
                result = await self.agent.compact()
                out(f"  {GREEN}✓{RESET} {result}")
            else:
                out(f"  {DIM}No active session to compact.{RESET}")
            return True

        if verb == "/undo":
            if self.agent:
                result = await self.agent.tools.undo_last_turn()
                out(f"  {result}")
            return True

        if verb == "/rollback":
            if self.agent:
                ok, msg = await self.agent.git.restore()
                icon = GREEN + "✓" if ok else RED + "✗"
                out(f"  {icon}{RESET} {msg}")
            return True

        if verb == "/model":
            from nvagent.config import SUPPORTED_MODELS
            out(f"\n  {BOLD}Available models:{RESET}")
            for i, m in enumerate(SUPPORTED_MODELS):
                current = " ← current" if m == self.config.models.default else ""
                out(f"  {DIM}{i + 1}.{RESET} {m}{GREEN}{current}{RESET}")
            out()
            try:
                choice = await _ainput(f"  {BOLD}Select model (1-{len(SUPPORTED_MODELS)}, or Enter to keep): {RESET}")
                if choice.strip().isdigit():
                    idx = int(choice.strip()) - 1
                    if 0 <= idx < len(SUPPORTED_MODELS):
                        self.config.models.default = SUPPORTED_MODELS[idx]
                        from nvagent.config import save_config
                        save_config(self.config, self.workspace)
                        out(f"  {GREEN}✓{RESET} Model set to {self.config.models.default}")
            except (EOFError, KeyboardInterrupt):
                pass
            return True

        if verb == "/sessions":
            sessions = self.session_store.list_sessions(str(self.workspace), limit=10)
            out(f"\n  {BOLD}Recent sessions:{RESET}")
            for s in sessions:
                n = sum(1 for m in s.messages if m.get("role") == "user")
                current = " ← current" if s.id == self.session.id else ""
                out(f"  {DIM}#{s.id:<4}{RESET} {s.updated_at[:16]}  {DIM}{n} exchanges{RESET}{GREEN}{current}{RESET}")
            out()
            return True

        if verb == "/diff":
            import subprocess
            try:
                r = subprocess.run(
                    ["git", "diff", "HEAD"],
                    capture_output=True, text=True, cwd=self.workspace, timeout=10
                )
                diff_text = r.stdout.strip()
                if not diff_text:
                    # Try diff against initial checkpoint if no HEAD diff
                    r2 = subprocess.run(
                        ["git", "status", "--short"],
                        capture_output=True, text=True, cwd=self.workspace, timeout=5
                    )
                    out(f"  {DIM}No uncommitted changes.{RESET}")
                    if r2.stdout.strip():
                        out(f"  {DIM}Working tree status:{RESET}\n{r2.stdout.strip()}")
                else:
                    lines = diff_text.splitlines()
                    max_lines = 80
                    for line in lines[:max_lines]:
                        if line.startswith("+") and not line.startswith("+++"):
                            out(c(GREEN, f"  {line}"))
                        elif line.startswith("-") and not line.startswith("---"):
                            out(c(RED, f"  {line}"))
                        elif line.startswith("@@"):
                            out(c(VIOLET, f"  {line}"))
                        else:
                            out(c(DIM, f"  {line}"))
                    if len(lines) > max_lines:
                        out(c(DIM, f"  ... {len(lines) - max_lines} more lines (pipe to git diff for full output)"))
            except Exception as e:
                out(f"  {RED}git diff failed: {e}{RESET}")
            return True

        if verb == "/history":
            if not self._history_log:
                out(f"  {DIM}No activity recorded yet this session.{RESET}")
                return True
            out()
            out(f"  {BOLD}Session history{RESET}  {DIM}(#{self.session.id}){RESET}")
            out(rule())
            for entry in self._history_log:
                t = entry.get("turn", "?")
                msg_preview = entry.get("message", "")[:60]
                tokens = entry.get("tokens", 0)
                sha = entry.get("git_sha", "")
                files = entry.get("files", [])
                commands = entry.get("commands", [])

                out(f"  {BOLD}Turn {t}{RESET}  {DIM}{msg_preview}{RESET}")
                if tokens:
                    out(f"    {DIM}tokens:{RESET} ~{tokens:,}")
                if sha:
                    out(f"    {DIM}checkpoint:{RESET} {GREEN}{sha[:8]}{RESET}")
                if files:
                    out(f"    {DIM}files changed:{RESET}")
                    for f in files[:8]:
                        out(f"      {DIM}• {f}{RESET}")
                    if len(files) > 8:
                        out(f"      {DIM}… and {len(files) - 8} more{RESET}")
                if commands:
                    out(f"    {DIM}commands:{RESET}")
                    for cmd in commands[:4]:
                        out(f"      {DIM}$ {cmd[:70]}{RESET}")
                    if len(commands) > 4:
                        out(f"      {DIM}… and {len(commands) - 4} more{RESET}")
            out(rule())
            total_files = sum(len(e.get("files", [])) for e in self._history_log)
            total_tokens = sum(e.get("tokens", 0) for e in self._history_log)
            out(f"  {DIM}{len(self._history_log)} turn(s) · {total_files} file(s) changed · ~{total_tokens:,} tokens total{RESET}")
            out()
            return True
            out(f"  {DIM}Paste mode — enter your text. Type {BOLD}Ctrl+D{RESET}{DIM} on a new line to submit, {BOLD}Ctrl+C{RESET}{DIM} to cancel.{RESET}")
            lines: list[str] = []
            loop = asyncio.get_event_loop()
            while True:
                try:
                    if _PT_AVAILABLE:
                        from prompt_toolkit import PromptSession as _PS
                        from prompt_toolkit.formatted_text import ANSI as _PA
                        _ps = _PS()
                        line = await _ps.prompt_async(_PA(f"  {DIM}│{RESET} "))
                    else:
                        sys.stdout.write(f"  {DIM}│{RESET} ")
                        sys.stdout.flush()
                        line = await loop.run_in_executor(None, input)
                    lines.append(line)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    out(f"  {DIM}Paste cancelled.{RESET}")
                    return True
            if lines:
                pasted = "\n".join(lines)
                out(c(DIM, f"  ↗ Pasted {len(lines)} line(s). Submitting…"))
                await self._stream(pasted)
            return True

        return False

    async def _stream(self, message: str) -> None:
        """Send a message to the agent and stream the response."""
        self._interrupt_event.clear()

        # Install SIGINT handler
        _orig_sigint = signal.getsignal(signal.SIGINT)
        _agent_task: list[asyncio.Task] = []
        _spinner_ref: list[Spinner] = []

        def _sigint_handler(sig, frame):
            self._interrupt_event.set()
            # Kill any running subprocess immediately
            if self.agent is not None:
                self.agent.tools.kill_active_proc()
            if _agent_task:
                try:
                    asyncio.get_event_loop().call_soon_threadsafe(_agent_task[0].cancel)
                except Exception:
                    pass

        signal.signal(signal.SIGINT, _sigint_handler)

        async def _confirm_write(path: str, diff_str: str) -> bool:
            if _spinner_ref and _spinner_ref[0]._active:
                await _spinner_ref[0].stop()
            out()
            out(c(BOLD, YELLOW, f"  ⚠  Apply changes to {path}?"))
            for line in diff_str.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    out(c(GREEN, f"    {line}"))
                elif line.startswith("-") and not line.startswith("---"):
                    out(c(RED, f"    {line}"))
                else:
                    out(c(DIM, f"    {line}"))
            try:
                answer = await _ainput(c(YELLOW, BOLD, "  Apply? [Y/n] ") + RESET)
                return answer.strip().lower() not in ("n", "no")
            except (EOFError, KeyboardInterrupt):
                return False

        confirm_fn = _confirm_write if not self.no_confirm else None

        _is_new_agent = self.agent is None
        if self.agent is None:
            self.agent = Agent(
                config=self.config,
                workspace=self.workspace,
                session=self.session,
                session_store=self.session_store,
                confirm_fn=confirm_fn,
            )
        else:
            self.agent.tools.confirm_fn = confirm_fn

        out(c(BOLD, GREEN, f"  nvagent  {ts()}"))
        spinner = Spinner()
        _spinner_ref.append(spinner)
        spinner.start("thinking…")

        in_think = False
        line_buf: list[str] = []
        in_fence = [False]

        def flush_line(trailing: str = "") -> None:
            line = "".join(line_buf)
            line_buf.clear()
            rendered = _render_md_line(line, in_fence)
            sys.stdout.write(rendered + trailing + "\n")
            sys.stdout.flush()

        async def run_loop() -> None:
            nonlocal in_think
            _turn_files: list[str] = []
            _turn_commands: list[str] = []
            _turn_git_sha: str = ""
            _turn_tokens: int = 0
            async for event in self.agent.run(message):
                if self._interrupt_event.is_set():
                    # Prompt for correction
                    self._interrupt_event.clear()
                    if spinner._active:
                        await spinner.stop()
                    if line_buf:
                        flush_line(trailing=c(DIM, GRAY, "…"))
                    out()
                    out(c(YELLOW, DIM, "  Interrupted. Send a correction, or Enter to cancel:"))
                    def _hard_cancel(sig, frame):
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, _hard_cancel)
                    try:
                        correction = await _ainput(c(YELLOW, BOLD, "  ✎ ") + RESET)
                    except (EOFError, KeyboardInterrupt):
                        correction = ""
                    finally:
                        signal.signal(signal.SIGINT, _sigint_handler)
                    if correction.strip():
                        self.agent.correction_queue.put_nowait(correction.strip())
                        out(c(DIM, GREEN, f"  → Correction queued: {correction.strip()}"))
                    else:
                        self.agent.cancel()
                        return

                if event.type == "token":
                    if spinner._active:
                        await spinner.stop()
                        sys.stdout.write("\n")
                    if in_think:
                        sys.stdout.write(RESET + "\n")
                        in_think = False
                    text = event.data
                    for ch in text:
                        if ch == "\n":
                            flush_line()
                        else:
                            line_buf.append(ch)

                elif event.type == "think_token":
                    if spinner._active:
                        await spinner.stop()
                    if not in_think:
                        sys.stdout.write(f"\n  {DIM}{VIOLET}◌ thinking…\n  {VIOLET}")
                        in_think = True
                    sys.stdout.write(event.data)
                    sys.stdout.flush()

                elif event.type == "tool_start":
                    if spinner._active:
                        await spinner.stop()
                    if in_think:
                        sys.stdout.write(RESET + "\n")
                        in_think = False
                    if line_buf:
                        flush_line()
                    d = event.data
                    out()
                    out(c(ORANGE, BOLD, f"  ⚙ {_fmt_tool_args(d['name'], d['args'])}"))
                    spinner.start(f"{d['name']}…")
                    # Track commands for /history
                    if d.get("name") == "run_command":
                        cmd = d.get("args", {}).get("command", "")
                        if cmd:
                            _turn_commands.append(cmd[:80])

                elif event.type == "tool_stream":
                    if spinner._active:
                        await spinner.stop()
                    lines = event.data.get("lines", [])
                    for line in lines[-20:]:  # cap live display to 20 lines
                        out(f"  {DIM}{line.rstrip()}{RESET}")

                elif event.type == "tool_result":
                    if spinner._active:
                        await spinner.stop()
                    d = event.data
                    result = d.get("result", "")
                    preview = _preview_result(result)
                    if preview:
                        out(preview)

                elif event.type == "safety_violation":
                    if spinner._active:
                        await spinner.stop()
                    d = event.data if isinstance(event.data, dict) else {}
                    icon = BRED + "✗" if d.get("fatal") else YELLOW + "⚠"
                    out(f"  {icon}{RESET} {d.get('message', str(event.data))}")

                elif event.type == "error":
                    if spinner._active:
                        await spinner.stop()
                    d = event.data if isinstance(event.data, dict) else {}
                    msg = d.get("message", str(event.data))
                    if msg:
                        out(f"\n  {BRED}✗{RESET} {RED}{msg}{RESET}")

                elif event.type == "files_changed":
                    files = event.data if isinstance(event.data, list) else []
                    if files:
                        out(c(DIM, f"  ✎ Modified: {', '.join(str(f) for f in files[:5])}{'...' if len(files) > 5 else ''}"))
                        # Invalidate file index so @-completion reflects new files
                        if _file_index is not None:
                            _file_index.invalidate()
                        _turn_files.extend(f for f in files if f not in _turn_files)

                elif event.type == "status":
                    spinner.update(str(event.data))
                    # After first "thinking" status, MCP is started — refresh header to show status
                    if _is_new_agent and "thinking" in str(event.data).lower() and self.config.mcp.servers:
                        _is_new_agent = False  # only once
                        if spinner._active:
                            await spinner.stop()
                        self._print_header()
                        spinner.start(str(event.data))
                    # Track git checkpoint SHA for /history
                    status_str = str(event.data)
                    if status_str.startswith("Git checkpoint: "):
                        _turn_git_sha = status_str.split(": ", 1)[1].strip()

                elif event.type == "done":
                    if spinner._active:
                        await spinner.stop()
                    if in_think:
                        sys.stdout.write(RESET + "\n")
                        in_think = False
                    if line_buf:
                        flush_line()
                    d = event.data if isinstance(event.data, dict) else {}
                    turns = d.get("turns", 0)
                    tokens = d.get("tokens_used", 0)
                    files = d.get("files_changed", [])
                    self._session_tokens += tokens
                    _turn_tokens = tokens
                    parts = [f"{turns} turn{'s' if turns != 1 else ''}"]
                    if tokens:
                        parts.append(f"~{tokens:,} tokens this turn")
                    if self._session_tokens > tokens:
                        parts.append(f"~{self._session_tokens:,} session total")
                    if files:
                        parts.append(f"{len(files)} file{'s' if len(files) != 1 else ''} changed")
                    out()
                    out(c(DIM, f"  ✓ Done  ({', '.join(parts)})"))
                    # Save to history log
                    self._history_log.append({
                        "turn": len(self._history_log) + 1,
                        "message": message[:60],
                        "files": list(_turn_files),
                        "commands": list(_turn_commands),
                        "git_sha": _turn_git_sha,
                        "tokens": _turn_tokens,
                    })

        task = asyncio.ensure_future(run_loop())
        _agent_task.append(task)
        try:
            await task
        except asyncio.CancelledError:
            if spinner._active:
                await spinner.stop()
            if line_buf:
                flush_line(trailing=c(DIM, GRAY, "…"))
            out(c(DIM, "\n  Cancelled."))
        except Exception as e:
            if spinner._active:
                await spinner.stop()
            out(f"\n  {BRED}Error:{RESET} {e}")
        finally:
            if spinner._active:
                await spinner.stop()
            if in_think:
                sys.stdout.write(RESET + "\n")
            if line_buf:
                flush_line()
            out()
            signal.signal(signal.SIGINT, _orig_sigint)

    async def run(self) -> None:
        self._print_header()

        while True:
            try:
                raw = await _ainput(
                    f"{BOLD}{BLUE}  ❯ {RESET}",
                    workspace=self.workspace,
                )
            except (EOFError, KeyboardInterrupt):
                out(f"\n  {DIM}Goodbye.{RESET}\n")
                if self.agent:
                    await self.agent.stop()
                break

            message = raw.strip()
            if not message:
                continue

            if message.startswith("/"):
                try:
                    handled = await self._handle_slash_command(message)
                    if handled:
                        continue
                except SystemExit:
                    if self.agent:
                        await self.agent.stop()
                    raise

            # Expand @file mentions
            message, mentioned = _expand_at_mentions(message, self.workspace)
            if mentioned:
                out(c(DIM, f"  ↗ Included: {', '.join(mentioned)}"))

            try:
                await self._stream(message)
            except KeyboardInterrupt:
                out(c(DIM, "\n  Cancelled."))
                continue


def launch_tui(
    workspace: Path,
    config: Config,
    session: Session,
    session_store: SessionStore,
    no_confirm: bool = False,
    force_ansi: bool = False,
) -> None:
    """Entry point to launch the terminal UI.

    Tries the Textual two-panel TUI first; falls back to the ANSI REPL if
    textual is not installed or if ``force_ansi=True`` (or ``--no-tui`` flag).
    """
    if not force_ansi:
        try:
            from nvagent.tui.app import launch_textual_tui
            launch_textual_tui(
                workspace=workspace,
                config=config,
                session=session,
                session_store=session_store,
                no_confirm=no_confirm,
            )
            return
        except ImportError:
            # textual not installed — fall through to ANSI REPL
            out(f"  {DIM}(textual not installed — using classic terminal UI. "
                f"Install with: pip install 'textual>=0.70'){RESET}")
        except Exception as e:
            out(f"  {DIM}(Textual TUI failed to start: {e} — falling back to ANSI){RESET}")

    # ── ANSI REPL fallback ────────────────────────────────────────────────────
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    repl = NVAgentREPL(
        workspace=workspace,
        config=config,
        session=session,
        session_store=session_store,
        no_confirm=no_confirm,
    )

    try:
        asyncio.run(repl.run())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
