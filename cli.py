"""
nvagent CLI entrypoint.

Commands:
  nvagent               Launch TUI (interactive)
  nvagent chat          Launch TUI (interactive)
  nvagent run "task"    Run a task headlessly
  nvagent config set key val
  nvagent config show
  nvagent sessions
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer
from nvagent.config import Config, init_workspace, load_config, save_config
from nvagent.core.agent import Agent
from nvagent.core.session import SessionStore

app = typer.Typer(
    name="nvagent",
    help="⬛ NVIDIA NIM powered terminal coding agent",
    rich_markup_mode="rich",
    no_args_is_help=False,
)

config_app = typer.Typer(help="Manage nvagent configuration")
app.add_typer(config_app, name="config")


def _get_workspace(workspace_arg: str | None) -> Path:
    ws = Path(workspace_arg) if workspace_arg else Path.cwd()
    if not ws.exists():
        typer.echo(f"Error: workspace '{ws}' does not exist.", err=True)
        raise typer.Exit(1)
    return ws.resolve()


def _setup(workspace: Path) -> tuple[Config, SessionStore]:
    init_workspace(workspace)
    config = load_config(workspace)
    if not config.api_key:
        typer.echo(
            "\n⚠  No NVIDIA API key found.\n"
            "Set it with:\n"
            "  nvagent config set api_key nvapi-...\n"
            "Or:\n"
            "  export NVIDIA_API_KEY=nvapi-...\n",
            err=True,
        )
        raise typer.Exit(1)
    session_store = SessionStore(workspace / ".nvagent" / "sessions.db")
    return config, session_store


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace", "-w", help="Project directory")] = None,
    resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume last session")] = False,
    version: Annotated[bool, typer.Option("--version", "-v", help="Show version")] = False,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override default model")] = None,
):
    """⬛ nvagent — NVIDIA NIM powered terminal coding agent."""
    if version:
        from nvagent import __version__
        typer.echo(f"nvagent {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _launch_chat(workspace=workspace, resume=resume, model=model)


@app.command()
def chat(
    workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None,
    resume: Annotated[bool, typer.Option("--resume", "-r")] = False,
    new: Annotated[bool, typer.Option("--new", "-n")] = False,
    no_confirm: Annotated[bool, typer.Option("--no-confirm")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    no_tui: Annotated[bool, typer.Option("--no-tui", help="Use classic ANSI terminal instead of Textual TUI")] = False,
):
    """Launch the interactive TUI (Textual two-panel by default, --no-tui for classic ANSI)."""
    _launch_chat(workspace=workspace, resume=resume or not new, no_confirm=no_confirm, dry_run=dry_run, model=model, no_tui=no_tui)


def _launch_chat(
    workspace: str | None = None,
    resume: bool = True,
    no_confirm: bool = False,
    dry_run: bool = False,
    model: str | None = None,
    no_tui: bool = False,
) -> None:
    ws = _get_workspace(workspace)
    config, session_store = _setup(ws)

    if model:
        config.models.default = model

    if dry_run:
        config.agent.dry_run = True
        typer.echo("  ⚠  DRY RUN — no files will be written or commands executed.")

    session = None
    if resume:
        session = session_store.get_last_session(str(ws))
        if session:
            n = sum(1 for m in session.messages if m["role"] == "user")
            typer.echo(f"  ↩  Resuming session #{session.id}  ·  {n} exchange{'s' if n != 1 else ''}  ·  {session.updated_at[:16]}")

    if session is None:
        session = session_store.create_session(str(ws))
        typer.echo(f"  ✦  New session #{session.id}")

    from nvagent.tui.repl import launch_tui
    launch_tui(workspace=ws, config=config, session=session, session_store=session_store, no_confirm=no_confirm, force_ansi=no_tui)


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Task to perform")],
    workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Run a task non-interactively (headless mode)."""
    ws = _get_workspace(workspace)
    config, session_store = _setup(ws)

    if model:
        config.models.default = model

    if dry_run:
        config.agent.dry_run = True
        if not quiet:
            typer.echo("  ⚠  DRY RUN — no files will be written or commands executed.")

    session = session_store.create_session(str(ws))

    # Exit codes: 0=success, 1=error/fatal, 2=task incomplete/max-turns
    _exit_code = 0

    async def _run() -> None:
        nonlocal _exit_code
        agent = Agent(config=config, workspace=ws, session=session, session_store=session_store)
        response_parts = []
        files_changed: list[str] = []
        tools_used: dict[str, int] = {}
        commands_run: list[str] = []
        tests_result: str = ""
        task_complete = False

        async for event in agent.run(task):
            if event.type == "token":
                if not quiet:
                    print(event.data, end="", flush=True)
                response_parts.append(event.data)
            elif event.type == "tool_start":
                if not quiet:
                    d = event.data
                    path_hint = d["args"].get("path", d["args"].get("command", d["args"].get("query", "")))
                    print(f"\n⚙ {d['name']}({str(path_hint)[:50]!r})", flush=True)
                name = event.data.get("name", "")
                tools_used[name] = tools_used.get(name, 0) + 1
                if name == "run_command":
                    cmd = event.data.get("args", {}).get("command", "")
                    if cmd:
                        commands_run.append(cmd[:60])
            elif event.type == "tool_result":
                if not quiet:
                    result = event.data.get("result", "")
                    preview = result.splitlines()[0][:80] if result else ""
                    print(f"  → {preview}", flush=True)
                if event.data.get("name") == "run_tests":
                    tests_result = event.data.get("result", "")[:200]
            elif event.type == "files_changed":
                new_files = event.data if isinstance(event.data, list) else []
                for f in new_files:
                    if f not in files_changed:
                        files_changed.append(f)
            elif event.type == "status":
                if not quiet:
                    print(f"\r  {event.data:<60}", end="", flush=True)
            elif event.type == "error":
                d = event.data if isinstance(event.data, dict) else {}
                msg = d.get("message", str(event.data))
                is_max_turns = "maximum turns" in msg.lower()
                print(f"\n{'⚠' if is_max_turns else '✗'} {msg}", flush=True)
                _exit_code = 2 if is_max_turns else 1
            elif event.type == "safety_violation":
                d = event.data if isinstance(event.data, dict) else {}
                icon = "✗" if d.get("fatal") else "⚠"
                print(f"\n{icon} Safety [{d.get('kind', '?')}]: {d.get('message', '')}", flush=True)
                if d.get("fatal"):
                    _exit_code = 1
            elif event.type == "done":
                task_complete = True
                d = event.data if isinstance(event.data, dict) else {}
                if not quiet:
                    print(f"\n\n{'─' * 60}", flush=True)
                    print("✓  Task complete", flush=True)
                    print(f"   Turns:   {d.get('turns', 0)}", flush=True)
                    print(f"   Tokens:  ~{d.get('tokens_used', 0):,}", flush=True)
                    if files_changed:
                        print(f"   Files:   {len(files_changed)} changed", flush=True)
                        for f in files_changed[:10]:
                            print(f"            • {f}", flush=True)
                        if len(files_changed) > 10:
                            print(f"            … and {len(files_changed) - 10} more", flush=True)
                    if commands_run:
                        print(f"   Commands: {len(commands_run)} run", flush=True)
                        for c in commands_run[:5]:
                            print(f"            $ {c}", flush=True)
                    if tests_result:
                        first_line = tests_result.splitlines()[0] if tests_result else ""
                        print(f"   Tests:   {first_line}", flush=True)
                    print(f"{'─' * 60}", flush=True)

        if quiet and response_parts:
            print("".join(response_parts))

        # Incomplete if agent never emitted "done"
        if not task_complete and _exit_code == 0:
            _exit_code = 2

    asyncio.run(_run())
    if _exit_code != 0:
        sys.exit(_exit_code)


@config_app.command("show")
def config_show(workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None):
    """Show current configuration."""
    ws = _get_workspace(workspace)
    init_workspace(ws)
    config = load_config(ws)
    typer.echo(f"\n⬛ nvagent config — {ws}/.nvagent/config.toml\n")
    typer.echo(f"  API key:  {'set ✓' if config.api_key else 'not set ✗'}")
    typer.echo(f"  Base URL: {config.api.base_url}")
    typer.echo(f"  Model:    {config.models.default}")
    typer.echo(f"  Safe mode: {config.agent.safe_mode}")
    typer.echo(f"  Max tokens: {config.agent.max_tokens}\n")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (e.g. api_key, models.default)")],
    value: Annotated[str, typer.Argument(help="Value to set")],
    workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None,
):
    """Set a configuration value.

    Examples:
      nvagent config set api_key nvapi-...
      nvagent config set models.default nvidia/llama-3.1-nemotron-70b-instruct
    """
    ws = _get_workspace(workspace)
    init_workspace(ws)
    config = load_config(ws)

    parts = key.split(".")
    if len(parts) == 1:
        if key == "api_key":
            config.api.api_key = value
        elif key == "base_url":
            config.api.base_url = value
        else:
            typer.echo(f"Unknown key: {key}. Use section.key format.", err=True)
            raise typer.Exit(1)
    elif len(parts) == 2:
        section, attr = parts
        section_obj = getattr(config, section, None)
        if section_obj is None or not hasattr(section_obj, attr):
            typer.echo(f"Unknown config path: {key}", err=True)
            raise typer.Exit(1)
        current = getattr(section_obj, attr)
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(section_obj, attr, value)
    else:
        typer.echo(f"Invalid key format: {key}", err=True)
        raise typer.Exit(1)

    save_config(config, ws)
    typer.echo(f"✓ Set {key} = {value}")


@config_app.command("init")
def config_init(
    workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Non-interactive: accept defaults")] = False,
):
    """Interactive setup wizard — configure API key, model, and preferences."""
    ws = _get_workspace(workspace)
    config_dir = init_workspace(ws)
    config = load_config(ws)

    typer.echo(f"\n⬛ nvagent setup — {ws}\n")

    if yes:
        typer.echo(f"✓ Initialized at {config_dir}")
        typer.echo("  Run `nvagent config set api_key nvapi-...` to set your API key.")
        return

    # ── Step 1: API key ─────────────────────────────────────────────────────
    import os as _os
    existing_key = config.api.api_key or _os.environ.get("NVIDIA_API_KEY", "")
    if existing_key:
        typer.echo(f"  API key:  already set ({existing_key[:8]}…)")
    else:
        typer.echo("  Get your free API key at: https://build.nvidia.com")
        key = typer.prompt("  NVIDIA API key (nvapi-...)", default="", show_default=False)
        if key.strip():
            config.api.api_key = key.strip()
            typer.echo("  ✓ API key saved")
        else:
            typer.echo("  ⚠  Skipped — set later with: nvagent config set api_key nvapi-...")

    # ── Step 2: Model selection ──────────────────────────────────────────────
    from nvagent.config import SUPPORTED_MODELS
    typer.echo("\n  Available models:")
    for i, m in enumerate(SUPPORTED_MODELS):
        current = " ← default" if m == config.models.default else ""
        typer.echo(f"    {i + 1}. {m}{current}")
    choice = typer.prompt(
        f"  Choose model [1-{len(SUPPORTED_MODELS)}, Enter to keep default]",
        default="",
        show_default=False,
    )
    if choice.strip().isdigit():
        idx = int(choice.strip()) - 1
        if 0 <= idx < len(SUPPORTED_MODELS):
            config.models.default = SUPPORTED_MODELS[idx]
            typer.echo(f"  ✓ Model set to {config.models.default}")

    # ── Step 3: Safety preferences ──────────────────────────────────────────
    typer.echo("\n  Safety preferences:")
    safe_mode = typer.confirm(
        "  Enable safe mode? (blocks writes outside workspace, dangerous commands)",
        default=config.agent.safe_mode,
    )
    config.agent.safe_mode = safe_mode

    git_checkpoint = typer.confirm(
        "  Enable git checkpoints? (auto-commit before each task for easy undo)",
        default=config.safety.git_checkpoint,
    )
    config.safety.git_checkpoint = git_checkpoint

    # ── Save ─────────────────────────────────────────────────────────────────
    save_config(config, ws)
    typer.echo(f"\n✓ Configuration saved to {config_dir}/config.toml")
    typer.echo(f"  Model:       {config.models.default}")
    typer.echo(f"  Safe mode:   {config.agent.safe_mode}")
    typer.echo(f"  Git checkpoints: {config.safety.git_checkpoint}")
    typer.echo("\n  Run `nvagent chat` to start coding.\n")


@app.command()
def sessions(
    workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
):
    """List recent sessions for this workspace."""
    ws = _get_workspace(workspace)
    init_workspace(ws)
    store = SessionStore(ws / ".nvagent" / "sessions.db")
    sessions_list = store.list_sessions(str(ws), limit=limit)

    if not sessions_list:
        typer.echo("No sessions found.")
        return

    typer.echo(f"\n⬛ nvagent sessions — {ws}\n")
    typer.echo(f"  {'ID':<6} {'Updated':<20} {'Messages':<10} {'Summary'}")
    typer.echo(f"  {'─' * 60}")
    for s in sessions_list:
        msg_count = len(s.messages)
        summary = s.summary[:40] if s.summary else "(no summary)"
        typer.echo(f"  {s.id:<6} {s.updated_at[:16]:<20} {msg_count:<10} {summary}")
    typer.echo("")


@app.command()
def models(workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None):
    """List available NVIDIA NIM models."""
    ws = _get_workspace(workspace)
    config, _ = _setup(ws)

    async def _list():
        from nvagent.core.client import NIMClient
        client = NIMClient(config)
        typer.echo("\n⬛ Available NVIDIA NIM models:\n")
        models_list = await client.get_models()
        for m in sorted(models_list):
            marker = " ← current" if m == config.models.default else ""
            typer.echo(f"  {m}{marker}")
        typer.echo("")

    asyncio.run(_list())


@app.command()
def watch(
    goal: Annotated[str, typer.Argument(help="Goal to keep achieving (e.g. 'keep tests green')")],
    workspace: Annotated[str | None, typer.Option("--workspace", "-w")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    debounce: Annotated[float, typer.Option("--debounce", help="Seconds to wait after last change before triggering")] = 2.0,
    patterns: Annotated[str | None, typer.Option("--patterns", help="Glob patterns to watch, comma-separated (default: *.py,*.ts,*.js,*.go,*.rs)")] = None,
    max_runs: Annotated[int, typer.Option("--max-runs", help="Max agent runs before stopping (0=unlimited)")] = 0,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Watch for file changes and run agent to achieve goal on each change.

    Examples:
      nvagent watch "keep tests green"
      nvagent watch "fix any type errors" --patterns "*.py,*.pyi"
      nvagent watch "keep the build passing" --debounce 5
    """
    import fnmatch
    import threading
    import time

    ws = _get_workspace(workspace)
    config, session_store = _setup(ws)

    if model:
        config.models.default = model
    if dry_run:
        config.agent.dry_run = True

    watch_patterns = [p.strip() for p in (patterns or "*.py,*.ts,*.js,*.go,*.rs,*.toml,*.json").split(",")]

    typer.echo(f"\n⬛ nvagent watch — {ws}")
    typer.echo(f"  goal:     {goal}")
    typer.echo(f"  patterns: {', '.join(watch_patterns)}")
    typer.echo(f"  debounce: {debounce}s")
    if max_runs:
        typer.echo(f"  max runs: {max_runs}")
    typer.echo("\n  Watching for changes… (Ctrl+C to stop)\n")

    # Use polling-based watcher (no watchdog dependency needed)
    _nvagent_dir = ws / ".nvagent"

    def _matches_patterns(path: Path) -> bool:
        # Skip .nvagent internals and common noise
        try:
            rel = path.relative_to(ws)
        except ValueError:
            return False
        parts = rel.parts
        skip_dirs = {".git", ".nvagent", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
        if any(p in skip_dirs for p in parts):
            return False
        return any(fnmatch.fnmatch(path.name, pat) for pat in watch_patterns)

    def _snapshot(root: Path) -> dict[str, float]:
        snap: dict[str, float] = {}
        try:
            for p in root.rglob("*"):
                if p.is_file() and _matches_patterns(p):
                    try:
                        snap[str(p)] = p.stat().st_mtime
                    except OSError:
                        pass
        except Exception:
            pass
        return snap

    def _changed(old: dict, new: dict) -> list[str]:
        changed = []
        for path, mtime in new.items():
            if path not in old or old[path] != mtime:
                changed.append(path)
        for path in old:
            if path not in new:
                changed.append(path)
        return changed

    run_count = 0
    stop_event = threading.Event()

    def _run_agent(changed_files: list[str]) -> int:
        """Run the agent for one watch cycle. Returns exit code."""
        nonlocal run_count
        run_count += 1
        changed_summary = ", ".join(Path(f).name for f in changed_files[:4])
        if len(changed_files) > 4:
            changed_summary += f" +{len(changed_files) - 4} more"

        typer.echo(f"\n  [{run_count}] Change detected: {changed_summary}")
        typer.echo(f"  Running: {goal}\n")

        session = session_store.create_session(str(ws))
        task_msg = (
            f"{goal}\n\nFiles that just changed: {', '.join(changed_files[:10])}"
            if changed_files else goal
        )
        exit_code = 0

        async def _do_run():
            nonlocal exit_code
            agent = Agent(config=config, workspace=ws, session=session, session_store=session_store)
            task_done = False
            async for event in agent.run(task_msg):
                if event.type == "token":
                    print(event.data, end="", flush=True)
                elif event.type == "tool_start":
                    d = event.data
                    hint = d["args"].get("path", d["args"].get("command", ""))
                    print(f"\n  ⚙ {d['name']}({str(hint)[:40]!r})", flush=True)
                elif event.type == "tool_result":
                    result = event.data.get("result", "")
                    preview = result.splitlines()[0][:80] if result else ""
                    print(f"    → {preview}", flush=True)
                elif event.type == "status":
                    print(f"\r  ◌ {event.data:<55}", end="", flush=True)
                elif event.type == "error":
                    d = event.data if isinstance(event.data, dict) else {}
                    msg = d.get("message", str(event.data))
                    is_max = "maximum turns" in msg.lower()
                    print(f"\n  {'⚠' if is_max else '✗'} {msg}", flush=True)
                    exit_code = 2 if is_max else 1
                elif event.type == "done":
                    task_done = True
                    d = event.data if isinstance(event.data, dict) else {}
                    tokens = d.get("tokens_used", 0)
                    files = d.get("files_changed", [])
                    parts = [f"{d.get('turns', 0)} turns", f"~{tokens:,} tokens"]
                    if files:
                        parts.append(f"{len(files)} files changed")
                    print(f"\n\n  ✓ Done ({', '.join(parts)})", flush=True)
            if not task_done and exit_code == 0:
                exit_code = 2

        asyncio.run(_do_run())
        return exit_code

    try:
        prev_snapshot = _snapshot(ws)
        poll_interval = min(debounce / 4, 0.5)
        pending_changes: list[str] = []
        last_change_time = 0.0

        while not stop_event.is_set():
            time.sleep(poll_interval)
            curr_snapshot = _snapshot(ws)
            new_changes = _changed(prev_snapshot, curr_snapshot)

            if new_changes:
                prev_snapshot = curr_snapshot
                for f in new_changes:
                    if f not in pending_changes:
                        pending_changes.append(f)
                last_change_time = time.monotonic()

            if pending_changes and (time.monotonic() - last_change_time) >= debounce:
                changes_to_process = list(pending_changes)
                pending_changes.clear()
                last_change_time = 0.0

                _run_agent(changes_to_process)

                if max_runs > 0 and run_count >= max_runs:
                    typer.echo(f"\n  Reached max-runs ({max_runs}). Stopping.")
                    break

                # Refresh snapshot after agent may have changed files
                prev_snapshot = _snapshot(ws)
                typer.echo("\n  Watching for changes… (Ctrl+C to stop)\n")

    except KeyboardInterrupt:
        typer.echo(f"\n\n  Stopped after {run_count} run(s).\n")


if __name__ == "__main__":
    app()
