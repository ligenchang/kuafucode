"""
nvagent CLI entrypoint.

Commands:
  nvagent chat              Launch TUI (interactive)
  nvagent run "task"        Run a task headlessly (non-interactive)
  nvagent config set key val
  nvagent config show
  nvagent config init
  nvagent sessions          List recent sessions
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from typing import Annotated

from nvagent.config import Config, load_config, save_config, init_workspace
from nvagent.core.loop import Agent, AgentEvent
from nvagent.core.session import Session, SessionStore

app = typer.Typer(
    name="nvagent",
    help="⬛ NVIDIA NIM powered terminal coding agent",
    rich_markup_mode="rich",
    no_args_is_help=False,
)

config_app = typer.Typer(help="Manage nvagent configuration")
app.add_typer(config_app, name="config")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_workspace(workspace_arg: Optional[str]) -> Path:
    ws = Path(workspace_arg) if workspace_arg else Path.cwd()
    if not ws.exists():
        typer.echo(f"Error: workspace '{ws}' does not exist.", err=True)
        raise typer.Exit(1)
    return ws.resolve()


def _setup(workspace: Path) -> tuple[Config, SessionStore]:
    """Initialize workspace, load config, create session store."""
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


# ─────────────────────────────────────────────────────────────────────────────
# nvagent (default — launches chat)
# ─────────────────────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Annotated[
        Optional[str], typer.Option("--workspace", "-w", help="Project directory")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume last session")] = False,
    version: Annotated[bool, typer.Option("--version", "-v", help="Show version")] = False,
    model: Annotated[
        Optional[str], typer.Option("--model", "-m", help="Override default model")
    ] = None,
):
    """⬛ nvagent — NVIDIA NIM powered terminal coding agent."""
    if version:
        from nvagent import __version__

        typer.echo(f"nvagent {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        # Default: launch TUI chat
        _launch_chat(workspace=workspace, resume=resume, model=model)


# ─────────────────────────────────────────────────────────────────────────────
# nvagent chat
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def chat(
    workspace: Annotated[
        Optional[str], typer.Option("--workspace", "-w", help="Project directory")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume last session")] = False,
    new: Annotated[bool, typer.Option("--new", "-n", help="Force new session")] = False,
    no_confirm: Annotated[
        bool, typer.Option("--no-confirm", help="Skip diff confirmation prompts")
    ] = False,
    model: Annotated[
        Optional[str], typer.Option("--model", "-m", help="Override default model")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Simulate — show what the agent would do without modifying any files"
        ),
    ] = False,
    plan_preview: Annotated[
        bool,
        typer.Option(
            "--plan-preview",
            "-p",
            help="Pause after plan is ready and ask for approval before executing",
        ),
    ] = False,
    approve_writes: Annotated[
        bool,
        typer.Option(
            "--approve-writes",
            "-a",
            help="Ask for confirmation on every file write (shows full diff)",
        ),
    ] = False,
):
    """Launch the interactive TUI."""
    _launch_chat(
        workspace=workspace,
        resume=resume or not new,
        no_confirm=no_confirm,
        model=model,
        dry_run=dry_run,
        plan_preview=plan_preview,
        approve_writes=approve_writes,
    )


def _launch_chat(
    workspace: Optional[str] = None,
    resume: bool = True,
    no_confirm: bool = False,
    model: Optional[str] = None,
    dry_run: bool = False,
    plan_preview: bool = False,
    approve_writes: bool = False,
) -> None:
    ws = _get_workspace(workspace)
    config, session_store = _setup(ws)

    if model:
        config.models.default = model
    if dry_run:
        config.agent.dry_run = True
    if plan_preview:
        config.agent.plan_preview = True
    if approve_writes:
        config.agent.approve_writes = True

    # Session management
    session = None
    if resume:
        session = session_store.get_last_session(str(ws))
        if session:
            n = sum(1 for m in session.messages if m["role"] == "user")
            typer.echo(
                f"  ↩  Resuming session #{session.id}  ·  {n} exchange{'s' if n != 1 else ''}  ·  {session.updated_at[:16]}"
            )

    if session is None:
        session = session_store.create_session(str(ws))
        typer.echo(f"  ✦  New session #{session.id}")

    # Launch TUI
    from nvagent.tui.app import launch_tui

    launch_tui(
        workspace=ws,
        config=config,
        session=session,
        session_store=session_store,
        no_confirm=no_confirm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# nvagent run
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Task to perform")],
    workspace: Annotated[
        Optional[str], typer.Option("--workspace", "-w", help="Project directory")
    ] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Override model")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only show final output")] = False,
):
    """Run a task non-interactively (headless mode — good for CI/scripts)."""
    ws = _get_workspace(workspace)
    config, session_store = _setup(ws)

    if model:
        config.models.default = model

    session = session_store.create_session(str(ws))

    async def _run() -> None:
        agent = Agent(
            config=config,
            workspace=ws,
            session=session,
            session_store=session_store,
        )

        response_parts = []

        async for event in agent.run(task):
            if event.type == "token":
                if not quiet:
                    print(event.data, end="", flush=True)
                response_parts.append(event.data)

            elif event.type == "tool_start":
                if not quiet:
                    tc = event.data
                    name = tc["name"]
                    args = tc.get("args", {})
                    path_hint = args.get("path", args.get("command", args.get("query", "")))
                    print(f"\n⚙ {name}({path_hint!r:.50})", flush=True)

            elif event.type == "tool_result":
                if not quiet:
                    result = event.data.get("result", "")
                    preview = result.splitlines()[0][:80] if result else ""
                    print(f"  → {preview}", flush=True)

            elif event.type == "status":
                if not quiet:
                    print(f"\r  {event.data:<60}", end="", flush=True)

            elif event.type == "error":
                msg = event.data.get("message", str(event.data))
                print(f"\n✗ Error: {msg}", flush=True)
                sys.exit(1)

            elif event.type == "safety_violation":
                d = event.data if isinstance(event.data, dict) else {}
                kind = d.get("kind", "?")
                msg = d.get("message", str(event.data))
                fatal = d.get("fatal", False)
                icon = "✗" if fatal else "⚠"
                print(f"\n{icon} Safety [{kind}]: {msg}", flush=True)

            elif event.type == "done":
                data = event.data
                if not quiet:
                    print(
                        f"\n\n✓ Done ({data.get('turns', 0)} turns, ~{data.get('tokens_used', 0):,} tokens)"
                    )

        if quiet and response_parts:
            print("".join(response_parts))

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# nvagent config
# ─────────────────────────────────────────────────────────────────────────────


@config_app.command("show")
def config_show(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
):
    """Show current configuration."""
    ws = _get_workspace(workspace)
    init_workspace(ws)
    config = load_config(ws)

    typer.echo(f"\n⬛ nvagent config — {ws}/.nvagent/config.toml\n")
    typer.echo(f"  API key:    {'set ✓' if config.api.api_key else 'not set ✗'}")
    typer.echo(f"  Base URL:   {config.api.base_url}")
    typer.echo(f"  Default:    {config.models.default}")
    typer.echo(f"  Fast:       {config.models.fast}")
    typer.echo(f"  Code:       {config.models.code}")
    typer.echo(f"  Safe mode:  {config.agent.safe_mode}")
    typer.echo(f"  Max tokens: {config.agent.max_tokens}")
    typer.echo(f"  Temp:       {config.agent.temperature}")
    typer.echo("")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (e.g. api_key, models.default)")],
    value: Annotated[str, typer.Argument(help="Value to set")],
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
):
    """Set a configuration value.

    Examples:
      nvagent config set api_key nvapi-...
      nvagent config set models.default nvidia/llama-3.1-nemotron-70b-instruct
      nvagent config set agent.safe_mode true
    """
    ws = _get_workspace(workspace)
    init_workspace(ws)
    config = load_config(ws)

    # Parse nested key like "models.default"
    parts = key.split(".")
    if len(parts) == 1:
        # Top-level shortcuts
        if key == "api_key":
            config.api.api_key = value
        elif key == "base_url":
            config.api.base_url = value
        else:
            typer.echo(f"Unknown key: {key}. Use section.key format, e.g. api.api_key", err=True)
            raise typer.Exit(1)
    elif len(parts) == 2:
        section, attr = parts
        section_obj = getattr(config, section, None)
        if section_obj is None:
            typer.echo(f"Unknown section: {section}", err=True)
            raise typer.Exit(1)
        if not hasattr(section_obj, attr):
            typer.echo(f"Unknown attribute: {attr} in section {section}", err=True)
            raise typer.Exit(1)
        # Type coercion
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
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
):
    """Initialize .nvagent directory with default config."""
    ws = _get_workspace(workspace)
    config_dir = init_workspace(ws)
    typer.echo(f"✓ Initialized nvagent at {config_dir}")
    typer.echo(f"  Next: nvagent config set api_key nvapi-...")


# ─────────────────────────────────────────────────────────────────────────────
# nvagent sessions
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def sessions(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of sessions")] = 10,
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
    typer.echo(f"  {'ID':<6} {'Created':<20} {'Updated':<20} {'Messages':<10} {'Summary'}")
    typer.echo(f"  {'─' * 70}")
    for s in sessions_list:
        msg_count = len(s.messages)
        summary = s.summary[:40] if s.summary else "(no summary)"
        typer.echo(
            f"  {s.id:<6} {s.created_at[:16]:<20} {s.updated_at[:16]:<20} {msg_count:<10} {summary}"
        )
    typer.echo("")


# ─────────────────────────────────────────────────────────────────────────────
# nvagent models
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def models(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
):
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


if __name__ == "__main__":
    app()
