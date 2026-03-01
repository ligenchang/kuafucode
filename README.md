# nvagent

A terminal coding agent powered by NVIDIA NIM models.

## Installation

```bash
pip install -e .
```

For the full two-panel TUI (recommended):

```bash
pip install "textual>=0.70"
```

## Quick Start

```bash
export NIM_API_KEY=nvapi-...

nvagent chat                              # two-panel TUI (default)
nvagent chat --no-tui                     # classic ANSI terminal
nvagent run "add type hints to src/"     # headless one-shot task
nvagent watch "keep tests green"          # watch files, auto-fix on change
```

## The Two-Panel TUI

Running `nvagent chat` opens a split-screen interface:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ⬛ nvagent  │  nemotron-nano  │  session #3  │  myproject   main        │
├──────────────────────────────────┬──────────────────────────────────────┤
│                                  │  📄 src/utils.py                     │
│  ❯ add type hints to all funcs  │  ──────────────────────────────────  │
│                                  │   1  def parse(data: dict) -> dict:  │
│  ⚙  read_file('src/utils.py')   │   2      ...                         │
│  ⚙  write_file('src/utils.py')  │                                      │
│    ✎  Modified: src/utils.py     │  (live diff shown on each write)     │
│                                  │                                      │
│  ✓  Done  (3 turns, ~1,200 tok) │                                      │
├──────────────────────────────────┴──────────────────────────────────────┤
│  ❯  Message nvagent…  (/ for commands, @ for files)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Ready  ·  session #3  ·  ~1,200 tokens total                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Left panel** — scrollable chat log with tool calls, streamed responses, and status messages.

**Right panel** — automatically shows the file being read (syntax-highlighted) or a live unified diff when the agent writes a file.

**Confirm modal** — when `--no-confirm` is not set, every file write pauses the agent and shows a diff modal. Press `y` / Enter to apply or `n` / Esc to skip:

```
┌─────────────────────────────────────────────────┐
│  ⚠  Apply changes to src/utils.py?             │
│  ─────────────────────────────────────────────  │
│  --- a/src/utils.py                             │
│  +++ b/src/utils.py                             │
│  @@ -1,2 +1,2 @@                               │
│  -def parse(data):                              │
│  +def parse(data: dict) -> dict:                │
│       ...                                       │
│                                                 │
│       [ ✓ Apply  [y] ]   [ ✗ Skip  [n] ]       │
└─────────────────────────────────────────────────┘
```

### Keyboard shortcuts

| Key          | Action                                       |
|--------------|----------------------------------------------|
| `Ctrl+D`     | Toggle git diff in the file panel            |
| `Ctrl+L`     | Clear the chat log                           |
| `Ctrl+C`     | Quit                                         |
| `Esc`        | Interrupt a running agent turn               |
| `Tab`        | Cycle focus between chat log and input       |
| `y` / Enter  | Approve a file-write confirmation modal      |
| `n` / Esc    | Reject a file-write confirmation modal       |

## Slash Commands

| Command         | Description                                      |
|-----------------|--------------------------------------------------|
| `/help`         | Show all commands and keybindings                |
| `/clear`        | Clear the chat log                               |
| `/compact`      | Summarize and compress conversation history      |
| `/diff`         | Show uncommitted git diff in the file panel      |
| `/history`      | Show session activity (files, commands, tokens)  |
| `/model <n>`    | Switch model mid-session  e.g. `/model 1`        |
| `/undo`         | Undo last set of file changes                    |
| `/rollback`     | Git rollback to session start checkpoint         |
| `/sessions`     | List recent sessions for this workspace          |
| `/quit`         | Exit nvagent                                     |

## @-file mentions

In the input bar, type `@` followed by a filename to inline its contents:

```
❯ refactor @src/utils.py to use dataclasses
```

The status bar shows matching filenames as you type.

## Configuration

```bash
nvagent config set api_key nvapi-...    # set NVIDIA API key
nvagent config show                      # show current config
```

Or set via environment variable:

```bash
export NIM_API_KEY=nvapi-...
```

Config is saved to `.nvagent/config.toml` in your workspace.

## Project Memory

Create `.nvagent/memory.md` in your workspace to give the agent persistent context — conventions, architecture notes, things to avoid, etc.

## CLI Reference

```
nvagent chat      Launch interactive TUI (Textual two-panel by default)
  --no-tui          Use classic ANSI terminal instead
  --no-confirm      Skip all file-write confirmation modals
  --resume / -r     Resume last session
  --new / -n        Force a new session
  --model / -m      Override default model
  --dry-run         Preview without writing files or running commands
  --workspace / -w  Set workspace directory (default: cwd)

nvagent run <task>   Run a task headlessly (non-interactive)
  --quiet / -q        Suppress tool output
  --dry-run           Preview only
  --model / -m        Override model

nvagent watch <goal>   Watch for file changes and auto-run agent
  --patterns          Glob patterns to watch (default: *.py,*.ts,*.js,…)
  --debounce          Seconds to wait after last change (default: 2.0)
  --max-runs          Stop after N agent runs (default: unlimited)

nvagent config set <key> <value>
nvagent config show
nvagent sessions
nvagent models
```

## Architecture

```
nvagent/
├── cli.py              # Typer CLI entrypoint
├── config.py           # Configuration (API keys, model, limits)
├── core/
│   ├── agent.py        # Main agent loop (yields AgentEvent)
│   ├── client.py       # NVIDIA NIM API client (streaming)
│   ├── context.py      # System prompt / project context builder
│   ├── execution.py    # Command execution utilities
│   ├── mcp.py          # Model Context Protocol client
│   ├── safety.py       # Git checkpoints, loop detection, resource guards
│   └── session.py      # Session persistence (SQLite)
├── tools/
│   ├── __init__.py     # Tool dispatcher
│   ├── schemas.py      # OpenAI-format tool schemas
│   ├── context.py      # Sandbox, simple cache
│   └── handlers/
│       ├── file.py     # read_file, write_file, edit_file, delete_file
│       ├── search.py   # grep, find, find_symbol
│       ├── exec.py     # run_command, run_tests, run_formatter
│       ├── code.py     # extract_symbols, get_diagnostics
│       ├── git.py      # git_status, git_diff, git_log, git_commit
│       ├── memory.py   # read_memory, write_memory
│       ├── url.py      # fetch_url
│       ├── todo.py     # todo list management
│       ├── notebook.py # Jupyter notebook support
│       └── vc.py       # Version control helpers
└── tui/
    ├── ansi.py         # Terminal utilities, ANSI colors (classic REPL)
    ├── repl.py         # Classic ANSI REPL (--no-tui fallback)
    └── app.py          # Textual two-panel TUI (default)
```

## Development

```bash
# Install with dev deps
pip install -e .
pip install pytest pytest-asyncio pytest-cov ruff black textual

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nvagent --cov-report=term-missing

# Lint + format
ruff check .
black --check .
```
