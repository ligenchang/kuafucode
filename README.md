# nvagent ⬛

**NVIDIA-powered terminal coding agent** — competes with Claude Code, Aider, Cursor.

```
pip install nvagent
nvagent chat
```

rm -rf build .nvagent dist nvagent.egg-info && pip install --upgrade --force-reinstall .
rm -rf build .nvagent dist nvagent.egg-info && pip install -e .
.venv/bin/python -m py_compile tui/app.py && echo "OK"


## Setup

```bash
# Get your NVIDIA NIM API key from https://build.nvidia.com
export NVIDIA_API_KEY="nvapi-..."

# Or set it via CLI
nvagent config set api_key nvapi-...

# Launch the TUI
nvagent chat

# Run a one-shot task
nvagent run "add type hints to all functions in src/auth.py"

# Run on a specific project
nvagent chat --workspace /path/to/project
```

## Features

- **Full codebase context** — reads your entire project, understands structure
- **File operations** — read, write, create, delete with diff preview
- **Shell execution** — runs commands, streams output live
- **Code search** — fast grep/ripgrep across codebase
- **Git integration** — understands diffs, status, history
- **Claude Code-style TUI** — full-width chat, inline tool output, zero clutter
- **Session persistence** — remembers context across restarts
- **Project memory** — `.nvagent/memory.md` survives sessions
- **Streaming** — every token streams in real time

## Models (NVIDIA NIM)

Default models (configurable in `.nvagent/config.toml`):

| Task | Model |
|------|-------|
| Default | `minimaxai/minimax-m2.1` |
| Fast | `minimaxai/minimax-m2.1` |
| Code | `minimaxai/minimax-m2.1` |

## Architecture

3 dependencies. Hand-rolled agent loop. No LangChain. No RAG. No magic.

```
openai    → NVIDIA NIM API (AsyncOpenAI, streaming)
textual   → TUI (async-native, Rich built-in)
typer     → CLI (nvagent chat / run / config)
```
