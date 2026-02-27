"""Configuration management for nvagent."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Curated roster of supported models (used for the /model menu in the TUI).
# Each entry is a full NIM model ID.  The first entry is the default.
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_MODELS: list[str] = [
    "qwen/qwen3.5-397b-a17b",    # Qwen 3.5 397B MoE — strong reasoning + tools
    "minimaxai/minimax-m2.1",    # MiniMax M2.1 — long-context, reasoning
    "moonshotai/kimi-k2.5",      # Kimi K2.5 — fast MoE, tool-calling
    "z-ai/glm4.7",               # GLM-4.7 — fast thinking + tool-calling
]


DEFAULT_CONFIG = """\
# nvagent configuration
# Get your NVIDIA NIM API key from https://build.nvidia.com
#
# ── Quick model switcher ──────────────────────────────────────────────────
# Paste any model ID from https://build.nvidia.com/models into the fields
# below.  All three can be the same model (simplest setup).
# Use '/model' in the TUI to switch between supported models interactively.
#
# Supported models (switchable via /model in chat):
#
#   qwen/qwen3.5-397b-a17b       ← Qwen 3.5 397B MoE (strong reasoning + tools)
#   minimaxai/minimax-m2.1       ← MiniMax M2.1 (long-context, reasoning)
#   moonshotai/kimi-k2.5         ← Kimi K2.5 (fast MoE, tool-calling)
#   z-ai/glm4.7                  ← GLM-4.7 (fast thinking + tool-calling)
#
# To enable MCP (Model Context Protocol) servers, add one or more [[mcp.servers]]
# sections.  Example:
#
# [[mcp.servers]]
# name    = "filesystem"
# command = "npx"
# args    = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
# optional = true

[api]
api_key = ""
base_url = "https://integrate.api.nvidia.com/v1"

[models]
# Main model for complex coding tasks (must support tool/function calling)
default = "qwen/qwen3.5-397b-a17b"
# Fast model for simple queries — can be the same as default
fast = "qwen/qwen3.5-397b-a17b"
# Code-specialized model — can be the same as default
code = "qwen/qwen3.5-397b-a17b"

[agent]
# Max tokens per LLM response (lower = faster, less truncation risk)
max_tokens = 16384
# Temperature (0.0 = deterministic, 1.0 = creative)
temperature = 0.2
# Ask before writing/deleting files (safe mode)
safe_mode = false
# Dry-run mode: show what the agent would do without actually writing or running anything
dry_run = false
# Show the plan and ask for approval before the first LLM call
plan_preview = false
# Require explicit approval for every file write (shows diff)
approve_writes = false
# Max file size to read (bytes)
max_file_bytes = 512000
# Max files to include in context tree
max_context_files = 50
# Max reasoning/think chars before the loop retries with a concise directive
# (≈ 4 chars per token; set 0 to disable)
think_budget_tokens = 8000

[context]
# File patterns to always ignore
ignore_patterns = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "*.pyc", "*.pyo", "*.egg-info", "dist", "build",
    ".DS_Store", "*.min.js", "*.min.css", "package-lock.json",
    "yarn.lock", "*.lock", ".env", ".env.*"
]

[safety]
# Automatically create a git commit before the agent modifies files
git_checkpoint = true
# Syntax-check every file immediately after it is written
validate_writes = true
# Also run the linter on changed files (slower, requires ruff / eslint)
lint_on_write = false
# Detect and break infinite tool-call loops
loop_detection = true
# How many identical consecutive calls trigger loop detection
max_identical_calls = 3
# Rolling window size for loop inspection
loop_window = 8
# Hard token budget for a single run() invocation
max_tokens_per_task = 120000
# Hard wall-clock limit in seconds (0 = unlimited)
max_wall_seconds = 900.0
# Warn (non-fatal) when more than this many files are changed
max_files_per_task = 60
# Hard limit on total tool invocations per task
max_tool_calls = 150
# Block task completion if tests fail after code changes
require_tests_pass = false
# Override auto-detected test command (empty = auto-detect)
test_command = ""
# Seconds the test runner is allowed before it is killed
test_timeout = 120
"""


@dataclass
class ApiConfig:
    api_key: str = ""
    base_url: str = "https://integrate.api.nvidia.com/v1"


@dataclass
class ModelsConfig:
    default: str = "qwen/qwen3.5-397b-a17b"
    fast: str    = "qwen/qwen3.5-397b-a17b"
    code: str    = "qwen/qwen3.5-397b-a17b"


@dataclass
class AgentConfig:
    max_tokens: int = 16384
    temperature: float = 0.2
    safe_mode: bool = False
    dry_run: bool = False          # show what would happen — never write or execute
    plan_preview: bool = False     # pause after plan is ready, ask user to confirm
    approve_writes: bool = False   # ask for confirmation on every file write
    max_file_bytes: int = 512000   # 500KB
    max_context_files: int = 50
    # Reasoning/think-token budget in *tokens* (≈ chars / 4).
    # If the model generates more think tokens than this before emitting any
    # output, the stream is cut and the call is retried with a directive to
    # reason more concisely.  Set to 0 to disable.
    think_budget_tokens: int = 8000


@dataclass
class SafetyConfig:
    """All tunable safety / guardrail parameters."""
    # Git checkpointing
    git_checkpoint: bool = True
    git_checkpoint_push: bool = False
    # Post-write validation
    validate_writes: bool = True
    lint_on_write: bool = False
    # Loop detection
    loop_detection: bool = True
    max_identical_calls: int = 3
    loop_window: int = 8
    # Resource guards
    max_tokens_per_task: int = 120_000
    max_wall_seconds: float = 900.0
    max_files_per_task: int = 60
    max_tool_calls: int = 150
    max_tool_calls_per_turn: int = 30   # hard cap per single LLM response batch
    max_output_bytes: int = 2_097_152
    # Test enforcement
    require_tests_pass: bool = False
    test_command: str = ""
    test_timeout: int = 120


@dataclass
class McpServerConfig:
    """Configuration for a single stdio-based MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    optional: bool = True


@dataclass
class McpConfig:
    """Configuration for MCP (Model Context Protocol) server integration."""
    servers: list[McpServerConfig] = field(default_factory=list)


@dataclass
class ContextConfig:
    ignore_patterns: list[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        "*.pyc", "*.pyo", "*.egg-info", "dist", "build",
        ".DS_Store", "*.min.js", "*.min.css", "package-lock.json",
        "yarn.lock", "*.lock", ".env", ".env.*"
    ])


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    mcp: McpConfig = field(default_factory=McpConfig)

    @property
    def api_key(self) -> str:
        """Resolve API key from config or environment."""
        return self.api.api_key or os.environ.get("NVIDIA_API_KEY", "")


def get_config_dir(workspace: Optional[Path] = None) -> Path:
    """Get the .nvagent config directory for a workspace."""
    base = workspace or Path.cwd()
    return base / ".nvagent"


def load_config(workspace: Optional[Path] = None) -> Config:
    """Load config from .nvagent/config.toml, falling back to defaults."""
    config_dir = get_config_dir(workspace)
    config_file = config_dir / "config.toml"

    cfg = Config()

    if config_file.exists():
        with open(config_file, "rb") as f:
            raw = tomllib.load(f)

        if "api" in raw:
            for k, v in raw["api"].items():
                if hasattr(cfg.api, k):
                    setattr(cfg.api, k, v)

        if "models" in raw:
            for k, v in raw["models"].items():
                if hasattr(cfg.models, k):
                    setattr(cfg.models, k, v)

        if "agent" in raw:
            for k, v in raw["agent"].items():
                if hasattr(cfg.agent, k):
                    setattr(cfg.agent, k, v)

        if "context" in raw:
            for k, v in raw["context"].items():
                if hasattr(cfg.context, k):
                    setattr(cfg.context, k, v)

        if "safety" in raw:
            for k, v in raw["safety"].items():
                if hasattr(cfg.safety, k):
                    setattr(cfg.safety, k, v)

        # [[mcp.servers]] is an array of tables
        mcp_section = raw.get("mcp", {})
        for srv in mcp_section.get("servers", []):
            if "name" not in srv or "command" not in srv:
                continue
            cfg.mcp.servers.append(
                McpServerConfig(
                    name=srv["name"],
                    command=srv["command"],
                    args=srv.get("args", []),
                    env=srv.get("env", {}),
                    optional=srv.get("optional", True),
                )
            )

    return cfg


def save_config(cfg: Config, workspace: Optional[Path] = None) -> None:
    """Save config to .nvagent/config.toml."""
    config_dir = get_config_dir(workspace)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"

    # Write TOML manually to preserve comments
    lines = [
        "# nvagent configuration\n\n",
        "[api]\n",
        f'api_key = "{cfg.api.api_key}"\n',
        f'base_url = "{cfg.api.base_url}"\n\n',
        "[models]\n",
        f'default = "{cfg.models.default}"\n',
        f'fast = "{cfg.models.fast}"\n',
        f'code = "{cfg.models.code}"\n\n',
        "[agent]\n",
        f"max_tokens = {cfg.agent.max_tokens}\n",
        f"temperature = {cfg.agent.temperature}\n",
        f"safe_mode = {'true' if cfg.agent.safe_mode else 'false'}\n",
        f"dry_run = {'true' if cfg.agent.dry_run else 'false'}\n",
        f"plan_preview = {'true' if cfg.agent.plan_preview else 'false'}\n",
        f"approve_writes = {'true' if cfg.agent.approve_writes else 'false'}\n",
        f"max_file_bytes = {cfg.agent.max_file_bytes}\n",
        f"max_context_files = {cfg.agent.max_context_files}\n",
        f"think_budget_tokens = {cfg.agent.think_budget_tokens}\n\n",
        "[context]\n",
        "ignore_patterns = [\n",
    ]
    for p in cfg.context.ignore_patterns:
        lines.append(f'    "{p}",\n')
    lines.append("]\n")

    s = cfg.safety
    lines += [
        "\n[safety]\n",
        f"git_checkpoint = {'true' if s.git_checkpoint else 'false'}\n",
        f"validate_writes = {'true' if s.validate_writes else 'false'}\n",
        f"lint_on_write = {'true' if s.lint_on_write else 'false'}\n",
        f"loop_detection = {'true' if s.loop_detection else 'false'}\n",
        f"max_identical_calls = {s.max_identical_calls}\n",
        f"loop_window = {s.loop_window}\n",
        f"max_tokens_per_task = {s.max_tokens_per_task}\n",
        f"max_wall_seconds = {s.max_wall_seconds}\n",
        f"max_files_per_task = {s.max_files_per_task}\n",
        f"max_tool_calls = {s.max_tool_calls}\n",
        f"max_tool_calls_per_turn = {s.max_tool_calls_per_turn}\n",
        f"require_tests_pass = {'true' if s.require_tests_pass else 'false'}\n",
        f'test_command = "{s.test_command}"\n',
        f"test_timeout = {s.test_timeout}\n",
    ]

    config_file.write_text("".join(lines))


def init_workspace(workspace: Optional[Path] = None) -> Path:
    """Initialize .nvagent directory with default config and memory file."""
    config_dir = get_config_dir(workspace)
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "config.toml"
    if not config_file.exists():
        config_file.write_text(DEFAULT_CONFIG)

    memory_file = config_dir / "memory.md"
    if not memory_file.exists():
        memory_file.write_text(
            "# Project Memory\n\n"
            "This file is maintained by nvagent to remember important project context.\n"
            "You can edit it manually. The agent reads and writes this file.\n\n"
            "## Key Facts\n\n"
            "## Architecture Notes\n\n"
            "## Decisions & Rationale\n\n"
            "## TODO / Known Issues\n\n"
        )

    # Create .gitignore for .nvagent (keep memory, ignore sessions db)
    gitignore = config_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("sessions.db\n")

    return config_dir
