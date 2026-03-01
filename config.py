"""Configuration management for nvagent."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SUPPORTED_MODELS: list[str] = [
    "qwen/qwen3.5-397b-a17b",
    "minimaxai/minimax-m2.1",
    "moonshotai/kimi-k2.5",
    "z-ai/glm4.7",
    "nvidia/nemotron-3-nano-30b-a3b",
]

DEFAULT_CONFIG = """\
# nvagent configuration
# Get your NVIDIA NIM API key from https://build.nvidia.com

[api]
api_key = ""
base_url = "https://integrate.api.nvidia.com/v1"

[models]
default = "nvidia/nemotron-3-nano-30b-a3b"

[agent]
max_tokens = 16384
temperature = 0.2
safe_mode = false
dry_run = false
max_file_bytes = 512000
max_context_files = 50

[context]
ignore_patterns = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "*.pyc", "*.egg-info", "dist", "build",
    ".DS_Store", "*.min.js", "*.min.css", "package-lock.json",
    "yarn.lock", "*.lock", ".env", ".env.*"
]

[safety]
git_checkpoint = true
loop_detection = true
max_identical_calls = 3
loop_window = 8
max_tokens_per_task = 120000
max_wall_seconds = 900.0
max_tool_calls = 150
"""


@dataclass
class ApiConfig:
    api_key: str = ""
    base_url: str = "https://integrate.api.nvidia.com/v1"


@dataclass
class ModelsConfig:
    default: str = "nvidia/nemotron-3-nano-30b-a3b"


@dataclass
class AgentConfig:
    max_tokens: int = 16384
    temperature: float = 0.2
    safe_mode: bool = False
    dry_run: bool = False
    max_file_bytes: int = 512000
    max_context_files: int = 50


@dataclass
class SafetyConfig:
    git_checkpoint: bool = True
    loop_detection: bool = True
    max_identical_calls: int = 3
    loop_window: int = 8
    max_tokens_per_task: int = 120_000
    max_wall_seconds: float = 900.0
    max_tool_calls: int = 150


@dataclass
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    optional: bool = True


@dataclass
class McpConfig:
    servers: list[McpServerConfig] = field(default_factory=list)


@dataclass
class ContextConfig:
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "*.pyc", "*.egg-info", "dist", "build",
            ".DS_Store", "*.min.js", "*.min.css",
            "package-lock.json", "yarn.lock", "*.lock", ".env", ".env.*",
        ]
    )


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
        return self.api.api_key or os.environ.get("NVIDIA_API_KEY", "")


def load_config(workspace: Optional[Path] = None) -> Config:
    config_dir = (workspace or Path.cwd()) / ".nvagent"
    config_file = config_dir / "config.toml"
    cfg = Config()

    if not config_file.exists():
        return cfg

    with open(config_file, "rb") as f:
        raw = tomllib.load(f)

    for section, obj in [("api", cfg.api), ("models", cfg.models), ("agent", cfg.agent),
                          ("context", cfg.context), ("safety", cfg.safety)]:
        for k, v in raw.get(section, {}).items():
            if hasattr(obj, k):
                setattr(obj, k, v)

    for srv in raw.get("mcp", {}).get("servers", []):
        if "name" in srv and "command" in srv:
            cfg.mcp.servers.append(McpServerConfig(
                name=srv["name"], command=srv["command"],
                args=srv.get("args", []), env=srv.get("env", {}),
                optional=srv.get("optional", True),
            ))

    return cfg


def save_config(cfg: Config, workspace: Optional[Path] = None) -> None:
    config_dir = (workspace or Path.cwd()) / ".nvagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"

    lines = [
        "# nvagent configuration\n\n",
        "[api]\n",
        f'api_key = "{cfg.api.api_key}"\n',
        f'base_url = "{cfg.api.base_url}"\n\n',
        "[models]\n",
        f'default = "{cfg.models.default}"\n\n',
        "[agent]\n",
        f"max_tokens = {cfg.agent.max_tokens}\n",
        f"temperature = {cfg.agent.temperature}\n",
        f"safe_mode = {'true' if cfg.agent.safe_mode else 'false'}\n",
        f"dry_run = {'true' if cfg.agent.dry_run else 'false'}\n",
        f"max_file_bytes = {cfg.agent.max_file_bytes}\n",
        f"max_context_files = {cfg.agent.max_context_files}\n\n",
        "[context]\nignore_patterns = [\n",
    ]
    for p in cfg.context.ignore_patterns:
        lines.append(f'    "{p}",\n')
    lines.append("]\n\n")
    s = cfg.safety
    lines += [
        "[safety]\n",
        f"git_checkpoint = {'true' if s.git_checkpoint else 'false'}\n",
        f"loop_detection = {'true' if s.loop_detection else 'false'}\n",
        f"max_identical_calls = {s.max_identical_calls}\n",
        f"loop_window = {s.loop_window}\n",
        f"max_tokens_per_task = {s.max_tokens_per_task}\n",
        f"max_wall_seconds = {s.max_wall_seconds}\n",
        f"max_tool_calls = {s.max_tool_calls}\n",
    ]
    config_file.write_text("".join(lines))


def init_workspace(workspace: Optional[Path] = None) -> Path:
    config_dir = (workspace or Path.cwd()) / ".nvagent"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "config.toml"
    if not config_file.exists():
        config_file.write_text(DEFAULT_CONFIG)

    memory_file = config_dir / "memory.md"
    if not memory_file.exists():
        memory_file.write_text(
            "# Project Memory\n\nThis file is maintained by nvagent.\n"
            "You can edit it manually. The agent reads and writes this file.\n\n"
            "## Key Facts\n\n## Architecture Notes\n\n## TODO / Known Issues\n\n"
        )

    gitignore = config_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("sessions.db\n")

    return config_dir
