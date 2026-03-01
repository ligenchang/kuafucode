"""Tests for nvagent configuration loading and saving."""

import tempfile
from pathlib import Path

try:
    try:
        import pytest
    except ImportError:
        pytest = None
except ImportError:
    pytest = None

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # /home/claude -> nvagent symlink

from nvagent.config import (
    SUPPORTED_MODELS,
    Config,
    init_workspace,
    load_config,
    save_config,
)


def _make_workspace() -> Path:
    tmp = tempfile.mkdtemp()
    return Path(tmp)


def test_default_config_is_valid():
    cfg = Config()
    assert cfg.api.base_url.startswith("https://")
    assert cfg.agent.max_tokens > 0
    assert cfg.agent.temperature >= 0
    assert len(cfg.context.ignore_patterns) > 0
    assert cfg.safety.max_tool_calls > 0


def test_api_key_from_env():
    import os
    orig = os.environ.get("NVIDIA_API_KEY")
    try:
        os.environ["NVIDIA_API_KEY"] = "nvapi-test-env"
        cfg = Config()
        cfg.api.api_key = ""
        assert cfg.api_key == "nvapi-test-env"
    finally:
        if orig is None:
            os.environ.pop("NVIDIA_API_KEY", None)
        else:
            os.environ["NVIDIA_API_KEY"] = orig


def test_api_key_from_config_takes_precedence():
    import os
    orig = os.environ.get("NVIDIA_API_KEY")
    try:
        os.environ["NVIDIA_API_KEY"] = "nvapi-from-env"
        cfg = Config()
        cfg.api.api_key = "nvapi-from-config"
        assert cfg.api_key == "nvapi-from-config"
    finally:
        if orig is None:
            os.environ.pop("NVIDIA_API_KEY", None)
        else:
            os.environ["NVIDIA_API_KEY"] = orig


def test_load_config_missing_file():
    ws = _make_workspace()
    cfg = load_config(ws)
    # Should return defaults without error
    assert isinstance(cfg, Config)
    assert cfg.agent.max_tokens > 0


def test_save_and_reload_config():
    ws = _make_workspace()
    init_workspace(ws)
    cfg = Config()
    cfg.api.api_key = "nvapi-save-test"
    cfg.models.default = SUPPORTED_MODELS[0]
    cfg.agent.safe_mode = True
    cfg.agent.dry_run = False
    cfg.safety.git_checkpoint = False

    save_config(cfg, ws)

    reloaded = load_config(ws)
    assert reloaded.api.api_key == "nvapi-save-test"
    assert reloaded.models.default == SUPPORTED_MODELS[0]
    assert reloaded.agent.safe_mode is True
    assert reloaded.safety.git_checkpoint is False


def test_init_workspace_creates_files():
    ws = _make_workspace()
    config_dir = init_workspace(ws)

    assert (config_dir / "config.toml").exists()
    assert (config_dir / "memory.md").exists()
    assert (config_dir / ".gitignore").exists()


def test_init_workspace_idempotent():
    ws = _make_workspace()
    d1 = init_workspace(ws)
    d2 = init_workspace(ws)
    assert d1 == d2
    # Running twice shouldn't overwrite existing files
    memory = (d1 / "memory.md")
    memory.write_text("custom content")
    init_workspace(ws)
    assert memory.read_text() == "custom content"


def test_config_ignore_patterns_loaded():
    ws = _make_workspace()
    init_workspace(ws)
    cfg = load_config(ws)
    assert ".git" in cfg.context.ignore_patterns
    assert "__pycache__" in cfg.context.ignore_patterns


def test_mcp_servers_load():
    ws = _make_workspace()
    init_workspace(ws)
    toml = (ws / ".nvagent" / "config.toml")
    toml.write_text(toml.read_text() + """
[mcp]
[[mcp.servers]]
name = "test-server"
command = "npx"
args = ["-y", "@test/mcp"]
optional = true
""")
    cfg = load_config(ws)
    assert len(cfg.mcp.servers) == 1
    assert cfg.mcp.servers[0].name == "test-server"
    assert cfg.mcp.servers[0].optional is True
