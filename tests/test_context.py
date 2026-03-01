"""Tests for context assembly — file tree, project detection, ignore patterns."""

import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # /home/claude -> nvagent symlink

from nvagent.config import Config, ContextConfig
from nvagent.core.context import (
    build_file_tree, detect_project_type, build_system_prompt,
    load_nvagent_ignore, is_ignored,
)


def _make_workspace() -> Path:
    tmp = Path(tempfile.mkdtemp())
    return tmp


def test_file_tree_basic():
    ws = _make_workspace()
    (ws / "main.py").write_text("# main")
    (ws / "utils.py").write_text("# utils")
    tree = build_file_tree(ws, ignore_patterns=[])
    assert "main.py" in tree
    assert "utils.py" in tree


def test_file_tree_respects_ignore():
    ws = _make_workspace()
    (ws / "main.py").write_text("")
    node_mods = ws / "node_modules"
    node_mods.mkdir()
    (node_mods / "dep.js").write_text("")

    tree = build_file_tree(ws, ignore_patterns=["node_modules"])
    assert "main.py" in tree
    assert "node_modules" not in tree
    assert "dep.js" not in tree


def test_file_tree_max_files():
    ws = _make_workspace()
    for i in range(20):
        (ws / f"file_{i}.py").write_text("")
    tree = build_file_tree(ws, ignore_patterns=[], max_files=5)
    # Should be truncated
    assert "truncated" in tree.lower() or len(tree.splitlines()) <= 10


def test_detect_project_type_python():
    ws = _make_workspace()
    (ws / "pyproject.toml").write_text("[project]\nname='test'")
    result = detect_project_type(ws)
    assert "Python" in result


def test_detect_project_type_node():
    ws = _make_workspace()
    (ws / "package.json").write_text('{"name": "test"}')
    result = detect_project_type(ws)
    assert "Node" in result


def test_detect_project_type_multiple():
    ws = _make_workspace()
    (ws / "pyproject.toml").write_text("")
    (ws / "Dockerfile").write_text("")
    result = detect_project_type(ws)
    assert "Python" in result
    assert "Docker" in result


def test_detect_project_type_unknown():
    ws = _make_workspace()
    result = detect_project_type(ws)
    assert "Unknown" in result


def test_is_ignored_basic():
    p = Path("/workspace/node_modules/dep.js")
    assert is_ignored(p, ["node_modules"])


def test_is_ignored_glob():
    p = Path("/workspace/dist/bundle.min.js")
    assert is_ignored(p, ["*.min.js"])


def test_is_ignored_not_matched():
    p = Path("/workspace/src/main.py")
    assert not is_ignored(p, ["node_modules", "__pycache__"])


def test_load_nvagent_ignore_missing():
    ws = _make_workspace()
    result = load_nvagent_ignore(ws)
    assert result == []


def test_load_nvagent_ignore_reads_patterns():
    ws = _make_workspace()
    nvagent_dir = ws / ".nvagent"
    nvagent_dir.mkdir()
    (nvagent_dir / "ignore").write_text("# comment\n*.log\ndist/\nbuild/\n\n")
    result = load_nvagent_ignore(ws)
    assert "*.log" in result
    assert "dist/" in result
    assert "build/" in result
    assert "# comment" not in result


def test_build_system_prompt_contains_workspace():
    ws = _make_workspace()
    (ws / "main.py").write_text("print('hello')")
    cfg = Config()
    prompt = build_system_prompt(ws, cfg)
    assert str(ws) in prompt


def test_build_system_prompt_contains_file_tree():
    ws = _make_workspace()
    (ws / "important.py").write_text("")
    cfg = Config()
    prompt = build_system_prompt(ws, cfg)
    assert "important.py" in prompt


def test_build_system_prompt_includes_memory():
    ws = _make_workspace()
    cfg = Config()
    prompt = build_system_prompt(ws, cfg, memory="## Rules\n- Never break tests")
    assert "Never break tests" in prompt


def test_build_system_prompt_excludes_ignored():
    ws = _make_workspace()
    secret_dir = ws / ".secrets"
    secret_dir.mkdir()
    (secret_dir / "keys.txt").write_text("SECRET")
    cfg = Config()
    cfg.context.ignore_patterns = [".secrets"]
    prompt = build_system_prompt(ws, cfg)
    assert "keys.txt" not in prompt


def test_nvagent_ignore_applied_in_system_prompt():
    ws = _make_workspace()
    nvagent_dir = ws / ".nvagent"
    nvagent_dir.mkdir()
    (nvagent_dir / "ignore").write_text("*.log\n")
    (ws / "debug.log").write_text("log content")
    (ws / "main.py").write_text("code")

    cfg = Config()
    prompt = build_system_prompt(ws, cfg)
    assert "main.py" in prompt
    assert "debug.log" not in prompt
