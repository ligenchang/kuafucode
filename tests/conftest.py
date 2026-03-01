"""Shared pytest fixtures for nvagent tests."""

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


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A temporary directory to use as a workspace."""
    return tmp_path


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    """A temporary workspace with an initialized git repo."""
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@nvagent.test"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "nvagent Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / ".gitkeep").write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path, capture_output=True)
    return tmp_path


@pytest.fixture
def executor(tmp_workspace: Path):
    """A ToolExecutor over a fresh workspace with safe_mode=False."""
    from nvagent.tools import ToolExecutor
    return ToolExecutor(workspace=tmp_workspace, safe_mode=False)


@pytest.fixture
def safe_executor(tmp_workspace: Path):
    """A ToolExecutor with safe_mode=True."""
    from nvagent.tools import ToolExecutor
    return ToolExecutor(workspace=tmp_workspace, safe_mode=True)
