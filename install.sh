#!/usr/bin/env bash
# nvagent install script
# Usage: ./install.sh

set -e

echo ""
echo "⬛ nvagent — NVIDIA NIM Coding Agent"
echo "======================================"
echo ""

# Check Python version
python3 --version | grep -E "3\.(1[2-9]|[2-9][0-9])" > /dev/null 2>&1 || {
    echo "Error: Python 3.12+ required."
    echo "Install from https://python.org"
    exit 1
}

# Check for uv (preferred) or pip
if command -v uv &> /dev/null; then
    echo "Using uv..."
    uv pip install -e . 2>/dev/null || pip install -e .
elif command -v pip &> /dev/null; then
    echo "Using pip..."
    pip install -e .
elif command -v pip3 &> /dev/null; then
    echo "Using pip3..."
    pip3 install -e .
else
    echo "Error: pip not found. Install pip or uv."
    exit 1
fi

echo ""
echo "✓ nvagent installed!"
echo ""
echo "Next steps:"
echo ""
echo "  1. Get your NVIDIA NIM API key:"
echo "     https://build.nvidia.com → Sign in → Get API key"
echo ""
echo "  2. Set your API key:"
echo "     export NVIDIA_API_KEY=nvapi-..."
echo "     # Or: nvagent config set api_key nvapi-..."
echo ""
echo "  3. Launch nvagent in your project:"
echo "     cd /your/project"
echo "     nvagent chat"
echo ""
echo "  4. Or run a one-shot task:"
echo "     nvagent run 'add type hints to all Python files'"
echo ""

# Optional: install ripgrep for faster search
if ! command -v rg &> /dev/null; then
    echo "Tip: Install ripgrep for faster code search:"
    if command -v apt &> /dev/null; then
        echo "  sudo apt install ripgrep"
    elif command -v brew &> /dev/null; then
        echo "  brew install ripgrep"
    elif command -v cargo &> /dev/null; then
        echo "  cargo install ripgrep"
    fi
    echo ""
fi
