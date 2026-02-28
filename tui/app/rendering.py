"""Markdown rendering to ANSI output."""

from __future__ import annotations

import re
from nvagent.tui.app.ansi import RESET, BOLD, DIM, WHITE, c, cols

# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer colors (light code-highlighting palette)
# ─────────────────────────────────────────────────────────────────────────────

CODE_BLOCK = "\033[38;2;180;180;180m"  # light gray for code bodies
CODE_BORDER = "\033[38;2;100;100;120m"  # muted purple-gray for fence lines
HEADING1 = "\033[38;2;255;215;0m"  # gold
HEADING2 = "\033[38;2;121;192;255m"  # same as BLUE
HEADING3 = WHITE
QUOTE_COLOR = "\033[38;2;167;139;250m"  # VIOLET
BULLET_COL = "\033[38;2;118;185;0m"  # GREEN

# Pre-compiled inline markdown patterns (avoid re-compiling on every line)
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_BOLD1 = re.compile(r"\*\*(.+?)\*\*")
_RE_BOLD2 = re.compile(r"__(.+?)__")
_RE_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")

# Pre-compiled block markdown patterns
# Suppress model meta-commentary lines emitted as free text
_META_LINE_RE = re.compile(
    r"^\[(?:Used \d+ tools?|Tool(?:s)? used|I used|Tool call)[\s:]",
    re.IGNORECASE,
)

# Pre-formatted ANSI replacement strings
_INLINE_CODE_FMT = f"\033[38;2;240;136;62m{{}}{RESET}".format
_BOLD_FMT = f"{BOLD}{WHITE}{{}}{RESET}".format
_ITALIC_FMT = f"{DIM}{WHITE}{{}}{RESET}".format


def render_inline_md(text: str) -> str:
    """Convert inline markdown marks to ANSI (bold, italic, inline code)."""
    # Inline code  `...`
    text = _RE_INLINE_CODE.sub(
        lambda m: f"\033[38;2;240;136;62m{m.group(1)}{RESET}",
        text,
    )
    # Bold  **...**  or  __...__
    text = _RE_BOLD1.sub(lambda m: f"{BOLD}{WHITE}{m.group(1)}{RESET}", text)
    text = _RE_BOLD2.sub(lambda m: f"{BOLD}{WHITE}{m.group(1)}{RESET}", text)
    # Italic  *...*  (but not ** already consumed)
    text = _RE_ITALIC.sub(lambda m: f"{DIM}{WHITE}{m.group(1)}{RESET}", text)
    return text


def rule(char: str = "─", color: str = DIM) -> str:
    """Return a horizontal rule filling the terminal width."""
    return c(color, char * cols())


def render_line(
    text: str,
    in_fence: list,  # mutable [bool]
    fence_lang: list,  # mutable [str]
) -> str:
    """
    Render a single line of model output as ANSI text.
    
    *in_fence* and *fence_lang* carry state across calls.
    Returns the fully-styled string (no trailing newline).
    """
    stripped = text.strip()

    # ── Code fence boundary ───────────────────────────────────────────────
    if stripped.startswith("```"):
        if not in_fence[0]:
            in_fence[0] = True
            fence_lang[0] = stripped[3:].strip() or ""
            lang_tag = f" {fence_lang[0]}" if fence_lang[0] else ""
            return f"{CODE_BORDER}  ╭─{lang_tag}{RESET}"
        else:
            in_fence[0] = False
            fence_lang[0] = ""
            return f"{CODE_BORDER}  ╰─{RESET}"

    if in_fence[0]:
        return f"{CODE_BLOCK}  {text}{RESET}"

    # ── Headings ──────────────────────────────────────────────────────────
    if stripped.startswith("#"):
        _hi = 0
        while _hi < len(stripped) and stripped[_hi] == "#":
            _hi += 1
        if 1 <= _hi <= 3 and _hi < len(stripped) and stripped[_hi] == " ":
            level = _hi
            h_text = stripped[_hi + 1 :].strip()
            content = render_inline_md(h_text)
            if level == 1:
                pad = "═" * min(cols() - 4, len(h_text) + 2)
                return f"\n{BOLD}{HEADING1}  {content}  {RESET}\n{DIM}{HEADING1}  {pad}{RESET}"
            if level == 2:
                return f"\n{BOLD}{HEADING2}  {content}{RESET}"
            return f"{BOLD}{HEADING3}  {content}{RESET}"

    # ── Horizontal rule ───────────────────────────────────────────────────
    if stripped in ("---", "***", "___") and len(stripped) >= 3:
        return rule()

    # ── Blockquote ────────────────────────────────────────────────────────
    if stripped.startswith("> "):
        return f"{QUOTE_COLOR}  │ {render_inline_md(stripped[2:])}{RESET}"

    # ── Bullet / Numbered list (pure-string, no regex) ───────────────────
    _ls = text.lstrip("\t ")
    if _ls and _ls[0] in "-*•" and len(_ls) > 1 and _ls[1] in " \t":
        _body = _ls[1:].lstrip(" \t")
        if _body:
            _indent = (len(text) - len(_ls)) // 2
            _bullet = "◦" if _indent else "•"
            _pad = "  " * _indent
            return f"{BULLET_COL}  {_pad}{_bullet}{RESET} {render_inline_md(_body)}"

    # ── Numbered list ─────────────────────────────────────────────────────
    _dot = _ls.find(". ")
    if _dot > 0 and _ls[:_dot].isdigit():
        _indent = (len(text) - len(_ls)) // 2
        _pad = "  " * _indent
        return f"{BULLET_COL}  {_pad}{_ls[:_dot]}.{RESET} {render_inline_md(_ls[_dot + 2:])}"

    # ── Normal text ───────────────────────────────────────────────────────
    if text.strip():
        return f"  {render_inline_md(text)}"
    return ""
