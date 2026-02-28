"""Interactive arrow-key menu selection."""

from __future__ import annotations

import asyncio
import os
import select
import sys

try:
    import tty
    import termios
except ImportError:
    tty = None  # type: ignore[assignment]
    termios = None  # type: ignore[assignment]


def run_arrow_menu(
    title: str, options: list[str], current: str | None
) -> str | None:
    """Blocking arrow-key menu. Must be called from the main thread.

    Uses os.read/os.write on raw fds to avoid fighting with readline's
    internal buffering of sys.stdin.
    
    Returns the chosen option string, or None on cancel (Esc/Ctrl+C/q).
    """
    if tty is None or termios is None:
        return None  # non-Unix fallback

    n = len(options)
    cur = next((i for i, o in enumerate(options) if o == current), 0)
    ifd = sys.stdin.fileno()
    ofd = sys.stdout.fileno()

    # ANSI codes as byte strings for raw fd output
    _R = b"\x1b[0m"
    _B = b"\x1b[1m"
    _D = b"\x1b[2m"
    _G = b"\x1b[32m"
    _W = b"\x1b[97m"
    _BG = b"\x1b[48;5;236m"
    _NL = b"\r\n"
    _EL = b"\r\x1b[2K"  # carriage-return + erase line

    def _enc(s: str) -> bytes:
        return s.encode("utf-8", errors="replace")

    def _render() -> bytes:
        buf = bytearray()
        buf += _EL + _B + _G + _enc(f"  {title}") + _R + _NL
        buf += _EL + _NL
        for i, opt in enumerate(options):
            is_sel = i == cur
            is_active = opt == current
            tag = b"  " + _D + b"\xe2\x86\x90 current" + _R if is_active else b""  # ← current
            if is_sel:
                buf += (
                    _EL + _G + b"  \xe2\x96\xb6 " + _R + _BG + _W + _B + _enc(opt) + _R + tag + _NL
                )
            else:
                buf += _EL + b"    " + _D + _enc(opt) + _R + tag + _NL
        buf += _EL + _NL
        buf += (
            _EL + _D + b"  \xe2\x86\x91 \xe2\x86\x93  move   Enter confirm   Esc cancel" + _R + _NL
        )
        return bytes(buf)

    total_lines = n + 4  # title + blank + items + blank + hint

    def _draw(redraw: bool) -> None:
        out = bytearray()
        if redraw:
            out += f"\x1b[{total_lines}A".encode()  # cursor up N lines
        out += _render()
        os.write(ofd, bytes(out))

    old = termios.tcgetattr(ifd)
    try:
        tty.setraw(ifd)
        _draw(redraw=False)
        while True:
            ch = os.read(ifd, 1)
            if ch == b"\x1b":
                # Check if more bytes follow (arrow = ESC [ A/B)
                r, _, _ = select.select([ifd], [], [], 0.05)
                if r:
                    nxt = os.read(ifd, 2)  # read "[A" or "[B"
                    if nxt == b"[A":  # ↑
                        cur = (cur - 1) % n
                        _draw(redraw=True)
                        continue
                    if nxt == b"[B":  # ↓
                        cur = (cur + 1) % n
                        _draw(redraw=True)
                        continue
                    # Some other escape sequence — ignore, don't cancel
                    continue
                # Plain Esc → cancel
                os.write(ofd, b"\r\n")
                return None
            if ch in (b"\r", b"\n"):
                os.write(ofd, b"\r\n")
                return options[cur]
            if ch in (b"\x03", b"\x04", b"q", b"Q"):
                os.write(ofd, b"\r\n")
                return None
    finally:
        try:
            termios.tcsetattr(ifd, termios.TCSADRAIN, old)
        except Exception:
            pass


async def arrow_menu(
    title: str, options: list[str], current: str | None = None
) -> str | None:
    """Thin async shim — calls run_arrow_menu on the calling (main) thread."""
    return run_arrow_menu(title, options, current)
