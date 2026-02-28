"""Shared constants used across agent modules.

Kept in one place so they can be tuned without hunting through large files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Planning heuristics
# ──────────────────────────────────────────────────────────────────────────────

#: Task messages that start with any of these words/phrases skip the planning
#: phase (plan decomposition is waste for greetings, read-only questions, etc.)
SKIP_PLAN_WORDS: frozenset[str] = frozenset({
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
    "sure", "yes", "no", "bye", "goodbye",
    "what is", "what's", "what are", "what does", "what do",
    "who are", "who is", "how are", "how does", "how do",
    "why is", "why does", "when did", "where is",
    "help", "explain", "describe", "show", "list", "check",
    "analyze", "analyse", "review", "find", "search", "look",
    "summarize", "summarise", "tell me", "show me", "give me",
    "print", "display", "read", "open", "view", "inspect",
    "diagnose", "trace", "debug", "profile", "investigate",
})

#: Phrases that indicate the model is talking about work instead of doing it.
PREMATURE_STOP_PHRASES: tuple[str, ...] = (
    "i'll", "i will", "let me", "next i", "now i", "i need to",
    "continuing", "i'll continue", "i should", "i'll now", "let's",
)


# ──────────────────────────────────────────────────────────────────────────────
# Session JSONL logging
# ──────────────────────────────────────────────────────────────────────────────

def make_session_logger(log_path: Path):
    """Return a callable that appends a JSONL entry to *log_path*."""
    import datetime as _dt

    def _log_event(kind: str, data: object) -> None:
        try:
            entry = {
                "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                "kind": kind,
                "data": data,
            }
            with open(log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    return _log_event


# ──────────────────────────────────────────────────────────────────────────────
# Performance timing helpers
# ──────────────────────────────────────────────────────────────────────────────

def write_perf_log(workspace: Path, label: str, perf: dict) -> None:
    """Append a one-line perf record to ``<workspace>/.nvagent/perf.log``."""
    import datetime as _dt
    import time as _time

    try:
        log_dir = workspace / ".nvagent"
        log_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for k, v in perf.items():
            if isinstance(v, float):
                suffix = "ch" if "_chars_" in k else "s"
                parts.append(
                    f"{k}={v:.0f}{suffix}" if suffix == "ch" else f"{k}={v:.2f}{suffix}"
                )
            else:
                parts.append(f"{k}={v}")
        line = (
            f"{_dt.datetime.now().strftime('%H:%M:%S')} "
            f"msg={label!r}  "
            + "  ".join(parts)
            + "\n"
        )
        with open(log_dir / "perf.log", "a", encoding="utf-8") as _f:
            _f.write(line)
    except Exception:
        pass
