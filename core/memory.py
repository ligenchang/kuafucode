"""
Long-term memory system — structured project knowledge that persists across sessions.

## Memory Tiers

Working memory   → Session.messages  (managed by loop.py — in-context conversation)
Long-term memory → .nvagent/memory.json — facts, file notes, task history

## Design

- Zero external dependencies  (pure Python, stdlib JSON)
- Content-addressed storage  (SHA-1 key for dedup)
- Relevance scoring          (BM25-lite keyword match for recall)
- Automatic eviction         (LRU + total-size cap)
- Thread-safe singleton      (one Memory object per workspace)

## Public API

  get_memory(workspace)             → Memory  (singleton)
  Memory.learn(content, tags, file) → str  (key)
  Memory.recall(query, max_k)       → list[MemoryEntry]
  Memory.forget(key)                → bool
  Memory.file_note(path, note)      → attach structured note to a file
  Memory.get_file_note(path)        → str | None
  Memory.task_done(summary, files)  → record a completed task
  Memory.to_context_block(query)    → str  (LLM-ready injection, query-ranked)
  Memory.save() / load()
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


_MEMORY_VERSION = 2
_MAX_ENTRIES    = 500    # cap total long-term entries
_MAX_FILE_NOTES = 300    # cap file-level notes
_MAX_TASK_HIST  = 50     # cap task history entries


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    key:        str           # SHA-1 of content  (dedup key)
    content:    str
    tags:       list[str]
    file:       Optional[str] # workspace-relative path this fact belongs to, or None
    created_at: float         # epoch seconds
    accessed_at: float        # last recalled
    access_count: int = 0

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400


@dataclass
class FileNote:
    """Structured notes attached to a specific file path."""
    path:       str           # workspace-relative path
    summary:    str           # one-line summary of what this file does
    notes:      list[str]     # arbitrary notes (e.g. "handles JWT tokens")
    updated_at: float


@dataclass
class TaskRecord:
    """Brief record of a completed agent task."""
    summary:    str
    files:      list[str]     # files changed
    created_at: float


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser  (used for relevance scoring — no external deps)
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens (identifiers + numbers)."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def _score_entry(entry: MemoryEntry, query_tokens: set[str]) -> float:
    """
    Lightweight relevance score: intersection over query tokens,
    boosted by recency and access frequency.
    """
    if not query_tokens:
        return 0.0
    entry_tokens = set(_tokenize(entry.content + " " + " ".join(entry.tags)))
    overlap = len(query_tokens & entry_tokens) / max(len(query_tokens), 1)
    recency  = max(0.0, 1.0 - entry.age_days() / 30)   # decay over 30 days
    freq_boost = min(entry.access_count / 10, 0.3)
    return overlap + recency * 0.2 + freq_boost


# ─────────────────────────────────────────────────────────────────────────────
# Memory  (the main class)
# ─────────────────────────────────────────────────────────────────────────────

class Memory:
    """
    Thread-safe long-term memory for a workspace.

    Persisted to .nvagent/memory.json  (loaded on first access, saved on mutations).
    """

    MEMORY_FILE = ".nvagent/memory.json"

    def __init__(self, workspace: Path) -> None:
        self.workspace   = workspace.resolve()
        self._entries:   dict[str, MemoryEntry] = {}     # key → entry
        self._file_notes: dict[str, FileNote]  = {}      # rel_path → note
        self._task_hist: list[TaskRecord]       = []
        self._lock       = threading.Lock()
        self._dirty      = False

    # ── Core operations ───────────────────────────────────────────────────────

    def learn(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        file: Optional[str] = None,
    ) -> str:
        """
        Store a new fact in long-term memory.
        Returns the dedup key (SHA-1 prefix of content).
        Silently updates the existing entry if the same content is stored twice.
        """
        content = content.strip()
        if not content:
            return ""
        key = hashlib.sha1(content.encode()).hexdigest()[:12]

        with self._lock:
            if key in self._entries:
                # Update tags and bump access
                existing = self._entries[key]
                for t in (tags or []):
                    if t not in existing.tags:
                        existing.tags.append(t)
                existing.accessed_at = time.time()
                existing.access_count += 1
            else:
                self._entries[key] = MemoryEntry(
                    key          = key,
                    content      = content,
                    tags         = list(tags or []),
                    file         = file,
                    created_at   = time.time(),
                    accessed_at  = time.time(),
                    access_count = 0,
                )
                # Evict oldest entries if over cap
                if len(self._entries) > _MAX_ENTRIES:
                    self._evict()
            self._dirty = True

        return key

    def recall(
        self,
        query: str,
        max_k: int = 10,
        min_score: float = 0.05,
        tags: Optional[list[str]] = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve memory entries relevant to *query*, ranked by relevance.
        Optionally filter by *tags*.
        Updates access_count on recalled entries.
        """
        query_tokens = set(_tokenize(query))

        with self._lock:
            candidates = list(self._entries.values())

        if tags:
            tag_set = set(tags)
            candidates = [e for e in candidates if any(t in tag_set for t in e.tags)]

        scored = [
            (e, _score_entry(e, query_tokens))
            for e in candidates
        ]
        # Use heapq.nlargest to find top-k without sorting all entries (O(n log k) vs O(n log n))
        top = heapq.nlargest(max_k, scored, key=lambda x: x[1])

        results = []
        for entry, score in top:
            if score < min_score:
                break
            entry.accessed_at = time.time()
            entry.access_count += 1
            results.append(entry)

        if results:
            self._dirty = True

        return results

    def forget(self, key: str) -> bool:
        """Remove a memory entry by key. Returns True if it existed."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                self._dirty = True
                return True
        return False

    def forget_file(self, file: str) -> int:
        """Remove all entries associated with a specific file. Returns count removed."""
        removed = 0
        with self._lock:
            keys_to_del = [k for k, e in self._entries.items() if e.file == file]
            for k in keys_to_del:
                del self._entries[k]
                removed += 1
            if removed:
                self._dirty = True
        return removed

    # ── File notes ────────────────────────────────────────────────────────────

    def file_note(
        self,
        path: str,
        summary: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        """
        Attach a summary or append a note to a file.
        *path* should be workspace-relative (e.g. "core/symbols.py").
        """
        with self._lock:
            existing = self._file_notes.get(path)
            if existing is None:
                existing = FileNote(
                    path=path, summary="", notes=[], updated_at=time.time()
                )
                self._file_notes[path] = existing

            if summary:
                existing.summary = summary.strip()
            if note:
                note = note.strip()
                if note not in existing.notes:
                    existing.notes.append(note)
            existing.updated_at = time.time()

            if len(self._file_notes) > _MAX_FILE_NOTES:
                # Evict least recently updated
                oldest = sorted(self._file_notes.values(), key=lambda n: n.updated_at)
                for dead in oldest[: len(self._file_notes) - _MAX_FILE_NOTES]:
                    del self._file_notes[dead.path]

            self._dirty = True

    def get_file_note(self, path: str) -> Optional[str]:
        """Return formatted note for a file, or None."""
        with self._lock:
            fn = self._file_notes.get(path)
        if not fn:
            return None
        parts = []
        if fn.summary:
            parts.append(fn.summary)
        for note in fn.notes:
            parts.append(f"• {note}")
        return "\n".join(parts) if parts else None

    # ── Task history ──────────────────────────────────────────────────────────

    def task_done(self, summary: str, files: Optional[list[str]] = None) -> None:
        """Record a completed agent task (auto-evicts oldest if over cap)."""
        with self._lock:
            self._task_hist.append(TaskRecord(
                summary    = summary.strip()[:400],
                files      = [str(f) for f in (files or [])],
                created_at = time.time(),
            ))
            if len(self._task_hist) > _MAX_TASK_HIST:
                self._task_hist = self._task_hist[-_MAX_TASK_HIST:]
            self._dirty = True

    def recent_tasks(self, n: int = 5) -> list[TaskRecord]:
        """Return the N most recent task records."""
        with self._lock:
            return list(self._task_hist[-n:])

    # ── Context block for LLM injection ──────────────────────────────────────

    def to_context_block(
        self,
        query: str = "",
        max_entries: int = 8,
        max_file_notes: int = 5,
        max_tasks: int = 3,
        max_chars: int = 3000,
    ) -> str:
        """
        Build a compact `## Memory` block for the system prompt.
        Entries are ranked by relevance to *query* (if provided).
        Returns empty string if nothing useful to inject.
        """
        parts: list[str] = []
        total_chars = 0

        # ── Relevant facts ────────────────────────────────────────────────
        recalled = self.recall(query or "", max_k=max_entries, min_score=0.0)
        if recalled:
            fact_lines = []
            for e in recalled:
                snippet = e.content[:200].replace("\n", "  ")
                tag_str = f"  [{','.join(e.tags)}]" if e.tags else ""
                fact_lines.append(f"• {snippet}{tag_str}")
            block = "### Facts\n" + "\n".join(fact_lines)
            total_chars += len(block)
            if total_chars <= max_chars:
                parts.append(block)

        # ── File notes for active paths ───────────────────────────────────
        with self._lock:
            fn_items = sorted(
                self._file_notes.values(), key=lambda n: n.updated_at, reverse=True
            )[:max_file_notes]

        if fn_items:
            note_lines = []
            for fn in fn_items:
                if fn.summary:
                    note_lines.append(f"• {fn.path}: {fn.summary}")
                for n in fn.notes[:2]:
                    note_lines.append(f"  → {n}")
            block = "### File notes\n" + "\n".join(note_lines)
            total_chars += len(block)
            if total_chars <= max_chars:
                parts.append(block)

        # ── Recent tasks ──────────────────────────────────────────────────
        recent = self.recent_tasks(max_tasks)
        if recent:
            task_lines = []
            for t in reversed(recent):
                when = f" ({int((time.time() - t.created_at) / 3600)}h ago)" \
                    if time.time() - t.created_at < 86400 else ""
                f_str = f"  [{', '.join(t.files[:3])}]" if t.files else ""
                task_lines.append(f"• {t.summary[:120]}{when}{f_str}")
            block = "### Recent tasks\n" + "\n".join(task_lines)
            total_chars += len(block)
            if total_chars <= max_chars:
                parts.append(block)

        if not parts:
            return ""

        return "\n## Long-term Memory\n" + "\n\n".join(parts) + "\n"

    def stats(self) -> str:
        with self._lock:
            return (
                f"{len(self._entries)} facts, "
                f"{len(self._file_notes)} file notes, "
                f"{len(self._task_hist)} task records"
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist memory to .nvagent/memory.json (best-effort)."""
        if not self._dirty:
            return

        path = self.workspace / self.MEMORY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "version": _MEMORY_VERSION,
                "workspace": str(self.workspace),
                "saved_at": time.time(),
                "entries": {
                    k: {
                        "key":          e.key,
                        "content":      e.content,
                        "tags":         e.tags,
                        "file":         e.file,
                        "created_at":   e.created_at,
                        "accessed_at":  e.accessed_at,
                        "access_count": e.access_count,
                    }
                    for k, e in self._entries.items()
                },
                "file_notes": {
                    k: {
                        "path":       fn.path,
                        "summary":    fn.summary,
                        "notes":      fn.notes,
                        "updated_at": fn.updated_at,
                    }
                    for k, fn in self._file_notes.items()
                },
                "task_history": [
                    {
                        "summary":    t.summary,
                        "files":      t.files,
                        "created_at": t.created_at,
                    }
                    for t in self._task_hist
                ],
            }

        try:
            path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            with self._lock:
                self._dirty = False
        except Exception:
            pass

    def load(self) -> int:
        """Load memory from .nvagent/memory.json. Returns number of entries loaded."""
        path = self.workspace / self.MEMORY_FILE
        if not path.exists():
            return 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if raw.get("version") != _MEMORY_VERSION:
            return 0

        with self._lock:
            for k, e in raw.get("entries", {}).items():
                self._entries[k] = MemoryEntry(
                    key          = e["key"],
                    content      = e["content"],
                    tags         = e.get("tags", []),
                    file         = e.get("file"),
                    created_at   = e.get("created_at", time.time()),
                    accessed_at  = e.get("accessed_at", time.time()),
                    access_count = e.get("access_count", 0),
                )

            for k, fn in raw.get("file_notes", {}).items():
                self._file_notes[k] = FileNote(
                    path       = fn["path"],
                    summary    = fn.get("summary", ""),
                    notes      = fn.get("notes", []),
                    updated_at = fn.get("updated_at", time.time()),
                )

            for t in raw.get("task_history", []):
                self._task_hist.append(TaskRecord(
                    summary    = t["summary"],
                    files      = t.get("files", []),
                    created_at = t.get("created_at", time.time()),
                ))

        return len(self._entries)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Remove least-recently-accessed entries down to 80% of cap. Called with lock held."""
        target = int(_MAX_ENTRIES * 0.8)
        if len(self._entries) <= target:
            return
        sorted_entries = sorted(
            self._entries.values(), key=lambda e: (e.access_count, e.accessed_at)
        )
        for dead in sorted_entries[: len(self._entries) - target]:
            del self._entries[dead.key]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_MEMORIES:     dict[str, Memory] = {}
_MEMORIES_LOCK = threading.Lock()


def get_memory(workspace: Path) -> Memory:
    """Return (and lazily create + load) the singleton Memory for *workspace*."""
    key = str(workspace.resolve())
    with _MEMORIES_LOCK:
        if key not in _MEMORIES:
            m = Memory(workspace)
            m.load()
            _MEMORIES[key] = m
        return _MEMORIES[key]
