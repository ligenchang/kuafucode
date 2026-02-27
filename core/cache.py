"""
Persistent two-level tool-result cache for nvagent.

Architecture
────────────
  L1  in-process dict  (zero-overhead hits within a single session)
  L2  SQLite file at .nvagent/tool_cache.db
       (warm starts across restarts; survives process exit)

Cache key: (abs_path_str, mtime_ns)
  → automatically invalidated whenever the file changes on disk.

Usage
─────
  cache = get_tool_cache(workspace)
  result = cache.get(abs_path_str, mtime_ns)  # None on miss
  cache.put(abs_path_str, mtime_ns, result_str)
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

# ── module-level singleton so every Agent in the same process shares L1 ──────
_instances: dict[str, "ToolCache"] = {}
_instances_lock = threading.Lock()


def get_tool_cache(workspace: Path) -> "ToolCache":
    key = str(workspace)
    with _instances_lock:
        if key not in _instances:
            _instances[key] = ToolCache(workspace)
        return _instances[key]


# ── SQLite schema ─────────────────────────────────────────────────────────────

_CREATE_SQL = """
PRAGMA journal_mode=WAL;       -- concurrent readers don't block the writer
PRAGMA synchronous=NORMAL;     -- flush at checkpoints, not every commit (~2x faster than FULL)
PRAGMA busy_timeout=30000;     -- wait up to 30s instead of instantly erroring on lock contention
CREATE TABLE IF NOT EXISTS read_cache (
    path_key  TEXT    NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    result    TEXT    NOT NULL,
    accessed  INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
    PRIMARY KEY (path_key, mtime_ns)
);
CREATE INDEX IF NOT EXISTS idx_accessed ON read_cache(accessed);
"""

# Prune when table grows past _MAX_ENTRIES; keep the _PRUNE_TO most-recently-used.
_MAX_ENTRIES = 2_000
_PRUNE_TO    = 1_500

# Only cache results up to this many chars  (~300 KB) to avoid a giant DB.
_MAX_RESULT_CHARS = 300_000


class ToolCache:
    """
    Two-level read cache backed by SQLite.

    Thread-safe: a single threading.Lock serialises all SQLite writes.
    The L1 dict is only ever accessed from the asyncio event loop thread
    (ToolExecutor is not shared across threads), so no lock is needed there.
    """

    def __init__(self, workspace: Path) -> None:
        db_dir = workspace / ".nvagent"
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / "tool_cache.db"
        # L1 — in-memory, lives for the process lifetime
        self._l1: dict[tuple[str, int], str] = {}
        self._write_lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,  # matches PRAGMA busy_timeout=30000
            )
            conn.executescript(_CREATE_SQL)
            conn.commit()
            self._conn = conn
        except Exception:
            self._conn = None  # degrade gracefully — cache disabled, no crash

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, path: str, mtime_ns: int) -> str | None:
        """Return cached result or None on miss."""
        key = (path, mtime_ns)
        # L1 — instant dict lookup
        if key in self._l1:
            return self._l1[key]
        # L2 — SQLite
        if self._conn is None:
            return None
        try:
            with self._write_lock:
                row = self._conn.execute(
                    "SELECT result FROM read_cache WHERE path_key=? AND mtime_ns=?",
                    (path, mtime_ns),
                ).fetchone()
                if row:
                    # Refresh access time so LRU pruning keeps hot entries
                    self._conn.execute(
                        "UPDATE read_cache SET accessed=CAST(strftime('%s','now') AS INTEGER) "
                        "WHERE path_key=? AND mtime_ns=?",
                        (path, mtime_ns),
                    )
                    self._conn.commit()
        except Exception:
            return None
        if row:
            self._l1[key] = row[0]
            return row[0]
        return None

    def put(self, path: str, mtime_ns: int, result: str) -> None:
        """Store a result.  Silently skips results that are too large."""
        if len(result) > _MAX_RESULT_CHARS:
            # Still populate L1 so the current session benefits
            self._l1[(path, mtime_ns)] = result
            return
        key = (path, mtime_ns)
        self._l1[key] = result
        if self._conn is None:
            return
        try:
            with self._write_lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO read_cache(path_key, mtime_ns, result) "
                    "VALUES (?, ?, ?)",
                    (path, mtime_ns, result),
                )
                # LRU prune
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM read_cache"
                ).fetchone()[0]
                if count > _MAX_ENTRIES:
                    self._conn.execute(
                        "DELETE FROM read_cache WHERE (path_key, mtime_ns) IN ("
                        "  SELECT path_key, mtime_ns FROM read_cache "
                        "  ORDER BY accessed ASC LIMIT ?"
                        ")",
                        (_MAX_ENTRIES - _PRUNE_TO,),
                    )
                self._conn.commit()
        except Exception:
            pass  # never let cache writes break tool execution

    def put_many(
        self,
        entries: list[tuple[str, int, str]],
    ) -> None:
        """Bulk-insert multiple (path, mtime_ns, result) entries.

        Uses a single ``executemany`` + one ``commit`` regardless of batch size,
        so it is significantly cheaper than calling ``put()`` in a loop when
        warming the cache with many files at once.

        Results that exceed ``_MAX_RESULT_CHARS`` are still stored in L1 (for
        the current session) but skipped from SQLite, matching ``put`` behaviour.
        """
        if not entries:
            return

        db_rows: list[tuple[str, int, str]] = []
        for path, mtime_ns, result in entries:
            self._l1[(path, mtime_ns)] = result
            if len(result) <= _MAX_RESULT_CHARS:
                db_rows.append((path, mtime_ns, result))

        if not db_rows or self._conn is None:
            return

        try:
            with self._write_lock:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO read_cache(path_key, mtime_ns, result) "
                    "VALUES (?, ?, ?)",
                    db_rows,
                )
                # Single prune check for the whole batch
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM read_cache"
                ).fetchone()[0]
                if count > _MAX_ENTRIES:
                    self._conn.execute(
                        "DELETE FROM read_cache WHERE (path_key, mtime_ns) IN ("
                        "  SELECT path_key, mtime_ns FROM read_cache "
                        "  ORDER BY accessed ASC LIMIT ?"
                        ")",
                        (_MAX_ENTRIES - _PRUNE_TO,),
                    )
                self._conn.commit()
        except Exception:
            pass  # never let cache writes break tool execution

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
