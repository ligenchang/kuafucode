"""
Workspace-wide symbol index with mtime-based in-memory cache and JSON persistence.

Features
--------
- symbols_for(path)   → SymbolIndex  (cached; re-parses only when mtime changes)
- find_symbol(name)   → list[SymbolMatch]  (O(N files) scan of cache)
- invalidate(path)    → drop one file from cache  (call on file-watcher events)
- build(max_files)    → index whole workspace eagerly
- save() / load()     → .nvagent/index.json  (instant cold start)
- get_workspace_index(workspace) → module-level singleton per workspace

Public API
----------
  get_workspace_index(workspace)  → WorkspaceIndex
  WorkspaceIndex.symbols_for(path)
  WorkspaceIndex.find_symbol(name, kinds)
  WorkspaceIndex.invalidate(path)
  WorkspaceIndex.build()
  WorkspaceIndex.save() / load()
  WorkspaceIndex.stats()
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

# Optional fast JSON backend (pip install orjson)
try:
    import orjson as _orjson
    def _json_dumps(obj: object) -> bytes:
        return _orjson.dumps(obj)
    def _json_loads(data: str | bytes) -> object:
        return _orjson.loads(data)
except ImportError:
    def _json_dumps(obj: object) -> bytes:  # type: ignore[misc]
        return json.dumps(obj, indent=None, separators=(",", ":")).encode()
    def _json_loads(data: str | bytes) -> object:  # type: ignore[misc]
        if isinstance(data, bytes):
            data = data.decode()
        return json.loads(data)

# Import only what we need from symbols — no circular deps possible since
# symbols.py does NOT import from index.py
from nvagent.core.symbols import (
    Symbol, SymbolIndex,
    extract_symbols, _iter_workspace_files,
    get_dependency_graph,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cache entry
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_VERSION = 2


@dataclass
class _CacheEntry:
    mtime:    float
    language: str
    symbols:  list[Symbol]
    imports:  list[str]     # raw import strings (for quick re-export resolution)


# ─────────────────────────────────────────────────────────────────────────────
# SymbolMatch  (result type for find_symbol)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolMatch:
    file:      Path
    line:      int
    kind:      str
    name:      str
    signature: str
    docstring: str = ""

    def render(self, workspace: Optional[Path] = None) -> str:
        try:
            ws_r = workspace.resolve() if workspace else None
            rel  = self.file.relative_to(ws_r) if ws_r else self.file
        except ValueError:
            rel = self.file
        doc = f"  # {self.docstring[:80]}" if self.docstring else ""
        return f"{rel}:{self.line}  [{self.kind}]  {self.signature.strip()}{doc}"


# ─────────────────────────────────────────────────────────────────────────────
# WorkspaceIndex
# ─────────────────────────────────────────────────────────────────────────────

class WorkspaceIndex:
    """
    Thread-safe, mtime-aware symbol index for an entire workspace.

    Usage (typical)
    ---------------
    idx = get_workspace_index(workspace)
    # Use as symbol_fetcher in build_symbol_context:
    ctx = build_symbol_context(paths, workspace, symbol_fetcher=idx.symbols_for)
    # Or search across workspace:
    matches = idx.find_symbol("Agent")
    """

    CACHE_FILE = ".nvagent/index.json"

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._cache:  dict[str, _CacheEntry] = {}   # abs_str → entry
        self._lock    = threading.Lock()
        self._dirty   = False   # cache has unsaved changes

    # ── Core cache access ────────────────────────────────────────────────────

    def symbols_for(self, path: Path) -> SymbolIndex:
        """
        Return a SymbolIndex for *path*.
        Served from cache when the file mtime is unchanged; re-parses otherwise.
        Zero extra work for files that haven't changed.
        """
        rpath = path.resolve()
        key   = str(rpath)

        try:
            current_mtime = rpath.stat().st_mtime
        except Exception:
            return SymbolIndex(path=rpath, language="unknown")

        with self._lock:
            entry = self._cache.get(key)
            if entry and entry.mtime == current_mtime:
                return SymbolIndex(
                    path=rpath,
                    language=entry.language,
                    symbols=list(entry.symbols),
                    imports=list(entry.imports),
                )

        # Cache miss (or stale) — parse from disk
        idx = extract_symbols(rpath)

        with self._lock:
            self._cache[key] = _CacheEntry(
                mtime=current_mtime,
                language=idx.language,
                symbols=list(idx.symbols),
                imports=list(idx.imports),
            )
            self._dirty = True

        return idx

    def invalidate(self, path: Path) -> None:
        """Remove *path* from cache. Next access will re-parse it."""
        key = str(path.resolve())
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._dirty = True

    def warm(self, paths: list[Path]) -> None:
        """Pre-warm cache for the given paths (e.g. active files at session start)."""
        for p in paths:
            self.symbols_for(p)

    # ── Global symbol search ─────────────────────────────────────────────────

    def find_symbol(
        self,
        name: str,
        kinds: Optional[list[str]] = None,
    ) -> list[SymbolMatch]:
        """
        Search all cached files for symbols named *name*.
        Optionally filter by kind (e.g. ["function", "class"]).
        Returns results sorted by (file, line).
        """
        matches: list[SymbolMatch] = []

        with self._lock:
            snapshot = list(self._cache.items())

        for key, entry in snapshot:
            for sym in entry.symbols:
                if sym.name != name:
                    continue
                if kinds and sym.kind not in kinds:
                    continue
                matches.append(SymbolMatch(
                    file      = Path(key),
                    line      = sym.line,
                    kind      = sym.kind,
                    name      = sym.name,
                    signature = sym.signature,
                    docstring = sym.docstring,
                ))

        matches.sort(key=lambda m: (str(m.file), m.line))
        return matches

    def search_symbols(
        self,
        query: str,
        kinds: Optional[list[str]] = None,
        max_results: int = 50,
    ) -> list[SymbolMatch]:
        """
        Case-insensitive substring search across all symbol names.
        Useful for fuzzy lookup: search_symbols("agent") finds Agent, AgentEvent, etc.
        """
        query_lower = query.lower()
        matches: list[SymbolMatch] = []

        with self._lock:
            snapshot = list(self._cache.items())

        for key, entry in snapshot:
            if len(matches) >= max_results:
                break
            for sym in entry.symbols:
                if query_lower not in sym.name.lower():
                    continue
                if kinds and sym.kind not in kinds:
                    continue
                matches.append(SymbolMatch(
                    file      = Path(key),
                    line      = sym.line,
                    kind      = sym.kind,
                    name      = sym.name,
                    signature = sym.signature,
                    docstring = sym.docstring,
                ))

        matches.sort(key=lambda m: (len(m.name), str(m.file), m.line))
        return matches[:max_results]

    # ── Workspace build ──────────────────────────────────────────────────────

    def build(self, max_files: int = 2000) -> int:
        """
        Eagerly index all source files in the workspace.
        Skips files whose mtime hasn't changed (uses existing cache entries).
        Returns the number of files indexed (including cache hits).

        Uses a ThreadPoolExecutor so AST parsing of independent files overlaps
        with I/O wait — typically 4-8× faster than sequential on large workspaces.
        """
        dep_graph = get_dependency_graph(self.workspace)

        # Collect file list first so we can respect max_files deterministically.
        files = []
        for fpath in _iter_workspace_files(self.workspace):
            if len(files) >= max_files:
                break
            files.append(fpath)

        def _index_one(fpath: Path) -> None:
            self.symbols_for(fpath)     # prime symbol cache (thread-safe)
            dep_graph.build_file(fpath) # prime dep graph  (thread-safe)

        n_workers = min(max(1, (os.cpu_count() or 1)), 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_index_one, fp): fp for fp in files}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    pass  # bad file — silently skip, same as sequential build

        return len(files)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist the current cache to .nvagent/index.json.
        Only writes if the cache has changed since the last save.
        """
        if not self._dirty:
            return

        cache_path = self.workspace / self.CACHE_FILE
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        files_data: dict = {}
        with self._lock:
            snapshot = dict(self._cache)

        for key, entry in snapshot.items():
            # Only save workspace-local files (skip system paths)
            try:
                Path(key).relative_to(self.workspace)
            except ValueError:
                continue
            files_data[key] = {
                "mtime":    entry.mtime,
                "language": entry.language,
                "symbols":  [
                    {
                        "kind":      s.kind,
                        "name":      s.name,
                        "signature": s.signature,
                        "line":      s.line,
                        "docstring": s.docstring,
                    }
                    for s in entry.symbols
                ],
                "imports": entry.imports,
            }

        data = {
            "version":   _CACHE_VERSION,
            "workspace": str(self.workspace),
            "built_at":  time.time(),
            "files":     files_data,
        }

        try:
            # Atomic write: write to .tmp then rename to prevent partial reads
            tmp_path = cache_path.with_suffix(".tmp")
            tmp_path.write_bytes(_json_dumps(data))
            os.replace(tmp_path, cache_path)
            with self._lock:
                self._dirty = False
        except Exception:
            pass   # Persistence is best-effort

    def load(self) -> int:
        """
        Load cache from .nvagent/index.json.
        Returns the number of entries loaded.
        Entries whose on-disk mtime has changed are silently dropped (will be
        re-parsed on next access).
        """
        cache_path = self.workspace / self.CACHE_FILE
        if not cache_path.exists():
            return 0

        try:
            raw = _json_loads(cache_path.read_bytes())
        except Exception:
            return 0

        if raw.get("version") != _CACHE_VERSION:
            return 0    # Stale format — ignore

        loaded = 0
        with self._lock:
            for key, fdata in raw.get("files", {}).items():
                fpath = Path(key)
                try:
                    current_mtime = fpath.stat().st_mtime
                except Exception:
                    continue

                # Drop entries whose file has changed since the cache was saved
                if current_mtime != fdata.get("mtime"):
                    continue

                symbols = [
                    Symbol(
                        kind      = s["kind"],
                        name      = s["name"],
                        signature = s["signature"],
                        line      = s.get("line", 0),
                        docstring = s.get("docstring", ""),
                    )
                    for s in fdata.get("symbols", [])
                ]
                self._cache[key] = _CacheEntry(
                    mtime    = fdata["mtime"],
                    language = fdata.get("language", "unknown"),
                    symbols  = symbols,
                    imports  = fdata.get("imports", []),
                )
                loaded += 1

        return loaded

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> str:
        with self._lock:
            n_files   = len(self._cache)
            n_symbols = sum(len(e.symbols) for e in self._cache.values())
        return f"{n_files} files cached, {n_symbols} symbols indexed"

    def cached_files(self) -> list[Path]:
        with self._lock:
            return [Path(k) for k in self._cache]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton  (one index per workspace path)
# ─────────────────────────────────────────────────────────────────────────────

_INDICES:      dict[str, WorkspaceIndex] = {}
_INDICES_LOCK  = threading.Lock()


def get_workspace_index(workspace: Path) -> WorkspaceIndex:
    """
    Return (and create if needed) the singleton WorkspaceIndex for *workspace*.
    Automatically loads the on-disk cache on first call.
    Wires the DependencyGraph to use this index's symbol cache, eliminating
    redundant extract_symbols() calls when both are used together.
    """
    key = str(workspace.resolve())
    with _INDICES_LOCK:
        if key not in _INDICES:
            idx = WorkspaceIndex(workspace)
            idx.load()          # instant cold-start from .nvagent/index.json
            _INDICES[key] = idx
            # Share symbol cache with DependencyGraph so both singletons
            # call extract_symbols() at most once per file per mtime change.
            get_dependency_graph(workspace)._symbol_fetcher = idx.symbols_for
        return _INDICES[key]
