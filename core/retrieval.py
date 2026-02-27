"""
Semantic file retrieval using BM25 (zero external dependencies).

Indexes workspace files over:
  - file path tokens         (high weight — filename/directory names are very informative)
  - symbol names             (from WorkspaceIndex — functions, classes, types)
  - content snippets         (first 300 chars of each file)
  - import names             (what the file imports / depends on)

BM25 parameters (k1=1.5, b=0.75) matching standard literature defaults.

Public API
----------
  retrieve_files(query, workspace, top_k, score_threshold) → list[ScoredFile]
  score_files(query, paths, workspace)                     → list[ScoredFile]
  get_retrieval_index(workspace)                           → RetrievalIndex  (singleton)
  RetrievalIndex.build(max_files)                          → int
  RetrievalIndex.invalidate(path)
  RetrievalIndex.search(query, top_k)                      → list[ScoredFile]
"""

from __future__ import annotations

import heapq
import math
import re
import threading
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import WorkspaceIndex to get per-file symbols
from nvagent.core.index import get_workspace_index
from nvagent.core.symbols import _iter_workspace_files


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{1,}|[0-9]+")


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase tokens.
    Also splits CamelCase identifiers (e.g. WorkspaceIndex → workspace, index).
    """
    raw = _TOKEN_RE.findall(text)
    expanded: list[str] = []
    for tok in raw:
        lower = tok.lower()
        expanded.append(lower)
        # Split CamelCase: WorkspaceIndex → workspace index
        parts = re.sub(r"([A-Z][a-z]+)", r" \1", tok).split()
        for p in parts:
            pl = p.lower()
            if pl != lower and len(pl) > 2:
                expanded.append(pl)
    return expanded


def _path_tokens(path: Path, workspace: Path) -> list[str]:
    """Extract tokens from a file path — path segments, stem, extension."""
    try:
        rel = path.relative_to(workspace)
    except ValueError:
        rel = path
    tokens: list[str] = []
    for part in rel.parts:
        tokens.extend(_tokenize(part))
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# BM25 implementation
# ─────────────────────────────────────────────────────────────────────────────

_BM25_K1 = 1.5
_BM25_B  = 0.75


class _BM25Scorer:
    """
    In-memory BM25 index over a document corpus.
    Each "document" is identified by an integer doc_id.
    """

    def __init__(self) -> None:
        self._docs:   list[list[str]] = []         # doc_id → token list
        self._df:     dict[str, int]  = {}         # term → document frequency
        self._avg_dl: float           = 1.0
        self._built:  bool            = False

    def add_document(self, tokens: list[str]) -> int:
        """Add a document (list of tokens). Returns doc_id."""
        doc_id = len(self._docs)
        self._docs.append(tokens)
        for term in set(tokens):
            self._df[term] = self._df.get(term, 0) + 1
        return doc_id

    def build(self) -> None:
        """Pre-compute average document length (call after all docs added)."""
        if not self._docs:
            self._avg_dl = 1.0
            return
        self._avg_dl = sum(len(d) for d in self._docs) / len(self._docs)
        self._built = True

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        """Return BM25 score for a query against a document."""
        if not self._built or doc_id >= len(self._docs):
            return 0.0
        doc    = self._docs[doc_id]
        dl     = len(doc)
        n      = len(self._docs)
        score  = 0.0
        tf_map: dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term in set(query_tokens):
            df  = self._df.get(term, 0)
            if df == 0:
                continue
            tf  = tf_map.get(term, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (_BM25_K1 + 1)) / (
                tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / self._avg_dl)
            )
            score += idf * tf_norm

        return score

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        """Return top-k (doc_id, score) pairs, sorted descending.

        Uses heapq.nlargest → O(n log k) vs O(n log n) for a full sort.
        The generator avoids allocating the full scored list before selection.
        """
        if not self._built:
            return []
        scored = (
            (doc_id, self.score(query_tokens, doc_id))
            for doc_id in range(len(self._docs))
        )
        top = heapq.nlargest(k, scored, key=lambda x: x[1])
        return [(did, s) for did, s in top if s > 0]


# ─────────────────────────────────────────────────────────────────────────────
# ScoredFile  (result type)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredFile:
    path:  Path
    score: float
    reason: str = ""   # human-readable explanation of why it was retrieved

    def render(self, workspace: Optional[Path] = None) -> str:
        try:
            ws_r = workspace.resolve() if workspace else None
            rel  = self.path.relative_to(ws_r) if ws_r else self.path
        except ValueError:
            rel = self.path
        reason_str = f"  # {self.reason}" if self.reason else ""
        return f"{rel}  ({self.score:.2f}){reason_str}"


# ─────────────────────────────────────────────────────────────────────────────
# RetrievalIndex
# ─────────────────────────────────────────────────────────────────────────────

# Field weights: how much each document field contributes to the token bag
_FIELD_WEIGHTS = {
    "path":    4,   # filename/directory names are very informative
    "symbols": 3,   # function/class names
    "imports": 2,   # what it imports (relates via dependency)
    "content": 1,   # raw content snippets
}


class RetrievalIndex:
    """
    Workspace-wide BM25 retrieval index.
    Each file is a weighted "document" composed of its path tokens,
    symbol names, import names, and content snippet.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace  = workspace.resolve()
        self._paths:    list[Path]    = []       # doc_id → Path
        self._bm25:     _BM25Scorer   = _BM25Scorer()
        self._path_idx: dict[str, int] = {}      # abs_str → doc_id
        self._lock      = threading.Lock()
        self._built     = False
        self._dirty     = False

    def build(self, max_files: int = 2000) -> int:
        """
        Index all workspace source files.
        Uses WorkspaceIndex for symbol info (cheap — uses its mtime cache).
        Returns number of files indexed.
        """
        ws_index = get_workspace_index(self.workspace)

        # Collect file list first (generator is not thread-safe to share)
        all_files: list[Path] = []
        for fpath in _iter_workspace_files(self.workspace):
            if len(all_files) >= max_files:
                break
            all_files.append(fpath)

        # Build token bags in parallel — I/O + symbol lookups dominate here
        def _tokens_safe(fpath: Path):
            try:
                return fpath, self._file_tokens(fpath, ws_index)
            except Exception:
                return fpath, None

        n_workers = min(16, max(1, len(all_files)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            token_results = list(ex.map(_tokens_safe, all_files))

        bm25     = _BM25Scorer()
        paths:   list[Path]    = []
        path_idx: dict[str, int] = {}

        for fpath, tokens in token_results:
            if tokens is None:
                continue
            doc_id = bm25.add_document(tokens)
            paths.append(fpath)
            path_idx[str(fpath.resolve())] = doc_id

        bm25.build()

        with self._lock:
            self._paths   = paths
            self._bm25    = bm25
            self._path_idx = path_idx
            self._built   = True

        return len(paths)

    def invalidate(self, path: Path) -> None:
        """
        Mark a file as stale.  The full index will be rebuilt lazily on the next
        search call if any file has been invalidated.
        (Full rebuild is fast — BM25 over symbols/paths, no I/O per file.)
        """
        with self._lock:
            self._dirty = True

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.1,
    ) -> list[ScoredFile]:
        """
        Return top-k files most relevant to *query*.
        Rebuilds index lazily if dirty.
        """
        with self._lock:
            needs_build = not self._built or self._dirty

        if needs_build:
            self.build()
            with self._lock:
                self._dirty = False

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            top = self._bm25.top_k(query_tokens, top_k * 2)
            paths = list(self._paths)

        results: list[ScoredFile] = []
        for doc_id, score in top:
            if score < score_threshold or doc_id >= len(paths):
                continue
            results.append(ScoredFile(path=paths[doc_id], score=score))
            if len(results) >= top_k:
                break

        return results

    def score_paths(
        self,
        query: str,
        paths: list[Path],
    ) -> list[ScoredFile]:
        """
        Score and rank an existing list of paths against *query*.
        Useful for re-ranking already-known active files.
        """
        with self._lock:
            needs_build = not self._built or self._dirty

        if needs_build:
            self.build()

        query_tokens = _tokenize(query)
        results: list[ScoredFile] = []

        with self._lock:
            for p in paths:
                key    = str(p.resolve())
                doc_id = self._path_idx.get(key)
                if doc_id is None:
                    # File not indexed yet — give it a path-only score
                    pt = _path_tokens(p, self.workspace)
                    overlap = len(set(query_tokens) & set(pt))
                    score = overlap / max(len(query_tokens), 1) * 2.0
                else:
                    score = self._bm25.score(query_tokens, doc_id)
                results.append(ScoredFile(path=p, score=score))

        results.sort(key=lambda s: s.score, reverse=True)
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _file_tokens(self, path: Path, ws_index) -> list[str]:
        """Build the weighted token bag for a single file."""
        tokens: list[str] = []

        # Path tokens (high weight)
        pt = _path_tokens(path, self.workspace)
        tokens.extend(pt * _FIELD_WEIGHTS["path"])

        # Symbol tokens (use index cache)
        try:
            sym_idx = ws_index.symbols_for(path)
            sym_tokens = []
            for sym in sym_idx.symbols:
                sym_tokens.extend(_tokenize(sym.name))
                sym_tokens.extend(_tokenize(sym.signature[:80]))
            tokens.extend(sym_tokens * _FIELD_WEIGHTS["symbols"])

            # Import tokens
            import_tokens: list[str] = []
            for imp in sym_idx.imports[:20]:
                import_tokens.extend(_tokenize(imp))
            tokens.extend(import_tokens * _FIELD_WEIGHTS["imports"])
        except Exception:
            pass

        # Content snippet tokens
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:500]
            tokens.extend(_tokenize(content) * _FIELD_WEIGHTS["content"])
        except Exception:
            pass

        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_RETRIEVAL_INDICES:      dict[str, RetrievalIndex] = {}
_RETRIEVAL_INDICES_LOCK  = threading.Lock()


def get_retrieval_index(workspace: Path) -> RetrievalIndex:
    """Return (and create if needed) the singleton RetrievalIndex for *workspace*."""
    key = str(workspace.resolve())
    with _RETRIEVAL_INDICES_LOCK:
        if key not in _RETRIEVAL_INDICES:
            _RETRIEVAL_INDICES[key] = RetrievalIndex(workspace)
        return _RETRIEVAL_INDICES[key]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_files(
    query: str,
    workspace: Path,
    top_k: int = 8,
    score_threshold: float = 0.1,
) -> list[ScoredFile]:
    """
    Return the top-k workspace files most relevant to *query*.
    Uses the singleton retrieval index (built lazily on first call).
    """
    return get_retrieval_index(workspace).search(
        query, top_k=top_k, score_threshold=score_threshold
    )


def score_files(
    query: str,
    paths: list[Path],
    workspace: Path,
) -> list[ScoredFile]:
    """
    Rank *paths* by relevance to *query*.
    Useful for prioritizing which active files to include in context.
    """
    return get_retrieval_index(workspace).score_paths(query, paths)
