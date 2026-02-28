"""Semantic file retrieval and indexing.

Exports:
  - retrieve_files - BM25-based file retrieval
  - RetrievalIndex - Workspace file index with incremental building
  - ScoredFile - Scored file result
  - get_retrieval_index - Retrieval index singleton accessor
  - WorkspaceIndex - Symbol index across all files
"""

from .core import retrieve_files, RetrievalIndex, ScoredFile, get_retrieval_index
from .index import WorkspaceIndex

__all__ = [
    "retrieve_files",
    "RetrievalIndex",
    "ScoredFile",
    "get_retrieval_index",
    "WorkspaceIndex",
]
