"""Symbol extraction data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Symbol:
    """A code symbol (function, class, import, type, const)."""

    kind: str  # "function" | "class" | "method" | "import" | "type" | "const"
    name: str
    signature: str  # one-line human-readable signature
    line: int = 0
    docstring: str = ""

    def __str__(self) -> str:
        doc = f"  # {self.docstring[:80]}" if self.docstring else ""
        return f"{self.signature}{doc}"


@dataclass
class SymbolIndex:
    """Index of symbols in a file."""

    path: Path
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # raw import strings

    def functions(self) -> list[Symbol]:
        """Return all functions and methods."""
        return [s for s in self.symbols if s.kind in ("function", "method")]

    def classes(self) -> list[Symbol]:
        """Return all class definitions."""
        return [s for s in self.symbols if s.kind == "class"]

    def types(self) -> list[Symbol]:
        """Return all type definitions."""
        return [s for s in self.symbols if s.kind == "type"]

    def is_empty(self) -> bool:
        """Check if index has any symbols or imports."""
        return len(self.symbols) == 0 and len(self.imports) == 0

    def render(self, max_symbols: int = 60) -> str:
        """Render as a compact text block suitable for LLM context."""
        lines: list[str] = []
        shown = 0
        for sym in self.symbols:
            if shown >= max_symbols:
                remaining = len(self.symbols) - shown
                lines.append(f"  ... ({remaining} more symbols)")
                break
            lines.append(f"  {sym}")
            shown += 1
        return "\n".join(lines)
