"""Data models for safety validation and violations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Violation:
    """A safety limit that was breached."""

    kind: str  # "loop" | "resource" | "validation" | "tests"
    message: str
    fatal: bool = True  # fatal → abort task; non-fatal → warn only

    def __str__(self) -> str:
        tag = "FATAL" if self.fatal else "WARNING"
        return f"[Safety {tag} — {self.kind}] {self.message}"


@dataclass
class ValidationResult:
    """Result of a code validation check."""

    ok: bool
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_str(self) -> str:
        if self.ok and not self.warnings:
            return f"✓ {self.path}: validation passed"
        lines = [f"{'✓' if self.ok else '✗'} {self.path}:"]
        for e in self.errors:
            lines.append(f"  ✗ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)
