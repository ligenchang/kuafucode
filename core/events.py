"""Core event definitions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentEvent:
    """Typed event emitted by the Agent loop."""
    type: str
    data: Any