# tui/ui.py
from textual.widgets import ScrollableContainer


class UnifiedView(ScrollableContainer):
    """
    A single scrollable view that merges chat, diff, and tool output.
    """

    def __init__(self) -> None:
        super().__init__(id="unified")
        self._content: str = ""

    def append_chat(self, text: str) -> None:
        """Append a chat/message line."""
        self._content += f"[CHAT] {text}\n"
        self.write(self._content)

    def append_diff(self, diff: str) -> None:
        """Append a diff block."""
        self._content += f"[DIFF] {diff}\n"
        self.write(self._content)

    def append_tool(self, out: str) -> None:
        """Append tool output."""
        self._content += f"[TOOL] {out}\n"
        self.write(self._content)
