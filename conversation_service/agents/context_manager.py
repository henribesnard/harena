"""Simple context manager to maintain conversational state."""

from typing import Any, Dict, List


class ContextManager:
    """Maintain conversation history between user and agents."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full conversation history."""
        return list(self._history)

    def clear(self) -> None:
        """Clear the conversation history."""
        self._history.clear()
