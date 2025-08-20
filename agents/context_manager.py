from typing import Dict, List

from .base_agent import BaseAgent


class ContextManager(BaseAgent):
    """Manages simple in-memory conversation context."""

    def __init__(self) -> None:
        super().__init__("context_manager")
        self._context: Dict[str, List[str]] = {}

    def _process(self, user_id: str, message: str) -> Dict[str, List[str]]:
        history = self._context.setdefault(user_id, [])
        history.append(message)
        self.logger.debug(
            "Updated context", extra={"user_id": user_id, "messages": len(history)}
        )
        return {"user_id": user_id, "history": list(history)}
