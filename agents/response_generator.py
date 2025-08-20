from typing import Any

from .base_agent import BaseAgent


class ResponseGenerator(BaseAgent):
    """Generates a textual response from data."""

    def __init__(self) -> None:
        super().__init__("response_generator")

    def _process(self, data: Any) -> str:
        response = f"Result: {data}"
        self.logger.debug("Generated response", extra={"response": response})
        return response
