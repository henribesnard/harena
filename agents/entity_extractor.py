import re
from typing import Dict

from .base_agent import BaseAgent


class EntityExtractor(BaseAgent):
    """Extracts simple entities like numbers from text."""

    def __init__(self) -> None:
        super().__init__("entity_extractor")

    def _process(self, text: str) -> Dict[str, str]:
        entities: Dict[str, str] = {}
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if match:
            entities["number"] = match.group(1)
            self.logger.debug("Extracted number entity", extra={"value": match.group(1)})
        return entities
