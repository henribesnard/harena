from typing import Dict

from .base_agent import BaseAgent


class QueryGenerator(BaseAgent):
    """Generates a query string based on intent and entities."""

    def __init__(self) -> None:
        super().__init__("query_generator")

    def _process(self, intent: str, entities: Dict[str, str]) -> str:
        if intent == "get_stock_price" and "number" in entities:
            query = f"SELECT price FROM stocks WHERE id={entities['number']}"
        elif intent == "get_weather":
            query = "SELECT * FROM weather"
        else:
            query = ""
        self.logger.debug("Generated query", extra={"query": query})
        return query
