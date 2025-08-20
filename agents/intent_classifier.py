from .base_agent import BaseAgent
from .metrics import metrics


class IntentClassifier(BaseAgent):
    """Simple keyword-based intent classifier."""

    def __init__(self) -> None:
        super().__init__("intent_classifier")

    def _process(self, text: str) -> str:
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["price", "stock"]):
            intent = "get_stock_price"
        elif "weather" in text_lower:
            intent = "get_weather"
        else:
            intent = "unknown"
        # Additional metrics per intent
        metrics.increment(f"intent_classifier.intent.{intent}")
        self.logger.debug("Classified intent", extra={"intent": intent})
        return intent
