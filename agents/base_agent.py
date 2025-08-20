import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from .metrics import metrics


class BaseAgent(ABC):
    """Base class providing a common interface for agents.

    Handles instrumentation via logging and metrics collection.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(name)

    def process(self, *args, **kwargs) -> Any:
        """Public entry point for agent execution with instrumentation."""
        start_time = time.time()
        self.logger.debug("Starting processing")
        try:
            result = self._process(*args, **kwargs)
            duration = time.time() - start_time
            metrics.increment(f"{self.name}.calls")
            metrics.observe(f"{self.name}.latency", duration)
            self.logger.info("Processing completed", extra={"duration": duration})
            return result
        except Exception:
            duration = time.time() - start_time
            metrics.increment(f"{self.name}.errors")
            metrics.observe(f"{self.name}.latency", duration)
            self.logger.exception("Processing failed", extra={"duration": duration})
            raise

    @abstractmethod
    def _process(self, *args, **kwargs) -> Any:
        """Concrete implementation of agent logic."""
        raise NotImplementedError

    @staticmethod
    def get_metrics() -> Dict[str, Dict[str, float]]:
        """Expose collected metrics for external reporting."""
        return metrics.export()
