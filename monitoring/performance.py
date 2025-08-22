"""Simple performance monitoring utilities."""
from __future__ import annotations

import contextlib
import logging
import time

logger = logging.getLogger("performance")

@contextlib.contextmanager
def track_operation(name: str):
    """Context manager to measure and log operation duration in ms."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = (time.perf_counter() - start) * 1000
        logger.info("%s took %.2fms", name, duration)

_total_openai_cost = 0.0

def record_openai_cost(cost: float) -> None:
    """Record and log cumulative OpenAI API cost."""
    global _total_openai_cost
    _total_openai_cost += float(cost)
    logger.info("OpenAI cost +%.4f -> total %.4f", cost, _total_openai_cost)
