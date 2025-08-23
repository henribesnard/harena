"""Prompt templates and helper utilities for query generation.

This module defines the system message for the ``QueryGenerator`` assistant. The
message maps user intentions to query templates and explains optimisation
strategies such as adaptive filtering, fallbacks and timeout handling.

It also provides a few-shot dataset showcasing multi-criteria queries involving
date ranges, amount thresholds and aggregation operations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List

# ---------------------------------------------------------------------------
# Intent to query mapping
# ---------------------------------------------------------------------------

INTENT_QUERY_MAP: Dict[str, str] = {
    "transactions_by_date": (
        "SELECT * FROM transactions "
        "WHERE date BETWEEN :start_date AND :end_date"
    ),
    "sum_by_category": (
        "SELECT category, SUM(amount) AS total "
        "FROM transactions WHERE date BETWEEN :start_date AND :end_date "
        "GROUP BY category"
    ),
    "average_amount": (
        "SELECT AVG(amount) FROM transactions "
        "WHERE date BETWEEN :start_date AND :end_date"
    ),
}

QUERY_GENERATOR_SYSTEM_MESSAGE = (
    "You are QueryGenerator, an assistant that converts high level intentions "
    "into SQL-like queries.\n"
    "Available intentions and their query templates:\n"
    "- transactions_by_date: SELECT * FROM transactions "
    "WHERE date BETWEEN :start_date AND :end_date\n"
    "- sum_by_category: SELECT category, SUM(amount) AS total FROM transactions "
    "WHERE date BETWEEN :start_date AND :end_date GROUP BY category\n"
    "- average_amount: SELECT AVG(amount) FROM transactions WHERE date BETWEEN "
    ":start_date AND :end_date\n\n"
    "Optimisation strategies:\n"
    "- Apply adaptive filters: only include predicates supplied by the user.\n"
    "- Provide a safe fallback query when the intent is unknown or malformed.\n"
    "- Abort generation if it exceeds the timeout and return the fallback."
)

# ---------------------------------------------------------------------------
# Few-shot examples for multi-criteria queries
# ---------------------------------------------------------------------------

@dataclass
class FewShotExample:
    """A single training example for the query generator."""

    user: str
    intent: str
    query: str


FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    FewShotExample(
        user="Show expenses above 1000 between 2024-01-01 and 2024-03-31",
        intent="transactions_by_date",
        query=(
            "SELECT * FROM transactions WHERE amount > 1000 "
            "AND date BETWEEN '2024-01-01' AND '2024-03-31'"
        ),
    ),
    FewShotExample(
        user="Total sales per category for March 2024",
        intent="sum_by_category",
        query=(
            "SELECT category, SUM(amount) AS total FROM transactions "
            "WHERE date BETWEEN '2024-03-01' AND '2024-03-31' GROUP BY category"
        ),
    ),
    FewShotExample(
        user="Average donation over 500 last year",
        intent="average_amount",
        query=(
            "SELECT AVG(amount) FROM transactions "
            "WHERE amount > 500 AND date BETWEEN '2023-01-01' AND '2023-12-31'"
        ),
    ),
]

# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def apply_adaptive_filters(base_query: str, filters: Dict[str, Any]) -> str:
    """Attach ``filters`` to ``base_query`` when values are provided.

    Parameters
    ----------
    base_query:
        The query template without optional conditions.
    filters:
        Mapping of column names to values. ``None`` values are ignored.

    Returns
    -------
    str
        The query with added conditions.
    """

    conditions: List[str] = []
    for column, value in filters.items():
        if value is not None:
            conditions.append(f"{column} = :{column}")
    if not conditions:
        return base_query
    connector = " WHERE " if "WHERE" not in base_query.upper() else " AND "
    return base_query + connector + " AND ".join(conditions)

def fallback_query() -> str:
    """Return a minimal fallback query."""

    return "SELECT 1"

async def run_with_timeout(
    fn: Callable[[], Awaitable[str]],
    timeout: float,
    *,
    fallback: str | None = None,
) -> str:
    """Execute ``fn`` with ``timeout`` seconds and use ``fallback`` on timeout."""

    try:
        return await asyncio.wait_for(fn(), timeout)
    except asyncio.TimeoutError:
        return fallback or fallback_query()

