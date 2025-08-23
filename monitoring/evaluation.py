"""Utilities to evaluate intent detection and extraction.

This module provides helpers to run prompts through a model function and
collect metrics such as latency, token usage and success rate.  It is
used by scripts/evaluate_templates.py and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

import tiktoken

# Prices in dollars per token (approx for gpt-4o-mini)
PROMPT_TOKEN_PRICE = 0.00000015  # $0.15 per 1M tokens
COMPLETION_TOKEN_PRICE = 0.00000060  # $0.60 per 1M tokens


@dataclass
class Sample:
    """Single evaluation example."""

    prompt: str
    expected_intent: str
    expected_entities: Mapping[str, Any]


def _count_tokens(text: str, model: str) -> int:
    """Return token count for *text* using *model* encoding."""

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return len(text.split())
    return len(enc.encode(text))


def evaluate_variant(
    samples: Sequence[Sample],
    template: str,
    call_model: Callable[[str], Mapping[str, Any]],
    *,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Evaluate *template* on *samples* using *call_model*.

    ``call_model`` must accept a prompt string and return a mapping with at
    least the keys ``intent``, ``entities`` and ``raw_response``.
    """

    results: List[Dict[str, Any]] = []
    for sample in samples:
        full_prompt = template.format(prompt=sample.prompt)
        start = time.perf_counter()
        output = call_model(full_prompt)
        latency = time.perf_counter() - start

        prompt_tokens = _count_tokens(full_prompt, model)
        completion_tokens = _count_tokens(output.get("raw_response", ""), model)

        success = (
            output.get("intent") == sample.expected_intent
            and output.get("entities") == sample.expected_entities
        )
        cost = (
            prompt_tokens * PROMPT_TOKEN_PRICE
            + completion_tokens * COMPLETION_TOKEN_PRICE
        )

        results.append(
            {
                "prompt": sample.prompt,
                "intent": sample.expected_intent,
                "prompt_length": len(full_prompt),
                "latency": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "success": success,
            }
        )
    return results


def aggregate_metrics(
    results_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate metrics for each variant."""

    summary: Dict[str, Any] = {}
    for variant, results in results_by_variant.items():
        total = len(results)
        successes = sum(1 for r in results if r["success"])
        avg_latency = sum(r["latency"] for r in results) / total if total else 0
        avg_prompt_len = (
            sum(r["prompt_length"] for r in results) / total if total else 0
        )
        avg_cost = sum(r["cost"] for r in results) / total if total else 0

        # Accuracy per intent
        intent_totals: Dict[str, int] = {}
        intent_success: Dict[str, int] = {}
        for r in results:
            intent = r["intent"]
            intent_totals[intent] = intent_totals.get(intent, 0) + 1
            if r["success"]:
                intent_success[intent] = intent_success.get(intent, 0) + 1
        accuracy_by_intent = {
            intent: intent_success.get(intent, 0) / total
            for intent, total in intent_totals.items()
        }

        summary[variant] = {
            "success_rate": successes / total if total else 0,
            "avg_latency": avg_latency,
            "avg_prompt_len": avg_prompt_len,
            "avg_cost": avg_cost,
            "accuracy_by_intent": accuracy_by_intent,
        }
    return summary


def best_variant(summary: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the variant name with highest success rate."""

    return max(summary.items(), key=lambda kv: kv[1]["success_rate"])[0]
