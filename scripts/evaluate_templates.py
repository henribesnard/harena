"""Evaluate prompt templates against intent fixtures.

This script loads a fixture file describing template variants and
examples with expected intents and entities.  For each template variant it
runs the examples through an LLM (OpenAI by default) and reports metrics
such as accuracy by intent, average latency and token cost.

Example:
    python scripts/evaluate_templates.py tests/fixtures/intent_samples.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Mapping

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from monitoring.evaluation import (
    Sample,
    aggregate_metrics,
    best_variant,
    evaluate_variant,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai optional for tests
    OpenAI = None  # type: ignore


def _openai_call(prompt: str, model: str, api_key: str | None = None) -> Mapping[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not available")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY manquante : chargez-la depuis .env ou exportez-la."
        )
    client = OpenAI()
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    resp = client.responses.create(model=model, input=prompt)
    text = resp.output_text
    data = json.loads(text)
    return {
        "intent": data.get("intent"),
        "entities": data.get("entities", {}),
        "raw_response": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate template variants")
    parser.add_argument("fixture", help="JSON fixture with templates and samples")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--api-key", help="OpenAI API key", default=None)
    args = parser.parse_args()

    with open(args.fixture, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    templates: Dict[str, str] = data["template_variants"]
    samples = [Sample(**s) for s in data["samples"]]

    results_by_variant: Dict[str, Any] = {}
    for name, template in templates.items():
        results_by_variant[name] = evaluate_variant(
            samples,
            template,
            lambda p: _openai_call(p, args.model, args.api_key),
            model=args.model,
        )

    summary = aggregate_metrics(results_by_variant)
    for name, metrics in summary.items():
        print(
            f"Variant {name}: success={metrics['success_rate']:.2%}, "
            f"latency={metrics['avg_latency']*1000:.1f}ms, "
            f"cost=${metrics['avg_cost']:.6f}"
        )
        for intent, acc in metrics["accuracy_by_intent"].items():
            print(f"  - {intent}: {acc:.2%}")

    winner = best_variant(summary)
    print(f"\nBest variant: {winner}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
