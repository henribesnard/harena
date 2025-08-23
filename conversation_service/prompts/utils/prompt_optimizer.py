"""Prompt optimization utilities for conversation intents.

This module provides helpers to analyze prompt length and cost per
intent, detect redundant sections across prompts, and suggest
alternatives for A/B testing.  It also exposes real-time Prometheus
metrics to monitor potential token and cost savings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from prometheus_client import Counter

from conversation_service.config.openai_config import DEFAULT_OPENAI_CONFIG

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

TOKENS_SAVED = Counter(
    "prompt_optimizer_tokens_saved_total",
    "Total number of prompt tokens saved after optimisation",
)
COST_SAVED = Counter(
    "prompt_optimizer_cost_saved_usd_total",
    "Total USD saved thanks to prompt optimisation",
)
OPTIMIZATION_CALLS = Counter(
    "prompt_optimizer_calls_total", "Number of optimisation evaluations run"
)


@dataclass
class PromptAnalysis:
    """Summary information about a single prompt."""

    intent: str
    tokens: int
    cost: float
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Very small tokenizer splitting on words and punctuation."""

    return re.findall(r"\w+|[^\w\s]", text)


def analyse_prompts(
    prompts: Dict[str, str], model: str = "gpt-4o-mini"
) -> Dict[str, PromptAnalysis]:
    """Return token/cost analysis for *prompts* indexed by intent.

    Parameters
    ----------
    prompts:
        Mapping of intent -> prompt content.
    model:
        Name of the model used to determine cost per token.  Defaults to
        ``gpt-4o-mini`` as defined in :mod:`conversation_service.config.openai_config`.
    """

    model_cfg = DEFAULT_OPENAI_CONFIG.get_model(model)
    cost_per_token = model_cfg.prompt_cost_per_million / 1_000_000
    analyses: Dict[str, PromptAnalysis] = {}
    for intent, prompt in prompts.items():
        tokens = len(_tokenize(prompt))
        cost = tokens * cost_per_token
        suggestions: List[str] = []
        if tokens > 150:
            suggestions.append(
                "Prompt length exceeds 150 tokens; consider compressing the description."
            )
        analyses[intent] = PromptAnalysis(intent, tokens, cost, suggestions)
    return analyses


def suggest_compressions(
    analyses: Dict[str, PromptAnalysis], max_tokens: int = 120
) -> Dict[str, List[str]]:
    """Generate compression suggestions for each analysed prompt."""

    suggestions: Dict[str, List[str]] = {}
    for intent, analysis in analyses.items():
        tips = list(analysis.suggestions)
        if analysis.tokens > max_tokens:
            tips.append(f"Reduce prompt to under {max_tokens} tokens where possible.")
        suggestions[intent] = tips
    return suggestions


# ---------------------------------------------------------------------------
# Redundancy detection and A/B testing helpers
# ---------------------------------------------------------------------------


def identify_redundant_sections(prompts: Dict[str, str]) -> Dict[str, List[str]]:
    """Return sections appearing in more than one prompt.

    The returned mapping is ``section -> [intents...]``.
    """

    sections: Dict[str, List[str]] = {}
    for intent, prompt in prompts.items():
        # Split into sentences and normalise spacing
        pieces = {
            s.strip()
            for s in re.split(r"[\n\.]+", prompt)
            if s.strip()
        }
        for piece in pieces:
            sections.setdefault(piece, []).append(intent)
    return {s: i for s, i in sections.items() if len(i) > 1}


def generate_variants(prompt: str) -> List[str]:
    """Produce naive prompt variants for A/B testing."""

    variants = {prompt}
    if "\n" in prompt:
        variants.add(prompt.replace("\n", " "))
        variants.add("\n".join(line.strip() for line in prompt.splitlines()))
    if "Please" in prompt:
        variants.add(prompt.replace("Please", "Kindly"))
    if len(prompt) > 100:
        variants.add(prompt[:100].rstrip() + "...")
    return list(variants)


# ---------------------------------------------------------------------------
# ROI and monitoring
# ---------------------------------------------------------------------------


def calculate_roi(baseline_cost: float, optimised_cost: float) -> float:
    """Return the ROI ratio given baseline and optimised costs."""

    if optimised_cost <= 0:
        return 0.0
    return (baseline_cost - optimised_cost) / optimised_cost


def record_optimisation(
    baseline_tokens: int, optimised_tokens: int, model: str = "gpt-4o-mini"
) -> Dict[str, float]:
    """Record optimisation metrics and return cost/ROI information."""

    model_cfg = DEFAULT_OPENAI_CONFIG.get_model(model)
    cost_per_token = model_cfg.prompt_cost_per_million / 1_000_000

    baseline_cost = baseline_tokens * cost_per_token
    optimised_cost = optimised_tokens * cost_per_token
    tokens_saved = max(baseline_tokens - optimised_tokens, 0)
    cost_saved = max(baseline_cost - optimised_cost, 0.0)

    OPTIMIZATION_CALLS.inc()
    if tokens_saved:
        TOKENS_SAVED.inc(tokens_saved)
    if cost_saved:
        COST_SAVED.inc(cost_saved)

    roi = calculate_roi(baseline_cost, optimised_cost)
    return {
        "baseline_cost": baseline_cost,
        "optimised_cost": optimised_cost,
        "tokens_saved": tokens_saved,
        "cost_saved": cost_saved,
        "roi": roi,
    }


__all__ = [
    "PromptAnalysis",
    "analyse_prompts",
    "suggest_compressions",
    "identify_redundant_sections",
    "generate_variants",
    "calculate_roi",
    "record_optimisation",
]
