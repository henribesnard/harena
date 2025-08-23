from monitoring.evaluation import (
    Sample,
    aggregate_metrics,
    best_variant,
    evaluate_variant,
)


def dummy_model(prompt: str):
    if "bonjour" in prompt:
        return {"intent": "GREETING", "entities": {}, "raw_response": "{}"}
    return {
        "intent": "SPENDING_ANALYSIS",
        "entities": {"merchant": "Carrefour"},
        "raw_response": "{}",
    }


def test_aggregate_metrics():
    samples = [
        Sample(prompt="bonjour", expected_intent="GREETING", expected_entities={}),
        Sample(
            prompt="Combien j'ai dépensé chez Carrefour ?",
            expected_intent="SPENDING_ANALYSIS",
            expected_entities={"merchant": "Carrefour"},
        ),
    ]
    results = evaluate_variant(samples, "{prompt}", dummy_model)
    summary = aggregate_metrics({"A": results})
    assert summary["A"]["success_rate"] == 1.0
    assert best_variant(summary) == "A"
