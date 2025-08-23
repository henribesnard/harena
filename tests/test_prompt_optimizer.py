import pytest

from conversation_service.prompts.utils import prompt_optimizer as po


def test_analyse_and_suggest():
    prompts = {"GREETING": "Hello world"}
    analyses = po.analyse_prompts(prompts)
    assert "GREETING" in analyses
    analysis = analyses["GREETING"]
    assert analysis.tokens == 2
    expected_cost = 2 * 0.15 / 1_000_000
    assert analysis.cost == pytest.approx(expected_cost)

    suggestions = po.suggest_compressions(analyses, max_tokens=1)
    assert "GREETING" in suggestions
    assert any("Reduce prompt" in s for s in suggestions["GREETING"])


def test_redundant_sections_and_variants():
    prompts = {
        "A": "Intro. Common part.",
        "B": "Different. Common part.",
    }
    redundant = po.identify_redundant_sections(prompts)
    assert "Common part" in redundant
    assert set(redundant["Common part"]) == {"A", "B"}

    variants = po.generate_variants("Please stay.\nThank you")
    assert len(variants) >= 2
    assert any("Kindly" in v for v in variants)


def test_record_optimisation_metrics():
    po.TOKENS_SAVED._value.set(0)
    po.COST_SAVED._value.set(0)
    po.OPTIMIZATION_CALLS._value.set(0)

    metrics = po.record_optimisation(100, 80)
    assert metrics["tokens_saved"] == 20
    assert metrics["cost_saved"] == pytest.approx(20 * 0.15 / 1_000_000)
    assert metrics["roi"] > 0
    assert po.TOKENS_SAVED._value.get() == 20
    assert po.COST_SAVED._value.get() == pytest.approx(20 * 0.15 / 1_000_000)
    assert po.OPTIMIZATION_CALLS._value.get() == 1
