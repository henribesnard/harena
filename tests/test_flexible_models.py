import pytest
from decimal import Decimal
from datetime import date

from search_service.models import (
    FlexibleFinancialTransaction,
    DynamicSpendingAnalysis,
    FlexibleSearchCriteria,
    LLMExtractedInsights,
)


def test_transaction_amount_decimal_and_user_validation():
    tx = FlexibleFinancialTransaction(
        user_id=1,
        amount="10.50",
        currency_code="EUR",
        date=date(2024, 1, 1),
    )
    assert isinstance(tx.amount, Decimal)

    with pytest.raises(ValueError):
        FlexibleFinancialTransaction(user_id=0, amount="1", currency_code="EUR", date=date(2024, 1, 1))


def test_search_criteria_date_validation():
    with pytest.raises(ValueError):
        FlexibleSearchCriteria(
            user_id=1,
            start_date=date(2024, 5, 1),
            end_date=date(2024, 4, 1),
        )

    crit = FlexibleSearchCriteria(user_id=1, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
    assert crit.start_date < crit.end_date


def test_dynamic_analysis_and_insights():
    tx = FlexibleFinancialTransaction(
        user_id=1,
        amount="20",
        currency_code="EUR",
        date=date(2024, 1, 1),
    )
    analysis = DynamicSpendingAnalysis(user_id=1, total_spent="20", transactions=[tx])
    assert analysis.total_spent == Decimal("20")

    insights = LLMExtractedInsights(criteria=FlexibleSearchCriteria(user_id=1), analysis=analysis)
    assert insights.criteria.user_id == 1
    assert insights.analysis.total_spent == Decimal("20")
