import asyncio

from conversation_service.teams import FinancialAnalysisTeamPhase2


def test_run_returns_intent_and_entities():
    team = FinancialAnalysisTeamPhase2()
    result = asyncio.run(team.run("Analyse AAPL price 100"))
    assert result["intent"] is not None
    assert result["entities"] is not None
    assert isinstance(result["entities"], dict)
