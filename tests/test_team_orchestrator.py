import pytest
from teams.team_orchestrator import TeamOrchestrator, orchestrator
from teams.financial_team import FinancialTeam  # triggers registration


def test_register_and_process_financial_team():
    orchestrator = TeamOrchestrator()
    orchestrator.register_team("financial", FinancialTeam)
    assert "financial" in orchestrator.available_teams()
    team = orchestrator.get_team("financial")
    result = team.process("stock prices")
    assert result["analysis"] == "analysis of stock prices"
    assert result["report"] == "report based on analysis of stock prices"


def test_duplicate_registration_raises():
    orchestrator = TeamOrchestrator()
    orchestrator.register_team("financial", FinancialTeam)
    with pytest.raises(ValueError):
        orchestrator.register_team("financial", FinancialTeam)


def test_global_registration():
    assert "financial" in orchestrator.available_teams()
    team = orchestrator.get_team("financial")
    assert team.process("budget")["analysis"].startswith("analysis")
