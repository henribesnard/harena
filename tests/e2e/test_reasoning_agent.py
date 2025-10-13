"""
E2E Tests for Reasoning Agent (Phase 2)

Tests the complete Chain-of-Thought reasoning pipeline for complex queries:
- Query decomposition into execution plans
- Multi-step plan execution (sequential and parallel)
- Period comparison workflows
- Integration with Analytics Agent
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from conversation_service.agents.reasoning import ReasoningAgent, ExecutionStep, ReasoningPlan, StepType
from conversation_service.agents.analytics.analytics_agent import AnalyticsAgent


class MockLLMManager:
    """Mock LLM Manager for testing"""

    def __init__(self):
        self.call_count = 0

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """Return mock reasoning plan based on prompt"""
        self.call_count += 1

        # Detect period comparison query
        if "Compare mes dépenses" in prompt or "period_comparison" in prompt:
            return """{
                "steps": [
                    {
                        "step_id": "query_current",
                        "step_type": "query_data",
                        "description": "Retrieve transactions for current period",
                        "depends_on": [],
                        "parameters": {
                            "date_range": {"gte": "2024-07-01", "lte": "2024-07-31"}
                        },
                        "output_key": "transactions_current"
                    },
                    {
                        "step_id": "query_previous",
                        "step_type": "query_data",
                        "description": "Retrieve transactions for previous period",
                        "depends_on": [],
                        "parameters": {
                            "date_range": {"gte": "2024-06-01", "lte": "2024-06-30"}
                        },
                        "output_key": "transactions_previous"
                    },
                    {
                        "step_id": "compare",
                        "step_type": "compare_periods",
                        "description": "Compare spending between periods",
                        "depends_on": ["query_current", "query_previous"],
                        "parameters": {
                            "transactions_current_key": "transactions_current",
                            "transactions_previous_key": "transactions_previous",
                            "metric": "sum"
                        },
                        "output_key": "comparison_result"
                    }
                ],
                "execution_mode": "parallel",
                "metadata": {
                    "can_parallelize": ["query_current", "query_previous"],
                    "requires_analytics": true
                }
            }"""

        # Simple query
        return """{
            "steps": [
                {
                    "step_id": "query_transactions",
                    "step_type": "query_data",
                    "description": "Retrieve transactions",
                    "depends_on": [],
                    "parameters": {},
                    "output_key": "transactions"
                }
            ],
            "execution_mode": "sequential",
            "metadata": {}
        }"""


class MockQueryExecutor:
    """Mock Query Executor for testing"""

    def __init__(self):
        self.call_count = 0
        self.last_params = None

    async def execute_query(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return mock transactions based on date range"""
        self.call_count += 1
        self.last_params = params

        date_range = params.get("date_range", {})

        # Generate mock transactions for date range
        if "gte" in date_range:
            # Extract month from date
            start_date = date_range["gte"]
            if "2024-07" in start_date:
                # July transactions
                return [
                    {"amount": 50.0, "date": "2024-07-10", "merchant": "Carrefour", "category": "Alimentation"},
                    {"amount": 30.0, "date": "2024-07-15", "merchant": "Shell", "category": "Transport"},
                    {"amount": 100.0, "date": "2024-07-20", "merchant": "Amazon", "category": "Shopping"},
                ]
            elif "2024-06" in start_date:
                # June transactions
                return [
                    {"amount": 45.0, "date": "2024-06-10", "merchant": "Carrefour", "category": "Alimentation"},
                    {"amount": 25.0, "date": "2024-06-15", "merchant": "Shell", "category": "Transport"},
                    {"amount": 80.0, "date": "2024-06-20", "merchant": "Amazon", "category": "Shopping"},
                ]

        # Default empty
        return []


class MockAnalyticsAgent:
    """Mock Analytics Agent for testing"""

    def __init__(self):
        self.call_count = 0

    async def compare_periods(
        self,
        transactions_current: List[Dict[str, Any]],
        transactions_previous: List[Dict[str, Any]],
        metric: str = "sum"
    ) -> Dict[str, Any]:
        """Return mock comparison result"""
        self.call_count += 1

        current_sum = sum(t.get("amount", 0) for t in transactions_current)
        previous_sum = sum(t.get("amount", 0) for t in transactions_previous)

        delta = current_sum - previous_sum
        percent_change = (delta / previous_sum * 100) if previous_sum > 0 else 0

        return {
            "current": {
                "value": current_sum,
                "count": len(transactions_current)
            },
            "previous": {
                "value": previous_sum,
                "count": len(transactions_previous)
            },
            "delta": delta,
            "percent_change": percent_change
        }


@pytest.fixture
def mock_llm_manager():
    """Fixture for mock LLM manager"""
    return MockLLMManager()


@pytest.fixture
def mock_query_executor():
    """Fixture for mock query executor"""
    return MockQueryExecutor()


@pytest.fixture
def mock_analytics_agent():
    """Fixture for mock analytics agent"""
    return MockAnalyticsAgent()


@pytest.fixture
def reasoning_agent(mock_llm_manager, mock_query_executor, mock_analytics_agent):
    """Fixture for reasoning agent with mocks"""
    return ReasoningAgent(
        llm_manager=mock_llm_manager,
        query_executor=mock_query_executor,
        analytics_agent=mock_analytics_agent
    )


@pytest.mark.asyncio
async def test_decompose_simple_query(reasoning_agent, mock_llm_manager):
    """Test decomposition of simple query into execution plan"""

    plan = await reasoning_agent.decompose_query(
        query="Mes dépenses ce mois-ci",
        intent_group="TRANSACTION_SEARCH",
        intent_subtype="simple",
        entities={"date_range": {"gte": "2024-07-01", "lte": "2024-07-31"}},
        user_context={}
    )

    # Verify plan structure
    assert isinstance(plan, ReasoningPlan)
    assert plan.query == "Mes dépenses ce mois-ci"
    assert plan.intent_group == "TRANSACTION_SEARCH"
    assert plan.intent_subtype == "simple"
    assert len(plan.steps) >= 1
    assert plan.execution_mode in ["sequential", "parallel"]

    # Verify LLM was called
    assert mock_llm_manager.call_count == 1


@pytest.mark.asyncio
async def test_decompose_period_comparison(reasoning_agent, mock_llm_manager):
    """Test decomposition of period comparison query"""

    entities = {
        "periode_1": {"date": {"gte": "2024-07-01", "lte": "2024-07-31"}},
        "periode_2": {"date": {"gte": "2024-06-01", "lte": "2024-06-30"}},
        "analysis_type": "multi_period_comparison"
    }

    plan = await reasoning_agent.decompose_query(
        query="Compare mes dépenses de juillet avec juin",
        intent_group="ANALYSIS_INSIGHTS",
        intent_subtype="period_comparison",
        entities=entities,
        user_context={}
    )

    # Verify plan has multiple steps
    assert len(plan.steps) == 3, "Period comparison should have 3 steps (2 queries + 1 comparison)"

    # Verify step types
    step_types = [step.step_type for step in plan.steps]
    assert StepType.QUERY_DATA in step_types
    assert StepType.COMPARE_PERIODS in step_types

    # Verify execution mode is parallel
    assert plan.execution_mode == "parallel"

    # Verify dependencies
    compare_step = [s for s in plan.steps if s.step_type == StepType.COMPARE_PERIODS][0]
    assert len(compare_step.depends_on) == 2, "Comparison step should depend on both query steps"


@pytest.mark.asyncio
async def test_execute_plan_sequential(reasoning_agent, mock_query_executor):
    """Test sequential execution of simple plan"""

    # Create simple plan
    plan = ReasoningPlan(
        query="Test query",
        intent_group="TRANSACTION_SEARCH",
        intent_subtype="simple",
        steps=[
            ExecutionStep(
                step_id="query_transactions",
                step_type=StepType.QUERY_DATA,
                description="Retrieve transactions",
                depends_on=[],
                parameters={"date_range": {"gte": "2024-07-01", "lte": "2024-07-31"}},
                output_key="transactions"
            )
        ],
        execution_mode="sequential"
    )

    result = await reasoning_agent.execute_plan(
        plan=plan,
        user_id=1,
        context={}
    )

    # Verify execution success
    assert result["success"] is True
    assert "transactions" in result["results"]
    assert len(result["results"]["transactions"]) > 0

    # Verify query executor was called
    assert mock_query_executor.call_count == 1


@pytest.mark.asyncio
async def test_execute_plan_parallel_comparison(reasoning_agent, mock_query_executor, mock_analytics_agent):
    """Test parallel execution of period comparison plan"""

    # Create period comparison plan
    plan = ReasoningPlan(
        query="Compare juillet vs juin",
        intent_group="ANALYSIS_INSIGHTS",
        intent_subtype="period_comparison",
        steps=[
            ExecutionStep(
                step_id="query_current",
                step_type=StepType.QUERY_DATA,
                description="Retrieve current period transactions",
                depends_on=[],
                parameters={"date_range": {"gte": "2024-07-01", "lte": "2024-07-31"}},
                output_key="transactions_current"
            ),
            ExecutionStep(
                step_id="query_previous",
                step_type=StepType.QUERY_DATA,
                description="Retrieve previous period transactions",
                depends_on=[],
                parameters={"date_range": {"gte": "2024-06-01", "lte": "2024-06-30"}},
                output_key="transactions_previous"
            ),
            ExecutionStep(
                step_id="compare",
                step_type=StepType.COMPARE_PERIODS,
                description="Compare periods",
                depends_on=["query_current", "query_previous"],
                parameters={
                    "transactions_current_key": "transactions_current",
                    "transactions_previous_key": "transactions_previous",
                    "metric": "sum"
                },
                output_key="comparison_result"
            )
        ],
        execution_mode="parallel"
    )

    result = await reasoning_agent.execute_plan(
        plan=plan,
        user_id=1,
        context={}
    )

    # Verify execution success
    assert result["success"] is True
    assert "transactions_current" in result["results"]
    assert "transactions_previous" in result["results"]
    assert "comparison_result" in result["results"]

    # Verify parallel execution (both queries called)
    assert mock_query_executor.call_count == 2

    # Verify analytics agent was called
    assert mock_analytics_agent.call_count == 1

    # Verify comparison result structure
    comparison = result["results"]["comparison_result"]
    assert "current" in comparison
    assert "previous" in comparison
    assert "delta" in comparison
    assert "percent_change" in comparison


@pytest.mark.asyncio
async def test_e2e_period_comparison_workflow(reasoning_agent, mock_llm_manager, mock_query_executor, mock_analytics_agent):
    """Test complete E2E workflow for period comparison"""

    # Step 1: Decompose query
    entities = {
        "periode_1": {"date": {"gte": "2024-07-01", "lte": "2024-07-31"}},
        "periode_2": {"date": {"gte": "2024-06-01", "lte": "2024-06-30"}},
        "analysis_type": "multi_period_comparison"
    }

    plan = await reasoning_agent.decompose_query(
        query="Compare mes dépenses de juillet avec juin",
        intent_group="ANALYSIS_INSIGHTS",
        intent_subtype="period_comparison",
        entities=entities,
        user_context={}
    )

    # Verify plan
    assert len(plan.steps) == 3

    # Step 2: Execute plan
    result = await reasoning_agent.execute_plan(
        plan=plan,
        user_id=1,
        context={}
    )

    # Verify execution
    assert result["success"] is True

    # Step 3: Verify results contain comparison data
    comparison = result["results"]["comparison_result"]

    # July: 50 + 30 + 100 = 180
    # June: 45 + 25 + 80 = 150
    # Delta: 180 - 150 = 30
    # Percent: (30/150)*100 = 20%

    assert comparison["current"]["value"] == 180.0
    assert comparison["previous"]["value"] == 150.0
    assert comparison["delta"] == 30.0
    assert abs(comparison["percent_change"] - 20.0) < 0.01

    # Verify all components were used
    assert mock_llm_manager.call_count == 1  # Decomposition
    assert mock_query_executor.call_count == 2  # Two parallel queries
    assert mock_analytics_agent.call_count == 1  # Comparison


@pytest.mark.asyncio
async def test_fallback_plan_on_llm_failure(reasoning_agent):
    """Test that fallback plan is created when LLM decomposition fails"""

    # Replace LLM manager with one that raises exception
    class FailingLLMManager:
        async def generate_completion(self, **kwargs):
            raise Exception("LLM service unavailable")

    reasoning_agent.llm_manager = FailingLLMManager()

    # Should fallback to simple plan
    entities = {
        "periode_1": {"date": {"gte": "2024-07-01", "lte": "2024-07-31"}},
        "periode_2": {"date": {"gte": "2024-06-01", "lte": "2024-06-30"}},
        "analysis_type": "multi_period_comparison"
    }

    plan = await reasoning_agent.decompose_query(
        query="Compare juillet vs juin",
        intent_group="ANALYSIS_INSIGHTS",
        intent_subtype="period_comparison",
        entities=entities,
        user_context={}
    )

    # Verify fallback plan was created
    assert plan is not None
    assert len(plan.steps) == 3  # Fallback creates comparison plan
    assert plan.metadata.get("fallback") is True


@pytest.mark.asyncio
async def test_validate_step_dependencies(reasoning_agent):
    """Test that execution validates step dependencies"""

    # Create plan with invalid dependencies
    plan = ReasoningPlan(
        query="Test",
        intent_group="TEST",
        intent_subtype="test",
        steps=[
            ExecutionStep(
                step_id="step2",
                step_type=StepType.QUERY_DATA,
                description="Step that depends on non-existent step",
                depends_on=["step1"],  # This step doesn't exist!
                parameters={},
                output_key="result"
            )
        ],
        execution_mode="sequential"
    )

    result = await reasoning_agent.execute_plan(
        plan=plan,
        user_id=1,
        context={}
    )

    # Should fail with dependency error
    assert result["success"] is False
    assert "depends on" in result["error"].lower()


@pytest.mark.asyncio
async def test_execute_calculate_metric_step(reasoning_agent):
    """Test execution of CALCULATE_METRIC step"""

    # Mock transactions
    mock_transactions = [
        {"amount": 100.0},
        {"amount": 200.0},
        {"amount": 150.0}
    ]

    # Create plan with metric calculation
    plan = ReasoningPlan(
        query="Test",
        intent_group="TEST",
        intent_subtype="test",
        steps=[
            ExecutionStep(
                step_id="query",
                step_type=StepType.QUERY_DATA,
                description="Get transactions",
                depends_on=[],
                parameters={"date_range": {"gte": "2024-07-01", "lte": "2024-07-31"}},
                output_key="transactions"
            ),
            ExecutionStep(
                step_id="calculate",
                step_type=StepType.CALCULATE_METRIC,
                description="Calculate sum",
                depends_on=["query"],
                parameters={
                    "transactions_key": "transactions",
                    "metric_type": "sum"
                },
                output_key="total"
            )
        ],
        execution_mode="sequential"
    )

    result = await reasoning_agent.execute_plan(
        plan=plan,
        user_id=1,
        context={}
    )

    # Verify metric was calculated
    assert result["success"] is True
    assert "total" in result["results"]

    # July transactions: 50 + 30 + 100 = 180
    assert result["results"]["total"] == 180.0


@pytest.mark.asyncio
async def test_reasoning_agent_statistics(reasoning_agent, mock_llm_manager):
    """Test that statistics are tracked correctly"""

    # Execute multiple queries
    for i in range(3):
        await reasoning_agent.decompose_query(
            query=f"Test query {i}",
            intent_group="TRANSACTION_SEARCH",
            intent_subtype="simple",
            entities={},
            user_context={}
        )

    # Verify LLM was called for each
    assert mock_llm_manager.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
