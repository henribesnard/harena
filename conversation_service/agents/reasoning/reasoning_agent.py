"""
Reasoning Agent - Chain-of-Thought query decomposition for complex queries

Implements Phase 2 (Sprint 4-5) of the Harena implementation plan.
Handles multi-step queries like period comparisons, trend analysis, and complex aggregations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import json
import asyncio

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of execution steps in reasoning plan"""
    QUERY_DATA = "query_data"
    CALCULATE_METRIC = "calculate_metric"
    COMPARE_PERIODS = "compare_periods"
    DETECT_ANOMALIES = "detect_anomalies"
    CALCULATE_TREND = "calculate_trend"
    AGGREGATE_RESULTS = "aggregate_results"


class ExecutionStep(BaseModel):
    """Single step in execution plan"""
    step_id: str = Field(..., description="Unique identifier for this step")
    step_type: StepType = Field(..., description="Type of operation to execute")
    description: str = Field(..., description="Human-readable description")
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this step depends on")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for execution")
    output_key: str = Field(..., description="Key to store result in execution context")


class ReasoningPlan(BaseModel):
    """Complete execution plan for complex query"""
    query: str = Field(..., description="Original user query")
    intent_group: str = Field(..., description="Detected intent group")
    intent_subtype: str = Field(..., description="Detected intent subtype")
    steps: List[ExecutionStep] = Field(..., description="Ordered execution steps")
    execution_mode: str = Field(default="sequential", description="sequential or parallel")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ReasoningAgent:
    """
    Agent that decomposes complex queries into executable plans using Chain-of-Thought reasoning.

    Features:
    - LLM-based query decomposition with few-shot prompting
    - Support for parallel and sequential execution
    - Integration with QueryExecutor and Analytics Agent
    - Handles period comparisons, trend analysis, anomaly detection
    """

    def __init__(
        self,
        llm_manager: Any,
        query_executor: Optional[Any] = None,
        analytics_agent: Optional[Any] = None
    ):
        """
        Initialize Reasoning Agent.

        Args:
            llm_manager: LLM manager for Chain-of-Thought prompting
            query_executor: QueryExecutor for data retrieval
            analytics_agent: Analytics Agent for calculations
        """
        self.llm_manager = llm_manager
        self.query_executor = query_executor
        self.analytics_agent = analytics_agent
        self.logger = logging.getLogger(__name__)

    async def decompose_query(
        self,
        query: str,
        intent_group: str,
        intent_subtype: str,
        entities: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> ReasoningPlan:
        """
        Decompose complex query into execution plan using Chain-of-Thought.

        Args:
            query: Original user query
            intent_group: Detected intent group (e.g., ANALYSIS_INSIGHTS)
            intent_subtype: Detected intent subtype (e.g., period_comparison)
            entities: Extracted entities from intent classification
            user_context: Additional user context

        Returns:
            ReasoningPlan with ordered execution steps
        """
        self.logger.info(f"Decomposing query: {query} (intent: {intent_group}.{intent_subtype})")

        # Build Chain-of-Thought prompt
        prompt = self._build_decomposition_prompt(query, intent_group, intent_subtype, entities, user_context)

        try:
            # Get LLM decomposition
            response = await self.llm_manager.generate_completion(
                prompt=prompt,
                system_prompt="You are a query planning agent that decomposes complex financial queries into executable steps. Always respond with valid JSON.",
                temperature=0.3,
                max_tokens=2000
            )

            # Parse LLM response
            plan_dict = self._parse_llm_response(response)

            # Validate and create ReasoningPlan
            plan = ReasoningPlan(
                query=query,
                intent_group=intent_group,
                intent_subtype=intent_subtype,
                steps=[ExecutionStep(**step) for step in plan_dict.get("steps", [])],
                execution_mode=plan_dict.get("execution_mode", "sequential"),
                metadata=plan_dict.get("metadata", {})
            )

            self.logger.info(f"Generated plan with {len(plan.steps)} steps (mode: {plan.execution_mode})")
            return plan

        except Exception as e:
            self.logger.error(f"Failed to decompose query: {str(e)}")
            # Fallback to simple plan
            return self._create_fallback_plan(query, intent_group, intent_subtype, entities)

    def _build_decomposition_prompt(
        self,
        query: str,
        intent_group: str,
        intent_subtype: str,
        entities: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build Chain-of-Thought prompt for query decomposition"""

        # Few-shot examples for period comparison
        examples = """
Example 1:
Query: "Compare mes dépenses ce mois vs mois dernier"
Intent: ANALYSIS_INSIGHTS.period_comparison
Entities: {
  "periode_1": {"date": {"gte": "2025-01-01", "lte": "2025-01-31"}},
  "periode_2": {"date": {"gte": "2024-12-01", "lte": "2024-12-31"}},
  "analysis_type": "multi_period_comparison"
}

Plan:
{
  "steps": [
    {
      "step_id": "query_current",
      "step_type": "query_data",
      "description": "Retrieve transactions for current period (January 2025)",
      "depends_on": [],
      "parameters": {
        "date_range": {"gte": "2025-01-01", "lte": "2025-01-31"}
      },
      "output_key": "transactions_current"
    },
    {
      "step_id": "query_previous",
      "step_type": "query_data",
      "description": "Retrieve transactions for previous period (December 2024)",
      "depends_on": [],
      "parameters": {
        "date_range": {"gte": "2024-12-01", "lte": "2024-12-31"}
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
}

Example 2:
Query: "Mes dépenses à Carrefour en décembre"
Intent: TRANSACTION_SEARCH.by_merchant
Entities: {
  "merchant_name": "Carrefour",
  "date_range": {"gte": "2024-12-01", "lte": "2024-12-31"}
}

Plan:
{
  "steps": [
    {
      "step_id": "query_transactions",
      "step_type": "query_data",
      "description": "Retrieve transactions for Carrefour in December",
      "depends_on": [],
      "parameters": {
        "merchant_name": "Carrefour",
        "date_range": {"gte": "2024-12-01", "lte": "2024-12-31"}
      },
      "output_key": "transactions"
    }
  ],
  "execution_mode": "sequential",
  "metadata": {
    "requires_analytics": false
  }
}
"""

        prompt = f"""You are a query planning agent for a financial assistant. Decompose the following query into executable steps.

{examples}

Now decompose this query:
Query: "{query}"
Intent: {intent_group}.{intent_subtype}
Entities: {json.dumps(entities, indent=2)}
User Context: {json.dumps(user_context or {}, indent=2)}

Generate a complete execution plan in JSON format with the structure shown in the examples.
For period comparisons, create parallel QUERY_DATA steps followed by COMPARE_PERIODS step.
For simple queries, create a single QUERY_DATA step.

Respond ONLY with valid JSON, no additional text.
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract plan JSON"""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            # Parse JSON
            plan_dict = json.loads(response)
            return plan_dict

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            self.logger.debug(f"Response was: {response}")
            raise

    def _create_fallback_plan(
        self,
        query: str,
        intent_group: str,
        intent_subtype: str,
        entities: Dict[str, Any]
    ) -> ReasoningPlan:
        """Create simple fallback plan when LLM decomposition fails"""

        self.logger.warning("Creating fallback plan due to LLM decomposition failure")

        # For period comparisons, create a two-query plan
        if intent_subtype == "period_comparison":
            steps = [
                ExecutionStep(
                    step_id="query_current",
                    step_type=StepType.QUERY_DATA,
                    description="Retrieve transactions for current period",
                    depends_on=[],
                    parameters={
                        "date_range": entities.get("periode_1", {}).get("date", {})
                    },
                    output_key="transactions_current"
                ),
                ExecutionStep(
                    step_id="query_previous",
                    step_type=StepType.QUERY_DATA,
                    description="Retrieve transactions for previous period",
                    depends_on=[],
                    parameters={
                        "date_range": entities.get("periode_2", {}).get("date", {})
                    },
                    output_key="transactions_previous"
                ),
                ExecutionStep(
                    step_id="compare",
                    step_type=StepType.COMPARE_PERIODS,
                    description="Compare spending between periods",
                    depends_on=["query_current", "query_previous"],
                    parameters={
                        "transactions_current_key": "transactions_current",
                        "transactions_previous_key": "transactions_previous",
                        "metric": "sum"
                    },
                    output_key="comparison_result"
                )
            ]
            execution_mode = "parallel"
        else:
            # Simple query - single step
            steps = [
                ExecutionStep(
                    step_id="query_transactions",
                    step_type=StepType.QUERY_DATA,
                    description="Retrieve transactions",
                    depends_on=[],
                    parameters=entities,
                    output_key="transactions"
                )
            ]
            execution_mode = "sequential"

        return ReasoningPlan(
            query=query,
            intent_group=intent_group,
            intent_subtype=intent_subtype,
            steps=steps,
            execution_mode=execution_mode,
            metadata={"fallback": True}
        )

    async def execute_plan(
        self,
        plan: ReasoningPlan,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute reasoning plan and return results.

        Args:
            plan: ReasoningPlan to execute
            user_id: User ID for query execution
            context: Additional execution context

        Returns:
            Dictionary with execution results and metadata
        """
        self.logger.info(f"Executing plan with {len(plan.steps)} steps (mode: {plan.execution_mode})")

        execution_context = context or {}
        execution_context["user_id"] = user_id
        results = {}

        try:
            if plan.execution_mode == "parallel":
                results = await self._execute_parallel(plan, execution_context)
            else:
                results = await self._execute_sequential(plan, execution_context)

            self.logger.info(f"Plan execution completed successfully with {len(results)} results")

            return {
                "success": True,
                "results": results,
                "plan": plan.model_dump() if hasattr(plan, 'model_dump') else plan.dict(),
                "metadata": {
                    "steps_executed": len(plan.steps),
                    "execution_mode": plan.execution_mode
                }
            }

        except Exception as e:
            self.logger.error(f"Plan execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": results,
                "plan": plan.model_dump() if hasattr(plan, 'model_dump') else plan.dict()
            }

    async def _execute_sequential(
        self,
        plan: ReasoningPlan,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute steps sequentially"""
        results = {}
        executed_step_ids = set()

        for step in plan.steps:
            self.logger.debug(f"Executing step: {step.step_id} ({step.step_type})")

            # Check dependencies (by step_id)
            for dep in step.depends_on:
                if dep not in executed_step_ids:
                    raise ValueError(f"Step {step.step_id} depends on {dep} which has not been executed")

            # Execute step
            result = await self._execute_step(step, results, execution_context)
            results[step.output_key] = result
            executed_step_ids.add(step.step_id)

        return results

    async def _execute_parallel(
        self,
        plan: ReasoningPlan,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute steps in parallel where possible"""
        results = {}
        executed_steps = set()

        # Build dependency graph
        remaining_steps = plan.steps.copy()

        while remaining_steps:
            # Find steps that can execute now (dependencies satisfied)
            ready_steps = [
                step for step in remaining_steps
                if all(dep in executed_steps for dep in step.depends_on)
            ]

            if not ready_steps:
                raise ValueError("Circular dependency detected in execution plan")

            # Execute ready steps in parallel
            self.logger.debug(f"Executing {len(ready_steps)} steps in parallel")

            tasks = [
                self._execute_step(step, results, execution_context)
                for step in ready_steps
            ]

            step_results = await asyncio.gather(*tasks)

            # Store results
            for step, result in zip(ready_steps, step_results):
                results[step.output_key] = result
                executed_steps.add(step.step_id)
                remaining_steps.remove(step)

        return results

    async def _execute_step(
        self,
        step: ExecutionStep,
        results: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Any:
        """Execute a single step based on step type"""

        if step.step_type == StepType.QUERY_DATA:
            return await self._execute_query_data(step, execution_context)

        elif step.step_type == StepType.COMPARE_PERIODS:
            return await self._execute_compare_periods(step, results, execution_context)

        elif step.step_type == StepType.DETECT_ANOMALIES:
            return await self._execute_detect_anomalies(step, results, execution_context)

        elif step.step_type == StepType.CALCULATE_TREND:
            return await self._execute_calculate_trend(step, results, execution_context)

        elif step.step_type == StepType.CALCULATE_METRIC:
            return await self._execute_calculate_metric(step, results, execution_context)

        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

    async def _execute_query_data(
        self,
        step: ExecutionStep,
        execution_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute QUERY_DATA step"""

        if not self.query_executor:
            raise ValueError("QueryExecutor not available")

        user_id = execution_context["user_id"]
        params = step.parameters.copy()
        params["user_id"] = user_id

        self.logger.debug(f"Querying data with params: {params}")

        # Execute query via QueryExecutor
        result = await self.query_executor.execute_query(params)

        # Handle both dict response and list response (for flexibility)
        if isinstance(result, list):
            transactions = result
        elif isinstance(result, dict):
            transactions = result.get("transactions", result.get("results", []))
        else:
            transactions = []

        self.logger.debug(f"Query returned {len(transactions)} transactions")

        return transactions

    async def _execute_compare_periods(
        self,
        step: ExecutionStep,
        results: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute COMPARE_PERIODS step"""

        if not self.analytics_agent:
            raise ValueError("Analytics Agent not available")

        # Get transactions from previous steps
        current_key = step.parameters.get("transactions_current_key", "transactions_current")
        previous_key = step.parameters.get("transactions_previous_key", "transactions_previous")

        transactions_current = results.get(current_key, [])
        transactions_previous = results.get(previous_key, [])

        metric = step.parameters.get("metric", "sum")

        self.logger.debug(f"Comparing periods: {len(transactions_current)} current vs {len(transactions_previous)} previous")

        # Call Analytics Agent
        comparison = await self.analytics_agent.compare_periods(
            transactions_current=transactions_current,
            transactions_previous=transactions_previous,
            metric=metric
        )

        # Convert to dict if it's a Pydantic model
        if hasattr(comparison, 'model_dump'):
            return comparison.model_dump()
        elif hasattr(comparison, 'dict'):
            return comparison.dict()
        else:
            return comparison

    async def _execute_detect_anomalies(
        self,
        step: ExecutionStep,
        results: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute DETECT_ANOMALIES step"""

        if not self.analytics_agent:
            raise ValueError("Analytics Agent not available")

        transactions_key = step.parameters.get("transactions_key", "transactions")
        transactions = results.get(transactions_key, [])

        anomalies = await self.analytics_agent.detect_anomalies(transactions)

        # Convert to dict if it's a Pydantic model
        if hasattr(anomalies, 'model_dump'):
            return anomalies.model_dump()
        elif hasattr(anomalies, 'dict'):
            return anomalies.dict()
        else:
            return anomalies

    async def _execute_calculate_trend(
        self,
        step: ExecutionStep,
        results: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute CALCULATE_TREND step"""

        if not self.analytics_agent:
            raise ValueError("Analytics Agent not available")

        transactions_key = step.parameters.get("transactions_key", "transactions")
        transactions = results.get(transactions_key, [])

        trend = await self.analytics_agent.calculate_trend(transactions)

        # Convert to dict if it's a Pydantic model
        if hasattr(trend, 'model_dump'):
            return trend.model_dump()
        elif hasattr(trend, 'dict'):
            return trend.dict()
        else:
            return trend

    async def _execute_calculate_metric(
        self,
        step: ExecutionStep,
        results: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Any:
        """Execute CALCULATE_METRIC step"""

        # Simple metric calculations (sum, avg, count, etc.)
        transactions_key = step.parameters.get("transactions_key", "transactions")
        transactions = results.get(transactions_key, [])
        metric_type = step.parameters.get("metric_type", "sum")

        if metric_type == "sum":
            return sum(t.get("amount", 0) for t in transactions)
        elif metric_type == "avg":
            return sum(t.get("amount", 0) for t in transactions) / len(transactions) if transactions else 0
        elif metric_type == "count":
            return len(transactions)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
