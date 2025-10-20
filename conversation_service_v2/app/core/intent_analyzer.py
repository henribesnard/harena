"""Module 1: Intent Analysis using DeepSeek API."""

from openai import AsyncOpenAI
import json
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
import os

# DeepSeek client (OpenAI-compatible API)
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-your-deepseek-api-key"),
    base_url="https://api.deepseek.com"
)


class TimePeriod(BaseModel):
    """Time period specification."""

    type: str = Field(
        ...,
        description="Type of time period",
        examples=["current_month", "last_month", "current_year", "last_n_days", "specific_date", "date_range"]
    )
    value: Optional[str] = Field(
        None,
        description="Value for the time period (e.g., '2025-04' for specific month)"
    )


class AmountFilter(BaseModel):
    """Amount filter specification."""

    min: Optional[float] = Field(None, description="Minimum amount")
    max: Optional[float] = Field(None, description="Maximum amount")
    operator: Optional[Literal["gt", "lt", "between", "eq"]] = Field(
        None,
        description="Comparison operator"
    )

    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v, info):
        """Validate that operator is provided if min or max is set."""
        data = info.data
        if (data.get('min') is not None or data.get('max') is not None) and v is None:
            # Default to 'gt' if min is set, 'lt' if max is set
            if data.get('min') is not None:
                return 'gt'
            elif data.get('max') is not None:
                return 'lt'
        return v


class IntentAnalysis(BaseModel):
    """Result of intent analysis."""

    intent_type: Literal[
        "aggregation",
        "comparison",
        "trend_analysis",
        "ranking",
        "filtering",
        "anomaly_detection"
    ] = Field(..., description="Type of intent")

    time_periods: List[TimePeriod] = Field(
        default_factory=list,
        description="Time periods mentioned in the query"
    )
    categories: List[str] = Field(
        default_factory=list,
        description="Transaction categories to filter"
    )
    merchants: List[str] = Field(
        default_factory=list,
        description="Merchant names to filter"
    )
    amount_filters: Optional[AmountFilter] = Field(
        None,
        description="Amount filters"
    )
    aggregation_type: Optional[Literal["sum", "avg", "count", "max", "min"]] = Field(
        None,
        description="Type of aggregation to perform"
    )
    grouping: List[str] = Field(
        default_factory=list,
        description="Fields to group by (e.g., category, merchant, month)"
    )
    ordering: Optional[str] = Field(
        None,
        description="Ordering clause (e.g., 'amount DESC', 'date ASC')"
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of results to return"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the analysis"
    )

    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


# Prompt templates
INTENT_ANALYSIS_SYSTEM_PROMPT = """
Tu es un expert en analyse d'intentions pour des questions financières.
Ton rôle est d'extraire la structure de la question pour générer du SQL PostgreSQL.

FORMAT DE SORTIE (JSON strict):
{
  "intent_type": "aggregation|comparison|trend_analysis|ranking|filtering|anomaly_detection",
  "time_periods": [
    {"type": "current_month|last_month|current_year|last_n_days|specific_date|date_range", "value": "..."}
  ],
  "categories": ["courses", "restaurants", "transport", ...],
  "merchants": ["Carrefour", "Amazon", ...],
  "amount_filters": {
    "min": 0,
    "max": null,
    "operator": "gt|lt|between"
  },
  "aggregation_type": "sum|avg|count|max|min",
  "grouping": ["category", "merchant", "month", "week"],
  "ordering": "amount DESC|date ASC",
  "limit": 50,
  "confidence_score": 0.95
}

EXEMPLES:

Question: "Combien j'ai dépensé en restaurants ce mois-ci ?"
Intent: {
  "intent_type": "aggregation",
  "time_periods": [{"type": "current_month"}],
  "categories": ["restaurants"],
  "aggregation_type": "sum",
  "confidence_score": 0.98
}

Question: "Mes transactions de plus de 100 euros"
Intent: {
  "intent_type": "filtering",
  "amount_filters": {"min": 100, "operator": "gt"},
  "ordering": "date DESC",
  "limit": 50,
  "confidence_score": 0.96
}

Question: "Compare mes dépenses courses entre avril et mai"
Intent: {
  "intent_type": "comparison",
  "time_periods": [
    {"type": "specific_month", "value": "2025-04"},
    {"type": "specific_month", "value": "2025-05"}
  ],
  "categories": ["courses"],
  "aggregation_type": "sum",
  "confidence_score": 0.94
}

Question: "Mes 5 commerces où je dépense le plus"
Intent: {
  "intent_type": "ranking",
  "grouping": ["merchant"],
  "aggregation_type": "sum",
  "ordering": "amount DESC",
  "limit": 5,
  "confidence_score": 0.97
}

Question: "Évolution de mes dépenses restaurants sur 6 mois"
Intent: {
  "intent_type": "trend_analysis",
  "time_periods": [{"type": "last_n_months", "value": "6"}],
  "categories": ["restaurants"],
  "grouping": ["month"],
  "aggregation_type": "sum",
  "ordering": "month ASC",
  "confidence_score": 0.93
}
"""

INTENT_ANALYSIS_USER_PROMPT = """
Question utilisateur: "{user_query}"

Date actuelle: {current_date}

Analyse cette question et génère le JSON d'intention correspondant.

Raisonne étape par étape:
1. Quel est le type d'intention principal ?
2. Quelles périodes temporelles sont mentionnées ?
3. Quels filtres (catégories, montants, commerces) ?
4. Quel type d'agrégation est nécessaire ?
5. Y a-t-il un groupement ou tri spécifique ?
6. Limite de résultats mentionnée ?

Réponds UNIQUEMENT avec le JSON, sans texte avant ni après.
"""


class IntentAnalyzer:
    """Analyzer for user intent using DeepSeek API."""

    def __init__(self):
        """Initialize the intent analyzer."""
        self.client = deepseek_client

    async def analyze(self, user_query: str) -> IntentAnalysis:
        """
        Analyze the user query and return structured intent.

        Args:
            user_query: User question in natural language

        Returns:
            IntentAnalysis: Extracted and validated intent

        Raises:
            Exception: If API call fails or JSON parsing fails
        """
        # Prepare the prompt
        user_prompt = INTENT_ANALYSIS_USER_PROMPT.format(
            user_query=user_query,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )

        # Call DeepSeek API
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": INTENT_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"}  # Force JSON output
        )

        # Parse JSON response
        intent_json = json.loads(response.choices[0].message.content)

        # Validate with Pydantic
        intent = IntentAnalysis(**intent_json)

        return intent
