"""Mock intent detection agent for workflow validation.

This agent bypasses any LLM calls and returns pre-defined intent
classification results for a fixed set of user questions.  It enables the
rest of the conversation workflow to be validated independently of the
actual intent detection model.

The dataset includes 30 typical financial questions covering all supported
intent types.  Each question maps to the structured response the real intent
detector is expected to produce.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

from .llm_intent_agent import LLMIntentAgent
from ..models.financial_models import (
    DetectionMethod,
    EntityType,
    FinancialEntity,
    IntentCategory,
    IntentResult,
)

# ---------------------------------------------------------------------------
# Dataset of mock intents
# ---------------------------------------------------------------------------

MOCK_INTENT_RESPONSES: Dict[str, Dict[str, Any]] = {
    # TRANSACTION_SEARCH
    "Mes transactions Netflix ce mois": {
        "intent_type": "TRANSACTION_SEARCH",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.95,
        "entities": [
            {
                "entity_type": "MERCHANT",
                "raw_value": "Netflix",
                "normalized_value": "netflix",
                "confidence": 0.98,
                "position": [16, 23],
            },
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "ce mois",
                "normalized_value": "current_month",
                "confidence": 0.90,
                "position": [24, 31],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 120.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_merchant", "filter_by_date"],
    },
    "Combien j'ai dépensé chez Carrefour ?": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.92,
        "entities": [
            {
                "entity_type": "MERCHANT",
                "raw_value": "Carrefour",
                "normalized_value": "carrefour",
                "confidence": 0.95,
                "position": [25, 34],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 110.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_merchant", "calculate_sum"],
    },
    "Mes achats Amazon janvier 2025": {
        "intent_type": "TRANSACTION_SEARCH",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.96,
        "entities": [
            {
                "entity_type": "MERCHANT",
                "raw_value": "Amazon",
                "normalized_value": "amazon",
                "confidence": 0.97,
                "position": [11, 17],
            },
            {
                "entity_type": "DATE",
                "raw_value": "janvier 2025",
                "normalized_value": "2025-01",
                "confidence": 0.93,
                "position": [18, 30],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 135.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_merchant", "filter_by_date"],
    },
    "Transactions supérieures à 100 euros": {
        "intent_type": "TRANSACTION_SEARCH",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.90,
        "entities": [
            {
                "entity_type": "AMOUNT",
                "raw_value": "100 euros",
                "normalized_value": 100.0,
                "confidence": 0.95,
                "position": [25, 34],
            },
            {
                "entity_type": "CURRENCY",
                "raw_value": "euros",
                "normalized_value": "EUR",
                "confidence": 0.98,
                "position": [29, 34],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 140.0,
        "requires_clarification": False,
        "suggested_actions": ["filter_by_amount_greater"],
    },

    # SPENDING_ANALYSIS
    "Mes dépenses restaurant cette semaine": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "SPENDING_ANALYSIS",
        "confidence": 0.88,
        "entities": [
            {
                "entity_type": "CATEGORY",
                "raw_value": "restaurant",
                "normalized_value": "restaurant",
                "confidence": 0.92,
                "position": [12, 22],
            },
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "cette semaine",
                "normalized_value": "current_week",
                "confidence": 0.85,
                "position": [23, 36],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 125.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_category", "calculate_total"],
    },
    "Analyse mes courses alimentaires": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "SPENDING_ANALYSIS",
        "confidence": 0.89,
        "entities": [
            {
                "entity_type": "CATEGORY",
                "raw_value": "courses alimentaires",
                "normalized_value": "alimentation",
                "confidence": 0.87,
                "position": [12, 32],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 115.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_category", "spending_breakdown"],
    },
    "Combien je dépense en transport par mois ?": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "SPENDING_ANALYSIS",
        "confidence": 0.93,
        "entities": [
            {
                "entity_type": "CATEGORY",
                "raw_value": "transport",
                "normalized_value": "transport",
                "confidence": 0.95,
                "position": [21, 30],
            },
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "par mois",
                "normalized_value": "monthly",
                "confidence": 0.88,
                "position": [31, 39],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 130.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_category", "monthly_average"],
    },

    # TREND_ANALYSIS
    "Évolution de mes dépenses ces 3 derniers mois": {
        "intent_type": "TREND_ANALYSIS",
        "intent_category": "TREND_ANALYSIS",
        "confidence": 0.91,
        "entities": [
            {
                "entity_type": "DATE_RANGE",
                "raw_value": "ces 3 derniers mois",
                "normalized_value": "last_3_months",
                "confidence": 0.89,
                "position": [32, 51],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 145.0,
        "requires_clarification": False,
        "suggested_actions": ["trend_analysis", "monthly_comparison"],
    },
    "Tendance dépenses loisirs depuis janvier": {
        "intent_type": "TREND_ANALYSIS",
        "intent_category": "TREND_ANALYSIS",
        "confidence": 0.87,
        "entities": [
            {
                "entity_type": "CATEGORY",
                "raw_value": "loisirs",
                "normalized_value": "loisirs",
                "confidence": 0.90,
                "position": [18, 25],
            },
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "depuis janvier",
                "normalized_value": "since_january",
                "confidence": 0.85,
                "position": [26, 40],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 155.0,
        "requires_clarification": False,
        "suggested_actions": ["trend_analysis", "category_evolution"],
    },

    # BALANCE_INQUIRY
    "Quel est mon solde actuel ?": {
        "intent_type": "BALANCE_INQUIRY",
        "intent_category": "BALANCE_INQUIRY",
        "confidence": 0.96,
        "entities": [
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "actuel",
                "normalized_value": "current",
                "confidence": 0.92,
                "position": [18, 24],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 100.0,
        "requires_clarification": False,
        "suggested_actions": ["get_current_balance"],
    },
    "Solde de mon compte épargne": {
        "intent_type": "BALANCE_INQUIRY",
        "intent_category": "ACCOUNT_INFORMATION",
        "confidence": 0.94,
        "entities": [
            {
                "entity_type": "ACCOUNT_TYPE",
                "raw_value": "compte épargne",
                "normalized_value": "savings_account",
                "confidence": 0.93,
                "position": [10, 24],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 105.0,
        "requires_clarification": False,
        "suggested_actions": ["get_account_balance"],
    },

    # COMPARISON_QUERY
    "Compare mes dépenses janvier vs février": {
        "intent_type": "COMPARISON_QUERY",
        "intent_category": "SPENDING_ANALYSIS",
        "confidence": 0.89,
        "entities": [
            {
                "entity_type": "DATE",
                "raw_value": "janvier",
                "normalized_value": "2025-01",
                "confidence": 0.90,
                "position": [21, 28],
            },
            {
                "entity_type": "DATE",
                "raw_value": "février",
                "normalized_value": "2025-02",
                "confidence": 0.90,
                "position": [32, 39],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 160.0,
        "requires_clarification": False,
        "suggested_actions": ["compare_periods", "spending_comparison"],
    },
    "Restaurant vs courses : quel budget ?": {
        "intent_type": "COMPARISON_QUERY",
        "intent_category": "SPENDING_ANALYSIS",
        "confidence": 0.85,
        "entities": [
            {
                "entity_type": "CATEGORY",
                "raw_value": "Restaurant",
                "normalized_value": "restaurant",
                "confidence": 0.95,
                "position": [0, 10],
            },
            {
                "entity_type": "CATEGORY",
                "raw_value": "courses",
                "normalized_value": "alimentation",
                "confidence": 0.88,
                "position": [14, 21],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 170.0,
        "requires_clarification": False,
        "suggested_actions": ["compare_categories", "budget_breakdown"],
    },

    # MERCHANT_INQUIRY
    "Toutes mes transactions Uber": {
        "intent_type": "MERCHANT_INQUIRY",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.94,
        "entities": [
            {
                "entity_type": "MERCHANT",
                "raw_value": "Uber",
                "normalized_value": "uber",
                "confidence": 0.97,
                "position": [24, 28],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 110.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_merchant", "list_transactions"],
    },
    "Historique paiements Orange": {
        "intent_type": "MERCHANT_INQUIRY",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.92,
        "entities": [
            {
                "entity_type": "MERCHANT",
                "raw_value": "Orange",
                "normalized_value": "orange",
                "confidence": 0.96,
                "position": [20, 26],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 115.0,
        "requires_clarification": False,
        "suggested_actions": ["search_by_merchant", "payment_history"],
    },

    # CATEGORY_ANALYSIS
    "Répartition par catégorie ce trimestre": {
        "intent_type": "CATEGORY_ANALYSIS",
        "intent_category": "SPENDING_ANALYSIS",
        "confidence": 0.90,
        "entities": [
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "ce trimestre",
                "normalized_value": "current_quarter",
                "confidence": 0.87,
                "position": [23, 35],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 140.0,
        "requires_clarification": False,
        "suggested_actions": ["category_breakdown", "spending_distribution"],
    },
    "Top 5 catégories de dépenses": {
        "intent_type": "CATEGORY_ANALYSIS",
        "intent_category": "SPENDING_ANALYSIS",
        "confidence": 0.88,
        "entities": [
            {
                "entity_type": "OTHER",
                "raw_value": "Top 5",
                "normalized_value": "top_5",
                "confidence": 0.85,
                "position": [0, 5],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 125.0,
        "requires_clarification": False,
        "suggested_actions": ["top_categories", "ranking_analysis"],
    },

    # BUDGET_INQUIRY
    "Mon budget alimentation est-il respecté ?": {
        "intent_type": "BUDGET_INQUIRY",
        "intent_category": "BUDGET_ANALYSIS",
        "confidence": 0.91,
        "entities": [
            {
                "entity_type": "CATEGORY",
                "raw_value": "alimentation",
                "normalized_value": "alimentation",
                "confidence": 0.94,
                "position": [11, 23],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 135.0,
        "requires_clarification": False,
        "suggested_actions": ["budget_check", "category_budget_status"],
    },
    "Où en suis-je dans mon budget mensuel ?": {
        "intent_type": "BUDGET_INQUIRY",
        "intent_category": "BUDGET_ANALYSIS",
        "confidence": 0.89,
        "entities": [
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "mensuel",
                "normalized_value": "monthly",
                "confidence": 0.90,
                "position": [32, 39],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 130.0,
        "requires_clarification": False,
        "suggested_actions": ["budget_status", "monthly_tracking"],
    },

    # CONVERSATIONAL
    "Bonjour, comment ça va ?": {
        "intent_type": "CONVERSATIONAL",
        "intent_category": "GREETING",
        "confidence": 0.97,
        "entities": [],
        "method": "llm_detection",
        "processing_time_ms": 80.0,
        "requires_clarification": False,
        "suggested_actions": ["greeting_response"],
    },
    "Merci pour l'information": {
        "intent_type": "CONVERSATIONAL",
        "intent_category": "CONFIRMATION",
        "confidence": 0.95,
        "entities": [],
        "method": "llm_detection",
        "processing_time_ms": 75.0,
        "requires_clarification": False,
        "suggested_actions": ["acknowledgment_response"],
    },
    "Peux-tu m'expliquer ça plus clairement ?": {
        "intent_type": "CONVERSATIONAL",
        "intent_category": "CLARIFICATION",
        "confidence": 0.92,
        "entities": [],
        "method": "llm_detection",
        "processing_time_ms": 90.0,
        "requires_clarification": True,
        "suggested_actions": ["clarification_request", "explain_previous"],
    },

    # FILTER_REQUEST
    "Filtre les dépenses > 50€ en janvier": {
        "intent_type": "FILTER_REQUEST",
        "intent_category": "FILTER_REQUEST",
        "confidence": 0.93,
        "entities": [
            {
                "entity_type": "AMOUNT",
                "raw_value": "50€",
                "normalized_value": 50.0,
                "confidence": 0.96,
                "position": [23, 26],
            },
            {
                "entity_type": "DATE",
                "raw_value": "janvier",
                "normalized_value": "2025-01",
                "confidence": 0.90,
                "position": [30, 37],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 145.0,
        "requires_clarification": False,
        "suggested_actions": ["apply_amount_filter", "apply_date_filter"],
    },
    "Montre seulement les débits": {
        "intent_type": "FILTER_REQUEST",
        "intent_category": "FILTER_REQUEST",
        "confidence": 0.88,
        "entities": [
            {
                "entity_type": "TRANSACTION_TYPE",
                "raw_value": "débits",
                "normalized_value": "debit",
                "confidence": 0.92,
                "position": [19, 25],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 120.0,
        "requires_clarification": False,
        "suggested_actions": ["filter_transaction_type"],
    },

    # EXPORT_REQUEST
    "Exporte mes transactions en CSV": {
        "intent_type": "EXPORT_REQUEST",
        "intent_category": "EXPORT_REQUEST",
        "confidence": 0.96,
        "entities": [
            {
                "entity_type": "OTHER",
                "raw_value": "CSV",
                "normalized_value": "csv_format",
                "confidence": 0.98,
                "position": [28, 31],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 110.0,
        "requires_clarification": False,
        "suggested_actions": ["export_transactions", "csv_format"],
    },
    "Télécharger rapport mensuel Excel": {
        "intent_type": "EXPORT_REQUEST",
        "intent_category": "EXPORT_REQUEST",
        "confidence": 0.94,
        "entities": [
            {
                "entity_type": "RELATIVE_DATE",
                "raw_value": "mensuel",
                "normalized_value": "monthly",
                "confidence": 0.88,
                "position": [20, 27],
            },
            {
                "entity_type": "OTHER",
                "raw_value": "Excel",
                "normalized_value": "excel_format",
                "confidence": 0.96,
                "position": [28, 33],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 130.0,
        "requires_clarification": False,
        "suggested_actions": ["export_report", "excel_format"],
    },

    # UNCLEAR_INTENT
    "Trucs bizarres dans mes comptes": {
        "intent_type": "UNCLEAR_INTENT",
        "intent_category": "UNCLEAR_INTENT",
        "confidence": 0.35,
        "entities": [
            {
                "entity_type": "ACCOUNT_TYPE",
                "raw_value": "comptes",
                "normalized_value": "accounts",
                "confidence": 0.65,
                "position": [21, 28],
            }
        ],
        "method": "llm_detection",
        "processing_time_ms": 200.0,
        "requires_clarification": True,
        "suggested_actions": ["clarification_request", "anomaly_detection"],
    },
    "Ça marche pas comme d'habitude": {
        "intent_type": "UNCLEAR_INTENT",
        "intent_category": "UNCLEAR_INTENT",
        "confidence": 0.25,
        "entities": [],
        "method": "llm_detection",
        "processing_time_ms": 180.0,
        "requires_clarification": True,
        "suggested_actions": ["clarification_request", "technical_support"],
    },

    # GOAL_TRACKING
    "Progression objectif épargne 1000€": {
        "intent_type": "GOAL_TRACKING",
        "intent_category": "GOAL_TRACKING",
        "confidence": 0.91,
        "entities": [
            {
                "entity_type": "ACCOUNT_TYPE",
                "raw_value": "épargne",
                "normalized_value": "savings",
                "confidence": 0.89,
                "position": [21, 28],
            },
            {
                "entity_type": "AMOUNT",
                "raw_value": "1000€",
                "normalized_value": 1000.0,
                "confidence": 0.97,
                "position": [29, 34],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 150.0,
        "requires_clarification": False,
        "suggested_actions": ["goal_progress", "savings_tracking"],
    },
    "Suivi budget vacances 2000€": {
        "intent_type": "GOAL_TRACKING",
        "intent_category": "GOAL_TRACKING",
        "confidence": 0.88,
        "entities": [
            {
                "entity_type": "CATEGORY",
                "raw_value": "vacances",
                "normalized_value": "vacances",
                "confidence": 0.90,
                "position": [13, 21],
            },
            {
                "entity_type": "AMOUNT",
                "raw_value": "2000€",
                "normalized_value": 2000.0,
                "confidence": 0.95,
                "position": [22, 27],
            },
        ],
        "method": "llm_detection",
        "processing_time_ms": 140.0,
        "requires_clarification": False,
        "suggested_actions": ["budget_tracking", "category_goal"],
    },
}


# ---------------------------------------------------------------------------
# Mock agent implementation
# ---------------------------------------------------------------------------


class MockIntentAgent(LLMIntentAgent):
    """Intent agent returning predefined results from ``MOCK_INTENT_RESPONSES``."""

    def __init__(self, deepseek_client: Optional[Any] = None, dataset: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        dummy_client = deepseek_client or SimpleNamespace(api_key="mock", base_url="http://mock")
        super().__init__(deepseek_client=dummy_client)
        self._dataset = dataset or MOCK_INTENT_RESPONSES

    async def detect_intent(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """Return the predefined intent for the exact ``user_message``."""

        data = self._dataset.get(user_message)
        if data is None:
            # Unknown question, return unclear intent
            data = {
                "intent_type": "UNCLEAR_INTENT",
                "intent_category": "UNCLEAR_INTENT",
                "confidence": 0.0,
                "entities": [],
                "method": "mock_detection",
                "processing_time_ms": 0.0,
                "requires_clarification": True,
                "suggested_actions": [],
            }

        start = time.perf_counter()

        entities = []
        for ent in data.get("entities", []):
            start_pos, end_pos = ent.get("position", (None, None))
            entities.append(
                FinancialEntity(
                    entity_type=EntityType(ent["entity_type"]),
                    raw_value=ent["raw_value"],
                    normalized_value=ent["normalized_value"],
                    confidence=ent["confidence"],
                    start_position=start_pos,
                    end_position=end_pos,
                    detection_method=DetectionMethod.LLM_BASED,
                )
            )

        intent_result = IntentResult(
            intent_type=data["intent_type"],
            intent_category=IntentCategory(data["intent_category"]),
            confidence=data["confidence"],
            entities=entities,
            method=DetectionMethod.LLM_BASED,
            processing_time_ms=data.get("processing_time_ms", (time.perf_counter() - start) * 1000),
            requires_clarification=data.get("requires_clarification", False),
            suggested_actions=data.get("suggested_actions", []),
        )

        result_payload = {
            "intent": intent_result.intent_type,
            "confidence": intent_result.confidence,
            "entities": [
                {
                    "entity_type": e.entity_type.value,
                    "value": e.normalized_value,
                    "confidence": e.confidence,
                }
                for e in intent_result.entities
            ],
        }

        return {
            "content": json.dumps(result_payload),
            "metadata": {
                "intent_result": intent_result,
                "detection_method": DetectionMethod.LLM_BASED,
                "confidence": intent_result.confidence,
                "intent_type": intent_result.intent_type,
                "entities": [
                    e.model_dump() if hasattr(e, "model_dump") else e.__dict__
                    for e in intent_result.entities
                ],
            },
            "confidence_score": intent_result.confidence,
        }

