import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import pytest
import sys
import types


class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass


sys.modules["openai"] = types.SimpleNamespace(
    OpenAI=_DummyClient, AsyncOpenAI=_DummyClient
)
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

from scripts.quick_intent_test import HarenaIntentAgent
from scripts.intent_utils import parse_intents_md

THRESHOLD = 0.8


# Representative user queries for each intent
INTENT_QUERIES: Dict[str, str] = {
    "TRANSACTION_SEARCH": "Liste toutes mes transactions",
    "SEARCH_BY_DATE": "Transactions de mars 2024",
    "SEARCH_BY_AMOUNT": "Transactions de 50 euros",
    "SEARCH_BY_MERCHANT": "Transactions chez Carrefour",
    "SEARCH_BY_CATEGORY": "Transactions de la catégorie restaurants",
    "SEARCH_BY_AMOUNT_AND_DATE": "Achats de plus de 100 euros en janvier 2024",
    "SEARCH_BY_OPERATION_TYPE": "Transactions par carte bancaire",
    "SEARCH_BY_TEXT": "Recherche les transactions contenant Netflix",
    "COUNT_TRANSACTIONS": "Combien de transactions en février ?",
    "MERCHANT_INQUIRY": "Analyse des dépenses chez Amazon",
    "FILTER_REQUEST": "Seulement les débits pas les crédits",
    "SPENDING_ANALYSIS": "Analyse de mes dépenses",
    "SPENDING_ANALYSIS_BY_CATEGORY": "Analyse des dépenses alimentaires",
    "SPENDING_ANALYSIS_BY_PERIOD": "Analyse de mes dépenses la semaine dernière",
    "SPENDING_COMPARISON": "Compare mes dépenses de janvier et février",
    "TREND_ANALYSIS": "Tendance de mes dépenses cette année",
    "CATEGORY_ANALYSIS": "Répartition de mes dépenses par catégorie",
    "COMPARISON_QUERY": "Compare restaurants et courses",
    "BALANCE_INQUIRY": "Quel est mon solde actuel ?",
    "ACCOUNT_BALANCE_SPECIFIC": "Solde de mon compte courant",
    "BALANCE_EVOLUTION": "Évolution de mon solde",
    "GREETING": "Bonjour",
    "CONFIRMATION": "Merci beaucoup",
    "CLARIFICATION": "Peux-tu préciser ?",
    "GENERAL_QUESTION": "Quel temps fait-il ?",
    "TRANSFER_REQUEST": "Transfère 100 euros à Paul",
    "PAYMENT_REQUEST": "Paye ma facture d'électricité",
    "CARD_BLOCK": "Bloque ma carte bancaire",
    "BUDGET_INQUIRY": "Où en est mon budget ?",
    "GOAL_TRACKING": "Quel est l'état de mon objectif d'épargne ?",
    "EXPORT_REQUEST": "Exporte mes transactions en CSV",
    "OUT_OF_SCOPE": "Donne-moi une recette de cuisine",
    "UNCLEAR_INTENT": "Je ne sais pas fais quelque chose",
    "UNKNOWN": "blabla ???",
    "TEST_INTENT": "[TEST] ping",
    "ERROR": "\ufffd\ufffd",
}


class DummyUsage:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0


class DummyResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = DummyUsage()


def make_dummy_client(mapping: Dict[str, Tuple[str, str]]):
    async def create(*, messages, **kwargs):
        user_prompt = messages[1]["content"]
        user_message = user_prompt.split("IntentResult: ", 1)[1]
        intent_type, category = mapping[user_message]
        response_json = json.dumps(
            {
                "intent_type": intent_type,
                "intent_category": category,
                "confidence": 0.95,
                "entities": [],
            }
        )
        return DummyResponse(response_json)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_intents_full():
    intents = parse_intents_md(Path("INTENTS.md"))
    missing = set(intents) - set(INTENT_QUERIES)
    assert not missing, f"Manque des requêtes pour: {missing}"

    # Build mapping from query to expected (intent, category)
    query_mapping = {INTENT_QUERIES[i]: (i, intents[i]) for i in intents}

    async def run_tests():
        agent = HarenaIntentAgent(api_key="test")
        agent.async_client = make_dummy_client(query_mapping)

        results = []
        for intent, query in INTENT_QUERIES.items():
            expected_category = intents[intent]
            res = await agent.detect_intent_async(query)
            results.append((intent, expected_category, res))

        success_count = 0
        confidence_total = 0.0
        per_intent_conf: Dict[str, list] = {i: [] for i in intents}

        for intent, expected_category, res in results:
            ok = (
                res.intent_type == intent
                and res.intent_category == expected_category
                and res.confidence >= THRESHOLD
            )
            if ok:
                success_count += 1
            confidence_total += res.confidence
            per_intent_conf[intent].append(res.confidence)

        overall_success = success_count / len(results)
        overall_conf = confidence_total / len(results)

        print("\n=== Rapport d'intentions ===")
        print(f"Taux de réussite global: {overall_success:.1%}")
        print(f"Confiance moyenne globale: {overall_conf:.2f}")
        for intent, confs in per_intent_conf.items():
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            print(f"- {intent}: {avg_conf:.2f}")

        for intent, expected_category, res in results:
            assert res.intent_type == intent
            assert res.intent_category == expected_category
            assert res.confidence >= THRESHOLD

    asyncio.run(run_tests())
