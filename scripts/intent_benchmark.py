import asyncio
import os
import time
from pathlib import Path
from typing import Dict

from quick_intent_test import HarenaIntentAgent, CATEGORY_MAP


def parse_intents_md(path: Path) -> Dict[str, str]:
    """Parse INTENTS.md and return mapping of intent_type to category."""
    intents: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| ---") or "Intent Type" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        intent, category = parts[0], parts[1]
        if category.startswith("UNSUPPORTED"):
            category = "UNCLEAR_INTENT"
        category = CATEGORY_MAP.get(category, category)
        intents[intent] = category
    return intents


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
    "ERROR": "��",
}


# Intents that are not yet supported by the system. If the model predicts
# ``UNSUPPORTED`` for any of these intents (with a matching category), the
# prediction is still considered correct.
UNSUPPORTED_INTENTS = {
    "TRANSFER_REQUEST",
    "PAYMENT_REQUEST",
    "CARD_BLOCK",
    "BUDGET_INQUIRY",
    "GOAL_TRACKING",
    "EXPORT_REQUEST",
    "OUT_OF_SCOPE",
}


async def run_benchmark() -> None:
    intents = parse_intents_md(Path("INTENTS.md"))

    api_key = os.getenv("OPENAI_API_KEY", "")
    agent = HarenaIntentAgent(api_key=api_key)

    results = []
    for intent_type, query in INTENT_QUERIES.items():
        expected_category = intents.get(intent_type, "UNKNOWN")
        start_time = time.perf_counter()
        res = await agent.detect_intent_async(query)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        results.append(
            {
                "intent": intent_type,
                "expected_category": expected_category,
                "predicted_intent": res.intent_type,
                "predicted_category": res.intent_category,
                "confidence": res.confidence,
                "latency_ms": elapsed_ms,
            }
        )

    total = len(results)
    success = sum(
        1
        for r in results
        if r["expected_category"] == r["predicted_category"]
        and (
            r["intent"] == r["predicted_intent"]
            or (
                r["predicted_intent"] == "UNSUPPORTED"
                and r["intent"] in UNSUPPORTED_INTENTS
            )
        )
    )
    avg_conf = sum(r["confidence"] for r in results) / total
    latencies = [r["latency_ms"] for r in results]
    avg_lat = sum(latencies) / total
    min_lat = min(latencies)
    max_lat = max(latencies)

    print("=== Intent Benchmark Report ===")
    print(f"Total tests: {total}")
    print(f"Overall success rate: {success / total:.1%}")
    print(f"Average confidence: {avg_conf:.2f}")
    print(
        f"Latency (ms) -> avg: {avg_lat:.1f}, min: {min_lat:.1f}, max: {max_lat:.1f}"
    )
    print()
    header = (
        f"{'Intent':<25} {'Expected Cat':<15} {'Predicted Intent':<25} "
        f"{'Pred Cat':<15} {'Conf':<6} {'Latency(ms)':<11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['intent']:<25} {r['expected_category']:<15} {r['predicted_intent']:<25} "
            f"{r['predicted_category']:<15} {r['confidence']:<6.2f} {r['latency_ms']:<11.1f}"
        )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
