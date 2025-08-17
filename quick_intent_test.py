#!/usr/bin/env python3
"""
Benchmark simple pour l'agent de détection d'intentions financières.
- Appelle l'API Responses avec Structured Outputs (JSON Schema strict)
- Mesure la latence réelle
- Valide le JSON retourné (jsonschema)
- Calcule un score de fiabilité (taux de réponses conformes au schéma)
- Imprime un rapport synthétique

Dépendances:
    pip install openai python-dotenv jsonschema

Usage:
    python intent_benchmark.py --model gpt-4.1-mini --runs 40 --shuffle
"""

import os
import json
import time
import math
import random
import argparse
import statistics as stats
from collections import Counter

import openai
from packaging import version
from dotenv import load_dotenv
from jsonschema import Draft7Validator, ValidationError
from openai import OpenAI

REQUIRED_OPENAI = "1.1.0"
if version.parse(openai.__version__) < version.parse(REQUIRED_OPENAI):
    raise RuntimeError(
        f"openai>={REQUIRED_OPENAI} required, but {openai.__version__} is installed. "
        "Please update the openai package."
    )

# -----------------------------
# 1) JSON Schema & validateur précompilé
# -----------------------------
INTENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "title": "IntentResult",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent_type", "intent_category", "confidence", "entities",
        "method", "processing_time_ms", "requires_clarification",
        "search_required", "raw_user_message"
    ],
    "properties": {
        "intent_type": {
            "type": "string",
            "enum": [
                "TRANSACTION_SEARCH", "TRANSACTION_SEARCH_BY_DATE", "TRANSACTION_SEARCH_BY_AMOUNT_AND_DATE",
                "SPENDING_ANALYSIS", "CATEGORY_ANALYSIS", "BUDGET_INQUIRY", "BUDGET_TRACKING", "TREND_ANALYSIS",
                "ACCOUNT_BALANCE", "BALANCE_INQUIRY", "MERCHANT_INQUIRY", "COMPARISON_QUERY", "GOAL_TRACKING",
                "GENERAL", "FINANCIAL_QUERY",
                "CONVERSATIONAL", "EXPORT_REQUEST", "UNCLEAR_INTENT", "GREETING", "OUT_OF_SCOPE",
                "FALLBACK_INTENT", "TEST_INTENT", "ERROR", "UNKNOWN"
            ]
        },
        "intent_category": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "entity_type", "raw_value", "normalized_value",
                    "confidence", "detection_method", "validation_status"
                ],
                "properties": {
                    "entity_type": {"type": "string"},
                    "raw_value": {"type": "string"},
                    "normalized_value": {},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "detection_method": {"type": "string"},
                    "validation_status": {"type": "string", "enum": ["valid", "invalid", "ambiguous"]}
                }
            }
        },
        "method": {"type": "string"},
        "processing_time_ms": {"type": "number", "minimum": 1},
        "requires_clarification": {"type": "boolean"},
        "search_required": {"type": "boolean"},
        "raw_user_message": {"type": "string"},
        "suggested_actions": {"type": "array", "items": {"type": "string"}},
        "additional_data": {"type": "object"}
    }
}

INTENT_VALIDATOR = Draft7Validator(INTENT_SCHEMA)

# -----------------------------
# 2) Prompt système
# -----------------------------
SYSTEM_PROMPT = """Tu es un classifieur d’intentions financières en français.
Tu dois produire STRICTEMENT un objet JSON valide respectant le JSON Schema fourni (toutes les clés requises).
Règles importantes:
- Choisis `intent_type` dans la liste autorisée (enum du schéma).
- `intent_category`:
    - Pour requêtes financières (recherche, analyse, budget, solde, etc.): "FINANCIAL_QUERY".
    - Pour salutation/conversation: "GREETING".
    - Pour hors-périmètre: "OUT_OF_SCOPE".
- `requires_clarification` = true si la requête est ambiguë/incomplète.
- `search_required` = true pour: TRANSACTION_SEARCH*, SPENDING_ANALYSIS, CATEGORY_ANALYSIS, BUDGET_*, TREND_ANALYSIS,
  ACCOUNT_BALANCE/BALANCE_INQUIRY, MERCHANT_INQUIRY, COMPARISON_QUERY, GOAL_TRACKING, GENERAL, FINANCIAL_QUERY.
  false pour: CONVERSATIONAL, EXPORT_REQUEST, UNCLEAR_INTENT, GREETING, OUT_OF_SCOPE, FALLBACK_INTENT, TEST_INTENT, ERROR, UNKNOWN.
- `entities` peut être vide si rien d’extractible avec confiance.
- `processing_time_ms` réaliste 50–300 pour ce classifieur (valeur indicative).
- `method` = "llm_based".
- `raw_user_message` = EXACTEMENT la question utilisateur.
- La sortie DOIT être uniquement l’objet JSON (pas de texte en dehors).
"""

# -----------------------------
# 3) Jeu de questions de test
# -----------------------------
TEST_QUESTIONS = [
    # SPENDING_ANALYSIS
    "Combien j'ai dépensé chez Carrefour le mois dernier ?",
    "Total des dépenses restaurants cette semaine ?",
    # CATEGORY_ANALYSIS
    "Analyse mes dépenses en transport sur les 30 derniers jours.",
    # TRANSACTION_SEARCH*
    "Montre-moi mes transactions Uber d’hier.",
    "Liste les transactions du 10 juillet 2025.",
    "Trouve les opérations supérieures à 100€ en juin.",
    # BUDGET_*
    "Quel est mon budget courses ce mois-ci ?",
    "Ai-je dépassé mon budget transport ?",
    # TREND_ANALYSIS
    "Mes factures d'électricité augmentent-elles depuis trois mois ?",
    # BALANCE / ACCOUNT_BALANCE
    "Quel est le solde de mon compte courant ?",
    # MERCHANT_INQUIRY
    "Combien ai-je dépensé chez Amazon en 2024 ?",
    # COMPARISON_QUERY
    "Ai-je dépensé plus en janvier qu'en février ?",
    # GOAL_TRACKING
    "Où en est mon objectif d’épargne vacances ?",
    # GENERAL & FINANCIAL_QUERY
    "Donne-moi un résumé de mes finances récentes.",
    "Combien j'ai dépensé récemment ?",
    # EXPORT_REQUEST
    "Exporte mes transactions de mai en CSV.",
    # CONVERSATIONAL / GREETING
    "Bonjour",
    "Merci beaucoup !",
    # UNCLEAR / OUT_OF_SCOPE / FALLBACK / TEST / ERROR / UNKNOWN
    "Peux-tu vérifier cela ?",
    "Quel temps fait-il à Paris ?",
    "blabla ???",
    "test test",
    "???"
]

# -----------------------------
# 4) Client & appel Responses API
# -----------------------------
def make_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant. Créez un fichier .env avec OPENAI_API_KEY=...")
    return OpenAI(api_key=api_key)


def detect_intent(client: OpenAI, question: str, model: str):
    """
    Appelle l'API Responses avec Structured Outputs (json_schema strict),
    renvoie (parsed_json, measured_latency_ms, raw_response).
    """
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": question}]},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "IntentResult",
                "schema": INTENT_SCHEMA,
                "strict": True,
            },
        },
    )
    t1 = time.perf_counter()
    measured_ms = round((t1 - t0) * 1000.0, 2)

    parsed = resp.output_parsed
    if parsed is None:
        try:
            parsed = json.loads(resp.output_text or "")
        except Exception as e:
            raise ValueError(f"Réponse non parsable en JSON: {e}")

    return parsed, measured_ms, resp

# -----------------------------
# 5) Validation & métriques
# -----------------------------
def validate_intent_result(obj: dict):
    """Valide la sortie via jsonschema. Retourne (ok: bool, error_msg: str|None)."""
    try:
        INTENT_VALIDATOR.validate(obj)
        return True, None
    except ValidationError as e:
        return False, str(e)


def p95(values):
    if not values:
        return 0.0
    s = sorted(values)
    k = int(math.ceil(0.95 * len(s))) - 1
    k = min(max(k, 0), len(s) - 1)
    return float(s[k])


def run_benchmark(model: str, runs: int, shuffle: bool):
    client = make_client()
    questions = TEST_QUESTIONS[:]
    if shuffle:
        random.shuffle(questions)

    if runs and runs < len(questions):
        questions = questions[:runs]
    elif runs and runs > len(questions):
        extra = []
        i = 0
        while len(questions) + len(extra) < runs:
            extra.append(TEST_QUESTIONS[i % len(TEST_QUESTIONS)])
            i += 1
        questions = questions + extra

    print(f"\n▶ Exécution: {len(questions)} questions — modèle: {model}\n")

    latencies = []
    confidences = []
    intents = []
    needs_clarif = 0
    search_true = 0

    ok_count = 0
    failures = []

    for idx, q in enumerate(questions, 1):
        print(f"[{idx:02d}/{len(questions)}] Q: {q}")
        try:
            parsed, measured_ms, _resp = detect_intent(client, q, model=model)
            ok, err = validate_intent_result(parsed)
            latencies.append(measured_ms)

            if ok:
                ok_count += 1
                confidences.append(parsed.get("confidence", 0.0))
                intents.append(parsed.get("intent_type", ""))
                if parsed.get("requires_clarification") is True:
                    needs_clarif += 1
                if parsed.get("search_required") is True:
                    search_true += 1
                print(f"    ✔ Valide | lat={measured_ms} ms | intent={parsed.get('intent_type')} | conf={parsed.get('confidence')}")
            else:
                failures.append({"question": q, "error": err, "parsed": parsed})
                print(f"    ✘ Invalide | lat={measured_ms} ms | erreur schéma: {err}")

        except Exception as e:
            failures.append({"question": q, "error": str(e), "parsed": None})
            print(f"    ✘ Erreur appel API: {e}")

    total = len(questions)
    reliability = (ok_count / total) if total else 0.0

    print("\n================= RAPPORT =================")
    print(f"Total questions     : {total}")
    print(f"Valides (schéma)    : {ok_count}")
    print(f"Invalides/Erreurs   : {total - ok_count}")
    print(f"Score de fiabilité  : {reliability:.2%}")

    if latencies:
        print(f"Latence moyenne     : {stats.mean(latencies):.1f} ms")
        print(f"Latence médiane     : {stats.median(latencies):.1f} ms")
        print(f"Latence p95         : {p95(latencies):.1f} ms")

    if confidences:
        print(f"Confiance moyenne   : {stats.mean(confidences):.3f}")

    if intents:
        cnt = Counter(intents)
        top = ", ".join(f"{k}:{v}" for k, v in cnt.most_common())
        print(f"Répartition intents : {top}")

    print(f"Clarifications (true): {needs_clarif}")
    print(f"search_required=true : {search_true}")

    if failures:
        print("\n-- Exemples d’échecs --")
        for f in failures[:3]:
            print(f"* Q: {f['question']}\n  Erreur: {f['error']}\n")

    print("===========================================\n")
    return reliability

# -----------------------------
# 6) CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark intent detector (OpenAI Responses + Structured Outputs)")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini",
                    help="Nom du modèle (ex: gpt-4.1-mini, o3-mini, etc.)")
    ap.add_argument("--runs", type=int, default=0,
                    help="Nombre de questions à exécuter (0 = toutes, >len => boucle).")
    ap.add_argument("--shuffle", action="store_true",
                    help="Mélanger l'ordre des questions.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(model=args.model, runs=args.runs, shuffle=args.shuffle)
