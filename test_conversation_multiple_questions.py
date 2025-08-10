"""
Test spécifique pour vérifier le fonctionnement du service de conversation
avec plusieurs échanges successifs.

Ce script envoie une série de messages au service `/conversation/chat`
et vérifie que chaque réponse est retournée avec succès et que l'ID de
conversation reste constant.

Usage:
    python test_conversation_multiple_questions.py
    python test_conversation_multiple_questions.py --base-url http://localhost:8000/api/v1
    python test_conversation_multiple_questions.py --messages "Bonjour" "Comment ça va?" "Merci"
"""

import argparse
import json
import sys
from datetime import datetime
from typing import List

import requests

DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
REQUEST_TIMEOUT = 30


def _print_step(step: int, message: str) -> None:
    print("\n" + "=" * 60)
    print(f"ÉTAPE {step}: {message}")
    print("=" * 60)


def _print_response(response: requests.Response) -> dict:
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print("Réponse JSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data
    except json.JSONDecodeError:
        print("❌ Réponse non JSON:")
        print(response.text[:500])
        return {}


def run_conversation_test(base_url: str, conversation_id: str, messages: List[str]) -> bool:
    print("🚀 DÉBUT DU TEST MULTI-CONVERSATION HARENA")
    print(f"Base URL: {base_url}")
    print(f"Conversation ID: {conversation_id}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    session = requests.Session()
    session.timeout = REQUEST_TIMEOUT

    successes = 0
    for idx, msg in enumerate(messages, start=1):
        _print_step(idx, f"ENVOI DU MESSAGE: {msg}")
        payload = {"conversation_id": conversation_id, "message": msg}
        try:
            response = session.post(
                f"{base_url.rstrip('/')}/conversation/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur de requête: {e}")
            continue

        data = _print_response(response)
        if (
            response.status_code == 200
            and data.get("success") is True
            and data.get("conversation_id") == conversation_id
        ):
            print("✅ Message traité avec succès")
            successes += 1
        else:
            print("❌ Échec du traitement du message")

    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DU TEST")
    print("=" * 60)
    print(f"Messages réussis: {successes}/{len(messages)}")

    if successes == len(messages):
        print("✅ TOUS LES MESSAGES ONT ÉTÉ TRAITÉS AVEC SUCCÈS")
        return True

    print("❌ Des messages n'ont pas été traités correctement")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de conversation multi-messages")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="URL de base de l'API Harena",
    )
    parser.add_argument(
        "--conversation-id",
        default="test-conversation-multi",
        help="Identifiant de conversation à utiliser",
    )
    parser.add_argument(
        "--messages",
        nargs="*",
        default=[
            "Bonjour",
            "Peux-tu me donner ton nom ?",
            "Merci et au revoir",
        ],
        help="Liste de messages à envoyer",
    )
    args = parser.parse_args()

    success = run_conversation_test(args.base_url, args.conversation_id, args.messages)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
