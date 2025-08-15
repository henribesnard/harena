"""
Test minimal pour Harena : login → chat → vérifications.
Affiche l'authentification, l'accès au chat, l'utilisation du MockIntentAgent,
l'intention détectée, la requête envoyée au SearchService et le résultat obtenu.
"""

import base64
import json
import requests
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/api/v1"
USERNAME = "test2@example.com"
PASSWORD = "password123"
QUESTION = "Mes transactions Netflix ce mois"


def _decode_jwt(token: str) -> dict:
    """Décodage manuel du payload JWT (sans vérification de signature)."""
    payload = token.split(".")[1]
    padding = "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload + padding).decode())


def main() -> None:
    session = requests.Session()

    # ----- AUTHENTIFICATION --------------------------------------------------
    data = f"username={USERNAME}&password={PASSWORD}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = session.post(f"{BASE_URL}/users/auth/login", data=data, headers=headers)
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print("✅ OK client authentifié")

    session.headers.update({"Authorization": f"Bearer {token}"})
    user_id = int(_decode_jwt(token)["sub"])

    # ----- CHAT DIRECT -------------------------------------------------------
    chat_payload = {"message": QUESTION, "conversation_id": "test-chat-direct"}
    chat_resp = session.post(f"{BASE_URL}/conversation/chat", json=chat_payload)
    chat_resp.raise_for_status()
    chat_data = chat_resp.json()
    print("✅ Ok accès au chat")
    print(f"🗨️ Question posée : {QUESTION}")
    print(f"💬 Réponse conversation : {chat_data['message']}")   # <— ligne ajoutée

    intent_result = chat_data["metadata"]["intent_result"]
    print(f"🤖 Intention détectée : {intent_result['intent_type']}")
    print(f"🧩 Entités : {json.dumps(intent_result.get('entities', []), indent=2, ensure_ascii=False)}")

    # ----- VÉRIFICATION DU MOCK ---------------------------------------------
    metrics_resp = session.get(f"{BASE_URL}/conversation/metrics")
    metrics_resp.raise_for_status()
    metrics_data = metrics_resp.json()
    if "intent_agent" in metrics_data["agent_metrics"]["agent_performance"]:
        intent_agent_type = metrics_data["agent_metrics"]["agent_performance"]["intent_agent"][
            "agent_type"
        ]
        if intent_agent_type == "MockIntentAgent":
            print("✅ Ok si mock utilisé")
        else:
            print(f"❌ Mock non utilisé (agent_type={intent_agent_type})")
    else:
        print("❌ Mock non utilisé (agent_type inconnu)")

    # ----- RECHERCHE EFFECTUÉE PAR L'AGENT ----------------------------------
    merchant = next(
        (e["normalized_value"] for e in intent_result.get("entities", []) if e["entity_type"] == "MERCHANT"),
        None,
    )

    # Le filtre ``merchant_name`` est volontairement omis lors de l'appel direct
    # au Search Service. Certaines bases de données peuvent ne pas renseigner ce
    # champ pour toutes les transactions, ce qui exclurait à tort des résultats
    # pertinents si le filtre était appliqué. Le texte de recherche reste
    # suffisant pour trouver les transactions correspondantes.
    search_payload = {
        "user_id": user_id,
        "query": merchant or QUESTION,
        "limit": 20,
        "offset": 0,
    }
    print(f"📨 Message envoyé à l'agent de recherche : {json.dumps(search_payload, ensure_ascii=False)}")

    search_resp = session.post(f"{BASE_URL}/search/search", json=search_payload)
    search_resp.raise_for_status()
    search_data = search_resp.json()
    meta = search_data.get("response_metadata", {})
    print(
        f"🔎 Recherche effectuée : {meta.get('returned_results', 0)}/{meta.get('total_results', 0)} résultat(s)"
    )

    conv_results = chat_data["metadata"].get("search_results_count", 0)
    print(f"📊 Résultat retourné par la conversation : {conv_results} résultat(s)")

    if search_data.get("results"):
        print("📄 Détails des résultats :")
        for r in search_data["results"]:
            print(
                f" - {r['primary_description']} | montant {r['amount']} {r['currency_code']} | date {r['date']}"
            )


if __name__ == "__main__":
    main()