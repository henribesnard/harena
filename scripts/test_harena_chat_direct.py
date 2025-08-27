"""
Test minimal pour Harena : login → chat → analyse du workflow.
ANALYSE PURE : récupère et affiche les données internes de l'agent sans refaire de recherche.
"""

import json
import time
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"
USERNAME = "test2@example.com"
PASSWORD = "password123"
QUESTIONS = [
    "Combien ai-je fait de virements en mai ?",
    "Combien ai-je dépensé en juin ?",
    "Combien j'ai eu d'entrée d'argent en juin ?",
    "Compare mes entrées et sorties d'argent en juin !",
]

def run_question(
    session: requests.Session, user_id: int, question: str
) -> tuple[dict | None, str, str, float]:
    """Exécute une question de chat et affiche le résultat."""

    chat_payload = {
        "message": question,
        "client_info": {"platform": "web", "version": "1.0.0"},
        "message_type": "text",
        "priority": "normal",
    }
    start_time = time.perf_counter()
    intent_type = "N/A"
    confidence = "N/A"

    try:
        chat_resp = session.post(
            f"{BASE_URL}/conversation/{user_id}", json=chat_payload
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if chat_resp.status_code // 100 != 2:
            return None, intent_type, confidence, elapsed_ms

        chat_data = chat_resp.json()
        intent_result = chat_data.get("metadata", {}).get("intent_result", {})
        intent_type = intent_result.get("intent_type", "N/A")
        confidence = intent_result.get("confidence", "N/A")

        print("✅ Conversation réussie")
        print(f"🗨️ Question posée : {question}")
        print(f"💬 Réponse générée : {chat_data['message']}")

        aggregations = (
            chat_data.get("metadata", {})
            .get("workflow_data", {})
            .get("search_results", {})
            .get("metadata", {})
            .get("search_response", {})
            .get("aggregations")
        )
        if aggregations:
            print("📊 Agrégats :", json.dumps(aggregations, indent=2, ensure_ascii=False))
        print()

        return chat_data, intent_type, confidence, elapsed_ms
    except requests.RequestException:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return None, intent_type, confidence, elapsed_ms

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

    # Récupération de l'ID utilisateur
    user_resp = session.get(f"{BASE_URL}/users/me")
    user_resp.raise_for_status()
    user_id = user_resp.json().get("id")
    print(f"✅ ID utilisateur récupéré : {user_id}")

    report = []
    last_chat_data = None

    for question in QUESTIONS:
        chat_data, intent_type, confidence, elapsed_ms = run_question(
            session, user_id, question
        )
        report.append(
            {
                "question": question,
                "intent_type": intent_type,
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
            }
        )
        if chat_data:
            last_chat_data = chat_data

    print("\n📄 RAPPORT :")
    for row in report:
        print(
            f"- {row['question']} | Intent: {row['intent_type']} | "
            f"Conf: {row['confidence']} | Temps: {row['elapsed_ms']:.2f}ms"
        )

    if last_chat_data is None:
        print("\n❌ Aucune conversation réussie, arrêt de l'analyse.")
        return

    chat_data = last_chat_data

    # ----- ANALYSE DE L'INTENTION DÉTECTÉE ----------------------------------
    intent_result = chat_data["metadata"]["intent_result"]
    print("🧠 ANALYSE DE L'INTENTION :")
    print(f"   🎯 Type : {intent_result['intent_type']}")
    print(f"   🎲 Confiance : {intent_result.get('confidence', 'N/A')}")
    print(f"   ⚡ Méthode : {intent_result.get('method', 'N/A')}")
    print()

    # ----- ANALYSE DES ENTITÉS EXTRAITES -----------------------------------
    entities = intent_result.get('entities', [])
    print("🧩 ENTITÉS EXTRAITES :")
    if entities:
        for i, entity in enumerate(entities, 1):
            print(f"   {i}. {entity['entity_type']} :")
            print(f"      📝 Valeur brute : '{entity['raw_value']}'")
            print(f"      🔄 Valeur normalisée : '{entity['normalized_value']}'")
            print(f"      🎯 Confiance : {entity['confidence']}")
            print(f"      🔍 Méthode : {entity.get('detection_method', 'N/A')}")
            print(f"      📍 Position : {entity.get('start_position', '?')}-{entity.get('end_position', '?')}")
    else:
        print("   ❌ Aucune entité détectée")
    print()

    # ----- ANALYSE DU TYPE D'AGENT UTILISÉ ----------------------------------
    metrics_resp = session.get(f"{BASE_URL}/conversation/metrics")
    metrics_resp.raise_for_status()
    metrics_data = metrics_resp.json()
    
    print("🤖 AGENTS UTILISÉS :")
    agent_perf = metrics_data.get("agent_metrics", {}).get("agent_performance", {})
    
    if "intent_agent" in agent_perf:
        intent_agent = agent_perf["intent_agent"]
        agent_type = intent_agent.get("agent_type", "Unknown")
        print(f"   🧠 Agent d'intention : {agent_type}")
        
        if agent_type == "MockIntentAgent":
            print("      ✅ Mock agent utilisé (mode test)")
        else:
            print(f"      ℹ️  Agent réel utilisé : {agent_type}")
    
    # Autres agents
    for agent_name, agent_data in agent_perf.items():
        if agent_name != "intent_agent":
            print(f"   🔧 {agent_name} : {agent_data.get('agent_type', 'Unknown')}")
    print()

    # ----- ANALYSE DE LA RECHERCHE EFFECTUÉE --------------------------------
    print("🔍 ANALYSE DE LA RECHERCHE :")
    
    # Extraire les informations de recherche depuis les métadonnées
    metadata = chat_data.get("metadata", {})
    workflow_data = metadata.get("workflow_data", {})

    search_results_count = 0
    if isinstance(workflow_data, dict):
        search_results_count = workflow_data.get("search_results_count", 0)

    print(f"   📊 Résultats trouvés par l'agent : {search_results_count}")
    
    # Analyser les entités pour comprendre la requête générée
    merchant = None
    date_filter = None
    
    for entity in entities:
        if entity['entity_type'] == 'MERCHANT':
            merchant = entity['normalized_value']
        elif entity['entity_type'] in ['RELATIVE_DATE', 'DATE']:
            date_filter = entity['normalized_value']
    
    print(f"   🏪 Marchand recherché : {merchant or 'N/A'}")
    print(f"   📅 Filtre temporel : {date_filter or 'N/A'}")
    
    # Déduire la requête probable de l'agent
    if date_filter == 'current_month':
        now = datetime.now()
        expected_filter = f"{now.year}-{now.month:02d}-01 à {now.year}-{now.month:02d}-31"
        print(f"   🗓️  Période déduite : {expected_filter}")
    
    print()

    # ----- ANALYSE DES PERFORMANCES -----------------------------------------
    processing_time = chat_data.get("processing_time_ms", 0)
    print("⚡ PERFORMANCES :")
    print(f"   ⏱️  Temps total : {processing_time}ms")
    
    if "orchestrator_performance" in metrics_data.get("agent_metrics", {}):
        orch_perf = metrics_data["agent_metrics"]["orchestrator_performance"]
        exec_times = orch_perf.get("execution_times", {})
        print(f"   🎭 Temps moyen orchestrateur : {exec_times.get('average_ms', 'N/A')}ms")
    
    # Temps par agent
    for agent_name, agent_data in agent_perf.items():
        exec_times = agent_data.get("execution_times", {})
        avg_time = exec_times.get("average_ms", 0)
        if avg_time > 0:
            print(f"   🔧 {agent_name} : {avg_time}ms")
    print()

    # ----- ANALYSE DE LA QUALITÉ DE LA RÉPONSE ------------------------------
    response_text = chat_data['message']
    print("✨ ANALYSE DE LA RÉPONSE :")
    print(f"   📏 Longueur : {len(response_text)} caractères")
    print(f"   🔤 Mots : ~{len(response_text.split())} mots")
    
    # Indicateurs de qualité
    quality_indicators = {
        "Structure markdown": "**" in response_text or "#" in response_text,
        "Actions suggérées": "Actions suggérées" in response_text or "suggestions" in response_text.lower(),
        "Insights/Observations": "insights" in response_text.lower() or "observations" in response_text.lower(),
        "Liens/références": "http" in response_text or "www." in response_text,
        "Contextualisation temporelle": any(month in response_text.lower() for month in ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'])
    }
    
    print("   📋 Indicateurs de qualité :")
    for indicator, present in quality_indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")
    print()

    # ----- RÉSUMÉ EXÉCUTIF ---------------------------------------------------
    print("📋 RÉSUMÉ EXÉCUTIF :")
    print(f"   🎯 Intention correctement détectée : {'✅' if intent_result['intent_type'] == 'SEARCH_BY_AMOUNT' else '❌'}")
    print(f"   🧩 Entités extraites : {'✅' if len(entities) > 0 else '❌'}")
    print(f"   🔍 Recherche exécutée : {'✅' if 'search_results_count' in chat_data['metadata'] else '❌'}")
    print(f"   💬 Réponse générée : {'✅' if len(response_text) > 50 else '❌'}")
    print(f"   ⚡ Performance acceptable : {'✅' if processing_time < 30000 else '❌'} ({processing_time}ms)")
    
    # Cohérence globale
    coherence_score = sum([
        intent_result['intent_type'] == 'SEARCH_BY_AMOUNT',
        len(entities) > 0,
        'search_results_count' in chat_data['metadata'],
        len(response_text) > 50,
        processing_time < 30000
    ])
    
    print(f"   🏆 Score de cohérence : {coherence_score}/5")
    
    if coherence_score >= 4:
        print("   🎉 Workflow fonctionnel et cohérent !")
    elif coherence_score >= 3:
        print("   ⚠️  Workflow fonctionnel avec améliorations possibles")
    else:
        print("   ❌ Problèmes détectés dans le workflow")

if __name__ == "__main__":
    main()

