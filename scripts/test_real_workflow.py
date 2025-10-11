"""
Test Real Workflow - Vérification intégration complète avec base de données

Ce script teste:
1. Requête réelle à la base de données via l'API
2. Vérification que les insights Analytics Agent apparaissent
3. Vérification que les recommendations apparaissent
4. Affichage structuré des résultats pour validation
"""

import asyncio
import os
import sys
from datetime import datetime

# Token JWT pour l'authentification
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjA2NzgwNTMsInN1YiI6IjEwMCIsInBlcm1pc3Npb25zIjpbImNoYXQ6d3JpdGUiXX0.M1WTO7bX9MelzBcp0IszPCiqAoDQB-EeWda1MkB14wc"

# Questions de test complexes
TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": "Montre moi toutes mes transactions",
        "expected_insights": ["trend_analysis", "unusual_transaction", "recommendation"],
        "description": "Test détection tendances + anomalies + recommendations"
    },
    {
        "id": "Q2",
        "question": "Quelles sont mes transactions anormales?",
        "expected_insights": ["unusual_transaction"],
        "description": "Test détection anomalies avec Analytics Agent"
    },
    {
        "id": "Q3",
        "question": "Recommande-moi des économies",
        "expected_insights": ["recommendation"],
        "description": "Test génération recommendations avec Recommendation Engine"
    }
]


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def print_subsection(title):
    """Print formatted subsection"""
    print("\n" + "-"*80)
    print(title)
    print("-"*80)


async def test_question(question_data, base_url="http://localhost:8000"):
    """Test une question et analyse la réponse"""

    import aiohttp

    question_id = question_data["id"]
    question = question_data["question"]
    expected = question_data["expected_insights"]
    description = question_data["description"]

    print_subsection(f"{question_id}: {description}")
    print(f"Question: \"{question}\"")

    url = f"{base_url}/api/v1/conversation/100"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "message": question
    }

    try:
        async with aiohttp.ClientSession() as session:
            print(f"\n[INFO] Envoi requête à {url}...")

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"[ERROR] HTTP {response.status}: {error_text}")
                    return {
                        "success": False,
                        "question_id": question_id,
                        "error": f"HTTP {response.status}"
                    }

                data = await response.json()

                # Analyser la réponse
                result = analyze_response(question_id, question, data, expected)
                return result

    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "question_id": question_id,
            "error": str(e)
        }


def analyze_response(question_id, question, data, expected_insights):
    """Analyse la réponse API et vérifie les insights"""

    print("\n[RESPONSE ANALYSIS]")

    # 1. Status général
    status = data.get("status", "unknown")
    success = status == "completed"

    print(f"Status: {status}")
    print(f"Architecture: {data.get('architecture', 'unknown')}")
    print(f"Processing time: {data.get('processing_time_ms', 0)}ms")

    # 2. Intent détecté
    intent = data.get("intent", {})
    if intent:
        print(f"\nIntent détecté:")
        print(f"  Type: {intent.get('type', 'UNKNOWN')}")
        print(f"  Confidence: {intent.get('confidence', 0):.2f}")
        entities = intent.get('entities', [])
        if entities:
            print(f"  Entities: {len(entities)} extraites")

    # 3. Recherche
    search_summary = data.get("search_summary", {})
    found_results = search_summary.get("found_results", False)
    total_results = search_summary.get("total_results", 0)

    print(f"\nRecherche:")
    print(f"  Résultats trouvés: {'Oui' if found_results else 'Non'}")
    print(f"  Total résultats: {total_results}")

    # 4. Réponse
    response = data.get("response", {})
    response_message = response.get("message", "")
    structured_data = response.get("structured_data", [])

    print(f"\nRéponse:")
    print(f"  Longueur message: {len(response_message)} caractères")
    print(f"  Insights générés: {len(structured_data)}")

    # 5. ANALYSE DES INSIGHTS (CRITÈRE PRINCIPAL)
    print("\n" + "="*60)
    print("INSIGHTS ANALYSIS".center(60))
    print("="*60)

    insights_found = {}
    insights_details = []

    if structured_data:
        for i, insight in enumerate(structured_data, 1):
            insight_type = insight.get("type", "unknown")
            title = insight.get("title", "N/A")
            description = insight.get("description", "")
            confidence = insight.get("confidence", 0)
            actionable = insight.get("actionable", False)
            data_support = insight.get("data_support", {})

            print(f"\n{i}. {insight_type.upper()}")
            print(f"   Title: {title}")
            print(f"   Description: {description[:100]}{'...' if len(description) > 100 else ''}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Actionable: {actionable}")

            # Détails spécifiques selon le type
            if insight_type == "trend_analysis":
                slope = data_support.get("slope", 0)
                r_squared = data_support.get("r_squared", 0)
                trend_direction = data_support.get("trend_direction", "unknown")
                print(f"   [Analytics Agent] Slope: {slope:.2f}%, R²: {r_squared:.2f}, Direction: {trend_direction}")

            elif insight_type == "unusual_transaction":
                anomaly_score = data_support.get("anomaly_score", 0)
                method = data_support.get("method", "unknown")
                amount = data_support.get("amount", 0)
                merchant = data_support.get("merchant", "N/A")
                print(f"   [Analytics Agent] Anomaly: {amount}€ at {merchant}, Score: {anomaly_score:.2f}, Method: {method}")

            elif insight_type == "recommendation":
                rec_type = data_support.get("recommendation_type", "unknown")
                estimated_savings = data_support.get("estimated_savings")
                cta_text = data_support.get("cta_text", "N/A")
                print(f"   [Recommendation Engine] Type: {rec_type}")
                if estimated_savings:
                    print(f"   [Recommendation Engine] Estimated savings: {estimated_savings:.2f}€")
                print(f"   [Recommendation Engine] CTA: {cta_text}")

            insights_found[insight_type] = True
            insights_details.append({
                "type": insight_type,
                "title": title,
                "confidence": confidence,
                "actionable": actionable
            })
    else:
        print("\n[WARNING] Aucun insight généré!")

    # 6. VÉRIFICATION DES INSIGHTS ATTENDUS
    print("\n" + "="*60)
    print("EXPECTED INSIGHTS CHECK".center(60))
    print("="*60 + "\n")

    all_expected_found = True

    for expected_type in expected_insights:
        found = insights_found.get(expected_type, False)
        status_icon = "[OK]" if found else "[MISSING]"
        print(f"{status_icon} {expected_type}: {'Found' if found else 'NOT FOUND'}")

        if not found:
            all_expected_found = False

    # 7. RÉSUMÉ
    print("\n" + "="*60)
    print("TEST RESULT".center(60))
    print("="*60 + "\n")

    test_passed = success and found_results and len(structured_data) > 0

    if test_passed and all_expected_found:
        print(f"[PASS] {question_id}: All criteria met!")
        print(f"  - API call successful")
        print(f"  - {len(structured_data)} insights generated")
        print(f"  - All expected insights found")
    elif test_passed:
        print(f"[PARTIAL] {question_id}: API works but missing expected insights")
        print(f"  - API call successful")
        print(f"  - {len(structured_data)} insights generated")
        print(f"  - Some expected insights missing")
    else:
        print(f"[FAIL] {question_id}: Test failed")
        if not success:
            print(f"  - API call failed")
        if not found_results:
            print(f"  - No search results")
        if len(structured_data) == 0:
            print(f"  - No insights generated")

    return {
        "success": test_passed,
        "question_id": question_id,
        "question": question,
        "status": status,
        "total_results": total_results,
        "insights_count": len(structured_data),
        "insights_found": insights_found,
        "insights_details": insights_details,
        "all_expected_found": all_expected_found,
        "test_passed": test_passed and all_expected_found
    }


async def main():
    """Run all tests"""

    print_section("TEST WORKFLOW COMPLET - INTÉGRATION ANALYTICS AGENT & RECOMMENDATION ENGINE")

    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User ID: 100")
    print(f"Base URL: http://localhost:8000")
    print(f"Tests: {len(TEST_QUESTIONS)} questions")

    results = []

    # Tester toutes les questions
    for question_data in TEST_QUESTIONS:
        try:
            result = await test_question(question_data)
            results.append(result)

            # Pause entre les questions
            await asyncio.sleep(1)

        except Exception as e:
            print(f"\n[ERROR] Test {question_data['id']} failed: {str(e)}")
            results.append({
                "success": False,
                "question_id": question_data["id"],
                "error": str(e)
            })

    # RAPPORT FINAL
    print_section("RAPPORT FINAL")

    passed = sum(1 for r in results if r.get("test_passed", False))
    partial = sum(1 for r in results if r.get("success", False) and not r.get("test_passed", False))
    failed = sum(1 for r in results if not r.get("success", False))

    print(f"Total tests: {len(results)}")
    print(f"[PASS] Full pass: {passed}/{len(results)}")
    print(f"[PARTIAL] Partial pass: {partial}/{len(results)}")
    print(f"[FAIL] Failed: {failed}/{len(results)}")

    print("\nDétails par question:")
    for result in results:
        q_id = result.get("question_id", "?")
        test_passed = result.get("test_passed", False)
        insights_count = result.get("insights_count", 0)

        status_icon = "[PASS]" if test_passed else "[PARTIAL]" if result.get("success", False) else "[FAIL]"

        print(f"  {status_icon} {q_id}: {insights_count} insights")

        if result.get("insights_details"):
            for insight in result["insights_details"]:
                print(f"      - {insight['type']}: {insight['title'][:50]}...")

    # INSIGHTS GLOBAUX
    print("\n" + "="*80)
    print("INSIGHTS GÉNÉRÉS PAR TYPE (TOUTES QUESTIONS)".center(80))
    print("="*80 + "\n")

    all_insights_types = {}
    for result in results:
        for insight_detail in result.get("insights_details", []):
            insight_type = insight_detail["type"]
            all_insights_types[insight_type] = all_insights_types.get(insight_type, 0) + 1

    for insight_type, count in sorted(all_insights_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {insight_type}: {count}x")

    print("\n" + "="*80)

    if passed == len(results):
        print("\n[SUCCESS] All tests passed! Analytics Agent & Recommendation Engine integration working!")
        return True
    elif passed + partial == len(results):
        print("\n[WARNING] Tests partially passed. Review insights generation.")
        return True
    else:
        print("\n[FAILED] Some tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
