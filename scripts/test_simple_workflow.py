"""
Test Simple Workflow - Test minimal du pipeline complet

Ce script teste juste si on peut appeler le ResponseGenerator avec des données
et vérifier que les insights Analytics Agent et Recommendation Engine sont bien générés.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def generate_test_data():
    """Génère des données de test réalistes"""
    transactions = []
    base_date = datetime(2024, 1, 1)

    # Transactions normales avec tendance croissante
    for i in range(15):
        transactions.append({
            "id": i + 1,
            "amount": -(50 + i * 3),  # Tendance croissante
            "date": (base_date + timedelta(days=i * 7)).isoformat(),
            "merchant_name": f"Supermarché {i % 3}",
            "category_name": "Alimentation",
            "transaction_type": "debit",
            "primary_description": f"Achat supermarché"
        })

    # Ajout d'une anomalie
    transactions.append({
        "id": 100,
        "amount": -500,  # ANOMALIE!
        "date": (base_date + timedelta(days=120)).isoformat(),
        "merchant_name": "Electronics Store",
        "category_name": "Shopping",
        "transaction_type": "debit",
        "primary_description": "Achat électronique important"
    })

    # Ajout de 2 subscriptions (pour recommendation)
    for month in range(3):
        transactions.append({
            "id": 200 + month,
            "amount": -15.99,
            "date": (base_date + timedelta(days=month * 30)).isoformat(),
            "merchant_name": "Netflix",
            "category_name": "streaming",
            "transaction_type": "debit",
            "primary_description": "Abonnement Netflix"
        })

        transactions.append({
            "id": 300 + month,
            "amount": -9.99,
            "date": (base_date + timedelta(days=month * 30)).isoformat(),
            "merchant_name": "Spotify",
            "category_name": "streaming",
            "transaction_type": "debit",
            "primary_description": "Abonnement Spotify"
        })

    return transactions


async def test_response_generator():
    """Test du ResponseGenerator avec Analytics Agent et Recommendation Engine"""

    print("="*80)
    print("TEST SIMPLE - RESPONSE GENERATOR AVEC AGENTS INTÉGRÉS")
    print("="*80)

    from conversation_service.agents.llm.response_generator import (
        ResponseGenerator,
        ResponseGenerationRequest
    )
    from conversation_service.agents.llm.llm_provider import LLMProviderManager

    # 1. Initialize components
    print("\n[1/4] Initializing LLM Provider...")

    # Mock LLM Manager (minimal config)
    class MockLLMManager:
        async def generate(self, request):
            # Mock response
            class MockResponse:
                content = "Voici vos transactions avec insights."
                error = None
                usage = {"total_tokens": 100}
                model_used = "mock"
            return MockResponse()

    llm_manager = MockLLMManager()

    # 2. Create ResponseGenerator (with integrated agents)
    print("[2/4] Creating ResponseGenerator (with Analytics Agent & Recommendation Engine)...")
    response_generator = ResponseGenerator(llm_manager=llm_manager)

    # Verify agents are initialized
    print(f"   - Analytics Agent: {'OK' if hasattr(response_generator, 'analytics_agent') else 'MISSING'}")
    print(f"   - Recommendation Engine: {'OK' if hasattr(response_generator, 'recommendation_engine') else 'MISSING'}")

    if not hasattr(response_generator, 'analytics_agent'):
        print("\n[ERROR] Analytics Agent NOT initialized!")
        return False

    if not hasattr(response_generator, 'recommendation_engine'):
        print("\n[ERROR] Recommendation Engine NOT initialized!")
        return False

    # 3. Generate test data
    print("\n[3/4] Generating test data...")
    transactions = generate_test_data()
    print(f"   - Generated {len(transactions)} transactions")
    print(f"   - Including 1 anomaly (500€)")
    print(f"   - Including 2 subscriptions (Netflix + Spotify)")

    # 4. Test insight generation
    print("\n[4/4] Testing automatic insights generation...")

    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="all",
        user_message="Montre moi toutes mes transactions",
        search_results=transactions,
        conversation_context=[],
        user_profile={"user_id": 100, "avg_monthly_spending": 500},
        user_id=100,
        conversation_id=None,
        generate_insights=True,
        stream_response=False,
        search_aggregations=None
    )

    # Generate insights
    insights = await response_generator._generate_automatic_insights(request)

    # Analyze results
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)

    print(f"\nTotal insights générés: {len(insights)}")

    if len(insights) == 0:
        print("\n[FAILED] Aucun insight généré!")
        return False

    # Analyze insights by type
    insights_by_type = {}

    for i, insight in enumerate(insights, 1):
        insight_type = insight.type.value
        insights_by_type[insight_type] = insights_by_type.get(insight_type, 0) + 1

        print(f"\n{i}. {insight_type.upper()}")
        print(f"   Title: {insight.title}")
        print(f"   Description: {insight.description[:100]}{'...' if len(insight.description) > 100 else ''}")
        print(f"   Confidence: {insight.confidence:.2f}")
        print(f"   Actionable: {insight.actionable}")

        # Details by type
        if insight_type == "trend_analysis":
            slope = insight.data_support.get("slope", 0)
            r_squared = insight.data_support.get("r_squared", 0)
            trend_direction = insight.data_support.get("trend_direction", "unknown")
            print(f"   [Analytics Agent] Trend: {trend_direction}, Slope: {slope:.2f}%, R²: {r_squared:.2f}")

        elif insight_type == "unusual_transaction":
            anomaly_score = insight.data_support.get("anomaly_score", 0)
            method = insight.data_support.get("method", "unknown")
            print(f"   [Analytics Agent] Anomaly score: {anomaly_score:.2f}, Method: {method}")

        elif insight_type == "recommendation":
            rec_type = insight.data_support.get("recommendation_type", "unknown")
            estimated_savings = insight.data_support.get("estimated_savings")
            print(f"   [Recommendation Engine] Type: {rec_type}")
            if estimated_savings:
                print(f"   [Recommendation Engine] Estimated savings: {estimated_savings:.2f}€")

    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    checks = {
        "Insights generated": len(insights) > 0,
        "Analytics Agent used (trend or anomaly)": any(
            t in insights_by_type for t in ["trend_analysis", "unusual_transaction"]
        ),
        "Recommendation Engine used": "recommendation" in insights_by_type
    }

    all_passed = True
    for check, passed in checks.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] Tous les checks passent! Intégration fonctionnelle!")
        print("\nInsights par type:")
        for insight_type, count in insights_by_type.items():
            print(f"  - {insight_type}: {count}x")
        return True
    else:
        print("[FAILED] Certains checks ont échoué")
        return False


async def main():
    try:
        success = await test_response_generator()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
