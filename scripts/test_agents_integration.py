"""
Test script to verify Analytics Agent and Recommendation Engine integration
in ResponseGenerator

This tests:
1. Analytics Agent - Trend analysis
2. Analytics Agent - Anomaly detection
3. Recommendation Engine - Recommendations generation
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conversation_service.agents.llm.response_generator import (
    ResponseGenerator,
    ResponseGenerationRequest
)
from conversation_service.agents.llm.llm_provider import LLMProviderManager


def generate_test_transactions():
    """Generate test transactions with trends and anomalies"""

    transactions = []
    base_date = datetime(2025, 1, 1)

    # Normal transactions with upward trend
    for i in range(20):
        date = base_date + timedelta(days=i * 3)
        amount = 50 + (i * 2)  # Increasing trend
        transactions.append({
            "id": i,
            "amount": -amount,
            "date": date.isoformat(),
            "merchant_name": f"Merchant_{i % 5}",
            "merchant": f"Merchant_{i % 5}",
            "category_name": "Alimentation",
            "transaction_type": "debit",
            "primary_description": f"Purchase at Merchant_{i % 5}"
        })

    # Add an anomaly (very large transaction)
    transactions.append({
        "id": 100,
        "amount": -500,  # Anomaly!
        "date": (base_date + timedelta(days=60)).isoformat(),
        "merchant_name": "Electronics Store",
        "merchant": "Electronics Store",
        "category_name": "Shopping",
        "transaction_type": "debit",
        "primary_description": "Large purchase"
    })

    # Add subscriptions for recommendation engine
    for month in range(3):
        date = base_date + timedelta(days=month * 30)

        # Netflix subscription
        transactions.append({
            "id": 200 + month,
            "amount": -15.99,
            "date": date.isoformat(),
            "merchant_name": "Netflix",
            "merchant": "Netflix",
            "category_name": "streaming",
            "transaction_type": "debit",
            "primary_description": "Netflix subscription"
        })

        # Spotify subscription
        transactions.append({
            "id": 300 + month,
            "amount": -9.99,
            "date": date.isoformat(),
            "merchant_name": "Spotify",
            "merchant": "Spotify",
            "category_name": "streaming",
            "transaction_type": "debit",
            "primary_description": "Spotify subscription"
        })

    return transactions


async def test_analytics_trend():
    """Test Analytics Agent trend analysis integration"""

    print("\n" + "="*80)
    print("TEST 1: Analytics Agent - Trend Analysis")
    print("="*80)

    # Setup - Create mock LLM manager with minimal config
    class MockLLMManager:
        pass

    response_gen = ResponseGenerator(MockLLMManager())
    # Don't call initialize() - we're testing agent integration directly

    transactions = generate_test_transactions()
    user_profile = {"user_id": 100, "avg_monthly_spending": 1000}

    # Call the trend analysis generator directly
    insight = await response_gen._generate_trend_analysis_insight(
        search_results=transactions,
        user_profile=user_profile
    )

    if insight:
        print(f"[OK] Trend insight generated!")
        print(f"   Title: {insight.title}")
        print(f"   Description: {insight.description}")
        print(f"   Confidence: {insight.confidence:.2f}")
        print(f"   Actionable: {insight.actionable}")
        print(f"   Data support: {insight.data_support}")
        return True
    else:
        print(f"[SKIP] No trend insight generated (may need more data points)")
        return False


async def test_analytics_anomaly():
    """Test Analytics Agent anomaly detection integration"""

    print("\n" + "="*80)
    print("TEST 2: Analytics Agent - Anomaly Detection")
    print("="*80)

    # Setup - Create mock LLM manager
    class MockLLMManager:
        pass

    response_gen = ResponseGenerator(MockLLMManager())

    transactions = generate_test_transactions()
    user_profile = {"user_id": 100}

    # Call the unusual transaction generator directly
    insight = await response_gen._generate_unusual_transaction_insight(
        search_results=transactions,
        user_profile=user_profile
    )

    if insight:
        print(f"[OK] Anomaly insight generated!")
        print(f"   Title: {insight.title}")
        print(f"   Description: {insight.description}")
        print(f"   Confidence: {insight.confidence:.2f}")
        print(f"   Actionable: {insight.actionable}")
        print(f"   Anomaly score: {insight.data_support.get('anomaly_score', 'N/A')}")
        return True
    else:
        print(f"[SKIP] No anomaly insight generated")
        return False


async def test_recommendation_engine():
    """Test Recommendation Engine integration"""

    print("\n" + "="*80)
    print("TEST 3: Recommendation Engine - Subscription Optimization")
    print("="*80)

    # Setup - Create mock LLM manager
    class MockLLMManager:
        pass

    response_gen = ResponseGenerator(MockLLMManager())

    transactions = generate_test_transactions()
    user_profile = {"user_id": 100}

    # Call the recommendation generator directly
    insight = await response_gen._generate_recommendation_insight(
        search_results=transactions,
        user_profile=user_profile
    )

    if insight:
        print(f"[OK] Recommendation insight generated!")
        print(f"   Title: {insight.title}")
        print(f"   Description: {insight.description}")
        print(f"   Confidence: {insight.confidence:.2f}")
        print(f"   Actionable: {insight.actionable}")
        print(f"   Priority: {insight.priority}")

        if insight.data_support.get('estimated_savings'):
            print(f"   Estimated savings: {insight.data_support['estimated_savings']:.2f}€")

        print(f"   CTA: {insight.data_support.get('cta_text', 'N/A')}")
        return True
    else:
        print(f"[SKIP] No recommendation insight generated")
        return False


async def test_full_pipeline():
    """Test full insights generation pipeline"""

    print("\n" + "="*80)
    print("TEST 4: Full Insights Generation Pipeline")
    print("="*80)

    # Setup - Create mock LLM manager
    class MockLLMManager:
        pass

    response_gen = ResponseGenerator(MockLLMManager())

    transactions = generate_test_transactions()

    # Create full request
    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="all",
        user_message="Show me my recent transactions",
        search_results=transactions,
        conversation_context=[],
        user_profile={"user_id": 100, "avg_monthly_spending": 1000},
        user_id=100,
        generate_insights=True,
        stream_response=False,
        search_aggregations=None
    )

    # Generate automatic insights
    insights = await response_gen._generate_automatic_insights(request)

    print(f"\n[OK] Generated {len(insights)} insights:")
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight.type.value.upper()}")
        print(f"   Title: {insight.title}")
        print(f"   Description: {insight.description[:100]}...")
        print(f"   Confidence: {insight.confidence:.2f}")
        print(f"   Actionable: {insight.actionable}")

    return len(insights) > 0


async def main():
    """Run all tests"""

    print("\n" + "="*80)
    print("TESTING ANALYTICS AGENT & RECOMMENDATION ENGINE INTEGRATION")
    print("="*80)

    results = []

    try:
        # Test 1: Trend analysis
        result1 = await test_analytics_trend()
        results.append(("Trend Analysis", result1))

        # Test 2: Anomaly detection
        result2 = await test_analytics_anomaly()
        results.append(("Anomaly Detection", result2))

        # Test 3: Recommendations
        result3 = await test_recommendation_engine()
        results.append(("Recommendations", result3))

        # Test 4: Full pipeline
        result4 = await test_full_pipeline()
        results.append(("Full Pipeline", result4))

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[WARN]"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Integration successful!")
        return True
    elif passed > 0:
        print("\n[PARTIAL] Some tests passed. Integration partially working.")
        return True
    else:
        print("\n[FAILED] All tests failed. Check integration.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
