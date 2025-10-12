"""
Sprint 1.4 - Phase 1 - Validation avec Questions Réelles

Tests du workflow complet avec 15 questions utilisateur réalistes.
Valide: Intent classification, Search, Analytics, Visualizations, Response quality
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from conversation_service.api.dependencies import get_context_manager
from conversation_service.agents.llm.response_generator import ResponseType

# ============================================
# TEST QUESTIONS
# ============================================

TEST_QUESTIONS = [
    # Catégorie 1: Transaction Search Simple
    {
        "category": "simple",
        "user_id": 1,
        "message": "Mes dépenses du mois dernier",
        "expected_intent": "transaction_search.simple",
        "expected_viz_min": 1,  # Au moins KPI Cards
    },
    {
        "category": "simple",
        "user_id": 1,
        "message": "Transactions de cette semaine",
        "expected_intent": "transaction_search",
        "expected_viz_min": 1,
    },
    {
        "category": "simple",
        "user_id": 1,
        "message": "Combien j'ai dépensé aujourd'hui ?",
        "expected_intent": "transaction_search",
        "expected_viz_min": 1,
    },

    # Catégorie 2: By Category
    {
        "category": "by_category",
        "user_id": 1,
        "message": "Mes dépenses alimentaires ce mois-ci",
        "expected_intent": "transaction_search.by_category",
        "expected_viz_min": 2,  # KPI + Pie Chart
    },
    {
        "category": "by_category",
        "user_id": 1,
        "message": "Combien j'ai dépensé en transport ?",
        "expected_intent": "transaction_search.by_category",
        "expected_viz_min": 1,
    },
    {
        "category": "by_category",
        "user_id": 1,
        "message": "Répartition de mes dépenses par catégorie",
        "expected_intent": "transaction_search.by_category",
        "expected_viz_min": 2,
    },

    # Catégorie 3: By Merchant
    {
        "category": "by_merchant",
        "user_id": 1,
        "message": "Mes dépenses chez Carrefour",
        "expected_intent": "transaction_search.by_merchant",
        "expected_viz_min": 1,
    },
    {
        "category": "by_merchant",
        "user_id": 1,
        "message": "Combien j'ai payé à Netflix ce mois ?",
        "expected_intent": "transaction_search.by_merchant",
        "expected_viz_min": 1,
    },
    {
        "category": "by_merchant",
        "user_id": 1,
        "message": "Transactions SNCF du dernier mois",
        "expected_intent": "transaction_search",
        "expected_viz_min": 1,
    },

    # Catégorie 4: Analytics & Insights
    {
        "category": "analytics",
        "user_id": 1,
        "message": "Quelles sont mes dépenses inhabituelles ?",
        "expected_intent": "transaction_search",
        "expected_viz_min": 1,
    },
    {
        "category": "analytics",
        "user_id": 1,
        "message": "Mes tendances de dépenses",
        "expected_intent": "transaction_search",
        "expected_viz_min": 1,
    },
    {
        "category": "analytics",
        "user_id": 1,
        "message": "Comparaison avec le mois dernier",
        "expected_intent": "transaction_search",
        "expected_viz_min": 1,
    },

    # Catégorie 5: Edge Cases
    {
        "category": "edge_case",
        "user_id": 1,
        "message": "Mes transactions du 1er janvier 1970",
        "expected_intent": "transaction_search",
        "expected_viz_min": 0,  # Peut être 0 si pas de résultats
        "allow_no_results": True,
    },
    {
        "category": "edge_case",
        "user_id": 1,
        "message": "Dépenses chez XYZ_MERCHANT_INEXISTANT",
        "expected_intent": "transaction_search",
        "expected_viz_min": 0,
        "allow_no_results": True,
    },
    {
        "category": "edge_case",
        "user_id": 1,
        "message": "Mes trucs",
        "expected_intent": None,  # Intent peut être CONVERSATIONAL
        "expected_viz_min": 0,
        "allow_no_results": True,
    },
]


# ============================================
# TEST WORKFLOW
# ============================================

async def test_single_question(context_manager, question_data):
    """Test une seule question et retourne les résultats"""

    question = question_data["message"]
    user_id = question_data["user_id"]

    print(f"\n{'='*80}")
    print(f"Testing: {question}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Process conversation
        result = await context_manager.process_conversation(
            user_id=user_id,
            message=question
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Extract info
        success = result.success
        intent = result.intent_group if hasattr(result, 'intent_group') else None
        response_text = result.response_text if hasattr(result, 'response_text') else None
        visualizations = result.data_visualizations if hasattr(result, 'data_visualizations') else []
        insights = result.insights if hasattr(result, 'insights') else []

        # Validation checks
        checks = {
            "success": success,
            "has_response": response_text is not None and len(response_text) > 0,
            "has_visualizations": len(visualizations) >= question_data["expected_viz_min"],
            "latency_ok": elapsed_ms < 5000,  # 5s max
        }

        # Intent check (if specified)
        if question_data.get("expected_intent"):
            checks["intent_match"] = question_data["expected_intent"] in (intent or "")

        all_passed = all(checks.values())

        # Print results
        print(f"\n✅ SUCCESS" if all_passed else f"\n❌ FAILED")
        print(f"Intent: {intent}")
        print(f"Response length: {len(response_text) if response_text else 0} chars")
        print(f"Visualizations: {len(visualizations)}")
        print(f"Insights: {len(insights)}")
        print(f"Latency: {elapsed_ms:.0f}ms")

        if not all_passed:
            print(f"\nFailed Checks:")
            for check_name, passed in checks.items():
                if not passed:
                    print(f"  ❌ {check_name}")

        return {
            "question": question,
            "category": question_data["category"],
            "success": success,
            "all_checks_passed": all_passed,
            "intent": intent,
            "response_length": len(response_text) if response_text else 0,
            "viz_count": len(visualizations),
            "insights_count": len(insights),
            "latency_ms": elapsed_ms,
            "checks": checks,
            "error": None
        }

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\n❌ EXCEPTION: {str(e)}")

        return {
            "question": question,
            "category": question_data["category"],
            "success": False,
            "all_checks_passed": False,
            "intent": None,
            "response_length": 0,
            "viz_count": 0,
            "insights_count": 0,
            "latency_ms": elapsed_ms,
            "checks": {},
            "error": str(e)
        }


# ============================================
# MAIN TEST SUITE
# ============================================

async def run_validation_tests():
    """Execute all validation tests"""

    print("="*80)
    print("Sprint 1.4 - Phase 1 - Validation avec Questions Réelles")
    print("="*80)
    print(f"Total questions: {len(TEST_QUESTIONS)}")
    print(f"Started at: {datetime.now().isoformat()}")

    # Initialize context manager
    context_manager = get_context_manager()

    # Run tests sequentially (to avoid overwhelming the system)
    results = []
    for question_data in TEST_QUESTIONS:
        result = await test_single_question(context_manager, question_data)
        results.append(result)

        # Small delay between tests
        await asyncio.sleep(0.5)

    # ============================================
    # GENERATE REPORT
    # ============================================

    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)

    total = len(results)
    passed = sum(1 for r in results if r["all_checks_passed"])
    failed = total - passed
    success_rate = (passed / total * 100) if total > 0 else 0

    # Latency stats
    latencies = [r["latency_ms"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    # Visualizations stats
    total_viz = sum(r["viz_count"] for r in results)
    questions_with_viz = sum(1 for r in results if r["viz_count"] > 0)

    # Insights stats
    total_insights = sum(r["insights_count"] for r in results)
    questions_with_insights = sum(1 for r in results if r["insights_count"] > 0)

    print(f"\n📊 SUMMARY:")
    print(f"  Total Tests: {total}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print(f"  Success Rate: {success_rate:.1f}%")

    print(f"\n⏱️  LATENCY:")
    print(f"  Average: {avg_latency:.0f}ms")
    print(f"  Max: {max_latency:.0f}ms")

    print(f"\n📈 VISUALIZATIONS:")
    print(f"  Total Generated: {total_viz}")
    print(f"  Questions with Viz: {questions_with_viz}/{total}")
    print(f"  Coverage: {questions_with_viz/total*100:.1f}%")

    print(f"\n💡 INSIGHTS:")
    print(f"  Total Generated: {total_insights}")
    print(f"  Questions with Insights: {questions_with_insights}/{total}")

    # Results by category
    print(f"\n📋 RESULTS BY CATEGORY:")
    categories = set(r["category"] for r in results)
    for category in sorted(categories):
        cat_results = [r for r in results if r["category"] == category]
        cat_passed = sum(1 for r in cat_results if r["all_checks_passed"])
        cat_total = len(cat_results)
        cat_rate = (cat_passed / cat_total * 100) if cat_total > 0 else 0
        print(f"  {category}: {cat_passed}/{cat_total} ({cat_rate:.0f}%)")

    # Failed tests details
    if failed > 0:
        print(f"\n❌ FAILED TESTS:")
        for i, result in enumerate(results):
            if not result["all_checks_passed"]:
                print(f"\n  {i+1}. {result['question']}")
                print(f"     Category: {result['category']}")
                if result["error"]:
                    print(f"     Error: {result['error']}")
                else:
                    print(f"     Failed checks: {[k for k, v in result['checks'].items() if not v]}")

    # ============================================
    # GO/NO-GO DECISION
    # ============================================

    print(f"\n{'='*80}")
    print("GO/NO-GO DECISION")
    print(f"{'='*80}")

    criteria = {
        "Success Rate ≥ 95%": success_rate >= 95,
        "Max Latency ≤ 5000ms": max_latency <= 5000,
        "Avg Latency ≤ 4000ms": avg_latency <= 4000,
        "No Critical Errors": failed == 0 or success_rate >= 90,
    }

    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}")

    go_decision = all(criteria.values())

    print(f"\n{'='*80}")
    if go_decision:
        print("✅ GO - Ready for Phase 2 (Production Deployment)")
    else:
        print("❌ NO-GO - Corrections needed before production")
    print(f"{'='*80}")

    # Save results to JSON
    output_file = project_root / "SPRINT_1.4_PHASE1_RESULTS.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
            },
            "go_decision": go_decision,
            "criteria": criteria,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Results saved to: {output_file}")

    return go_decision


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        go_decision = asyncio.run(run_validation_tests())
        sys.exit(0 if go_decision else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
