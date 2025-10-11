"""
Test Direct Workflow - Test intégration directe sans API

Ce script teste le workflow complet en important directement les modules Python
sans passer par l'API HTTP. Plus rapide et permet de tester même si le service n'est pas démarré.

Tests:
1. Initialisation de tous les composants du pipeline
2. Exécution du workflow complet avec une vraie requête
3. Vérification des insights Analytics Agent
4. Vérification des recommendations
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def print_subsection(title):
    """Print formatted subsection"""
    print("\n" + "-"*60)
    print(title)
    print("-"*60)


async def initialize_pipeline():
    """Initialize all pipeline components"""

    print_subsection("INITIALISATION DU PIPELINE")

    # Import all components
    from conversation_service.config.settings import ConfigManager
    from conversation_service.core.context_manager import ContextManager
    from conversation_service.core.query_builder import QueryBuilder
    from conversation_service.core.query_executor import QueryExecutor
    from conversation_service.agents.llm import (
        LLMProviderManager,
        IntentClassifier,
        ResponseGenerator
    )
    from conversation_service.core.conversation_orchestrator import (
        ConversationOrchestrator,
        ConversationRequest
    )

    # Initialize configuration
    print("[1/7] Loading configuration...")
    config_manager = ConfigManager()
    await config_manager.load_configuration()
    print(f"    Config loaded: {config_manager.service_config.get('service_name', 'N/A')}")

    # Initialize LLM Provider
    print("[2/7] Initializing LLM Provider...")
    llm_configs = config_manager.get_llm_configs()
    llm_manager = LLMProviderManager(configs=llm_configs)
    print(f"    LLM models available: {len(llm_configs)}")

    # Initialize Context Manager
    print("[3/7] Initializing Context Manager...")
    context_manager = ContextManager()
    print("    Context Manager ready")

    # Initialize Intent Classifier
    print("[4/7] Initializing Intent Classifier...")
    intent_classifier = IntentClassifier(llm_manager=llm_manager)
    await intent_classifier.initialize()
    print("    Intent Classifier ready")

    # Initialize Query Builder
    print("[5/7] Initializing Query Builder...")
    query_builder = QueryBuilder()
    print("    Query Builder ready")

    # Initialize Query Executor
    print("[6/7] Initializing Query Executor...")
    search_config = config_manager.get_search_service_config()
    query_executor = QueryExecutor(config=search_config)
    print("    Query Executor ready")

    # Initialize Response Generator (avec Analytics Agent et Recommendation Engine)
    print("[7/7] Initializing Response Generator...")
    response_generator = ResponseGenerator(llm_manager=llm_manager)
    await response_generator.initialize()
    print("    Response Generator ready (with Analytics Agent & Recommendation Engine)")

    # Create Orchestrator
    print("\n[FINAL] Creating Conversation Orchestrator...")
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        intent_classifier=intent_classifier,
        query_builder=query_builder,
        query_executor=query_executor,
        response_generator=response_generator,
        config_manager=config_manager
    )

    await orchestrator.initialize()
    print("    Orchestrator ready!")

    return orchestrator


async def test_workflow(orchestrator, question, user_id=100):
    """Test workflow complet avec une question"""

    print_subsection(f"TEST: \"{question}\"")

    from conversation_service.core.conversation_orchestrator import ConversationRequest

    # Create request
    request = ConversationRequest(
        user_id=user_id,
        user_message=question,
        conversation_id=None,
        include_insights=True
    )

    # Process conversation
    print(f"\n[INFO] Processing conversation...")
    start_time = datetime.now()

    try:
        result = await orchestrator.process_conversation(request)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        print(f"[OK] Conversation processed in {processing_time:.0f}ms")

        # Analyze result
        return analyze_result(result, question)

    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def analyze_result(result, question):
    """Analyze conversation result"""

    print("\n" + "="*60)
    print("RÉSULTAT".center(60))
    print("="*60)

    # 1. Status
    print(f"\nSuccess: {result.success}")
    print(f"Pipeline stage: {result.pipeline_stage.value if result.pipeline_stage else 'N/A'}")

    if result.error_message:
        print(f"Error: {result.error_message}")

    # 2. Intent
    if result.classified_intent:
        print(f"\nIntent:")
        print(f"  Type: {result.classified_intent.get('intent_group', 'N/A')}")
        print(f"  Subtype: {result.classified_intent.get('intent_subtype', 'N/A')}")
        print(f"  Confidence: {result.classified_intent.get('confidence', 0):.2f}")

    # 3. Search results
    print(f"\nSearch:")
    print(f"  Total hits: {result.search_total_hits}")
    print(f"  Results returned: {len(result.search_results) if result.search_results else 0}")

    # 4. Response
    print(f"\nResponse:")
    print(f"  Length: {len(result.response_text)} characters")
    if result.response_text:
        # Print first 200 chars
        preview = result.response_text[:200] + "..." if len(result.response_text) > 200 else result.response_text
        print(f"  Preview: {preview}")

    # 5. INSIGHTS (CRITÈRE PRINCIPAL)
    print("\n" + "="*60)
    print("INSIGHTS ANALYSIS".center(60))
    print("="*60)

    insights = result.insights
    print(f"\nTotal insights generated: {len(insights)}")

    insights_by_type = {}

    if insights:
        for i, insight in enumerate(insights, 1):
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

            # Détails spécifiques
            if insight_type == "trend_analysis":
                slope = data_support.get("slope", 0)
                r_squared = data_support.get("r_squared", 0)
                trend_direction = data_support.get("trend_direction", "unknown")
                print(f"   [Analytics Agent] Trend: {trend_direction}, Slope: {slope:.2f}%, R²: {r_squared:.2f}")

            elif insight_type == "unusual_transaction":
                anomaly_score = data_support.get("anomaly_score", 0)
                method = data_support.get("method", "unknown")
                amount = data_support.get("amount", 0)
                merchant = data_support.get("merchant", "N/A")
                print(f"   [Analytics Agent] Anomaly: {amount}€ at {merchant}")
                print(f"   [Analytics Agent] Score: {anomaly_score:.2f}, Method: {method}")

            elif insight_type == "recommendation":
                rec_type = data_support.get("recommendation_type", "unknown")
                estimated_savings = data_support.get("estimated_savings")
                cta_text = data_support.get("cta_text", "N/A")
                print(f"   [Recommendation Engine] Type: {rec_type}")
                if estimated_savings:
                    print(f"   [Recommendation Engine] Savings: {estimated_savings:.2f}€")
                print(f"   [Recommendation Engine] CTA: {cta_text}")

            insights_by_type[insight_type] = insights_by_type.get(insight_type, 0) + 1
    else:
        print("\n[WARNING] No insights generated!")

    # 6. Metrics
    print("\n" + "="*60)
    print("METRICS".center(60))
    print("="*60)

    if result.metrics:
        print(f"\nProcessing time: {result.metrics.total_processing_time_ms}ms")
        print(f"Tokens used: {result.metrics.tokens_used}")
        print(f"Model: {result.metrics.model_used}")

        if result.metrics.stage_timings:
            print("\nStage timings:")
            for stage, timing in result.metrics.stage_timings.items():
                print(f"  {stage}: {timing}ms")

    # 7. Validation
    print("\n" + "="*60)
    print("VALIDATION".center(60))
    print("="*60)

    validation_checks = {
        "Conversation processed": result.success,
        "Intent classified": result.classified_intent is not None,
        "Search executed": result.search_total_hits > 0,
        "Response generated": len(result.response_text) > 0,
        "Insights generated": len(insights) > 0,
        "Analytics Agent used": "trend_analysis" in insights_by_type or "unusual_transaction" in insights_by_type,
        "Recommendation Engine used": "recommendation" in insights_by_type
    }

    all_passed = True
    for check, passed in validation_checks.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    return {
        "success": result.success,
        "question": question,
        "insights_count": len(insights),
        "insights_by_type": insights_by_type,
        "search_hits": result.search_total_hits,
        "response_length": len(result.response_text),
        "all_checks_passed": all_passed,
        "validation_checks": validation_checks
    }


async def main():
    """Main test function"""

    print_section("TEST DIRECT WORKFLOW - ANALYTICS AGENT & RECOMMENDATION ENGINE")

    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")

    # Test questions
    test_questions = [
        "Montre moi toutes mes transactions",
        "Quelles sont mes transactions anormales?",
        "Recommande-moi des économies"
    ]

    try:
        # Initialize pipeline
        print_section("PHASE 1: INITIALISATION")
        orchestrator = await initialize_pipeline()

        # Test questions
        print_section("PHASE 2: TESTS")

        results = []

        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(test_questions)}".center(80))
            print('='*80)

            result = await test_workflow(orchestrator, question)
            results.append(result)

            # Pause between tests
            await asyncio.sleep(1)

        # Summary
        print_section("PHASE 3: RAPPORT FINAL")

        passed = sum(1 for r in results if r.get("all_checks_passed", False))
        total = len(results)

        print(f"Tests passed: {passed}/{total}")

        print("\nPar question:")
        for i, result in enumerate(results, 1):
            question = result.get("question", "N/A")
            insights_count = result.get("insights_count", 0)
            passed_checks = result.get("all_checks_passed", False)

            status = "[PASS]" if passed_checks else "[PARTIAL]" if insights_count > 0 else "[FAIL]"
            print(f"  {status} Q{i}: {question[:50]}... ({insights_count} insights)")

        print("\nInsights générés par type (toutes questions):")
        all_insights = {}
        for result in results:
            for insight_type, count in result.get("insights_by_type", {}).items():
                all_insights[insight_type] = all_insights.get(insight_type, 0) + count

        for insight_type, count in sorted(all_insights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {insight_type}: {count}x")

        # Final verdict
        print("\n" + "="*80)

        if passed == total:
            print("[SUCCESS] All tests passed! Integration working perfectly!")
            return True
        elif passed > 0:
            print("[PARTIAL] Some tests passed. Review failed checks.")
            return True
        else:
            print("[FAILED] All tests failed. Check errors above.")
            return False

    except Exception as e:
        print(f"\n[ERROR] Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
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
