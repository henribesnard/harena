"""Simple test script for Conversation Service V2."""

import asyncio
import sys
from app.core.intent_analyzer import IntentAnalyzer


async def test_intent_analyzer():
    """Test the intent analyzer module."""
    print("Testing Intent Analyzer...")
    print("-" * 50)

    analyzer = IntentAnalyzer()

    # Test queries
    test_queries = [
        "Combien j'ai dépensé en restaurants ce mois-ci ?",
        "Mes 5 commerces où je dépense le plus",
        "Mes transactions de plus de 100 euros",
        "Compare mes dépenses courses entre avril et mai"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            # Note: This will only work if DEEPSEEK_API_KEY is set
            # intent = await analyzer.analyze(query)
            # print(f"Intent Type: {intent.intent_type}")
            # print(f"Confidence: {intent.confidence_score}")
            # print(f"Categories: {intent.categories}")
            # print(f"Time Periods: {intent.time_periods}")
            print("  → (Skipped - requires DEEPSEEK_API_KEY)")
        except Exception as e:
            print(f"  → Error: {e}")

    print("\n" + "-" * 50)
    print("Intent Analyzer test completed!")


def test_structure():
    """Test the project structure."""
    print("Testing Project Structure...")
    print("-" * 50)

    import os
    from pathlib import Path

    base_path = Path(__file__).parent / "app"

    expected_files = [
        "main.py",
        "auth/middleware.py",
        "core/intent_analyzer.py",
        "core/sql_generator.py",
        "core/sql_validator.py",
        "core/sql_executor.py",
        "core/context_builder.py",
        "core/response_generator.py",
        "services/conversation_service.py",
        "api/v2/endpoints/conversation.py",
        "models/requests/conversation_requests.py",
        "models/responses/conversation_responses.py",
    ]

    all_exist = True
    for file in expected_files:
        file_path = base_path / file
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False

    print("-" * 50)
    if all_exist:
        print("All files exist! ✓")
    else:
        print("Some files are missing! ✗")

    return all_exist


def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting Module Imports...")
    print("-" * 50)

    modules = [
        ("Auth Middleware", "app.auth.middleware"),
        ("Intent Analyzer", "app.core.intent_analyzer"),
        ("SQL Generator", "app.core.sql_generator"),
        ("SQL Validator", "app.core.sql_validator"),
        ("SQL Executor", "app.core.sql_executor"),
        ("Context Builder", "app.core.context_builder"),
        ("Response Generator", "app.core.response_generator"),
        ("Conversation Service", "app.services.conversation_service"),
        ("Request Models", "app.models.requests"),
        ("Response Models", "app.models.responses"),
    ]

    all_imported = True
    for name, module in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_imported = False

    print("-" * 50)
    if all_imported:
        print("All modules imported successfully! ✓")
    else:
        print("Some modules failed to import! ✗")

    return all_imported


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("CONVERSATION SERVICE V2 - TEST SUITE")
    print("=" * 50 + "\n")

    # Test 1: Structure
    structure_ok = test_structure()

    # Test 2: Imports
    imports_ok = test_imports()

    # Test 3: Intent Analyzer (async)
    # asyncio.run(test_intent_analyzer())

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Imports:   {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print("=" * 50 + "\n")

    if structure_ok and imports_ok:
        print("✓ All basic tests passed!")
        print("\nNext steps:")
        print("1. Configure your .env file with API keys")
        print("2. Run: python run.py")
        print("3. Visit: http://localhost:3003/api/v2/docs")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
