"""
Test simple pour vérifier la classification de "Mes retraits espèces"
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from conversation_service.agents.llm.intent_classifier import IntentClassifier, ClassificationRequest
from conversation_service.agents.llm.llm_provider import LLMProviderManager, ProviderConfig, ProviderType
from conversation_service.config.settings import ConfigManager

async def test_retraits_especes():
    """Test la classification de 'Mes retraits espèces'"""

    print("\n" + "="*80)
    print("TEST: Mes retraits espèces")
    print("="*80 + "\n")

    # Initialiser le config manager et LLM provider
    config_manager = ConfigManager()
    await config_manager.load_configurations()

    # Obtenir la configuration LLM depuis le config manager
    llm_config = config_manager.get_llm_providers_config()
    providers = llm_config.get("providers", {})
    provider_configs = {}

    # Configuration DeepSeek (depuis la config)
    if providers.get("deepseek", {}).get("enabled", False):
        deepseek_config = providers["deepseek"]
        provider_configs[ProviderType.DEEPSEEK] = ProviderConfig(
            api_key=deepseek_config.get("api_key", ""),
            base_url=deepseek_config.get("base_url", "https://api.deepseek.com/v1"),
            models=[deepseek_config.get("model", "deepseek-chat")],
            capabilities=[],
            rate_limit_rpm=deepseek_config.get("rate_limit", 60),
            priority=deepseek_config.get("priority", 1)
        )

    llm_manager = LLMProviderManager(provider_configs)
    await llm_manager.initialize()

    # Initialiser le classifier avec le LLM manager (pas le config_manager!)
    classifier = IntentClassifier(llm_manager)
    await classifier.initialize()

    # Requête de test
    query = "Mes retraits espèces"

    print(f"Requête: '{query}'")
    print(f"Attendu: categories=['Retrait especes'], transaction_type='debit'")
    print(f"NE DEVRAIT PAS avoir: operation_type='Carte'\n")

    # Créer la requête de classification
    classification_request = ClassificationRequest(
        user_message=query,
        user_id=100,
        conversation_context=[]
    )

    # Classifier la requête
    result = await classifier.classify_intent(classification_request)

    # Afficher les résultats
    print(f"Intent: {result.intent_group}.{result.intent_subtype}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}\n")

    print("Entités extraites:")
    for entity in result.entities:
        print(f"  - {entity.name}: {entity.value}")

    # Vérifier le résultat
    print("\n" + "="*80)
    print("VÉRIFICATION")
    print("="*80 + "\n")

    has_categories = any(e.name == "categories" and "Retrait especes" in e.value for e in result.entities)
    has_transaction_type_debit = any(e.name == "transaction_type" and e.value == "debit" for e in result.entities)
    has_operation_type = any(e.name == "operation_type" for e in result.entities)

    success = True

    if has_categories:
        print("✓ categories contient 'Retrait especes'")
    else:
        print("✗ ERREUR: categories ne contient pas 'Retrait especes'")
        success = False

    if has_transaction_type_debit:
        print("✓ transaction_type = 'debit'")
    else:
        print("✗ ERREUR: transaction_type != 'debit'")
        success = False

    if not has_operation_type:
        print("✓ operation_type n'est pas extrait (correct)")
    else:
        operation_type_value = next(e.value for e in result.entities if e.name == "operation_type")
        print(f"✗ ERREUR: operation_type est extrait avec valeur '{operation_type_value}'")
        success = False

    print("\n" + "="*80)
    if success:
        print("SUCCÈS: La classification est correcte!")
    else:
        print("ÉCHEC: La classification n'est pas conforme aux attentes")
    print("="*80 + "\n")

    return success

if __name__ == "__main__":
    success = asyncio.run(test_retraits_especes())
    sys.exit(0 if success else 1)
