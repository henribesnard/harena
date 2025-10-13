"""
Test d'intégration pour vérifier qu'aucune régression n'a été introduite
Teste que response_generator peut correctement calculer les limites de transactions
"""
import sys
import os

# Fix encodage Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import après l'ajout au path
from conversation_service.agents.llm.response_generator import ResponseGenerator
from conversation_service.agents.llm import LLMProviderManager, ProviderConfig, ProviderType
from config_service.config import settings

def test_response_generator_instantiation():
    """Test que ResponseGenerator peut être instancié sans erreur"""

    print("=" * 80)
    print("TEST D'INTÉGRATION - ResponseGenerator avec nouvelle configuration")
    print("=" * 80)

    try:
        # Créer un mock LLM provider manager
        provider_configs = {
            ProviderType.DEEPSEEK: ProviderConfig(
                api_key="test_key",
                base_url="https://api.deepseek.com/v1",
                models=["deepseek-chat"],
                capabilities=[],
                rate_limit_rpm=60,
                priority=1
            )
        }

        llm_manager = LLMProviderManager(provider_configs)

        print("\n✓ Étape 1: Création du LLM Provider Manager")

        # Instancier ResponseGenerator
        response_generator = ResponseGenerator(
            llm_manager=llm_manager,
            response_templates_path=None,
            model="deepseek-chat",
            max_tokens=8000,
            temperature=0.7,
            enable_analytics=False,
            enable_visualizations=False
        )

        print("✓ Étape 2: Instanciation du ResponseGenerator")

        # Simuler des transactions filtrées
        test_transactions = [
            {
                "amount": -50.0,
                "date": "2024-01-15",
                "primary_description": "Restaurant ABC",
                "merchant_name": "Restaurant ABC",
                "category_name": "Restaurants",
                "operation_type": "CB"
            }
        ] * 150  # 150 transactions de test

        print(f"✓ Étape 3: Création de {len(test_transactions)} transactions de test")

        # Tester la méthode _calculate_max_transactions_for_context
        max_tx = response_generator._calculate_max_transactions_for_context(test_transactions)

        print(f"✓ Étape 4: Calcul de la limite de transactions: {max_tx}")

        # Vérifications
        expected_max = settings.MAX_TRANSACTIONS_IN_CONTEXT

        if max_tx == expected_max:
            print(f"\n✓ SUCCÈS: Limite calculée ({max_tx}) = limite attendue ({expected_max})")
        else:
            print(f"\n❌ ERREUR: Limite calculée ({max_tx}) != limite attendue ({expected_max})")
            return False

        # Tester avec moins de transactions que la limite
        small_test = test_transactions[:50]
        max_tx_small = response_generator._calculate_max_transactions_for_context(small_test)

        if max_tx_small == 50:
            print(f"✓ SUCCÈS: Avec 50 transactions, limite = 50 (correct)")
        else:
            print(f"❌ ERREUR: Avec 50 transactions, limite = {max_tx_small} (attendu: 50)")
            return False

        print("\n" + "=" * 80)
        print("✓ TOUS LES TESTS PASSENT - Aucune régression détectée")
        print("=" * 80)
        print("\nDétails de la configuration utilisée:")
        print("-" * 80)
        print(f"  MAX_CONTEXT_TOKENS              : {settings.MAX_CONTEXT_TOKENS:>10,}")
        print(f"  TRANSACTION_CONTEXT_BUDGET      : {settings.TRANSACTION_CONTEXT_BUDGET_TOKENS:>10,}")
        print(f"  MAX_TRANSACTIONS_IN_CONTEXT     : {settings.MAX_TRANSACTIONS_IN_CONTEXT:>10,}")
        print(f"  AVG_TOKENS_PER_TRANSACTION      : {settings.AVG_TOKENS_PER_TRANSACTION:>10}")
        print(f"  TOKEN_SAFETY_BUFFER             : {settings.TOKEN_SAFETY_BUFFER:>10,}")

        return True

    except Exception as e:
        print(f"\n❌ ERREUR lors du test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_response_generator_instantiation()
    sys.exit(0 if success else 1)
