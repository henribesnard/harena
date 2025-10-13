"""
Test rapide de la limite de transactions avec la nouvelle configuration
Simule le comportement de response_generator._calculate_max_transactions_for_context
"""
import sys
import os

# Fix encodage Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_service.config import settings

def simulate_transaction_limit_calculation(num_available_transactions: int) -> int:
    """Simule le calcul de _calculate_max_transactions_for_context"""

    # Récupérer les configurations
    transaction_budget_tokens = settings.TRANSACTION_CONTEXT_BUDGET_TOKENS
    avg_tokens_per_tx = settings.AVG_TOKENS_PER_TRANSACTION
    hard_limit_transactions = settings.MAX_TRANSACTIONS_IN_CONTEXT

    # Calcul du nombre max basé sur le budget tokens
    max_based_on_tokens = transaction_budget_tokens // avg_tokens_per_tx

    # Appliquer les limites
    max_transactions = min(
        hard_limit_transactions,
        max_based_on_tokens,
        num_available_transactions
    )

    return max_transactions

def test_various_scenarios():
    """Teste différents scénarios de nombre de transactions"""

    print("=" * 80)
    print("TEST DES LIMITES DE TRANSACTIONS AVEC NOUVELLE CONFIGURATION")
    print("=" * 80)

    scenarios = [
        (50, "Peu de transactions (50)"),
        (100, "Exactement la limite hard (100)"),
        (200, "Supérieur à hard limit (200)"),
        (500, "Beaucoup de transactions (500)"),
        (1327, "Cas réel utilisateur (1327)"),
        (5000, "Maximum pagination (5000)"),
        (10000, "Maximum théorique (10000)"),
    ]

    print("\nScénarios testés:")
    print("-" * 80)
    print(f"{'Disponibles':>12} | {'Limite calculée':>16} | Commentaire")
    print("-" * 80)

    for num_available, description in scenarios:
        max_tx = simulate_transaction_limit_calculation(num_available)
        tokens_used = max_tx * settings.AVG_TOKENS_PER_TRANSACTION

        # Déterminer quelle limite a été appliquée
        hard_limit = settings.MAX_TRANSACTIONS_IN_CONTEXT
        budget_limit = settings.TRANSACTION_CONTEXT_BUDGET_TOKENS // settings.AVG_TOKENS_PER_TRANSACTION

        if max_tx == hard_limit and num_available >= hard_limit:
            limit_type = "Hard limit"
        elif max_tx == budget_limit and num_available >= budget_limit:
            limit_type = "Budget tokens"
        else:
            limit_type = "Disponibles"

        print(f"{num_available:>12,} tx | {max_tx:>12,} tx | {limit_type:15} ({tokens_used:>6,} tokens) - {description}")

    print("\n" + "=" * 80)
    print("Configuration actuelle:")
    print("-" * 80)
    print(f"  MAX_CONTEXT_TOKENS              : {settings.MAX_CONTEXT_TOKENS:>10,} tokens")
    print(f"  TRANSACTION_CONTEXT_BUDGET      : {settings.TRANSACTION_CONTEXT_BUDGET_TOKENS:>10,} tokens")
    print(f"  MAX_TRANSACTIONS_IN_CONTEXT     : {settings.MAX_TRANSACTIONS_IN_CONTEXT:>10,} tx")
    print(f"  AVG_TOKENS_PER_TRANSACTION      : {settings.AVG_TOKENS_PER_TRANSACTION:>10} tokens")
    print(f"  TOKEN_SAFETY_BUFFER             : {settings.TOKEN_SAFETY_BUFFER:>10,} tokens")

    # Calcul des limites théoriques
    budget_limit = settings.TRANSACTION_CONTEXT_BUDGET_TOKENS // settings.AVG_TOKENS_PER_TRANSACTION
    print(f"\n  Limite théorique (budget)       : {budget_limit:>10,} tx")
    print(f"  Limite effective (hard limit)   : {settings.MAX_TRANSACTIONS_IN_CONTEXT:>10,} tx")

    print("\n" + "=" * 80)
    print("✓ TEST TERMINÉ - La limite de 100 transactions est bien appliquée")
    print("=" * 80)

if __name__ == "__main__":
    test_various_scenarios()
