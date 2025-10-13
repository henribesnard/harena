"""
Script de validation de la configuration des tokens
Vérifie que toutes les variables sont cohérentes et accessibles
"""
import sys
import os

# Fix pour l'encodage Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_service.config import settings

def validate_token_configuration():
    """Valide la configuration des tokens"""

    print("=" * 80)
    print("VALIDATION DE LA CONFIGURATION DES TOKENS")
    print("=" * 80)

    # Vérifier que toutes les variables sont définies
    required_vars = {
        'MAX_CONTEXT_TOKENS': settings.MAX_CONTEXT_TOKENS,
        'TRANSACTION_CONTEXT_BUDGET_TOKENS': settings.TRANSACTION_CONTEXT_BUDGET_TOKENS,
        'MAX_TRANSACTIONS_IN_CONTEXT': settings.MAX_TRANSACTIONS_IN_CONTEXT,
        'AVG_TOKENS_PER_TRANSACTION': settings.AVG_TOKENS_PER_TRANSACTION,
        'TOKEN_SAFETY_BUFFER': settings.TOKEN_SAFETY_BUFFER,
    }

    print("\n✓ Variables de configuration chargées:")
    print("-" * 80)
    for var_name, value in required_vars.items():
        print(f"  {var_name:40s} = {value:>10}")

    # Calculs de validation
    print("\n✓ Calculs de validation:")
    print("-" * 80)

    max_context = settings.MAX_CONTEXT_TOKENS
    tx_budget = settings.TRANSACTION_CONTEXT_BUDGET_TOKENS
    avg_tokens = settings.AVG_TOKENS_PER_TRANSACTION
    hard_limit = settings.MAX_TRANSACTIONS_IN_CONTEXT
    safety_buffer = settings.TOKEN_SAFETY_BUFFER

    # Budget restant pour prompts/aggregations/output
    other_budget = max_context - tx_budget - safety_buffer

    # Nombre max de transactions théorique basé sur budget
    max_tx_by_budget = tx_budget // avg_tokens

    # Nombre effectif (limité par hard_limit)
    effective_max_tx = min(hard_limit, max_tx_by_budget)

    # Tokens utilisés pour les transactions
    tokens_for_tx = effective_max_tx * avg_tokens

    # Tokens restants
    tokens_remaining = max_context - tokens_for_tx - safety_buffer

    print(f"  Contexte global maximum              : {max_context:>10,} tokens")
    print(f"  Budget transactions                  : {tx_budget:>10,} tokens")
    print(f"  Budget autre (prompts/agg/output)    : {other_budget:>10,} tokens")
    print(f"  Buffer sécurité                      : {safety_buffer:>10,} tokens")
    print()
    print(f"  Max transactions (budget tokens)     : {max_tx_by_budget:>10,} tx")
    print(f"  Max transactions (hard limit)        : {hard_limit:>10,} tx")
    print(f"  Max transactions effectif            : {effective_max_tx:>10,} tx")
    print()
    print(f"  Tokens utilisés pour transactions    : {tokens_for_tx:>10,} tokens")
    print(f"  Tokens restants après transactions   : {tokens_remaining:>10,} tokens")

    # Validations
    print("\n✓ Validations:")
    print("-" * 80)

    errors = []
    warnings = []

    # Validation 1: Budget transactions ne doit pas dépasser contexte
    if tx_budget > max_context:
        errors.append(f"Budget transactions ({tx_budget}) > Contexte max ({max_context})")
    else:
        print(f"  ✓ Budget transactions < Contexte max")

    # Validation 2: Budget restant suffisant pour prompts/output
    min_other_budget = 20000  # Minimum 20K pour prompts + aggregations + output
    if other_budget < min_other_budget:
        warnings.append(f"Budget autre ({other_budget}) < minimum recommandé ({min_other_budget})")
    else:
        print(f"  ✓ Budget autre suffisant ({other_budget:,} tokens)")

    # Validation 3: Hard limit raisonnable
    if hard_limit > max_tx_by_budget:
        warnings.append(f"Hard limit ({hard_limit}) > max budget ({max_tx_by_budget}) - la limite budget prévaudra")
    else:
        print(f"  ✓ Hard limit cohérent avec budget tokens")

    # Validation 4: Safety buffer raisonnable
    min_safety = 1000
    if safety_buffer < min_safety:
        warnings.append(f"Buffer sécurité ({safety_buffer}) < minimum ({min_safety})")
    else:
        print(f"  ✓ Buffer sécurité suffisant")

    # Validation 5: Tokens restants positifs
    if tokens_remaining < 0:
        errors.append(f"Tokens restants négatifs ({tokens_remaining}) - configuration incohérente")
    else:
        print(f"  ✓ Tokens restants positifs après allocation")

    # Afficher warnings et errors
    if warnings:
        print("\n⚠ Avertissements:")
        print("-" * 80)
        for warning in warnings:
            print(f"  ⚠ {warning}")

    if errors:
        print("\n❌ Erreurs:")
        print("-" * 80)
        for error in errors:
            print(f"  ❌ {error}")
        print("\n" + "=" * 80)
        print("VALIDATION ÉCHOUÉE")
        print("=" * 80)
        return False

    print("\n" + "=" * 80)
    print("✓ VALIDATION RÉUSSIE - Configuration cohérente")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = validate_token_configuration()
    sys.exit(0 if success else 1)
