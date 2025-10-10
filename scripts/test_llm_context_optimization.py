"""
Script de test de l'optimisation du contexte LLM

Teste la réduction des transactions envoyées au LLM avec des données réelles
"""

import json
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def filter_transaction_data(transaction: dict) -> dict:
    """Filtre une transaction selon les nouveaux critères (6 champs essentiels)"""

    essential_fields = {
        'amount',
        'date',
        'primary_description',
        'merchant_name',
        'category_name',
        'operation_type'
    }

    filtered = {}
    for key, value in transaction.items():
        if key in essential_fields and value is not None:
            filtered[key] = value

    return filtered


def estimate_tokens(text: str) -> int:
    """Estimation simple du nombre de tokens (règle approximative)"""
    # Règle approximative : 1 token ≈ 4 caractères
    return len(text) // 4


def main():
    # Charger les données exemple
    example_file = Path(__file__).parent.parent / "exemple_result.json"

    print("=" * 80)
    print("TEST DE L'OPTIMISATION DU CONTEXTE LLM")
    print("=" * 80)
    print()

    if not example_file.exists():
        print(f"[ERROR] Fichier {example_file} introuvable")
        return 1

    with open(example_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transactions_full = data.get('results', [])

    if not transactions_full:
        print("[ERROR] Aucune transaction dans le fichier exemple")
        return 1

    print(f"[DATA] Chargement des donnees : {len(transactions_full)} transactions\n")

    # Test AVANT optimisation (une transaction complète)
    sample_transaction_full = transactions_full[0]
    full_json = json.dumps([sample_transaction_full], ensure_ascii=False, indent=2)
    full_size = len(full_json)
    full_tokens = estimate_tokens(full_json)

    print("AVANT OPTIMISATION (transaction complète):")
    print(f"  - Champs par transaction: {len(sample_transaction_full)}")
    print(f"  - Taille 1 transaction: {len(json.dumps(sample_transaction_full))} bytes")
    print(f"  - Tokens estimés (1 transaction): {full_tokens}")

    # Test APRES optimisation
    filtered_transactions = [filter_transaction_data(tx) for tx in transactions_full]
    filtered_json = json.dumps(filtered_transactions, ensure_ascii=False, indent=2)
    filtered_size = len(filtered_json)
    filtered_tokens = estimate_tokens(filtered_json)

    sample_filtered = filtered_transactions[0]

    print()
    print("APRÈS OPTIMISATION (transaction filtrée - 6 champs):")
    print(f"  - Champs par transaction: {len(sample_filtered)}")
    print(f"  - Champs conservés: {', '.join(sample_filtered.keys())}")
    print(f"  - Taille 1 transaction: {len(json.dumps(sample_filtered))} bytes")

    # Calcul pour toutes les transactions
    full_all_size = len(json.dumps([sample_transaction_full] * len(transactions_full), ensure_ascii=False))
    full_all_tokens = estimate_tokens(json.dumps([sample_transaction_full] * len(transactions_full), ensure_ascii=False))

    print()
    print(f"COMPARAISON POUR {len(transactions_full)} TRANSACTIONS:")
    print("-" * 80)
    print(f"  AVANT (toutes transactions complètes):")
    print(f"    - Taille totale: {full_all_size:,} bytes")
    print(f"    - Tokens estimés: {full_all_tokens:,}")

    print()
    print(f"  APRÈS (toutes transactions filtrées):")
    print(f"    - Taille totale: {filtered_size:,} bytes")
    print(f"    - Tokens estimés: {filtered_tokens:,}")

    # Calcul de la réduction
    size_reduction = ((full_all_size - filtered_size) / full_all_size) * 100
    token_reduction = ((full_all_tokens - filtered_tokens) / full_all_tokens) * 100

    print()
    print(f"  RÉDUCTION:")
    print(f"    - Taille: {size_reduction:.1f}% (économie: {full_all_size - filtered_size:,} bytes)")
    print(f"    - Tokens: {token_reduction:.1f}% (économie: {full_all_tokens - filtered_tokens:,} tokens)")

    # Estimation coûts LLM (DeepSeek pricing)
    cost_before = (full_all_tokens / 1_000_000) * 0.14  # $0.14 per 1M tokens
    cost_after = (filtered_tokens / 1_000_000) * 0.14

    print()
    print(f"  COÛT LLM ESTIMÉ (DeepSeek @ $0.14/1M tokens):")
    print(f"    - Avant: ${cost_before:.6f} par requête")
    print(f"    - Après: ${cost_after:.6f} par requête")
    print(f"    - Économie: ${cost_before - cost_after:.6f} par requête ({token_reduction:.1f}%)")

    # Vérification limite tokens
    print()
    print("VÉRIFICATION LIMITES TOKENS:")
    print("-" * 80)

    MAX_TOKENS_LLM = 128_000
    MAX_TOKENS_FOR_TRANSACTIONS = 80_000

    print(f"  - Limite totale LLM (DeepSeek): {MAX_TOKENS_LLM:,} tokens")
    print(f"  - Budget alloué aux transactions: {MAX_TOKENS_FOR_TRANSACTIONS:,} tokens")
    print(f"  - Marge restante pour le reste: {MAX_TOKENS_LLM - MAX_TOKENS_FOR_TRANSACTIONS:,} tokens")

    if filtered_tokens > MAX_TOKENS_FOR_TRANSACTIONS:
        print(f"  [WARNING] DEPASSEMENT: {filtered_tokens:,} tokens > {MAX_TOKENS_FOR_TRANSACTIONS:,} limite")
        max_transactions = MAX_TOKENS_FOR_TRANSACTIONS // (filtered_tokens // len(filtered_transactions))
        print(f"  -> Reduction necessaire a {max_transactions} transactions")
    else:
        print(f"  [OK] OK: {filtered_tokens:,} tokens < {MAX_TOKENS_FOR_TRANSACTIONS:,} limite")
        remaining = MAX_TOKENS_FOR_TRANSACTIONS - filtered_tokens
        print(f"  -> Marge restante: {remaining:,} tokens")

    # Exemple de transaction filtrée
    print()
    print("EXEMPLE DE TRANSACTION FILTRÉE:")
    print("-" * 80)
    print(json.dumps(sample_filtered, ensure_ascii=False, indent=2))

    print()
    print("=" * 80)
    print("[SUCCESS] TEST TERMINE AVEC SUCCES")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
