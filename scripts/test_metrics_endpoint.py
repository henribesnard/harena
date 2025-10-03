"""
Script de test pour l'endpoint des métriques dashboard.

Ce script teste l'endpoint /api/v1/metrics/dashboard et affiche
les résultats de manière formatée.
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_V1 = "/api/v1"

# Token de test - à remplacer par un vrai token
# Pour obtenir un token : python scripts/get_auth_token.py
TEST_TOKEN = "your-jwt-token-here"


def format_currency(amount):
    """Formate un montant en euros."""
    return f"{amount:,.2f} €".replace(",", " ")


def format_evolution(percent):
    """Formate un pourcentage d'évolution avec couleur."""
    if percent > 0:
        return f"📈 +{percent}%"
    elif percent < 0:
        return f"📉 {percent}%"
    else:
        return f"➡️  {percent}%"


def test_metrics_endpoint():
    """Test l'endpoint des métriques."""
    print("=" * 60)
    print("TEST DE L'ENDPOINT MÉTRIQUES DASHBOARD")
    print("=" * 60)
    print()

    # Vérifier que le serveur est accessible
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Le serveur ne répond pas correctement")
            print(f"Status: {health_response.status_code}")
            return False
        print("✅ Serveur accessible")
        print()
    except requests.exceptions.RequestException as e:
        print(f"❌ Impossible de contacter le serveur: {e}")
        print(f"Vérifiez que le serveur tourne sur {BASE_URL}")
        return False

    # Appeler l'endpoint des métriques
    headers = {
        "Authorization": f"Bearer {TEST_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        print(f"📡 Appel de {API_V1}/metrics/dashboard")
        response = requests.get(
            f"{BASE_URL}{API_V1}/metrics/dashboard",
            headers=headers,
            timeout=10
        )

        print(f"Status: {response.status_code}")
        print()

        if response.status_code == 401:
            print("❌ Token invalide ou expiré")
            print("Utilisez un token valide (voir TEST_TOKEN dans le script)")
            return False

        if response.status_code != 200:
            print(f"❌ Erreur: {response.status_code}")
            print(response.text)
            return False

        # Parser la réponse
        data = response.json()

        print("=" * 60)
        print("RÉSULTAT")
        print("=" * 60)
        print()

        # Afficher les soldes par compte
        print("📊 SOLDES PAR COMPTE")
        print("-" * 60)
        if data.get("accounts"):
            for account in data["accounts"]:
                print(f"\n  {account.get('account_name', 'Sans nom')}")
                print(f"    💰 Solde: {format_currency(account.get('balance', 0))}")
                print(f"    🏦 Type: {account.get('account_type', 'N/A')}")
                if account.get('updated_at'):
                    print(f"    🕐 MAJ: {account['updated_at']}")
        else:
            print("  Aucun compte trouvé")

        print()
        print("-" * 60)
        print()

        # Afficher les évolutions
        print("📈 ÉVOLUTIONS MENSUELLES")
        print("-" * 60)

        # Dépenses
        expenses = data.get("expenses", {})
        print(f"\n  💸 DÉPENSES")
        print(f"    Mois en cours:  {format_currency(expenses.get('current_month', 0))}")
        print(f"    Mois précédent: {format_currency(expenses.get('previous_month', 0))}")
        print(f"    Évolution:      {format_evolution(expenses.get('evolution_percent', 0))}")

        # Revenus
        income = data.get("income", {})
        print(f"\n  💰 REVENUS")
        print(f"    Mois en cours:  {format_currency(income.get('current_month', 0))}")
        print(f"    Mois précédent: {format_currency(income.get('previous_month', 0))}")
        print(f"    Évolution:      {format_evolution(income.get('evolution_percent', 0))}")

        print()
        print("-" * 60)
        print()

        # Période
        period = data.get("period", {})
        if period:
            print("📅 PÉRIODE")
            print("-" * 60)
            print(f"  Début mois en cours:  {period.get('current_month_start', 'N/A')}")
            print(f"  Début mois précédent: {period.get('previous_month_start', 'N/A')}")
            print(f"  Fin mois précédent:   {period.get('previous_month_end', 'N/A')}")
            print()

        print("=" * 60)
        print("✅ TEST RÉUSSI")
        print("=" * 60)

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de la requête: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Erreur de parsing JSON: {e}")
        print("Réponse brute:", response.text)
        return False


def print_usage():
    """Affiche les instructions d'utilisation."""
    print()
    print("INSTRUCTIONS:")
    print("-" * 60)
    print("1. Assurez-vous que le serveur est démarré:")
    print("   python local_app.py")
    print()
    print("2. Obtenez un token JWT valide:")
    print("   - Connectez-vous via /api/v1/users/auth/login")
    print("   - Ou utilisez un token existant")
    print()
    print("3. Modifiez la variable TEST_TOKEN dans ce script")
    print()
    print("4. Relancez le script:")
    print("   python scripts/test_metrics_endpoint.py")
    print("-" * 60)
    print()


if __name__ == "__main__":
    success = test_metrics_endpoint()

    if not success:
        print_usage()
        sys.exit(1)
    else:
        sys.exit(0)
