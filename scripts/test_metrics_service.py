"""
Script de test pour le Metric Service
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8004"

def test_health():
    """Test du health check"""
    print("🔍 Test Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200

def test_mom(user_id: int = 1):
    """Test Month-over-Month"""
    print(f"🔍 Test MoM pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/trends/mom/{user_id}")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ MoM calculé:")
        print(f"   Mois actuel: {data['data']['current_month']}")
        print(f"   Montant actuel: {data['data']['current_amount']}€")
        print(f"   Variation: {data['data']['change_percent']}%")
        print(f"   Tendance: {data['data']['trend']}")
        print(f"   Cached: {data.get('cached', False)}\n")
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def test_yoy(user_id: int = 1):
    """Test Year-over-Year"""
    print(f"🔍 Test YoY pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/trends/yoy/{user_id}")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ YoY calculé:")
        print(f"   Année actuelle: {data['data']['current_year']}")
        print(f"   Montant actuel: {data['data']['current_amount']}€")
        print(f"   Variation: {data['data']['change_percent']}%")
        print(f"   Tendance: {data['data']['trend']}\n")
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def test_savings_rate(user_id: int = 1):
    """Test Savings Rate"""
    print(f"🔍 Test Savings Rate pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/health/savings-rate/{user_id}?period_days=30")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Savings Rate calculé:")
        print(f"   Taux d'épargne: {data['data']['savings_rate']}%")
        print(f"   Revenus: {data['data']['total_income']}€")
        print(f"   Dépenses: {data['data']['total_expenses']}€")
        print(f"   Épargne nette: {data['data']['net_savings']}€")
        print(f"   Statut: {data['data']['health_status']}")
        if data['data'].get('recommendation'):
            print(f"   💡 {data['data']['recommendation']}\n")
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def test_expense_ratio(user_id: int = 1):
    """Test Expense Ratio"""
    print(f"🔍 Test Expense Ratio pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/health/expense-ratio/{user_id}?period_days=30")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Expense Ratio calculé:")
        print(f"   Essentiels: {data['data']['essentials_percent']}% ({data['data']['essentials']}€)")
        print(f"   Lifestyle: {data['data']['lifestyle_percent']}% ({data['data']['lifestyle']}€)")
        print(f"   Épargne: {data['data']['savings_percent']}% ({data['data']['savings']}€)")
        print(f"   Équilibré (50/30/20): {'✅' if data['data']['is_balanced'] else '❌'}")
        if data['data']['recommendations']:
            print(f"   💡 Recommandations:")
            for rec in data['data']['recommendations']:
                print(f"      - {rec}")
        print()
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def test_burn_rate(user_id: int = 1):
    """Test Burn Rate"""
    print(f"🔍 Test Burn Rate pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/health/burn-rate/{user_id}?period_days=30")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Burn Rate calculé:")
        print(f"   Solde actuel: {data['data']['current_balance']}€")
        print(f"   Burn Rate mensuel: {data['data']['monthly_burn_rate']}€/mois")
        if data['data']['runway_days']:
            print(f"   Runway: {data['data']['runway_days']} jours ({data['data']['runway_months']} mois)")
        print(f"   Niveau de risque: {data['data']['risk_level']}")
        if data['data'].get('alert'):
            print(f"   ⚠️ {data['data']['alert']}")
        print()
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def test_balance_forecast(user_id: int = 1):
    """Test Balance Forecast"""
    print(f"🔍 Test Balance Forecast pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/health/balance-forecast/{user_id}?periods=30")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        forecast_data = data['data']
        print(f"✅ Balance Forecast calculé:")
        print(f"   Type: {forecast_data['forecast_type']}")
        print(f"   Période: {forecast_data['periods']} jours")
        print(f"   Solde actuel: {forecast_data['current_balance']}€")
        print(f"   Tendance: {forecast_data['trend']}")
        print(f"   Confiance: {forecast_data['confidence']}")

        # Afficher quelques prévisions
        predictions = forecast_data['predictions']
        print(f"   Prévisions (échantillon):")
        for i in [0, len(predictions)//2, -1]:
            if i < len(predictions):
                pred = predictions[i]
                print(f"      {pred['date']}: {pred['balance']:.2f}€ (±{pred['balance_upper'] - pred['balance']:.2f}€)")
        print()
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def test_expense_forecast(user_id: int = 1):
    """Test Expense Forecast"""
    print(f"🔍 Test Expense Forecast pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/health/expense-forecast/{user_id}?periods=30")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        forecast_data = data['data']
        print(f"✅ Expense Forecast calculé:")
        print(f"   Type: {forecast_data['forecast_type']}")
        print(f"   Période: {forecast_data['periods']} jours")
        print(f"   Total prévu: {forecast_data['total_forecast']:.2f}€")
        print(f"   Confiance: {forecast_data['confidence']}")

        # Afficher quelques prévisions
        predictions = forecast_data['predictions']
        print(f"   Prévisions (échantillon):")
        for i in [0, len(predictions)//2, -1]:
            if i < len(predictions):
                pred = predictions[i]
                print(f"      {pred['date']}: {pred['amount']:.2f}€ (±{pred['amount_upper'] - pred['amount']:.2f}€)")
        print()
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def test_recurring_expenses(user_id: int = 1):
    """Test Recurring Expenses Detection"""
    print(f"🔍 Test Recurring Expenses pour user_id={user_id}...")
    response = requests.get(f"{BASE_URL}/api/v1/metrics/patterns/recurring/{user_id}?min_occurrences=3&lookback_days=90")

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        recurring_data = data['data']
        print(f"✅ Recurring Expenses détectées:")
        print(f"   Période: {recurring_data['period_start']} → {recurring_data['period_end']}")
        print(f"   Total mensuel récurrent: {recurring_data['total_monthly_recurring']}€")
        print(f"   Pourcentage des dépenses: {recurring_data['recurring_percent_of_expenses']}%")

        expenses = recurring_data['recurring_expenses']
        print(f"   Nombre de dépenses récurrentes: {len(expenses)}")

        if expenses:
            print(f"\n   Top 5 dépenses récurrentes:")
            for exp in expenses[:5]:
                print(f"      • {exp['merchant']}: {exp['average_amount']}€ ({exp['frequency']})")
                print(f"        Dernière: {exp['last_occurrence']}, Prochaine: {exp['next_expected']}")
                print(f"        Confiance: {exp['confidence']:.0%}, Occurrences: {exp['occurrences']}")
        print()
        return True
    else:
        print(f"❌ Erreur: {response.text}\n")
        return False

def run_all_tests(user_id: int = 1):
    """Exécuter tous les tests"""
    print("=" * 80)
    print("🧪 TESTS DU METRIC SERVICE")
    print("=" * 80)
    print()

    results = {
        "Health Check": test_health(),
        "MoM": test_mom(user_id),
        "YoY": test_yoy(user_id),
        "Savings Rate": test_savings_rate(user_id),
        "Expense Ratio": test_expense_ratio(user_id),
        "Burn Rate": test_burn_rate(user_id),
        "Balance Forecast": test_balance_forecast(user_id),
        "Expense Forecast": test_expense_forecast(user_id),
        "Recurring Expenses": test_recurring_expenses(user_id),
    }

    print("=" * 80)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<50} {status}")

    total = len(results)
    passed = sum(results.values())
    print()
    print(f"Total: {passed}/{total} tests réussis ({passed/total*100:.0f}%)")
    print()

if __name__ == "__main__":
    import sys

    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print(f"🎯 Tests pour user_id={user_id}")
    print()

    try:
        run_all_tests(user_id)
    except requests.exceptions.ConnectionError:
        print("❌ Erreur: Impossible de se connecter au Metric Service")
        print(f"   Assurez-vous que le service tourne sur {BASE_URL}")
        sys.exit(1)
