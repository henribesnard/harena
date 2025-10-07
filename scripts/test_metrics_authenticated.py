"""
Test des endpoints de métriques avec authentification

IMPORTANT: Utilise le port 8004 (metric_service standalone)
Le port 8000 nécessite un redémarrage manuel de local_app.py
"""
import requests

# PORT 8004: metric_service standalone (FONCTIONNE)
url = "http://localhost:8004/api/v1/metrics/expenses/yoy"

# Token JWT pour user_id=100
headers = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjAzODg0NDcsInN1YiI6IjEwMCIsInBlcm1pc3Npb25zIjpbImNoYXQ6d3JpdGUiXX0.P6Uga0xm3RgRCWDv96stimmYv2Ow36As-Am4SVDKqMU'
}

print("Test de l'endpoint YoY Depenses")
print(f"URL: {url}")
print(f"Token: {headers['Authorization'][:50]}...\n")

response = requests.get(url, headers=headers)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("SUCCES!\n")
    data = response.json()

    if data.get('success'):
        metrics = data['data']
        print(f"Metriques YoY Depenses:")
        print(f"   Annee talon: {metrics['periode_talon']['annee']} -> {metrics['periode_talon']['total']:.2f} EUR")
        print(f"   Annee cible: {metrics['periode_cible']['annee']} -> {metrics['periode_cible']['total']:.2f} EUR")
        print(f"   Variation: {metrics['variation']['montant']:.2f} EUR ({metrics['variation']['pourcentage']:.2f}%)")
        print(f"   Direction: {metrics['variation']['direction']}")
else:
    print(f"ERREUR: {response.text}")

print("\n" + "="*60)
print("Pour utiliser le port 8000 (tous les services intégrés):")
print("1. Arrête local_app.py (Ctrl+C)")
print("2. Redémarre: python local_app.py")
print("3. Change l'URL vers: http://localhost:8000/api/v1/metrics/expenses/yoy")
print("="*60)
