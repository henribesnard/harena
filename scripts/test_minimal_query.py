#!/usr/bin/env python3
"""
Test minimal avec 2 questions pour valider l'extraction de queries
"""

import json
import requests
import time

class MinimalQueryTester:
    def __init__(self):
        self.base_url = "http://localhost:8000/api/v1"
        self.username = "henri@example.com"
        self.password = "hounwanou"
        self.session = requests.Session()
        self.user_id = None

    def authenticate(self):
        data = f"username={self.username}&password={self.password}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        resp = self.session.post(f"{self.base_url}/users/auth/login", data=data, headers=headers, timeout=30)
        resp.raise_for_status()

        token = resp.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {token}"})

        user_resp = self.session.get(f"{self.base_url}/users/me", timeout=30)
        user_resp.raise_for_status()
        self.user_id = user_resp.json().get("id")

        print(f"Authentification OK - User ID: {self.user_id}")

    def test_query(self, question):
        print(f"\nTest: {question}")

        start_time = time.time()
        payload = {"message": question}
        response = self.session.post(f"{self.base_url}/conversation/{self.user_id}", json=payload, timeout=60)
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            query = data.get("query")

            if query:
                print(f"  SUCCESS ({latency:.0f}ms) - Query found!")
                print(f"  Query size: {len(json.dumps(query))} chars")
                print(f"  Has filters: {'filters' in query}")
                print(f"  Has aggregations: {'aggregations' in query}")
                return True
            else:
                print(f"  SUCCESS ({latency:.0f}ms) - but NO QUERY")
                return False
        else:
            print(f"  ERROR: {response.status_code}")
            return False

    def run_minimal_test(self):
        print("Test minimal extraction queries")
        print("=" * 35)

        self.authenticate()

        questions = [
            "Mes dépenses de plus de 100 euros",
            "Mes achats Amazon"
        ]

        successes = 0
        for question in questions:
            if self.test_query(question):
                successes += 1
            time.sleep(1)  # Pause entre requêtes

        print(f"\n=== RÉSULTAT ===")
        print(f"Queries trouvées: {successes}/{len(questions)}")

        if successes == len(questions):
            print("EXCELLENT! L'extraction de queries fonctionne parfaitement!")
        elif successes > 0:
            print("BON! Quelques queries sont extraites.")
        else:
            print("PROBLÈME: Aucune query extraite.")

if __name__ == "__main__":
    tester = MinimalQueryTester()
    try:
        tester.run_minimal_test()
    except Exception as e:
        print(f"ERREUR: {e}")
        import traceback
        traceback.print_exc()