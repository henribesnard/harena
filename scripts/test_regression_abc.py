#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de régression pour les catégories A, B, C
Compare les résultats actuels avec les résultats de référence (resultat_ok/)
"""

import json
import requests
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RegressionResult:
    """Résultat de comparaison pour une question"""
    question_id: str
    question: str
    category: str
    has_regression: bool
    differences: List[str]
    current_result: Optional[Dict[str, Any]]
    reference_result: Optional[Dict[str, Any]]


class RegressionTester:
    """Testeur de régression pour catégories A, B, C"""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.user_id: Optional[int] = None

        # Configuration d'authentification
        self.username = "henri@example.com"
        self.password = "hounwanou"

        # Chemins
        self.reference_dir = Path("resultat_ok")
        self.questions_file = Path("test_suite_complete.json")

        # Résultats
        self.regression_results: List[RegressionResult] = []
        self.stats = {
            "total_tested": 0,
            "no_regression": 0,
            "regressions_found": 0,
            "categories": {"A": 0, "B": 0, "C": 0}
        }

    def authenticate(self) -> bool:
        """Authentification OAuth"""
        try:
            data = f"username={self.username}&password={self.password}"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            resp = self.session.post(
                f"{self.base_url}/users/auth/login",
                data=data,
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()

            token = resp.json()["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {token}"})

            # Récupération de l'ID utilisateur
            user_resp = self.session.get(f"{self.base_url}/users/me", timeout=30)
            user_resp.raise_for_status()
            self.user_id = user_resp.json().get("id")

            print(f"Authentification reussie - User ID: {self.user_id}\n")
            return True

        except Exception as e:
            print(f"Erreur authentification: {e}")
            return False

    def load_reference_results(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Charge les résultats de référence pour une catégorie"""
        category_map = {
            "A": "A_montant_results.json",
            "B": "B_vendeur_results.json",
            "C": "C_categorie_results.json"
        }

        reference_file = self.reference_dir / category_map[category]

        if not reference_file.exists():
            print(f"[!] Fichier de reference manquant: {reference_file}")
            return {}

        with open(reference_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Indexer par question_id
        results = {}
        for result in data.get("detailed_results", []):
            question_id = result.get("question_id")
            if question_id:
                results[question_id] = result

        return results

    def test_question(self, question_id: str, question: str, category: str) -> Dict[str, Any]:
        """Teste une question et retourne le résultat"""
        try:
            payload = {
                "user_id": self.user_id,
                "message": question
            }

            resp = self.session.post(
                f"{self.base_url}/conversation/{self.user_id}",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()

            # Extraire les données pertinentes
            return {
                "question_id": question_id,
                "question": question,
                "intent_detected": result.get("intent"),
                "intent_confidence": result.get("confidence", 0.0),
                "entities_structured": result.get("entities", {}),
                "entities_count": len(result.get("entities_list", [])),
                "query_found": "elasticsearch_query" in result,
                "query_data": result.get("elasticsearch_query"),
                "coherence_analysis": result.get("coherence_analysis")
            }

        except Exception as e:
            print(f"[X] Erreur test {question_id}: {e}")
            return {
                "question_id": question_id,
                "question": question,
                "error": str(e)
            }

    def compare_results(self, current: Dict[str, Any], reference: Dict[str, Any],
                       question_id: str, question: str, category: str) -> RegressionResult:
        """Compare les résultats actuel et de référence"""
        differences = []

        # Vérifier si erreur dans le résultat actuel
        if "error" in current:
            return RegressionResult(
                question_id=question_id,
                question=question,
                category=category,
                has_regression=True,
                differences=[f"Erreur API: {current['error']}"],
                current_result=current,
                reference_result=reference
            )

        # 1. Comparer l'intent
        if current.get("intent_detected") != reference.get("intent_detected"):
            differences.append(
                f"Intent différent: '{current.get('intent_detected')}' vs '{reference.get('intent_detected')}'"
            )

        # 2. Comparer les entités structurées
        current_entities = current.get("entities_structured", {})
        ref_entities = reference.get("entities_structured", {})

        # Clés manquantes ou en plus
        current_keys = set(current_entities.keys())
        ref_keys = set(ref_entities.keys())

        missing_keys = ref_keys - current_keys
        extra_keys = current_keys - ref_keys

        if missing_keys:
            differences.append(f"Entités manquantes: {', '.join(missing_keys)}")
        if extra_keys:
            differences.append(f"Entités supplémentaires: {', '.join(extra_keys)}")

        # Valeurs différentes pour les clés communes
        for key in current_keys & ref_keys:
            if current_entities[key] != ref_entities[key]:
                differences.append(
                    f"Entité '{key}': {current_entities[key]} vs {ref_entities[key]}"
                )

        # 3. Comparer la présence de query
        current_has_query = current.get("query_found", False)
        ref_has_query = reference.get("query_found", False)

        if current_has_query != ref_has_query:
            differences.append(
                f"Query trouvée: {current_has_query} vs {ref_has_query}"
            )

        # 4. Comparer les filtres dans la query (si présente)
        if current_has_query and ref_has_query:
            current_filters = current.get("query_data", {}).get("filters", {})
            ref_filters = reference.get("query_data", {}).get("filters", {})

            current_filter_keys = set(current_filters.keys())
            ref_filter_keys = set(ref_filters.keys())

            missing_filters = ref_filter_keys - current_filter_keys
            extra_filters = current_filter_keys - ref_filter_keys

            if missing_filters:
                differences.append(f"Filtres manquants: {', '.join(missing_filters)}")
            if extra_filters:
                differences.append(f"Filtres supplémentaires: {', '.join(extra_filters)}")

        has_regression = len(differences) > 0

        return RegressionResult(
            question_id=question_id,
            question=question,
            category=category,
            has_regression=has_regression,
            differences=differences,
            current_result=current,
            reference_result=reference
        )

    def run_regression_tests(self):
        """Exécute les tests de régression pour A, B, C"""

        # Charger les questions
        if not self.questions_file.exists():
            print(f"[X] Fichier de questions manquant: {self.questions_file}")
            return

        with open(self.questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extraire les catégories A, B, C
        all_categories = data.get("categories", {})
        questions_to_test = {
            cat_name: cat_data
            for cat_name, cat_data in all_categories.items()
            if cat_name in ["A_montant", "B_vendeur", "C_categorie"]
        }

        print("=" * 70)
        print("TEST DE RÉGRESSION - CATÉGORIES A, B, C")
        print("=" * 70)
        print()

        # Pour chaque catégorie
        for category_name, category_data in questions_to_test.items():
            category_letter = category_name[0]  # A, B ou C
            questions = category_data.get("questions", [])

            print(f"\n{'-' * 70}")
            print(f"CATEGORIE {category_letter} - {category_name}")
            print(f"{'-' * 70}")

            # Charger les résultats de référence
            reference_results = self.load_reference_results(category_letter)

            if not reference_results:
                print(f"[!] Pas de resultats de reference, skip categorie {category_letter}")
                continue

            # Tester chaque question
            for q in questions:
                question_id = q["id"]
                question_text = q["question"]

                print(f"\n  Test {question_id}: {question_text[:60]}...")

                # Résultat actuel
                current_result = self.test_question(question_id, question_text, category_letter)

                # Résultat de référence
                reference_result = reference_results.get(question_id)

                if not reference_result:
                    print(f"  [!] Pas de reference pour {question_id}")
                    continue

                # Comparaison
                regression = self.compare_results(
                    current_result, reference_result,
                    question_id, question_text, category_letter
                )

                self.regression_results.append(regression)
                self.stats["total_tested"] += 1
                self.stats["categories"][category_letter] += 1

                if regression.has_regression:
                    self.stats["regressions_found"] += 1
                    print(f"  [X] REGRESSION detectee:")
                    for diff in regression.differences:
                        print(f"     - {diff}")
                else:
                    self.stats["no_regression"] += 1
                    print(f"  [OK] Pas de regression")

    def print_summary(self):
        """Affiche le résumé des résultats"""
        print("\n" + "=" * 70)
        print("RÉSUMÉ DES TESTS DE RÉGRESSION")
        print("=" * 70)
        print(f"\nTotal teste: {self.stats['total_tested']}")
        print(f"  [OK] Sans regression: {self.stats['no_regression']}")
        print(f"  [X] Avec regression: {self.stats['regressions_found']}")

        print(f"\nPar catégorie:")
        for cat, count in self.stats["categories"].items():
            if count > 0:
                print(f"  - Catégorie {cat}: {count} questions testées")

        if self.stats["regressions_found"] > 0:
            print(f"\n{'-' * 70}")
            print("DETAIL DES REGRESSIONS")
            print(f"{'-' * 70}")

            for result in self.regression_results:
                if result.has_regression:
                    print(f"\n{result.question_id} - {result.question}")
                    for diff in result.differences:
                        print(f"  - {diff}")

        # Code de sortie
        exit_code = 0 if self.stats["regressions_found"] == 0 else 1
        print(f"\n{'=' * 70}")
        if exit_code == 0:
            print("[OK] TOUS LES TESTS PASSENT - Aucune regression detectee")
        else:
            print(f"[X] {self.stats['regressions_found']} REGRESSION(S) DETECTEE(S)")
        print("=" * 70)

        return exit_code


def main():
    tester = RegressionTester()

    # Authentification
    if not tester.authenticate():
        print("❌ Échec authentification")
        sys.exit(1)

    # Tests de régression
    tester.run_regression_tests()

    # Résumé et code de sortie
    exit_code = tester.print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
