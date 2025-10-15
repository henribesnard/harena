#!/usr/bin/env python3
"""
Test script pour récupérer les réponses finales textuelles du système
Génère des rapports par catégorie pour validation manuelle
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class FinalResponseResult:
    """Résultat de test focalisé sur la réponse finale textuelle"""
    question_id: str
    category: str
    question: str

    # Réponse finale
    response_text: str
    response_length: int
    has_response: bool

    # Métadonnées de la réponse
    intent_detected: Optional[str]
    transactions_found: int
    insights_count: int
    visualizations_count: int

    # Métriques
    latency_ms: float
    api_success: bool
    error_message: Optional[str]

class FinalResponseTester:
    """Testeur focalisé sur les réponses finales textuelles"""

    def __init__(self, user_service_url: str = "http://localhost:8000/api/v1",
                 conversation_service_url: str = "http://localhost:8000/api/v1"):
        self.user_service_url = user_service_url
        self.conversation_service_url = conversation_service_url
        self.session = requests.Session()
        self.user_id: Optional[int] = None

        # Configuration d'authentification
        self.username = "henri@example.com"
        self.password = "Henri123456"

        # Résultats par catégorie
        self.results_by_category: Dict[str, List[FinalResponseResult]] = defaultdict(list)
        self.global_stats = {
            "total_questions": 0,
            "api_success": 0,
            "responses_received": 0,
            "empty_responses": 0,
            "with_transactions": 0,
            "with_insights": 0,
            "with_visualizations": 0,
            "categories_tested": 0,
            "total_response_length": 0
        }

    def authenticate(self) -> bool:
        """Authentification OAuth"""
        try:
            data = f"username={self.username}&password={self.password}"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            # URL correcte: /users/auth/login sur le user_service (port 8000)
            resp = self.session.post(
                f"{self.user_service_url}/users/auth/login",
                data=data,
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()

            token = resp.json()["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {token}"})

            # Récupération de l'ID utilisateur
            user_resp = self.session.get(f"{self.user_service_url}/users/me", timeout=30)
            user_resp.raise_for_status()
            self.user_id = user_resp.json().get("id")

            print(f"Authentification reussie - User ID: {self.user_id}")
            return True

        except Exception as e:
            print(f"Erreur d'authentification: {e}")
            return False

    def load_test_suite(self) -> Dict[str, Any]:
        """Charge la suite de tests complète"""
        try:
            with open("test_suite_complete.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur de chargement de la suite de tests: {e}")
            return {}

    def test_single_question(self, category: str, question_data: Dict[str, Any]) -> FinalResponseResult:
        """Test une seule question et extrait la réponse finale"""
        question_id = question_data["id"]
        question = question_data["question"]

        # Préparation de la requête
        payload = {
            "client_info": {
                "platform": "web",
                "version": "1.0.0"
            },
            "message": question,
            "message_type": "text",
            "priority": "normal"
        }
        start_time = time.perf_counter()

        try:
            # Appel API sur le conversation_service (port 8001)
            response = self.session.post(
                f"{self.conversation_service_url}/conversation/{self.user_id}",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                return FinalResponseResult(
                    question_id=question_id,
                    category=category,
                    question=question,
                    response_text=f"[ERROR HTTP {response.status_code}]",
                    response_length=0,
                    has_response=False,
                    intent_detected=None,
                    transactions_found=0,
                    insights_count=0,
                    visualizations_count=0,
                    latency_ms=latency_ms,
                    api_success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )

            data = response.json()

            # EXTRACTION DE LA RÉPONSE FINALE
            # Gestion de la nouvelle structure de réponse avec objet { message, structured_data, visualizations }
            response_text = data.get("response", "")

            # Si response est un objet avec un champ "message", extraire ce champ
            if isinstance(response_text, dict) and "message" in response_text:
                actual_message = response_text.get("message", "")
                response_text = actual_message

            # Fallbacks pour d'autres formats
            if not response_text:
                response_text = data.get("response_text", "")
            if not response_text:
                response_text = data.get("message", "")
            if not response_text:
                response_text = data.get("text", "")

            # S'assurer que response_text est une chaîne
            if not isinstance(response_text, str):
                response_text = ""

            response_length = len(response_text)
            has_response = response_length > 0

            # Extraction des métadonnées
            intent_detected = None
            if "intent" in data:
                intent_data = data["intent"]
                if isinstance(intent_data, dict):
                    intent_detected = intent_data.get("type") or intent_data.get("intent_group", "UNKNOWN")
                elif isinstance(intent_data, str):
                    intent_detected = intent_data

            # Extraction des informations complémentaires
            transactions_found = 0
            if "transactions" in data:
                if isinstance(data["transactions"], list):
                    transactions_found = len(data["transactions"])
                elif isinstance(data["transactions"], int):
                    transactions_found = data["transactions"]

            # Nombre de résultats de recherche
            if "search_results" in data:
                if isinstance(data["search_results"], list):
                    transactions_found = max(transactions_found, len(data["search_results"]))

            # Total hits
            if "total_hits" in data:
                transactions_found = data["total_hits"]

            # Extraction de structured_data et visualizations depuis la nouvelle structure
            response_obj = data.get("response", {})
            insights_count = 0
            visualizations_count = 0

            if isinstance(response_obj, dict):
                # Extraire depuis la nouvelle structure
                insights_count = len(response_obj.get("structured_data", []))
                visualizations_count = len(response_obj.get("visualizations", []))
            else:
                # Ancienne structure (fallback)
                insights_count = len(data.get("insights", []))
                visualizations_count = len(data.get("data_visualizations", []))

            return FinalResponseResult(
                question_id=question_id,
                category=category,
                question=question,
                response_text=response_text,
                response_length=response_length,
                has_response=has_response,
                intent_detected=intent_detected,
                transactions_found=transactions_found,
                insights_count=insights_count,
                visualizations_count=visualizations_count,
                latency_ms=latency_ms,
                api_success=True,
                error_message=None
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return FinalResponseResult(
                question_id=question_id,
                category=category,
                question=question,
                response_text=f"[EXCEPTION: {str(e)}]",
                response_length=0,
                has_response=False,
                intent_detected=None,
                transactions_found=0,
                insights_count=0,
                visualizations_count=0,
                latency_ms=latency_ms,
                api_success=False,
                error_message=str(e)
            )

    def clean_response_data(self, data: Any, max_array_length: int = 10) -> Any:
        """Nettoie les données de réponse en limitant les grands tableaux et en supprimant les nulls inutiles"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # D'abord, vérifier si c'est un tableau de nulls avant nettoyage récursif
                if isinstance(value, list):
                    non_null_items = [item for item in value if item is not None]
                    if not non_null_items:
                        # Tableau vide ou que des nulls - indiquer combien il y en avait
                        if len(value) > 0:
                            cleaned[key] = f"[{len(value)} null values omitted]"
                        continue
                    # Si on a des items non-null, continuer avec le nettoyage récursif
                    cleaned_value = self.clean_response_data(value, max_array_length)
                else:
                    # Nettoyer récursivement pour les non-listes
                    cleaned_value = self.clean_response_data(value, max_array_length)

                cleaned[key] = cleaned_value
            return cleaned
        elif isinstance(data, list):
            # Limiter la longueur des tableaux longs
            if len(data) > max_array_length:
                return [self.clean_response_data(item, max_array_length) for item in data[:max_array_length]] + [f"... {len(data) - max_array_length} more items"]
            return [self.clean_response_data(item, max_array_length) for item in data]
        else:
            return data

    def save_individual_result(self, result: FinalResponseResult):
        """Sauvegarde immédiate d'un résultat individuel"""
        try:
            results_dir = Path("test_reports_final_responses/individual_results")
            results_dir.mkdir(exist_ok=True, parents=True)

            # Nom de fichier avec ID et statut
            status = "SUCCESS" if result.has_response else "EMPTY" if result.api_success else "ERROR"
            filename = f"{result.question_id}_{status}.json"
            filepath = results_dir / filename

            # Nettoyer response_text si c'est un objet complexe
            cleaned_response = result.response_text
            if isinstance(result.response_text, (dict, list)):
                cleaned_response = self.clean_response_data(result.response_text, max_array_length=5)

            # Données complètes du résultat
            result_dict = {
                "question_id": result.question_id,
                "category": result.category,
                "question": result.question,
                "response_text": cleaned_response,
                "response_length": result.response_length,
                "has_response": result.has_response,
                "intent_detected": result.intent_detected,
                "transactions_found": result.transactions_found,
                "insights_count": result.insights_count,
                "visualizations_count": result.visualizations_count,
                "latency_ms": result.latency_ms,
                "api_success": result.api_success,
                "error_message": result.error_message,
                "timestamp": datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"      Erreur sauvegarde individuelle {result.question_id}: {e}")

    def test_category_questions(self, category_name: str, questions: List[Dict]) -> List[FinalResponseResult]:
        """Test toutes les questions d'une catégorie"""

        print(f"\nCategorie: {category_name} - {len(questions)} questions")

        results = []

        for idx, question_data in enumerate(questions, 1):
            question_id = question_data["id"]
            question_text = question_data["question"]

            # Test de la question
            result = self.test_single_question(category_name, question_data)

            # Mise à jour des statistiques globales
            self.global_stats["total_questions"] += 1

            if result.api_success:
                self.global_stats["api_success"] += 1

                if result.has_response:
                    self.global_stats["responses_received"] += 1
                    self.global_stats["total_response_length"] += result.response_length
                else:
                    self.global_stats["empty_responses"] += 1

                if result.transactions_found > 0:
                    self.global_stats["with_transactions"] += 1

                if result.insights_count > 0:
                    self.global_stats["with_insights"] += 1

                if result.visualizations_count > 0:
                    self.global_stats["with_visualizations"] += 1

            # Affichage compact: seulement le statut
            status_symbol = "✓" if result.has_response else "○" if result.api_success else "✗"
            print(f"   [{idx}/{len(questions)}] {question_id}: {status_symbol}", end="")
            if not result.api_success:
                print(f" ERREUR", end="")
            print()  # Nouvelle ligne

            results.append(result)

            # SAUVEGARDE INDIVIDUELLE IMMÉDIATE après chaque question
            self.save_individual_result(result)

            # SAUVEGARDE DES RAPPORTS tous les 5 questions ou en cas d'erreur
            reports_dir = Path("test_reports_final_responses")
            if idx % 5 == 0 or not result.api_success:
                try:
                    # Mise à jour temporaire des résultats de la catégorie
                    self.results_by_category[category_name] = results
                    self.generate_category_report(reports_dir, category_name, results)
                    self.generate_global_report(reports_dir)
                    print(f"   -> Rapports sauvegardés ({idx}/{len(questions)})")
                except Exception as e:
                    print(f"   WARN: Erreur sauvegarde rapports: {e}")

            time.sleep(0.5)  # Pause entre requêtes

        return results

    def run_all_tests(self, categories_to_test: Optional[List[str]] = None):
        """Lance tous les tests de réponses finales

        Args:
            categories_to_test: Liste des catégories à tester (ex: ["A_montant", "B_marchands"])
                               Si None, teste toutes les catégories
        """

        print("Test des reponses finales - Questions categorisees")
        print("=" * 65)

        # Authentification
        if not self.authenticate():
            print("ERREUR: Impossible de s'authentifier")
            return

        # Charger la suite de tests
        test_suite = self.load_test_suite()
        if not test_suite:
            print("ERREUR: Impossible de charger la suite de tests")
            return

        print(f"Suite de tests chargee: {test_suite['metadata']['total_questions']} questions")

        # Créer le dossier de rapports dès le début
        reports_dir = Path("test_reports_final_responses")
        reports_dir.mkdir(exist_ok=True)
        print(f"Dossier de rapports cree: {reports_dir}")

        # Tester chaque catégorie (filtrer si nécessaire)
        categories = test_suite.get("categories", {})

        if categories_to_test:
            # Filtrer les catégories demandées
            categories = {k: v for k, v in categories.items() if k in categories_to_test}
            print(f"Filtrage: {len(categories)} categorie(s) selectionnee(s): {', '.join(categories.keys())}")
        else:
            print(f"Test de toutes les categories: {len(categories)} categorie(s)")

        for category_name, category_data in categories.items():
            try:
                print(f"\n{'='*50}")
                print(f"TRAITEMENT CATEGORIE: {category_name}")
                print(f"{'='*50}")

                questions = category_data.get("questions", [])

                if questions:
                    category_results = self.test_category_questions(category_name, questions)
                    self.results_by_category[category_name] = category_results
                    self.global_stats["categories_tested"] += 1

                    # SAUVEGARDE IMMÉDIATE après chaque catégorie
                    self.generate_category_report(reports_dir, category_name, category_results)
                    self.generate_global_report(reports_dir)  # Mise à jour du rapport global

                    print(f"OK Rapports sauvegardes pour {category_name}")

            except Exception as e:
                print(f"Erreur dans la catégorie {category_name}: {e}")
                # Même en cas d'erreur, sauvegarder ce qu'on a
                try:
                    self.generate_global_report(reports_dir)
                    print(f"OK Rapport global sauvegarde malgre l'erreur")
                except:
                    pass

        # Génération finale des rapports (pour s'assurer que tout est à jour)
        print(f"\n{'='*50}")
        print("GÉNÉRATION FINALE DES RAPPORTS")
        print(f"{'='*50}")
        self.generate_final_report_summary(reports_dir)

    def generate_category_report(self, reports_dir: Path, category_key: str, results: List[FinalResponseResult]):
        """Génère un rapport détaillé par catégorie"""
        category_dir = reports_dir / f"category_{category_key}"
        category_dir.mkdir(exist_ok=True)

        # Analyse par catégorie
        total = len(results)
        api_success = sum(1 for r in results if r.api_success)
        responses_received = sum(1 for r in results if r.has_response)

        report = {
            "category": category_key,
            "timestamp": datetime.now().isoformat(),
            "total_questions": total,
            "api_successful": api_success,
            "responses_received": responses_received,
            "success_rate": (responses_received / total * 100) if total > 0 else 0,
            "avg_response_length": sum(r.response_length for r in results) / len(results) if results else 0,
            "detailed_results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "response_text": self.clean_response_data(r.response_text, max_array_length=5) if isinstance(r.response_text, (dict, list)) else r.response_text,
                    "response_length": r.response_length,
                    "has_response": r.has_response,
                    "intent_detected": r.intent_detected,
                    "transactions_found": r.transactions_found,
                    "insights_count": r.insights_count,
                    "visualizations_count": r.visualizations_count,
                    "latency_ms": r.latency_ms,
                    "api_success": r.api_success,
                    "error_message": r.error_message
                }
                for r in results
            ]
        }

        # Sauvegarde rapport par catégorie
        with open(category_dir / f"{category_key}_results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def generate_global_report(self, reports_dir: Path):
        """Génère le rapport de synthèse global (mis à jour après chaque catégorie)"""
        total = self.global_stats["total_questions"]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "focus": "final_text_responses",
            "total_questions": total,
            "categories_tested": self.global_stats["categories_tested"],

            # Statistiques globales
            "api_success_rate": (self.global_stats["api_success"] / total * 100) if total > 0 else 0,
            "response_rate": (self.global_stats["responses_received"] / total * 100) if total > 0 else 0,
            "empty_response_rate": (self.global_stats["empty_responses"] / total * 100) if total > 0 else 0,

            # Contenu des réponses
            "with_transactions": self.global_stats["with_transactions"],
            "with_insights": self.global_stats["with_insights"],
            "with_visualizations": self.global_stats["with_visualizations"],
            "avg_response_length": (self.global_stats["total_response_length"] / self.global_stats["responses_received"]) if self.global_stats["responses_received"] > 0 else 0,

            "summary_by_category": {
                category: {
                    "total_questions": len(results),
                    "api_successful": sum(1 for r in results if r.api_success),
                    "responses_received": sum(1 for r in results if r.has_response),
                    "with_transactions": sum(1 for r in results if r.transactions_found > 0),
                    "with_insights": sum(1 for r in results if r.insights_count > 0),
                    "with_visualizations": sum(1 for r in results if r.visualizations_count > 0),
                    "avg_response_length": sum(r.response_length for r in results) / len(results) if results else 0,
                    "success_rate": (sum(1 for r in results if r.has_response) / len(results) * 100) if results else 0
                }
                for category, results in self.results_by_category.items()
            }
        }

        # Sauvegarde rapport global
        with open(reports_dir / "global_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def generate_final_report_summary(self, reports_dir: Path):
        """Affichage du rapport final de synthèse"""

        print("\n" + "=" * 65)
        print("RAPPORT FINAL - RÉPONSES FINALES")
        print("=" * 65)

        # Statistiques globales
        total = self.global_stats["total_questions"]
        api_success = self.global_stats["api_success"]
        responses_received = self.global_stats["responses_received"]

        api_success_rate = (api_success / total * 100) if total > 0 else 0
        response_rate = (responses_received / api_success * 100) if api_success > 0 else 0
        overall_success_rate = (responses_received / total * 100) if total > 0 else 0

        print(f"Questions totales: {total}")
        print(f"Reponses API reussies: {api_success} ({api_success_rate:.1f}%)")
        print(f"Reponses textuelles recues: {responses_received} ({response_rate:.1f}% des reponses API)")
        print(f"Reponses vides: {self.global_stats['empty_responses']}")

        print(f"\n--- CONTENU DES RÉPONSES ---")
        print(f"Avec transactions: {self.global_stats['with_transactions']} ({self.global_stats['with_transactions']/total*100:.1f}%)")
        print(f"Avec insights: {self.global_stats['with_insights']} ({self.global_stats['with_insights']/total*100:.1f}%)")
        print(f"Avec visualisations: {self.global_stats['with_visualizations']} ({self.global_stats['with_visualizations']/total*100:.1f}%)")

        if self.global_stats["responses_received"] > 0:
            avg_length = self.global_stats["total_response_length"] / self.global_stats["responses_received"]
            print(f"Longueur moyenne des reponses: {avg_length:.0f} caracteres")

        print(f"\nTaux de succes global: {overall_success_rate:.1f}%")
        print(f"Categories testees: {self.global_stats['categories_tested']}")

        # Résumé par catégorie
        print(f"\nResume par categorie:")
        for category, results in self.results_by_category.items():
            total_cat = len(results)
            responses_cat = sum(1 for r in results if r.has_response)
            success_rate_cat = (responses_cat / total_cat * 100) if total_cat > 0 else 0
            avg_length_cat = sum(r.response_length for r in results) / len(results) if results else 0
            print(f"   {category}: {responses_cat}/{total_cat} reponses ({success_rate_cat:.1f}%, moy: {avg_length_cat:.0f} chars)")

        print(f"\nRapports sauvegardes dans: test_reports_final_responses/")
        print(f"   - global_summary.json (rapport de synthese)")
        print(f"   - category_*/{{category}}_results.json (rapports detailles par categorie)")
        print(f"   - individual_results/*.json (resultats individuels)")

        # Affichage de quelques exemples de réponses
        print(f"\n=== EXEMPLES DE RÉPONSES ===")
        example_count = 0
        for category, results in self.results_by_category.items():
            for result in results:
                if result.has_response and example_count < 3:
                    print(f"\n{result.question_id}: {result.question}")
                    print(f"Reponse ({result.response_length} chars):")

                    # S'assurer que response_text est une chaîne
                    response_str = result.response_text if isinstance(result.response_text, str) else str(result.response_text)
                    response_preview = response_str[:200]
                    if len(response_str) > 200:
                        response_preview += "..."
                    print(f"\"{response_preview}\"")
                    example_count += 1
            if example_count >= 3:
                break

if __name__ == "__main__":
    import sys

    tester = FinalResponseTester()

    # Option pour tester une seule question par ID ou des catégories
    # Usage:
    #   python test_final_responses.py A001           # Une seule question
    #   python test_final_responses.py A_montant      # Une catégorie
    #   python test_final_responses.py A_montant B_marchands  # Plusieurs catégories
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Si l'argument ressemble à un ID de question (ex: A001, B016)
        if len(arg) <= 4 and arg[0].isalpha() and (arg[1:].isdigit() or arg[2:].isdigit()):
            print(f"Test de la question unique: {arg}")
            if not tester.authenticate():
                print("Echec d'authentification")
                sys.exit(1)

            # Charger toutes les questions pour trouver celle demandée
            questions_file = Path("test_suite_complete.json")
            with open(questions_file, 'r', encoding='utf-8') as f:
                test_suite = json.load(f)

            all_questions = test_suite.get("categories", {})

            # Trouver la question
            question_found = None
            category_found = None
            for category_key, category_data in all_questions.items():
                questions = category_data.get("questions", [])
                for q in questions:
                    if q.get("id") == arg:
                        question_found = q
                        category_found = category_key
                        break
                if question_found:
                    break

            if not question_found:
                print(f"Question {arg} non trouvée")
                sys.exit(1)

            print(f"Question trouvée dans la catégorie {category_found}")
            print(f"Question: {question_found['question']}")

            # Tester cette question
            result = tester.test_single_question(category_found, question_found)

            # Afficher le résultat
            print(f"\n{'='*60}")
            print(f"RÉSULTAT POUR {arg}")
            print(f"{'='*60}")
            print(f"Intent: {result.intent_detected}")
            print(f"Transactions: {result.transactions_found}")
            print(f"Insights: {result.insights_count}")
            print(f"Visualisations: {result.visualizations_count}")
            print(f"\nRéponse ({result.response_length} caractères):")
            print("=" * 60)
            print(result.response_text)
            print("=" * 60)

        # Sinon, c'est une liste de catégories
        else:
            if "," in arg:
                categories_to_test = arg.split(",")
            else:
                categories_to_test = sys.argv[1:]

            print(f"Categories demandees: {categories_to_test}")
            tester.run_all_tests(categories_to_test)
    else:
        print("Test de toutes les categories...")
        tester.run_all_tests()
