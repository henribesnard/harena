#!/usr/bin/env python3
"""
Test script focalisé sur l'extraction des queries générées
Génère des rapports par catégorie pour validation manuelle
Dupliqué depuis test_intent_entities_only.py
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
class QueryExtractionResult:
    """Résultat de test focalisé sur extraction de queries + cohérence intent/entités"""
    question_id: str
    category: str
    question: str

    # 1. Intentions détectées
    intent_detected: Optional[str]
    intent_confidence: float

    # 2. Entités détectées
    entities_raw: List[Dict[str, Any]]
    entities_structured: Dict[str, Any]
    entities_count: int
    entity_confidence: float

    # 3. Query construite
    query_found: bool
    query_data: Optional[Dict[str, Any]]
    query_size: int
    has_filters: bool
    has_aggregations: bool
    filter_count: int
    aggregation_count: int

    # 4. Analyse de cohérence intent->entités->query
    coherence_analysis: Optional[Dict[str, Any]]

    # Métriques
    latency_ms: float
    api_success: bool
    error_message: Optional[str]

class QueryExtractionTester:
    """Testeur focalisé sur extraction de queries"""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.user_id: Optional[int] = None

        # Configuration d'authentification
        self.username = "henri@example.com"
        self.password = "hounwanou"

        # Résultats par catégorie
        self.results_by_category: Dict[str, List[QueryExtractionResult]] = defaultdict(list)
        self.global_stats = {
            "total_questions": 0,
            "api_success": 0,
            "intent_detected": 0,
            "entities_detected": 0,
            "queries_found": 0,
            "queries_with_filters": 0,
            "queries_with_aggregations": 0,
            "coherence_high": 0,  # Bonne cohérence intent->entités->query
            "coherence_medium": 0,  # Cohérence partielle
            "coherence_low": 0,  # Incohérences détectées
            "categories_tested": 0
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

    def test_single_question(self, category: str, question_data: Dict[str, Any]) -> QueryExtractionResult:
        """Test une seule question et extrait la query générée"""
        question_id = question_data["id"]
        question = question_data["question"]

        # Préparation de la requête (payload complet comme test_intent_entities_only.py)
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
            # Appel API
            response = self.session.post(
                f"{self.base_url}/conversation/{self.user_id}",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                return QueryExtractionResult(
                    question_id=question_id,
                    category=category,
                    question=question,
                    intent_detected="HTTP_ERROR",
                    intent_confidence=0.0,
                    entities_raw=[],
                    entities_structured={},
                    entities_count=0,
                    entity_confidence=0.0,
                    query_found=False,
                    query_data=None,
                    query_size=0,
                    has_filters=False,
                    has_aggregations=False,
                    filter_count=0,
                    aggregation_count=0,
                    coherence_analysis=None,
                    latency_ms=latency_ms,
                    api_success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )

            data = response.json()

            # 1. EXTRACTION INTENTION (comme test_intent_entities_only.py)
            intent_detected = "UNKNOWN"
            intent_confidence = 0.0

            if "intent" in data:
                intent_data = data["intent"]
                if isinstance(intent_data, dict):
                    intent_detected = intent_data.get("type", "UNKNOWN")
                    intent_confidence = intent_data.get("confidence", 0.0)

                    if intent_detected == "UNKNOWN":
                        intent_detected = intent_data.get("intent_group", "UNKNOWN")
                        intent_confidence = intent_data.get("intent_confidence", intent_confidence)
                elif isinstance(intent_data, str):
                    intent_detected = intent_data
                    intent_confidence = 0.8

            # 2. EXTRACTION ENTITÉS (comme test_intent_entities_only.py)
            entities_raw = []
            entities_structured = {}
            entities_count = 0
            entity_confidence = 0.0

            if "intent" in data and "entities" in data["intent"]:
                entities_data = data["intent"]["entities"]
                if isinstance(entities_data, list):
                    entities_raw = entities_data
                    entities_count = len([e for e in entities_data if e.get("confidence", 0) > 0.6])

                    # Calcul de la confiance moyenne
                    confidences = [e.get("confidence", 0) for e in entities_data]
                    entity_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                    # Structure les entités
                    for entity in entities_data:
                        if entity.get("confidence", 0) > 0.6:
                            name = entity.get("name", "unknown")
                            value = entity.get("value", "")
                            entities_structured[name] = value

            # 3. EXTRACTION QUERY
            query = data.get("query")
            query_found = query is not None
            query_size = len(json.dumps(query)) if query else 0

            # Analyse de la query
            has_filters = False
            has_aggregations = False
            filter_count = 0
            aggregation_count = 0

            if query:
                has_filters = "filters" in query and bool(query["filters"])
                has_aggregations = "aggregations" in query and bool(query["aggregations"])
                filter_count = len(query.get("filters", {})) if isinstance(query.get("filters"), dict) else 0
                aggregation_count = len(query.get("aggregations", {})) if isinstance(query.get("aggregations"), dict) else 0

            # 4. ANALYSE DE COHÉRENCE intent->entités->query
            coherence_analysis = self._analyze_coherence(
                intent_detected,
                entities_structured,
                query
            )

            return QueryExtractionResult(
                question_id=question_id,
                category=category,
                question=question,
                intent_detected=intent_detected,
                intent_confidence=intent_confidence,
                entities_raw=entities_raw,
                entities_structured=entities_structured,
                entities_count=entities_count,
                entity_confidence=entity_confidence,
                query_found=query_found,
                query_data=query,
                query_size=query_size,
                has_filters=has_filters,
                has_aggregations=has_aggregations,
                filter_count=filter_count,
                aggregation_count=aggregation_count,
                coherence_analysis=coherence_analysis,
                latency_ms=latency_ms,
                api_success=True,
                error_message=None
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return QueryExtractionResult(
                question_id=question_id,
                category=category,
                question=question,
                intent_detected="EXCEPTION",
                intent_confidence=0.0,
                entities_raw=[],
                entities_structured={},
                entities_count=0,
                entity_confidence=0.0,
                query_found=False,
                query_data=None,
                query_size=0,
                has_filters=False,
                has_aggregations=False,
                filter_count=0,
                aggregation_count=0,
                coherence_analysis=None,
                latency_ms=latency_ms,
                api_success=False,
                error_message=str(e)
            )

    def _analyze_coherence(self, intent: str, entities: Dict[str, Any], query: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse la cohérence entre intention détectée, entités extraites et query construite"""

        analysis = {
            "coherence_score": "UNKNOWN",  # HIGH, MEDIUM, LOW, UNKNOWN
            "issues": [],
            "details": {}
        }

        if not query:
            analysis["coherence_score"] = "LOW"
            analysis["issues"].append("Aucune query générée malgré intent et entités détectées")
            return analysis

        # Extraction des filtres de la query
        query_filters = query.get("filters", {})

        # Vérification de la cohérence entités -> filtres query
        entities_translated = 0
        entities_missing = []

        # Mapping entités -> filtres ES
        entity_to_filter_mapping = {
            "merchant": "merchant_name",
            "merchant_name": "merchant_name",
            "amount": "amount_abs",
            "date": "date",
            "date_range": "date",
            "category": "category_name",
            "categories": "category_name",
            "transaction_type": "transaction_type",
            "operation_type": "operation_type",
            "description": "description"
        }

        for entity_name, entity_value in entities.items():
            # Ignorer les entités fusionnées dans amount_abs
            if entity_name in ["operator", "amount_min", "amount_max"]:
                # Vérifier si amount_abs existe dans la query (fusion réussie)
                if "amount_abs" in query_filters:
                    entities_translated += 1  # Comptée comme traduite (fusionnée)
                else:
                    entities_missing.append(entity_name)
                continue

            # transaction_type "all" signifie "pas de filtre" -> considéré comme traduit
            if entity_name == "transaction_type" and entity_value == "all":
                entities_translated += 1
                continue

            # Trouver le nom du filtre correspondant
            filter_name = entity_to_filter_mapping.get(entity_name, entity_name)

            if filter_name in query_filters:
                entities_translated += 1
            else:
                entities_missing.append(entity_name)

        # Calcul du score de cohérence
        total_entities = len(entities)

        if total_entities == 0:
            # Pas d'entités mais query générée = acceptable
            if intent in ["transaction_search", "financial_query"]:
                analysis["coherence_score"] = "MEDIUM"
                analysis["details"]["note"] = "Requête générique sans entités spécifiques"
            else:
                analysis["coherence_score"] = "HIGH"
        elif entities_translated == total_entities:
            # Toutes les entités traduites en filtres
            analysis["coherence_score"] = "HIGH"
            analysis["details"]["entities_translated"] = f"{entities_translated}/{total_entities}"
        elif entities_translated >= total_entities // 2:
            # Au moins la moitié des entités traduites
            analysis["coherence_score"] = "MEDIUM"
            analysis["details"]["entities_translated"] = f"{entities_translated}/{total_entities}"
            analysis["issues"].append(f"Entités manquantes dans query: {', '.join(entities_missing)}")
        else:
            # Moins de la moitié des entités traduites
            analysis["coherence_score"] = "LOW"
            analysis["details"]["entities_translated"] = f"{entities_translated}/{total_entities}"
            analysis["issues"].append(f"Mauvaise traduction entités->query. Manquantes: {', '.join(entities_missing)}")

        # Vérification de la cohérence intent -> query
        if intent == "transaction_search" or intent == "financial_query":
            if not query_filters and total_entities > 0:
                analysis["issues"].append("Intent transaction_search mais aucun filtre dans query")
                analysis["coherence_score"] = "LOW"

        return analysis

    def save_individual_result(self, result: QueryExtractionResult):
        """Sauvegarde immédiate d'un résultat individuel"""
        try:
            results_dir = Path("test_reports_query_extraction/individual_results")
            results_dir.mkdir(exist_ok=True, parents=True)

            # Nom de fichier avec ID et statut
            status = "SUCCESS" if result.query_found else "NO_QUERY" if result.api_success else "ERROR"
            filename = f"{result.question_id}_{status}.json"
            filepath = results_dir / filename

            # Données complètes du résultat (ENRICHI avec intent + entités + cohérence)
            result_dict = {
                "question_id": result.question_id,
                "category": result.category,
                "question": result.question,

                # 1. Intentions
                "intent_detected": result.intent_detected,
                "intent_confidence": result.intent_confidence,

                # 2. Entités
                "entities_structured": result.entities_structured,
                "entities_count": result.entities_count,
                "entity_confidence": result.entity_confidence,

                # 3. Query
                "query_found": result.query_found,
                "query_data": result.query_data,
                "query_size": result.query_size,
                "has_filters": result.has_filters,
                "has_aggregations": result.has_aggregations,
                "filter_count": result.filter_count,
                "aggregation_count": result.aggregation_count,

                # 4. Analyse de cohérence
                "coherence_analysis": result.coherence_analysis,

                # Métriques
                "latency_ms": result.latency_ms,
                "api_success": result.api_success,
                "error_message": result.error_message,
                "timestamp": datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"      Erreur sauvegarde individuelle {result.question_id}: {e}")

    def test_category_questions(self, category_name: str, questions: List[Dict]) -> List[QueryExtractionResult]:
        """Test toutes les questions d'une catégorie"""

        print(f"\nCategorie: {category_name}")
        print(f"   Questions a tester: {len(questions)}")

        results = []

        for idx, question_data in enumerate(questions, 1):
            question_id = question_data["id"]
            question_text = question_data["question"]

            print(f"   [{idx}/{len(questions)}] {question_id}: {question_text}")

            # Test de la question
            result = self.test_single_question(category_name, question_data)

            # Mise à jour des statistiques globales
            self.global_stats["total_questions"] += 1

            if result.api_success:
                self.global_stats["api_success"] += 1

                if result.intent_detected and result.intent_detected not in ["UNKNOWN", "HTTP_ERROR", "EXCEPTION"]:
                    self.global_stats["intent_detected"] += 1

                if result.entities_count > 0:
                    self.global_stats["entities_detected"] += 1

                if result.query_found:
                    self.global_stats["queries_found"] += 1
                    if result.has_filters:
                        self.global_stats["queries_with_filters"] += 1
                    if result.has_aggregations:
                        self.global_stats["queries_with_aggregations"] += 1

                # Statistiques de cohérence
                if result.coherence_analysis:
                    coherence_score = result.coherence_analysis.get("coherence_score", "UNKNOWN")
                    if coherence_score == "HIGH":
                        self.global_stats["coherence_high"] += 1
                    elif coherence_score == "MEDIUM":
                        self.global_stats["coherence_medium"] += 1
                    elif coherence_score == "LOW":
                        self.global_stats["coherence_low"] += 1

            # Affichage du résultat ENRICHI
            if result.api_success:
                coherence_score = result.coherence_analysis.get("coherence_score", "UNKNOWN") if result.coherence_analysis else "UNKNOWN"
                coherence_emoji = {"HIGH": "+", "MEDIUM": "~", "LOW": "-", "UNKNOWN": "?"}[coherence_score]

                if result.query_found:
                    print(f"      {coherence_emoji} SUCCESS ({result.latency_ms:.0f}ms) - Intent: {result.intent_detected}, Entities: {result.entities_count}, Query: {result.filter_count} filters / {result.aggregation_count} agg, Coherence: {coherence_score}")
                else:
                    print(f"      - API OK ({result.latency_ms:.0f}ms) - Intent: {result.intent_detected}, Entities: {result.entities_count}, mais NO QUERY")
            else:
                print(f"      - FAILED: {result.error_message}")

            results.append(result)

            # SAUVEGARDE INDIVIDUELLE IMMÉDIATE après chaque question
            self.save_individual_result(result)

            # SAUVEGARDE DES RAPPORTS tous les 5 questions ou en cas d'erreur
            reports_dir = Path("test_reports_query_extraction")
            if idx % 5 == 0 or not result.api_success:
                try:
                    # Mise à jour temporaire des résultats de la catégorie
                    self.results_by_category[category_name] = results
                    self.generate_category_report(reports_dir, category_name, results)
                    self.generate_global_report(reports_dir)
                    print(f"      -> Rapports mis a jour ({idx}/{len(questions)} questions de {category_name})")
                except Exception as e:
                    print(f"      WARN: Erreur sauvegarde rapports: {e}")

            time.sleep(0.5)  # Pause entre requêtes

        return results

    def run_all_tests(self, categories_to_test: Optional[List[str]] = None):
        """Lance tous les tests d'extraction de queries

        Args:
            categories_to_test: Liste des catégories à tester (ex: ["A_base", "B_avancees"])
                               Si None, teste toutes les catégories
        """

        print("Test extraction queries completes - Questions categorisees")
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
        reports_dir = Path("test_reports_query_extraction")
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

    def generate_category_report(self, reports_dir: Path, category_key: str, results: List[QueryExtractionResult]):
        """Génère un rapport détaillé par catégorie - structure identique à test_intent_entities_only.py"""
        category_dir = reports_dir / f"category_{category_key}"
        category_dir.mkdir(exist_ok=True)

        # Analyse par catégorie
        total = len(results)
        api_success = sum(1 for r in results if r.api_success)
        queries_found = sum(1 for r in results if r.query_found)

        report = {
            "category": category_key,
            "timestamp": datetime.now().isoformat(),
            "total_questions": total,
            "api_successful": api_success,
            "queries_found": queries_found,
            "success_rate": (queries_found / total * 100) if total > 0 else 0,
            "detailed_results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "intent_detected": r.intent_detected,
                    "intent_confidence": r.intent_confidence,
                    "entities_structured": r.entities_structured,
                    "entities_count": r.entities_count,
                    "entity_confidence": r.entity_confidence,
                    "query_found": r.query_found,
                    "query_data": r.query_data,
                    "query_size": r.query_size,
                    "has_filters": r.has_filters,
                    "has_aggregations": r.has_aggregations,
                    "filter_count": r.filter_count,
                    "aggregation_count": r.aggregation_count,
                    "coherence_analysis": r.coherence_analysis,
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
            "focus": "intent_entities_query_coherence",
            "total_questions": total,
            "categories_tested": self.global_stats["categories_tested"],

            # Statistiques globales
            "api_success_rate": (self.global_stats["api_success"] / total * 100) if total > 0 else 0,
            "intent_detection_rate": (self.global_stats["intent_detected"] / total * 100) if total > 0 else 0,
            "entities_detection_rate": (self.global_stats["entities_detected"] / total * 100) if total > 0 else 0,
            "query_extraction_rate": (self.global_stats["queries_found"] / self.global_stats["api_success"] * 100) if self.global_stats["api_success"] > 0 else 0,
            "overall_success_rate": (self.global_stats["queries_found"] / total * 100) if total > 0 else 0,

            # Détails queries
            "queries_with_filters": self.global_stats["queries_with_filters"],
            "queries_with_aggregations": self.global_stats["queries_with_aggregations"],

            # Statistiques de cohérence
            "coherence_stats": {
                "high": self.global_stats["coherence_high"],
                "medium": self.global_stats["coherence_medium"],
                "low": self.global_stats["coherence_low"],
                "high_rate": (self.global_stats["coherence_high"] / total * 100) if total > 0 else 0
            },

            "summary_by_category": {
                category: {
                    "total_questions": len(results),
                    "api_successful": sum(1 for r in results if r.api_success),
                    "intents_detected": sum(1 for r in results if r.intent_detected and r.intent_detected not in ["UNKNOWN", "HTTP_ERROR", "EXCEPTION"]),
                    "entities_detected": sum(1 for r in results if r.entities_count > 0),
                    "queries_found": sum(1 for r in results if r.query_found),
                    "coherence_high": sum(1 for r in results if r.coherence_analysis and r.coherence_analysis.get("coherence_score") == "HIGH"),
                    "success_rate": (sum(1 for r in results if r.query_found) / len(results) * 100) if results else 0
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
        print("RAPPORT FINAL - EXTRACTION DE QUERIES")
        print("=" * 65)

        # Statistiques globales
        total = self.global_stats["total_questions"]
        api_success = self.global_stats["api_success"]
        queries_found = self.global_stats["queries_found"]

        api_success_rate = (api_success / total * 100) if total > 0 else 0
        query_extraction_rate = (queries_found / api_success * 100) if api_success > 0 else 0
        overall_success_rate = (queries_found / total * 100) if total > 0 else 0

        print(f"Questions totales: {total}")
        print(f"Reponses API reussies: {api_success} ({api_success_rate:.1f}%)")
        print(f"\n--- DETECTION ---")
        print(f"Intentions detectees: {self.global_stats['intent_detected']} ({self.global_stats['intent_detected']/total*100:.1f}%)")
        print(f"Entites detectees: {self.global_stats['entities_detected']} ({self.global_stats['entities_detected']/total*100:.1f}%)")
        print(f"\n--- GENERATION QUERIES ---")
        print(f"Queries extraites: {queries_found} ({query_extraction_rate:.1f}% des reponses API)")
        print(f"Queries avec filtres: {self.global_stats['queries_with_filters']}")
        print(f"Queries avec aggregations: {self.global_stats['queries_with_aggregations']}")
        print(f"\n--- COHERENCE INTENT->ENTITIES->QUERY ---")
        print(f"Coherence HIGH: {self.global_stats['coherence_high']} ({self.global_stats['coherence_high']/total*100:.1f}%)")
        print(f"Coherence MEDIUM: {self.global_stats['coherence_medium']} ({self.global_stats['coherence_medium']/total*100:.1f}%)")
        print(f"Coherence LOW: {self.global_stats['coherence_low']} ({self.global_stats['coherence_low']/total*100:.1f}%)")
        print(f"\nTaux de succes global: {overall_success_rate:.1f}%")
        print(f"Categories testees: {self.global_stats['categories_tested']}")

        # Résumé par catégorie
        print(f"\nResume par categorie:")
        for category, results in self.results_by_category.items():
            total_cat = len(results)
            queries_found_cat = sum(1 for r in results if r.query_found)
            success_rate_cat = (queries_found_cat / total_cat * 100) if total_cat > 0 else 0
            print(f"   {category}: {queries_found_cat}/{total_cat} queries ({success_rate_cat:.1f}%)")

        print(f"\nRapports sauvegardes dans: test_reports_query_extraction/")
        print(f"   - global_summary.json (rapport de synthese)")
        print(f"   - category_*/{{category}}_results.json (rapports detailles par categorie)")

        # Conclusion
        if overall_success_rate >= 95:
            print(f"\nEXCELLENT! L'extraction de queries fonctionne parfaitement!")
        elif overall_success_rate >= 85:
            print(f"\nBON! Le systeme d'extraction de queries fonctionne bien.")
        elif overall_success_rate >= 70:
            print(f"\nCORRECT: Le systeme fonctionne mais peut etre ameliore.")
        else:
            print(f"\nA AMELIORER: Taux de succes insuffisant.")

        # Affichage de quelques exemples de queries
        print(f"\n=== EXEMPLES DE QUERIES GENEREES ===")
        example_count = 0
        for category, results in self.results_by_category.items():
            for result in results:
                if result.query_found and example_count < 3:
                    print(f"\n{result.question_id}: {result.question}")
                    print(f"Query ({result.query_size} chars):")
                    print(json.dumps(result.query_data, indent=2)[:300] + "..." if result.query_size > 300 else json.dumps(result.query_data, indent=2))
                    example_count += 1
            if example_count >= 3:
                break

if __name__ == "__main__":
    import sys

    tester = QueryExtractionTester()

    # Option pour tester seulement certaines catégories
    # Usage: python test_query_extraction_complete.py A_base B_avancees
    # Ou: python test_query_extraction_complete.py A_base,B_avancees
    if len(sys.argv) > 1:
        # Support de deux formats:
        # 1. python script.py A_base B_avancees C_temporelles
        # 2. python script.py A_base,B_avancees,C_temporelles
        if "," in sys.argv[1]:
            categories_to_test = sys.argv[1].split(",")
        else:
            categories_to_test = sys.argv[1:]

        print(f"Categories demandees: {categories_to_test}")
        tester.run_all_tests(categories_to_test)
    else:
        print("Test de toutes les categories...")
        tester.run_all_tests()