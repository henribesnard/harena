#!/usr/bin/env python3
"""
Test complet pour Harena : analyse des intentions sur 50+ questions - VERSION CORRIGÉE
Génère un rapport markdown détaillé des performances
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TestResult:
    """Structure pour stocker les résultats de test avec intent + entités"""
    question: str
    intent_type: str
    confidence: float
    category: str
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    performance_grade: Optional[str] = None
    efficiency_score: Optional[float] = None
    
    # Nouvelles données entités et timing
    entities_extracted: Optional[Dict[str, Any]] = None
    entities_count: int = 0
    entity_confidence: float = 0.0
    intent_processing_time_ms: int = 0
    entity_processing_time_ms: int = 0
    
    # Métriques détaillées
    amounts_found: int = 0
    merchants_found: int = 0
    dates_found: int = 0
    categories_found: int = 0
    operation_types_found: int = 0
    transaction_types_found: int = 0
    
    # Entités détaillées pour le rapport
    amounts_summary: str = ""
    merchants_summary: str = ""
    dates_summary: str = ""
    categories_summary: str = ""
    operation_types_summary: str = ""
    transaction_types_summary: str = ""


class HarenaTestSuite:
    """Suite de tests pour l'API Harena"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.user_id: Optional[int] = None
        self.results: List[TestResult] = []
        
    def authenticate(self, username: str, password: str) -> bool:
        """Authentifie l'utilisateur et configure la session"""
        try:
            data = f"username={username}&password={password}"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            resp = self.session.post(
                f"{self.base_url}/users/auth/login", 
                data=data, 
                headers=headers,
                timeout=10
            )
            resp.raise_for_status()
            
            token = resp.json()["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            
            # Récupération de l'ID utilisateur
            user_resp = self.session.get(f"{self.base_url}/users/me", timeout=10)
            user_resp.raise_for_status()
            self.user_id = user_resp.json().get("id")
            
            print(f"[OK] Authentification réussie - User ID: {self.user_id}")
            return True
            
        except Exception as e:
            print(f"[ERREUR] Erreur d'authentification: {e}")
            return False
    
    def run_single_test(self, question: str) -> TestResult:
        """Exécute un test sur une question donnée"""
        if not self.user_id:
            return TestResult(
                question=question,
                intent_type="ERROR",
                confidence=0.0,
                category="ERROR",
                latency_ms=0.0,
                success=False,
                error_message="Utilisateur non authentifié",
                entities_count=0,
                entity_confidence=0.0
            )
        
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
            response = self.session.post(
                f"{self.base_url}/conversation/{self.user_id}",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if response.status_code != 200:
                return TestResult(
                    question=question,
                    intent_type="HTTP_ERROR",
                    confidence=0.0,
                    category="ERROR",
                    latency_ms=latency_ms,
                    success=False,
                    error_message=f"HTTP {response.status_code}",
                    entities_count=0,
                    entity_confidence=0.0
                )
            
            data = response.json()
            intent = data.get("intent", {})
            agent_metrics = data.get("agent_metrics", {})
            entities = data.get("entities", {})
            
            # Extraction données entités si disponibles
            entities_extracted = None
            entities_count = 0
            entity_confidence = 0.0
            intent_processing_time_ms = intent.get("processing_time_ms", 0)
            entity_processing_time_ms = 0
            amounts_found = 0
            merchants_found = 0
            dates_found = 0
            categories_found = 0
            operation_types_found = 0
            transaction_types_found = 0
            
            # Analyse entités selon structure API
            if entities:
                entities_extracted = entities
                entity_confidence = entities.get("confidence", 0.0)
                
                # Support pour différents formats de réponse
                entities_data = entities.get("entities", {})
                
                # Phase 2 - comprehensive_entities 
                if "comprehensive_entities" in data:
                    comp_entities = data["comprehensive_entities"]
                    if comp_entities:
                        amounts_found = len(comp_entities.get("amounts", []))
                        merchants_found = len(comp_entities.get("merchants", []))
                        dates_found = len(comp_entities.get("date_ranges", [])) + len(comp_entities.get("dates", []))
                        categories_found = len(comp_entities.get("categories", []))
                        operation_types_found = len(comp_entities.get("operation_types", []))
                        transaction_types_found = len(comp_entities.get("transaction_types", []))
                        entity_confidence = comp_entities.get("overall_confidence", entity_confidence)
                        entities_count = amounts_found + merchants_found + dates_found + categories_found + operation_types_found + transaction_types_found
                        
                    # Timing Phase 2
                    if "multi_agent_insights" in data:
                        insights = data["multi_agent_insights"]
                        intent_processing_time_ms = insights.get("intent_processing_time_ms", intent_processing_time_ms)
                        entity_processing_time_ms = insights.get("entity_processing_time_ms", 0)
                
                # Phase 1 - format standard
                elif entities_data:
                    amounts_found = len(entities_data.get("amounts", []))
                    merchants_found = len(entities_data.get("merchants", []))
                    dates_found = len(entities_data.get("dates", []))
                    categories_found = len(entities_data.get("categories", []))
                    operation_types_found = len(entities_data.get("operation_types", []))
                    transaction_types_found = len(entities_data.get("transaction_types", []))
                    entities_count = amounts_found + merchants_found + dates_found + categories_found + operation_types_found + transaction_types_found
            
            # Extraction résumés entités
            entity_summaries = self._extract_entity_summaries(entities)
            
            return TestResult(
                question=question,
                intent_type=intent.get("intent_type", "UNKNOWN"),
                confidence=intent.get("confidence", 0.0),
                category=intent.get("category", "UNKNOWN"),
                latency_ms=latency_ms,
                success=True,
                performance_grade=agent_metrics.get("performance_grade"),
                efficiency_score=agent_metrics.get("efficiency_score"),
                entities_extracted=entities_extracted,
                entities_count=entities_count,
                entity_confidence=entity_confidence,
                intent_processing_time_ms=intent_processing_time_ms,
                entity_processing_time_ms=entity_processing_time_ms,
                amounts_found=amounts_found,
                merchants_found=merchants_found,
                dates_found=dates_found,
                categories_found=categories_found,
                operation_types_found=operation_types_found,
                transaction_types_found=transaction_types_found,
                amounts_summary=entity_summaries["amounts_summary"],
                merchants_summary=entity_summaries["merchants_summary"],
                dates_summary=entity_summaries["dates_summary"],
                categories_summary=entity_summaries["categories_summary"],
                operation_types_summary=entity_summaries["operation_types_summary"],
                transaction_types_summary=entity_summaries["transaction_types_summary"]
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return TestResult(
                question=question,
                intent_type="EXCEPTION",
                confidence=0.0,
                category="ERROR",
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
                entities_count=0,
                entity_confidence=0.0
            )
    
    def run_test_suite(self, questions: List[str]) -> None:
        """Exécute la suite de tests complète"""
        print(f"[DEBUT] Démarrage des tests sur {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"[TEST] {i}/{len(questions)}: {question[:50]}...")
            result = self.run_single_test(question)
            self.results.append(result)
            
            if result.success:
                entity_info = ""
                if result.entities_count > 0:
                    entity_info = f" | {result.entities_count} entités (conf: {result.entity_confidence:.2f})"
                timing_info = ""
                if result.entity_processing_time_ms > 0:
                    timing_info = f" | Intent: {result.intent_processing_time_ms}ms, Entités: {result.entity_processing_time_ms}ms"
                print(f"   [OK] {result.intent_type} ({result.confidence:.2f}){entity_info} - {result.latency_ms:.0f}ms{timing_info}")
            else:
                print(f"   [ERR] {result.error_message} - {result.latency_ms:.0f}ms")
            
            # Petite pause pour ne pas surcharger l'API
            time.sleep(0.1)
    
    def _extract_entity_summaries(self, entities_data: Dict[str, Any]) -> Dict[str, str]:
        """Extrait et résume les entités pour le rapport"""
        
        summaries = {
            "amounts_summary": "",
            "merchants_summary": "",
            "dates_summary": "",
            "categories_summary": "",
            "operation_types_summary": "",
            "transaction_types_summary": ""
        }
        
        if not entities_data:
            return summaries
        
        # Support pour différents formats de réponse
        entities = entities_data.get("entities", {})
        
        # Phase 2 - comprehensive_entities si disponible
        if "comprehensive_entities" in entities_data:
            comp_entities = entities_data["comprehensive_entities"]
            if comp_entities:
                entities = comp_entities
        
        # Extraction amounts - montrer toutes les entités
        amounts = entities.get("amounts", [])
        if amounts:
            amount_texts = []
            for amount in amounts:  # Tous les montants
                if isinstance(amount, dict):
                    value = amount.get("value", "")
                    currency = amount.get("currency", "EUR")
                    operator = amount.get("operator", "eq")
                    op_symbol = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "="}.get(operator, "=")
                    amount_texts.append(f"{op_symbol}{value}{currency}")
                else:
                    amount_texts.append(str(amount))
            summaries["amounts_summary"] = ", ".join(amount_texts)
        
        # Extraction merchants - montrer tous les marchands
        merchants = entities.get("merchants", [])
        if merchants:
            merchant_names = []
            for merchant in merchants:  # Tous les marchands
                if isinstance(merchant, dict):
                    merchant_names.append(merchant.get("name", merchant.get("text", str(merchant))))
                else:
                    merchant_names.append(str(merchant))
            summaries["merchants_summary"] = ", ".join(merchant_names)
        
        # Extraction dates - montrer toutes les dates
        dates = entities.get("dates", []) + entities.get("date_ranges", [])
        if dates:
            date_texts = []
            for date_item in dates:  # Toutes les dates
                if isinstance(date_item, dict):
                    value = date_item.get("value", date_item.get("text", str(date_item)))
                    date_type = date_item.get("type", "")
                    if date_type:
                        date_texts.append(f"{value}({date_type})")
                    else:
                        date_texts.append(str(value))
                else:
                    date_texts.append(str(date_item))
            summaries["dates_summary"] = ", ".join(date_texts)
        
        # Extraction categories - montrer toutes les catégories
        categories = entities.get("categories", [])
        if categories:
            cat_names = []
            for category in categories:  # Toutes les catégories
                if isinstance(category, dict):
                    cat_names.append(category.get("name", category.get("text", str(category))))
                else:
                    cat_names.append(str(category))
            summaries["categories_summary"] = ", ".join(cat_names)
        
        # Extraction operation_types - montrer tous les types d'opération
        operations = entities.get("operation_types", [])
        if operations:
            op_names = []
            for operation in operations:  # Tous les types d'opération
                if isinstance(operation, dict):
                    op_names.append(operation.get("type", operation.get("text", str(operation))))
                else:
                    op_names.append(str(operation))
            summaries["operation_types_summary"] = ", ".join(op_names)
        
        # Extraction transaction_types - montrer tous les types de transaction
        transaction_types = entities.get("transaction_types", [])
        if transaction_types:
            trans_names = []
            for trans_type in transaction_types:  # Tous les types de transaction
                if isinstance(trans_type, dict):
                    trans_names.append(trans_type.get("type", trans_type.get("text", str(trans_type))))
                else:
                    trans_names.append(str(trans_type))
            summaries["transaction_types_summary"] = ", ".join(trans_names)
        
        return summaries
    
    def generate_markdown_report(self, filename: str = "harena_test_report.md") -> None:
        """Génère un rapport détaillé en markdown avec support transaction_types"""
        if not self.results:
            print("[ERREUR] Aucun résultat à reporter")
            return
        
        # Statistiques de base
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        success_rate = (successful_tests / total_tests) * 100
        avg_latency = sum(r.latency_ms for r in self.results) / total_tests
        avg_confidence = sum(r.confidence for r in self.results if r.success) / max(successful_tests, 1)
        
        # Stats entités
        total_entities = sum(r.entities_count for r in self.results if r.success)
        avg_entities_per_query = total_entities / max(successful_tests, 1)
        avg_entity_confidence = sum(r.entity_confidence for r in self.results if r.success and r.entities_count > 0) / max(sum(1 for r in self.results if r.success and r.entities_count > 0), 1)
        
        successful_results = [r for r in self.results if r.success]
        total_amounts = sum(r.amounts_found for r in successful_results)
        total_merchants = sum(r.merchants_found for r in successful_results)  
        total_dates = sum(r.dates_found for r in successful_results)
        total_categories = sum(r.categories_found for r in successful_results)
        total_operation_types = sum(r.operation_types_found for r in successful_results)
        total_transaction_types = sum(r.transaction_types_found for r in successful_results)
        
        # Génération du rapport
        report_content = f"""# Rapport de Test Harena Chat API avec Transaction Types

**Date de génération**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Statistiques Globales

- **Total des tests**: {total_tests}
- **Tests réussis**: {successful_tests} 
- **Taux de réussite**: {success_rate:.1f}%
- **Latence moyenne**: {avg_latency:.0f}ms
- **Confiance intention moyenne**: {avg_confidence:.2f}

## Statistiques d'Extraction d'Entités

- **Total entités extraites**: {total_entities}
- **Moyenne entités par requête**: {avg_entities_per_query:.1f}
- **Confiance entités moyenne**: {avg_entity_confidence:.2f}

### Répartition par Type d'Entité

- **Montants**: {total_amounts} trouvés
- **Marchands**: {total_merchants} trouvés
- **Dates**: {total_dates} trouvées  
- **Catégories**: {total_categories} trouvées
- **Types d'opération**: {total_operation_types} trouvés
- **Types de transaction**: {total_transaction_types} trouvés

## Analyse Détaillée des Entités Extraites

| Question | Intention | Montants | Marchands | Dates | Catégories | Types Opération | Types Transaction | Status |
|----------|-----------|----------|-----------|-------|------------|-------------|----------------|--------|
"""
        
        for result in self.results:
            if not result.success:
                continue
                
            status = "[OK]"
            question_short = result.question[:35] + "..." if len(result.question) > 35 else result.question
            
            # Résumés d'entités complets
            amounts = result.amounts_summary or "-"
            merchants = result.merchants_summary or "-"
            dates = result.dates_summary or "-"
            categories = result.categories_summary or "-"
            operations = result.operation_types_summary or "-"
            transactions = result.transaction_types_summary or "-"
            
            report_content += f"| {question_short} | {result.intent_type} | {amounts} | {merchants} | {dates} | {categories} | {operations} | {transactions} | {status} |\n"
        
        # Sauvegarde
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"[OK] Rapport généré: {filename}")
        except Exception as e:
            print(f"[ERREUR] Erreur lors de la génération du rapport: {e}")


def get_test_questions() -> List[str]:
    """Retourne la liste des questions de test focalisées sur transaction_types"""
    return [
        # Questions entrées/sorties
        "Mes entrées d'argent en juin",
        "Mes sorties ce mois",
        "Combien j'ai reçu en mai",
        "Mes revenus de l'année",
        "Combien j'ai dépensé chez Amazon",
        
        # Questions virements
        "Mes virements reçus",
        "Mes virements effectués", 
        "Combien ai-je fait de virements en mai",
        
        # Questions dépenses/revenus
        "Mes dépenses de juin",
        "Mes revenus du trimestre",
        "Combien j'ai gagné ce mois",
        
        # Questions complexes
        "Compare mes entrées et sorties d'argent en juin",
        "Mes achats McDonald's",
        "Transactions d'hier",
        "Mon solde"
    ]


def main():
    """Fonction principale"""
    print("[INFO] Démarrage du script de test Harena...")
    
    # Configuration
    BASE_URL = "http://localhost:8000/api/v1"
    USERNAME = "test2@example.com"
    PASSWORD = "password123"
    
    print(f"[INFO] Configuration: {BASE_URL}")
    
    # Initialisation de la suite de tests
    test_suite = HarenaTestSuite(BASE_URL)
    print("[INFO] Suite de tests initialisée")
    
    # Authentification
    print("[INFO] Tentative d'authentification...")
    if not test_suite.authenticate(USERNAME, PASSWORD):
        print("[ERREUR] Impossible de continuer sans authentification")
        return
    
    print("[INFO] Authentification réussie")
    
    # Récupération des questions de test
    questions = get_test_questions()
    print(f"[INFO] {len(questions)} questions de test chargées")
    
    # Exécution des tests
    print("[INFO] Début de l'exécution des tests...")
    test_suite.run_test_suite(questions)
    
    # Génération du rapport
    print("[INFO] Génération du rapport...")
    report_filename = f"harena_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    test_suite.generate_markdown_report(report_filename)
    
    print(f"\n[TERMINE] Tests terminés ! Rapport disponible: {report_filename}")


if __name__ == "__main__":
    main()