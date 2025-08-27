"""
Test complet pour Harena : analyse des intentions sur 50+ questions
Génère un rapport markdown détaillé des performances
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TestResult:
    """Structure pour stocker les résultats de test"""
    question: str
    intent_type: str
    confidence: float
    category: str
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    performance_grade: Optional[str] = None
    efficiency_score: Optional[float] = None


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
                headers=headers
            )
            resp.raise_for_status()
            
            token = resp.json()["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            
            # Récupération de l'ID utilisateur
            user_resp = self.session.get(f"{self.base_url}/users/me")
            user_resp.raise_for_status()
            self.user_id = user_resp.json().get("id")
            
            print(f"✅ Authentification réussie - User ID: {self.user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur d'authentification: {e}")
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
                error_message="Utilisateur non authentifié"
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
                headers={'Content-Type': 'application/json'}
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
                    error_message=f"HTTP {response.status_code}"
                )
            
            data = response.json()
            intent = data.get("intent", {})
            agent_metrics = data.get("agent_metrics", {})
            
            return TestResult(
                question=question,
                intent_type=intent.get("intent_type", "UNKNOWN"),
                confidence=intent.get("confidence", 0.0),
                category=intent.get("category", "UNKNOWN"),
                latency_ms=latency_ms,
                success=True,
                performance_grade=agent_metrics.get("performance_grade"),
                efficiency_score=agent_metrics.get("efficiency_score")
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
                error_message=str(e)
            )
    
    def run_test_suite(self, questions: List[str]) -> None:
        """Exécute la suite de tests complète"""
        print(f"🚀 Démarrage des tests sur {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"📝 Test {i}/{len(questions)}: {question[:50]}...")
            result = self.run_single_test(question)
            self.results.append(result)
            
            if result.success:
                print(f"   ✅ {result.intent_type} ({result.confidence:.2f}) - {result.latency_ms:.0f}ms")
            else:
                print(f"   ❌ {result.error_message} - {result.latency_ms:.0f}ms")
            
            # Petite pause pour ne pas surcharger l'API
            time.sleep(0.1)
    
    def generate_markdown_report(self, filename: str = "harena_test_report.md") -> None:
        """Génère un rapport détaillé en markdown"""
        if not self.results:
            print("❌ Aucun résultat à reporter")
            return
        
        # Statistiques globales
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        success_rate = (successful_tests / total_tests) * 100
        
        avg_latency = sum(r.latency_ms for r in self.results) / total_tests
        avg_confidence = sum(r.confidence for r in self.results if r.success) / max(successful_tests, 1)
        
        # Comptage par intention
        intent_counts = {}
        category_counts = {}
        
        for result in self.results:
            if result.success:
                intent_counts[result.intent_type] = intent_counts.get(result.intent_type, 0) + 1
                category_counts[result.category] = category_counts.get(result.category, 0) + 1
        
        # Génération du rapport
        report_content = f"""# Rapport de Test Harena Chat API

**Date de génération**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Statistiques Globales

- **Total des tests**: {total_tests}
- **Tests réussis**: {successful_tests}
- **Taux de réussite**: {success_rate:.1f}%
- **Latence moyenne**: {avg_latency:.0f}ms
- **Confiance moyenne**: {avg_confidence:.2f}

## 🎯 Distribution des Intentions

| Intention | Nombre | Pourcentage |
|-----------|--------|-------------|
"""
        
        for intent_type, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / successful_tests) * 100
            report_content += f"| {intent_type} | {count} | {percentage:.1f}% |\n"
        
        report_content += f"""
## 📂 Distribution des Catégories

| Catégorie | Nombre | Pourcentage |
|-----------|--------|-------------|
"""
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / successful_tests) * 100
            report_content += f"| {category} | {count} | {percentage:.1f}% |\n"
        
        report_content += """
## 📋 Résultats Détaillés

| Question | Intention | Confiance | Catégorie | Latence (ms) | Status |
|----------|-----------|-----------|-----------|--------------|--------|
"""
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            question_short = result.question[:60] + "..." if len(result.question) > 60 else result.question
            
            report_content += f"| {question_short} | {result.intent_type} | {result.confidence:.2f} | {result.category} | {result.latency_ms:.0f} | {status} |\n"
        
        # Analyse des performances
        report_content += f"""
## ⚡ Analyse des Performances

### Latence par Quintiles
"""
        
        latencies = sorted([r.latency_ms for r in self.results if r.success])
        if latencies:
            report_content += f"""
- **Min**: {min(latencies):.0f}ms
- **P25**: {latencies[len(latencies)//4]:.0f}ms
- **Médiane**: {latencies[len(latencies)//2]:.0f}ms
- **P75**: {latencies[3*len(latencies)//4]:.0f}ms
- **Max**: {max(latencies):.0f}ms
"""
        
        # Erreurs et problèmes
        errors = [r for r in self.results if not r.success]
        if errors:
            report_content += f"""
## ❌ Erreurs Détectées ({len(errors)} erreurs)

| Question | Type d'Erreur | Message |
|----------|---------------|---------|
"""
            for error in errors:
                question_short = error.question[:50] + "..." if len(error.question) > 50 else error.question
                report_content += f"| {question_short} | {error.intent_type} | {error.error_message} |\n"
        
        # Recommandations
        report_content += """
## 💡 Recommandations

"""
        
        if success_rate < 90:
            report_content += "- ⚠️ Le taux de réussite est inférieur à 90%. Vérifier la stabilité de l'API.\n"
        
        if avg_latency > 5000:
            report_content += "- ⚠️ Latence moyenne élevée (>5s). Optimiser les performances.\n"
        
        if avg_confidence < 0.8:
            report_content += "- ⚠️ Confiance moyenne faible (<0.8). Améliorer le modèle de classification.\n"
        
        # Sauvegarde
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ Rapport généré: {filename}")
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport: {e}")


def get_test_questions() -> List[str]:
    """Retourne la liste des 50+ questions de test"""
    return [
        # Questions sur les virements et transferts
        "Combien ai-je fait de virements en mai ?",
        "Quels sont mes virements du mois dernier ?",
        "Peux-tu me montrer tous mes virements de juin ?",
        "Combien j'ai viré ce mois-ci ?",
        "Liste de mes transferts de la semaine passée",
        
        # Questions sur les dépenses
        "Combien ai-je dépensé en juin ?",
        "Mes dépenses de mai étaient de combien ?",
        "Peux-tu calculer mes dépenses totales ?",
        "Combien j'ai dépensé chez Carrefour ?",
        "Mes achats Amazon du mois dernier",
        
        # Questions sur les entrées d'argent
        "Combien j'ai eu d'entrée d'argent en juin ?",
        "Mes revenus de mai s'élevaient à combien ?",
        "Peux-tu me dire mes entrées d'argent ?",
        "Combien j'ai reçu ce mois-ci ?",
        "Mes revenus de la semaine dernière",
        
        # Questions comparatives
        "Compare mes entrées et sorties d'argent en juin !",
        "Différence entre mes revenus et dépenses de mai",
        "Est-ce que j'ai plus dépensé ou gagné ce mois ?",
        "Balance de mes comptes ce mois",
        "Comparaison revenus/dépenses sur 3 mois",
        
        # Questions sur des marchands spécifiques
        "Combien j'ai dépensé chez McDonald's ?",
        "Mes achats FNAC du trimestre",
        "Dépenses Total Station service",
        "Combien chez Leclerc ce mois ?",
        "Mes paiements Spotify de l'année",
        
        # Questions temporelles variées
        "Mes transactions d'hier",
        "Dépenses de la semaine dernière",
        "Revenus du trimestre passé",
        "Transactions de ce weekend",
        "Mes opérations de cette année",
        
        # Questions sur les catégories
        "Mes dépenses en alimentation",
        "Combien pour les transports ce mois ?",
        "Budget loisirs du trimestre",
        "Dépenses santé de l'année",
        "Coût des courses alimentaires",
        
        # Questions sur les montants
        "Transactions supérieures à 100€",
        "Petites dépenses inférieures à 10€",
        "Mes gros virements (>500€)",
        "Dépenses entre 50 et 100€",
        "Transactions de moins de 5€",
        
        # Questions sur les comptes
        "Solde de mon compte principal",
        "Historique compte épargne",
        "Mouvements compte joint",
        "Transactions carte bleue",
        "Opérations compte courant",
        
        # Questions complexes
        "Évolution de mes dépenses sur 6 mois",
        "Tendance de mes revenus cette année",
        "Prévision budget mois prochain",
        "Analyse de mes habitudes de consommation",
        "Répartition de mes dépenses par catégorie",
        
        # Questions inhabituelles ou edge cases
        "Salut, comment ça va ?",
        "Quelle heure est-il ?",
        "Peux-tu m'aider ?",
        "123456789",
        "€€€ !!! ???",
        "Transaction"
    ]


def main():
    """Fonction principale"""
    # Configuration
    BASE_URL = "http://localhost:8000/api/v1"
    USERNAME = "test2@example.com"
    PASSWORD = "password123"
    
    # Initialisation de la suite de tests
    test_suite = HarenaTestSuite(BASE_URL)
    
    # Authentification
    if not test_suite.authenticate(USERNAME, PASSWORD):
        print("❌ Impossible de continuer sans authentification")
        return
    
    # Récupération des questions de test
    questions = get_test_questions()
    
    # Exécution des tests
    test_suite.run_test_suite(questions)
    
    # Génération du rapport
    report_filename = f"harena_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    test_suite.generate_markdown_report(report_filename)
    
    print(f"\n🎉 Tests terminés ! Rapport disponible: {report_filename}")


if __name__ == "__main__":
    main()