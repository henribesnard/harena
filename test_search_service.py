#!/usr/bin/env python3
"""
Test Search Service - Catalogue de Requêtes JSON
==============================================

Script de test pour valider toutes les requêtes du catalogue
complete_search_types_catalog_extended.json avec search_service.

Objectif: Vérifier que toutes les requêtes retournent un code 200.

Usage: python test_search_service.py
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import sys
import os

# Configuration
SEARCH_SERVICE_URL = "http://localhost:8000/api/v1/search/search"
LOGIN_URL = "http://localhost:8000/api/v1/users/auth/login"
CATALOG_FILE = "complete_search_types_catalog_extended.json"
OUTPUT_FILE = "search_service_test_report.md"
TIMEOUT_SECONDS = 30

# Credentials pour l'authentification
TEST_USERNAME = "test2@example.com"
TEST_PASSWORD = "password123"

@dataclass
class TestResult:
    """Résultat d'un test individuel."""
    intent: str
    title: str
    questions: List[str]
    success: bool
    status_code: Optional[int]
    response_time_ms: float
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    query_payload: Optional[Dict[str, Any]] = None

class SearchServiceTester:
    """Testeur pour search_service basé sur le catalogue JSON."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.queries_catalog: Dict[str, Any] = {}
        self.auth_token: Optional[str] = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Authentification automatique
        await self.authenticate()
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def load_catalog(self) -> bool:
        """Charge le catalogue de requêtes JSON."""
        try:
            if not os.path.exists(CATALOG_FILE):
                print(f"[ERREUR] Fichier catalogue non trouvé: {CATALOG_FILE}")
                return False
                
            with open(CATALOG_FILE, 'r', encoding='utf-8') as f:
                self.queries_catalog = json.load(f)
                
            queries = self.queries_catalog.get('queries', [])
            print(f"[INFO] Catalogue chargé: {len(queries)} requêtes trouvées")
            return True
            
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le catalogue: {e}")
            return False

    async def authenticate(self) -> bool:
        """Authentifie l'utilisateur et récupère le token JWT."""
        try:
            print(f"[INFO] Authentification en cours...")
            
            # Données de connexion (form-encoded comme dans l'exemple)
            login_data = f"username={TEST_USERNAME}&password={TEST_PASSWORD}"
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            async with self.session.post(LOGIN_URL, data=login_data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    self.auth_token = result.get('access_token')
                    if self.auth_token:
                        print(f"[INFO] Authentification réussie")
                        return True
                    else:
                        print(f"[ERREUR] Token non trouvé dans la réponse")
                        return False
                else:
                    error_text = await response.text()
                    print(f"[ERREUR] Authentification échouée ({response.status}): {error_text}")
                    return False
                    
        except Exception as e:
            print(f"[ERREUR] Exception lors de l'authentification: {e}")
            return False

    async def test_single_query(self, query_data: Dict[str, Any]) -> TestResult:
        """Teste une requête individuelle du catalogue."""
        
        intent = query_data.get('intent', 'UNKNOWN')
        title = query_data.get('title', 'Sans titre')
        questions = query_data.get('questions', [])
        
        # Support pour les nouveaux endpoints GET
        endpoint = query_data.get('endpoint')
        method = query_data.get('method', 'POST')
        query_payload = query_data.get('query', {})
        
        print(f"[TEST] {intent}: {title}")
        start_time = time.perf_counter()
        
        try:
            # Headers avec authentification
            auth_headers = {
                'Content-Type': 'application/json'
            }
            if self.auth_token:
                auth_headers['Authorization'] = f'Bearer {self.auth_token}'
            
            if method == 'GET' and endpoint:
                # Nouvelle méthode : endpoints GET directs
                full_url = f"{SEARCH_SERVICE_URL.rstrip('/search')}{endpoint}"
                async with self.session.get(full_url, headers=auth_headers) as response:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    return await self._process_response(response, elapsed_ms, intent, title, questions, {"endpoint": endpoint, "method": method})
            else:
                # Méthode standard : POST /search avec authentification
                async with self.session.post(
                    SEARCH_SERVICE_URL, 
                    json=query_payload,
                    headers=auth_headers
                ) as response:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    return await self._process_response(response, elapsed_ms, intent, title, questions, query_payload)
                    
        except Exception as e:
            # Erreur de connexion ou autre
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return TestResult(
                intent=intent,
                title=title,
                questions=questions,
                success=False,
                status_code=None,
                response_time_ms=elapsed_ms,
                error_message=f"Erreur connexion: {str(e)}",
                query_payload=query_payload
            )

    async def _process_response(self, response, elapsed_ms: float, intent: str, title: str, questions: List[str], query_payload: Any) -> TestResult:
        """Traite la réponse HTTP et retourne un TestResult."""
        if response.status == 200:
            # Requête réussie
            try:
                response_data = await response.json()
                return TestResult(
                    intent=intent,
                    title=title,
                    questions=questions,
                    success=True,
                    status_code=200,
                    response_time_ms=elapsed_ms,
                    response_data=response_data,
                    query_payload=query_payload
                )
            except Exception as json_error:
                # Erreur de parsing JSON
                return TestResult(
                    intent=intent,
                    title=title,
                    questions=questions,
                    success=False,
                    status_code=200,
                    response_time_ms=elapsed_ms,
                    error_message=f"Erreur parsing JSON: {str(json_error)}",
                    query_payload=query_payload
                )
        else:
            # Erreur HTTP
            error_text = await response.text()
            return TestResult(
                intent=intent,
                title=title,
                questions=questions,
                success=False,
                status_code=response.status,
                response_time_ms=elapsed_ms,
                error_message=f"HTTP {response.status}: {error_text[:200]}",
                query_payload=query_payload
            )

    async def run_all_tests(self):
        """Lance tous les tests du catalogue."""
        queries = self.queries_catalog.get('queries', [])
        
        print(f"\n=> Démarrage des tests sur {len(queries)} requêtes")
        print(f"=> URL: {SEARCH_SERVICE_URL}")
        print("=" * 60)
        
        # Tester chaque requête séquentiellement
        for i, query_data in enumerate(queries, 1):
            print(f"[{i:2d}/{len(queries)}] ", end="")
            result = await self.test_single_query(query_data)
            self.results.append(result)
            
            # Affichage du statut
            status = "[✓]" if result.success else "[✗]"
            print(f"{status} {result.response_time_ms:6.1f}ms")
            
            # Petit délai pour éviter de surcharger le service
            await asyncio.sleep(0.1)

    def generate_report(self):
        """Génère le rapport markdown avec les résultats."""
        print(f"\n=> Génération du rapport: {OUTPUT_FILE}")
        
        # Statistiques globales
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Temps de réponse
        avg_response_time = sum(r.response_time_ms for r in self.results) / total_tests if total_tests > 0 else 0
        max_response_time = max(r.response_time_ms for r in self.results) if self.results else 0
        min_response_time = min(r.response_time_ms for r in self.results) if self.results else 0
        
        # Groupement par code de statut
        status_codes = {}
        for result in self.results:
            code = result.status_code or 0
            if code not in status_codes:
                status_codes[code] = 0
            status_codes[code] += 1
        
        # Génération du markdown
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# Rapport de Test - Search Service Catalog\n\n")
            f.write(f"**Date du test** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Catalogue testé** : {CATALOG_FILE}\n\n")
            f.write(f"**URL du service** : {SEARCH_SERVICE_URL}\n\n")
            f.write(f"**Timeout configuré** : {TIMEOUT_SECONDS}s\n\n")
            
            # Résumé exécutif
            f.write("## 📊 Résumé Exécutif\n\n")
            
            if success_rate == 100:
                status_emoji = "🟢"
                status_text = "EXCELLENT"
            elif success_rate >= 80:
                status_emoji = "🟡"
                status_text = "BON"
            elif success_rate >= 50:
                status_emoji = "🟠"
                status_text = "MOYEN"
            else:
                status_emoji = "🔴"
                status_text = "CRITIQUE"
            
            f.write(f"**Statut Global** : {status_emoji} {status_text}\n\n")
            f.write(f"- **Total des requêtes testées** : {total_tests}\n")
            f.write(f"- **Requêtes réussies (200)** : {successful_tests}\n")
            f.write(f"- **Requêtes échouées** : {failed_tests}\n")
            f.write(f"- **Taux de succès** : {success_rate:.1f}%\n")
            f.write(f"- **Temps de réponse moyen** : {avg_response_time:.1f}ms\n")
            f.write(f"- **Temps de réponse min/max** : {min_response_time:.1f}ms / {max_response_time:.1f}ms\n\n")
            
            # Répartition des codes de statut
            f.write("## 📈 Codes de Statut HTTP\n\n")
            f.write("| Code | Nombre | Pourcentage | Statut |\n")
            f.write("|------|---------|-------------|--------|\n")
            
            for code in sorted(status_codes.keys()):
                count = status_codes[code]
                percentage = (count / total_tests * 100) if total_tests > 0 else 0
                status_text = "✅ OK" if code == 200 else ("⚠️ Erreur Client" if 400 <= code < 500 else ("❌ Erreur Serveur" if code >= 500 else "🔌 Connexion"))
                f.write(f"| {code} | {count} | {percentage:.1f}% | {status_text} |\n")
            
            f.write("\n")
            
            # Liste des requêtes échouées
            failed_results = [r for r in self.results if not r.success]
            if failed_results:
                f.write("## ❌ Requêtes Échouées\n\n")
                f.write(f"**{len(failed_results)} requête(s) ont échoué :**\n\n")
                
                for result in failed_results:
                    f.write(f"### ❌ {result.intent}: {result.title}\n\n")
                    f.write(f"- **Code HTTP** : {result.status_code or 'Connexion échouée'}\n")
                    f.write(f"- **Temps de réponse** : {result.response_time_ms:.1f}ms\n")
                    f.write(f"- **Questions utilisateur** : {', '.join(result.questions)}\n")
                    f.write(f"- **Erreur** : {result.error_message or 'Erreur inconnue'}\n\n")
                    
                    # Payload pour debug
                    f.write("**Payload pour reproduction :**\n")
                    f.write("```json\n")
                    f.write(json.dumps(result.query_payload, indent=2, ensure_ascii=False))
                    f.write("\n```\n\n")
                    
                    # Commande curl pour reproduction manuelle
                    f.write("**Commande curl pour test manuel :**\n")
                    f.write("```bash\n")
                    f.write(f"curl -X POST '{SEARCH_SERVICE_URL}/search' \\\\\n")
                    f.write("  -H 'Content-Type: application/json' \\\\\n")
                    payload_str = json.dumps(result.query_payload, ensure_ascii=False).replace("'", "'\\''")
                    f.write(f"  -d '{payload_str}'\n")
                    f.write("```\n\n")
            
            # Statistiques des requêtes réussies
            successful_results = [r for r in self.results if r.success]
            if successful_results:
                f.write("## ✅ Requêtes Réussies\n\n")
                f.write(f"**{len(successful_results)} requête(s) ont réussi :**\n\n")
                
                # Regroupement par type d'intent
                intent_groups = {}
                for result in successful_results:
                    intent_prefix = result.intent.split('_')[0]
                    if intent_prefix not in intent_groups:
                        intent_groups[intent_prefix] = []
                    intent_groups[intent_prefix].append(result)
                
                for intent_prefix, results in intent_groups.items():
                    f.write(f"### ✅ {intent_prefix} ({len(results)} requêtes)\n\n")
                    
                    for result in results:
                        response_info = ""
                        if result.response_data:
                            # Extraire des infos utiles de la réponse
                            if 'results' in result.response_data:
                                results_count = len(result.response_data['results'])
                                response_info = f" | {results_count} résultats"
                            if 'aggregations' in result.response_data:
                                response_info += " | Agrégations OK"
                        
                        f.write(f"- **{result.intent}** : {result.title} ({result.response_time_ms:.1f}ms{response_info})\n")
                    
                    f.write("\n")
            
            # Recommandations
            f.write("## 🎯 Recommandations\n\n")
            
            if failed_tests == 0:
                f.write("🎉 **Parfait !** Toutes les requêtes du catalogue passent avec succès.\n\n")
                f.write("Le search_service est complètement fonctionnel pour tous les cas d'usage définis.\n\n")
            else:
                f.write("### 🔧 Actions Correctives Nécessaires\n\n")
                
                # Analyser les types d'erreurs
                error_4xx = len([r for r in failed_results if r.status_code and 400 <= r.status_code < 500])
                error_5xx = len([r for r in failed_results if r.status_code and r.status_code >= 500])
                error_conn = len([r for r in failed_results if not r.status_code])
                
                if error_conn > 0:
                    f.write(f"1. **🔌 Erreurs de connexion ({error_conn})** : Vérifier que search_service est démarré sur {SEARCH_SERVICE_URL}\n\n")
                
                if error_5xx > 0:
                    f.write(f"2. **🔥 Erreurs serveur ({error_5xx})** : Bugs dans search_service à corriger\n\n")
                
                if error_4xx > 0:
                    f.write(f"3. **⚠️ Erreurs client ({error_4xx})** : Requêtes invalides ou fonctionnalités non implémentées\n\n")
                
                # Priorités basées sur le taux de succès
                if success_rate < 50:
                    f.write("🆘 **CRITIQUE** : Moins de 50% des requêtes fonctionnent. Révision architecturale urgente nécessaire.\n\n")
                elif success_rate < 80:
                    f.write("⚠️ **IMPORTANT** : Plusieurs fonctionnalités manquent. Implémentation prioritaire recommandée.\n\n")
                else:
                    f.write("✅ **BON** : La majorité des fonctionnalités fonctionne. Corrections ponctuelles nécessaires.\n\n")
            
            # Informations de debug
            f.write("## 🔧 Informations de Debug\n\n")
            f.write(f"- **Fichier catalogue source** : `{CATALOG_FILE}`\n")
            f.write(f"- **Script de test** : `{__file__}`\n")
            f.write(f"- **Version Python** : {sys.version.split()[0]}\n")
            f.write(f"- **Commande pour relancer** : `python {os.path.basename(__file__)}`\n\n")

        print(f"[✅] Rapport généré : {OUTPUT_FILE}")

async def main():
    """Fonction principale du script."""
    print("Search Service - Test Catalogue JSON")
    print("======================================")
    
    try:
        async with SearchServiceTester() as tester:
            # Charger le catalogue
            if not tester.load_catalog():
                sys.exit(1)
            
            # Lancer tous les tests
            await tester.run_all_tests()
            
            # Générer le rapport
            tester.generate_report()
            
        # Afficher résumé final
        total = len(tester.results)
        success = len([r for r in tester.results if r.success])
        
        print(f"\n🏁 RÉSUMÉ FINAL")
        print(f"===============")
        print(f"Tests réussis : {success}/{total} ({success/total*100:.1f}%)")
        print(f"Rapport détaillé : {OUTPUT_FILE}")
        
        if success == total:
            print("✅ [SUCCESS] Toutes les requêtes passent ! Search service est opérationnel.")
            sys.exit(0)
        else:
            print("⚠️  [PARTIAL] Certaines requêtes échouent. Consultez le rapport pour corriger.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  [STOP] Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ [ERROR] Erreur critique : {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())