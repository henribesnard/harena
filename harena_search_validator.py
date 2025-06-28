"""
Validateur de recherche corrigé pour Harena Finance Platform.
Teste les recherches lexicales (Bonsai Elasticsearch) et sémantiques (Qdrant) 
sur les vraies données de transactions financières.

CORRECTIONS:
- Requêtes Elasticsearch optimisées pour améliorer la pertinence
- Correction des erreurs HTTP 400 Qdrant
- Vérification et réparation automatique des collections
- Analyse de pertinence améliorée
- Debug détaillé des problèmes

Usage:
    python harena_search_validator.py [--user-id USER_ID] [--repair-collections]

Ce script valide:
1. Configuration et connectivité
2. Structure et compatibilité des collections
3. Recherches lexicales optimisées
4. Recherches sémantiques corrigées
5. Performance et pertinence améliorées
"""
import asyncio
import aiohttp
import json
import time
import ssl
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Ajouter le répertoire parent au path pour importer config_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config_service.config import settings
    print("✅ Configuration Harena chargée avec succès")
except ImportError as e:
    print(f"❌ Impossible d'importer la configuration Harena: {e}")
    print("💡 Assurez-vous que le script est dans le répertoire racine du projet")
    sys.exit(1)

class HarenaSearchValidatorFixed:
    """Validateur corrigé pour les recherches de transactions financières Harena."""
    
    def __init__(self, test_user_id: int = 34, repair_mode: bool = False):
        # Configuration depuis config_service centralisé
        self.bonsai_url = settings.BONSAI_URL
        self.qdrant_url = settings.QDRANT_URL
        self.qdrant_api_key = settings.QDRANT_API_KEY
        self.openai_api_key = settings.OPENAI_API_KEY
        
        # Utilisateur de test
        self.test_user_id = test_user_id
        self.repair_mode = repair_mode
        
        # Index/Collection Harena
        self.elasticsearch_index = "harena_transactions"
        self.qdrant_collection = "financial_transactions"
        
        # Requêtes de test financières optimisées
        self.financial_test_queries = [
            {
                "query": "virement",
                "description": "Virements bancaires",
                "expected_categories": ["virement", "transfert", "salaire"],
                "search_type": "banking",
                "expected_results": "high"
            },
            {
                "query": "restaurant",
                "description": "Restaurants et restauration",
                "expected_categories": ["restaurant", "alimentation", "resto"],
                "search_type": "commercial",
                "expected_results": "medium"
            },
            {
                "query": "carte bancaire",
                "description": "Paiements par carte",
                "expected_categories": ["carte", "paiement", "cb"],
                "search_type": "payment_method",
                "expected_results": "high"
            },
            {
                "query": "courses supermarché",
                "description": "Achats alimentaires",
                "expected_categories": ["courses", "supermarché", "alimentation"],
                "search_type": "grocery",
                "expected_results": "medium"
            },
            {
                "query": "essence carburant",
                "description": "Achats de carburant",
                "expected_categories": ["essence", "carburant", "station"],
                "search_type": "transport",
                "expected_results": "low"
            },
            {
                "query": "pharmacie médicaments",
                "description": "Dépenses de santé",
                "expected_categories": ["pharmacie", "santé", "médical"],
                "search_type": "health",
                "expected_results": "low"
            }
        ]
        
        # Synonymes financiers pour expansion de requêtes
        self.financial_synonyms = {
            "restaurant": ["restaurant", "resto", "brasserie", "cafeteria", "fast", "food"],
            "courses": ["courses", "achats", "shopping", "supermarché", "hypermarché"],
            "supermarché": ["supermarché", "hypermarché", "grande", "surface", "magasin"],
            "pharmacie": ["pharmacie", "parapharmacie", "medical", "santé"],
            "essence": ["essence", "carburant", "station", "service", "petrole", "gazole"],
            "virement": ["virement", "transfer", "transfert", "salaire", "paie"],
            "carte": ["carte", "cb", "paiement", "achat", "bancaire"],
            "abonnement": ["abonnement", "subscription", "souscription", "mensuel"]
        }
        
        self.results = {
            "lexical_results": [],
            "semantic_results": [],
            "embedding_tests": [],
            "collection_diagnostics": {},
            "performance_metrics": {}
        }
    
    def log(self, level: str, message: str):
        """Log avec format coloré et timestamps."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            "INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", 
            "WARNING": "⚠️", "TEST": "🧪", "SEARCH": "🔍",
            "FINANCE": "💰", "PERFORMANCE": "⚡", "DEBUG": "🔧"
        }
        icon = icons.get(level, "📝")
        print(f"[{timestamp}] {icon} {message}")
    
    async def run_harena_search_validation(self) -> Dict[str, Any]:
        """Lance la validation complète des recherches Harena corrigée."""
        self.log("INFO", "🚀 VALIDATION RECHERCHES HARENA FINANCE PLATFORM - VERSION CORRIGÉE")
        print("=" * 85)
        
        # 1. Vérifier la configuration Harena
        self.log("INFO", "📋 Étape 1: Vérification configuration Harena")
        config_status = await self.verify_harena_configuration()
        
        if not config_status["valid"]:
            return {"error": "Configuration Harena incomplète", "config": config_status}
        
        # 2. Tester la connectivité des services
        self.log("INFO", "🌐 Étape 2: Test connectivité services Harena")
        connectivity = await self.test_harena_connectivity()
        
        # 3. Vérifier et diagnostiquer les collections
        self.log("INFO", "🔧 Étape 3: Diagnostic des collections et index")
        collection_status = await self.diagnose_collections()
        
        # 4. Réparer les collections si nécessaire et autorisé
        if self.repair_mode and collection_status.get("needs_repair"):
            self.log("INFO", "🛠️ Étape 4: Réparation des collections")
            await self.repair_collections(collection_status)
        
        # 5. Tester les embeddings OpenAI si disponible
        if self.openai_api_key and connectivity.get("can_test_embeddings"):
            self.log("INFO", "🧠 Étape 5: Test génération embeddings OpenAI")
            await self.test_openai_embeddings()
        
        # 6. Tests de recherche lexicale optimisée (Bonsai)
        if connectivity["bonsai"]["connected"]:
            self.log("INFO", "🔍 Étape 6: Tests recherche lexicale optimisée")
            await self.test_optimized_lexical_search()
        
        # 7. Tests de recherche sémantique corrigée (Qdrant)
        if connectivity["qdrant"]["connected"] and collection_status.get("qdrant_ready"):
            self.log("INFO", "🧠 Étape 7: Tests recherche sémantique corrigée")
            await self.test_corrected_semantic_search()
        
        # 8. Analyse des performances améliorée
        self.log("INFO", "⚡ Étape 8: Analyse des performances")
        performance = await self.analyze_enhanced_performance()
        
        # 9. Comparaison et recommandations
        self.log("INFO", "📊 Étape 9: Comparaison et recommandations")
        comparison = await self.compare_search_approaches_enhanced()
        
        # 10. Compilation du rapport final
        final_report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.test_user_id,
                "platform": "harena_finance_fixed",
                "repair_mode": self.repair_mode,
                "configuration": config_status,
                "connectivity": connectivity,
                "collection_diagnostics": collection_status
            },
            "search_tests": {
                "lexical_search": self.results["lexical_results"],
                "semantic_search": self.results["semantic_results"],
                "embedding_tests": self.results["embedding_tests"]
            },
            "performance": performance,
            "comparison": comparison,
            "recommendations": self.generate_enhanced_recommendations(connectivity, collection_status, performance)
        }
        
        # 11. Affichage du rapport amélioré
        self.display_enhanced_report(final_report)
        
        return final_report
    
    async def verify_harena_configuration(self) -> Dict[str, Any]:
        """Vérifie la configuration spécifique à Harena."""
        config = {
            "valid": False,
            "services": {},
            "issues": []
        }
        
        # Vérifier Bonsai Elasticsearch
        if self.bonsai_url and "your-" not in self.bonsai_url and self.bonsai_url.startswith("https://"):
            config["services"]["bonsai"] = True
            self.log("SUCCESS", f"✅ BONSAI_URL configurée")
        else:
            config["services"]["bonsai"] = False
            config["issues"].append("BONSAI_URL manquante ou invalide")
            self.log("ERROR", "❌ BONSAI_URL manquante ou invalide")
        
        # Vérifier Qdrant
        if self.qdrant_url and "your-" not in self.qdrant_url and self.qdrant_url.startswith("https://"):
            config["services"]["qdrant"] = True
            self.log("SUCCESS", f"✅ QDRANT_URL configurée")
        else:
            config["services"]["qdrant"] = False
            config["issues"].append("QDRANT_URL manquante ou invalide")
            self.log("ERROR", "❌ QDRANT_URL manquante ou invalide")
        
        # Vérifier clé API Qdrant
        if self.qdrant_api_key and len(self.qdrant_api_key) > 10:
            config["services"]["qdrant_auth"] = True
            self.log("SUCCESS", "✅ QDRANT_API_KEY configurée")
        else:
            config["services"]["qdrant_auth"] = False
            config["issues"].append("QDRANT_API_KEY manquante")
            self.log("ERROR", "❌ QDRANT_API_KEY manquante")
        
        # Vérifier OpenAI pour embeddings
        if self.openai_api_key and self.openai_api_key.startswith("sk-"):
            config["services"]["openai"] = True
            self.log("SUCCESS", "✅ OPENAI_API_KEY configurée")
        else:
            config["services"]["openai"] = False
            config["issues"].append("OPENAI_API_KEY manquante ou invalide")
            self.log("WARNING", "⚠️ OPENAI_API_KEY manquante - embeddings non testés")
        
        # Configuration valide si au moins un service de recherche est configuré
        config["valid"] = config["services"]["bonsai"] or (
            config["services"]["qdrant"] and config["services"]["qdrant_auth"]
        )
        
        return config
    
    async def test_harena_connectivity(self) -> Dict[str, Any]:
        """Teste la connectivité vers les services Harena."""
        connectivity = {
            "bonsai": {"connected": False, "response_time": 0, "error": None, "cluster_info": {}},
            "qdrant": {"connected": False, "response_time": 0, "error": None, "version": None},
            "can_test_embeddings": False
        }
        
        # Test Bonsai Elasticsearch avec détails du cluster
        if self.bonsai_url and "your-" not in self.bonsai_url:
            try:
                ssl_context = ssl.create_default_context()
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                timeout = aiohttp.ClientTimeout(total=15)
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    start_time = time.time()
                    async with session.get(self.bonsai_url) as response:
                        connectivity["bonsai"]["response_time"] = time.time() - start_time
                        
                        if response.status == 200:
                            connectivity["bonsai"]["connected"] = True
                            data = await response.json()
                            connectivity["bonsai"]["cluster_info"] = {
                                "name": data.get('cluster_name', 'unknown'),
                                "version": data.get('version', {}).get('number', 'unknown'),
                                "lucene_version": data.get('version', {}).get('lucene_version', 'unknown')
                            }
                            cluster_name = connectivity["bonsai"]["cluster_info"]["name"]
                            version = connectivity["bonsai"]["cluster_info"]["version"]
                            self.log("SUCCESS", f"✅ Bonsai connecté: {cluster_name} elasticsearch v{version}")
                        else:
                            connectivity["bonsai"]["error"] = f"HTTP {response.status}"
                            self.log("ERROR", f"❌ Bonsai: HTTP {response.status}")
            except Exception as e:
                connectivity["bonsai"]["error"] = str(e)
                self.log("ERROR", f"❌ Bonsai inaccessible: {e}")
        
        # Test Qdrant avec version
        if self.qdrant_url and "your-" not in self.qdrant_url:
            try:
                headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}
                timeout = aiohttp.ClientTimeout(total=15)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    start_time = time.time()
                    async with session.get(f"{self.qdrant_url}/", headers=headers) as response:
                        connectivity["qdrant"]["response_time"] = time.time() - start_time
                        
                        if response.status == 200:
                            connectivity["qdrant"]["connected"] = True
                            data = await response.json()
                            connectivity["qdrant"]["version"] = data.get("version", "unknown")
                            self.log("SUCCESS", f"✅ Qdrant connecté: v{connectivity['qdrant']['version']}")
                        else:
                            connectivity["qdrant"]["error"] = f"HTTP {response.status}"
                            self.log("ERROR", f"❌ Qdrant: HTTP {response.status}")
            except Exception as e:
                connectivity["qdrant"]["error"] = str(e)
                self.log("ERROR", f"❌ Qdrant inaccessible: {e}")
        
        # Capacité de test d'embeddings
        connectivity["can_test_embeddings"] = bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))
        
        return connectivity
    
    async def diagnose_collections(self) -> Dict[str, Any]:
        """Diagnostique détaillé des collections et index."""
        diagnostics = {
            "elasticsearch_status": {"exists": False, "transaction_count": 0, "mapping_ok": False, "sample_data": []},
            "qdrant_status": {"exists": False, "point_count": 0, "vector_config": {}, "compatible": False},
            "needs_repair": False,
            "repair_actions": [],
            "qdrant_ready": False
        }
        
        # Diagnostic Elasticsearch
        await self._diagnose_elasticsearch(diagnostics)
        
        # Diagnostic Qdrant
        await self._diagnose_qdrant(diagnostics)
        
        # Déterminer si une réparation est nécessaire
        if not diagnostics["qdrant_status"]["compatible"]:
            diagnostics["needs_repair"] = True
            diagnostics["repair_actions"].append("Recréer collection Qdrant avec dimension 1536")
        
        if diagnostics["elasticsearch_status"]["transaction_count"] == 0:
            diagnostics["repair_actions"].append("Synchroniser les données avec l'API enrichment")
        
        # Qdrant prêt si existe et compatible
        diagnostics["qdrant_ready"] = (
            diagnostics["qdrant_status"]["exists"] and 
            diagnostics["qdrant_status"]["compatible"]
        )
        
        return diagnostics
    
    async def _diagnose_elasticsearch(self, diagnostics: Dict[str, Any]):
        """Diagnostique spécifique à Elasticsearch."""
        if not self.bonsai_url or "your-" in self.bonsai_url:
            return
        
        try:
            ssl_context = ssl.create_default_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=15)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # 1. Vérifier existence index
                async with session.head(f"{self.bonsai_url}/{self.elasticsearch_index}") as response:
                    if response.status == 200:
                        diagnostics["elasticsearch_status"]["exists"] = True
                        self.log("SUCCESS", f"✅ Index Elasticsearch existe: {self.elasticsearch_index}")
                        
                        # 2. Compter les transactions utilisateur
                        count_query = {"query": {"term": {"user_id": self.test_user_id}}}
                        async with session.post(
                            f"{self.bonsai_url}/{self.elasticsearch_index}/_count",
                            json=count_query
                        ) as count_response:
                            if count_response.status == 200:
                                count_data = await count_response.json()
                                transaction_count = count_data.get("count", 0)
                                diagnostics["elasticsearch_status"]["transaction_count"] = transaction_count
                                self.log("SUCCESS", f"✅ {transaction_count} transactions pour user {self.test_user_id}")
                        
                        # 3. Examiner le mapping
                        async with session.get(f"{self.bonsai_url}/{self.elasticsearch_index}/_mapping") as mapping_response:
                            if mapping_response.status == 200:
                                mapping_data = await mapping_response.json()
                                properties = mapping_data.get(self.elasticsearch_index, {}).get("mappings", {}).get("properties", {})
                                
                                # Vérifier champs critiques
                                required_fields = ["searchable_text", "primary_description", "merchant_name", "user_id"]
                                missing_fields = [field for field in required_fields if field not in properties]
                                
                                if not missing_fields:
                                    diagnostics["elasticsearch_status"]["mapping_ok"] = True
                                    self.log("SUCCESS", "✅ Mapping Elasticsearch correct")
                                else:
                                    self.log("WARNING", f"⚠️ Champs manquants dans mapping: {missing_fields}")
                        
                        # 4. Récupérer échantillons de données
                        sample_query = {
                            "query": {"term": {"user_id": self.test_user_id}},
                            "size": 3,
                            "_source": ["searchable_text", "primary_description", "merchant_name", "amount"]
                        }
                        async with session.post(
                            f"{self.bonsai_url}/{self.elasticsearch_index}/_search",
                            json=sample_query
                        ) as sample_response:
                            if sample_response.status == 200:
                                sample_data = await sample_response.json()
                                hits = sample_data.get("hits", {}).get("hits", [])
                                diagnostics["elasticsearch_status"]["sample_data"] = [
                                    hit["_source"] for hit in hits
                                ]
                    else:
                        self.log("WARNING", f"⚠️ Index {self.elasticsearch_index} n'existe pas")
                        
        except Exception as e:
            self.log("ERROR", f"❌ Erreur diagnostic Elasticsearch: {e}")
    
    async def _diagnose_qdrant(self, diagnostics: Dict[str, Any]):
        """Diagnostique spécifique à Qdrant."""
        if not self.qdrant_url or "your-" in self.qdrant_url:
            return
        
        try:
            headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}
            timeout = aiohttp.ClientTimeout(total=15)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 1. Vérifier existence collection
                async with session.get(
                    f"{self.qdrant_url}/collections/{self.qdrant_collection}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        collection_info = await response.json()
                        result = collection_info.get("result", {})
                        
                        diagnostics["qdrant_status"]["exists"] = True
                        diagnostics["qdrant_status"]["point_count"] = result.get("points_count", 0)
                        
                        # Examiner configuration vectorielle
                        config = result.get("config", {})
                        vector_config = config.get("params", {}).get("vectors", {})
                        
                        if isinstance(vector_config, dict):
                            vector_size = vector_config.get("size", 0)
                            distance = vector_config.get("distance", "unknown")
                            
                            diagnostics["qdrant_status"]["vector_config"] = {
                                "size": vector_size,
                                "distance": distance
                            }
                            
                            # Vérifier compatibilité avec OpenAI embeddings (1536 dimensions)
                            if vector_size == 1536:
                                diagnostics["qdrant_status"]["compatible"] = True
                                self.log("SUCCESS", f"✅ Collection Qdrant compatible: {vector_size} dimensions, distance {distance}")
                            else:
                                self.log("WARNING", f"⚠️ Collection Qdrant incompatible: {vector_size} dimensions au lieu de 1536")
                        
                        point_count = diagnostics["qdrant_status"]["point_count"]
                        self.log("SUCCESS", f"✅ Collection Qdrant existe: {point_count} points")
                        
                    elif response.status == 404:
                        self.log("WARNING", f"⚠️ Collection {self.qdrant_collection} n'existe pas")
                    else:
                        error_text = await response.text()
                        self.log("ERROR", f"❌ Erreur Qdrant: HTTP {response.status} - {error_text}")
                        
        except Exception as e:
            self.log("ERROR", f"❌ Erreur diagnostic Qdrant: {e}")
    
    async def repair_collections(self, collection_status: Dict[str, Any]):
        """Répare les collections si nécessaire."""
        if not self.repair_mode:
            return
        
        # Réparer Qdrant si nécessaire
        if not collection_status["qdrant_status"]["compatible"]:
            await self._repair_qdrant_collection()
    
    async def _repair_qdrant_collection(self):
        """Répare la collection Qdrant avec la bonne dimension."""
        if not self.qdrant_url or "your-" in self.qdrant_url:
            return
        
        try:
            headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                self.log("DEBUG", "🔧 Réparation collection Qdrant...")
                
                # 1. Supprimer l'ancienne collection
                async with session.delete(
                    f"{self.qdrant_url}/collections/{self.qdrant_collection}",
                    headers=headers
                ) as delete_response:
                    if delete_response.status in [200, 404]:
                        self.log("SUCCESS", "✅ Ancienne collection supprimée")
                    else:
                        self.log("WARNING", f"⚠️ Erreur suppression: HTTP {delete_response.status}")
                
                # 2. Créer nouvelle collection avec dimension 1536
                collection_config = {
                    "vectors": {
                        "size": 1536,  # Dimension OpenAI text-embedding-3-small
                        "distance": "Cosine"
                    },
                    "optimizers_config": {
                        "default_segment_number": 2
                    }
                }
                
                async with session.put(
                    f"{self.qdrant_url}/collections/{self.qdrant_collection}",
                    json=collection_config,
                    headers=headers
                ) as create_response:
                    if create_response.status == 200:
                        self.log("SUCCESS", "✅ Nouvelle collection créée (1536 dims)")
                        self.log("WARNING", "⚠️ ATTENTION: Vous devez re-synchroniser les données!")
                        self.log("INFO", f"   → Lancez: POST /api/v1/enrichment/dual/sync-user?user_id={self.test_user_id}&force_refresh=true")
                    else:
                        error_text = await create_response.text()
                        self.log("ERROR", f"❌ Erreur création collection: {error_text}")
                        
        except Exception as e:
            self.log("ERROR", f"❌ Erreur réparation Qdrant: {e}")
    
    async def test_openai_embeddings(self):
        """Teste la génération d'embeddings avec OpenAI."""
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            self.log("WARNING", "⚠️ Pas de clé OpenAI valide - test d'embeddings ignoré")
            return
        
        test_texts = [
            "restaurant gastronomique Paris",
            "virement bancaire salaire",
            "essence station service",
            "courses supermarché bio"
        ]
        
        for test_text in test_texts:
            try:
                self.log("TEST", f"🧠 Test embedding: '{test_text}'")
                
                start_time = time.time()
                
                # Appel API OpenAI pour embedding
                timeout = aiohttp.ClientTimeout(total=30)
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "input": test_text,
                    "model": "text-embedding-3-small"
                }
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        "https://api.openai.com/v1/embeddings",
                        json=payload,
                        headers=headers
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            embedding = data["data"][0]["embedding"]
                            
                            embedding_result = {
                                "text": test_text,
                                "success": True,
                                "embedding_length": len(embedding),
                                "response_time": response_time,
                                "model": "text-embedding-3-small"
                            }
                            
                            self.results["embedding_tests"].append(embedding_result)
                            self.log("SUCCESS", f"   ✅ Embedding généré: {len(embedding)} dimensions en {response_time:.2f}s")
                        else:
                            error_text = await response.text()
                            self.log("ERROR", f"   ❌ OpenAI API error: HTTP {response.status} - {error_text}")
                            
                            self.results["embedding_tests"].append({
                                "text": test_text,
                                "success": False,
                                "error": f"HTTP {response.status}",
                                "response_time": response_time
                            })
                
                # Délai entre les requêtes pour respecter les limites de taux
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.log("ERROR", f"   ❌ Exception embedding: {e}")
                self.results["embedding_tests"].append({
                    "text": test_text,
                    "success": False,
                    "error": str(e),
                    "response_time": 0
                })
    
    def expand_financial_query(self, query: str) -> str:
        """Expand les requêtes avec des synonymes financiers."""
        query_lower = query.lower()
        expanded_terms = [query]
        
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        return " ".join(set(expanded_terms))
    
    def build_optimized_elasticsearch_query(self, query_text: str, user_id: int) -> Dict[str, Any]:
        """Construit une requête Elasticsearch optimisée pour les transactions."""
        
        # Expansion de la requête avec synonymes
        expanded_query = self.expand_financial_query(query_text)
        query_words = query_text.lower().split()
        
        # Requête multi-stratégie optimisée
        optimized_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}}
                    ],
                    "should": [
                        # 1. Correspondance exacte de phrase (boost très élevé)
                        {
                            "match_phrase": {
                                "searchable_text": {
                                    "query": query_text,
                                    "boost": 8.0
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "primary_description": {
                                    "query": query_text,
                                    "boost": 6.0
                                }
                            }
                        },
                        
                        # 2. Correspondance dans merchant_name (très important)
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["merchant_name^4.0", "merchant_name.keyword^5.0"],
                                "type": "best_fields",
                                "boost": 4.0
                            }
                        },
                        
                        # 3. Correspondance multi-champs avec fuzziness
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": [
                                    "searchable_text^3.0",
                                    "primary_description^2.5",
                                    "clean_description^2.0",
                                    "provider_description^1.5"
                                ],
                                "type": "best_fields",
                                "operator": "or",
                                "fuzziness": "AUTO",
                                "boost": 3.0
                            }
                        },
                        
                        # 4. Correspondance avec requête étendue (synonymes)
                        {
                            "multi_match": {
                                "query": expanded_query,
                                "fields": [
                                    "searchable_text^2.0",
                                    "primary_description^1.5"
                                ],
                                "type": "cross_fields",
                                "operator": "or",
                                "boost": 2.0
                            }
                        },
                        
                        # 5. Correspondance partielle avec wildcards
                        {
                            "bool": {
                                "should": [
                                    {
                                        "wildcard": {
                                            "searchable_text": {
                                                "value": f"*{word}*",
                                                "boost": 1.5
                                            }
                                        }
                                    } for word in query_words if len(word) > 3
                                ]
                            }
                        },
                        
                        # 6. Correspondance simple pour fallback
                        {
                            "simple_query_string": {
                                "query": query_text,
                                "fields": ["searchable_text", "primary_description", "merchant_name"],
                                "default_operator": "or",
                                "boost": 1.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": 20,
            "sort": [
                {"_score": {"order": "desc"}},
                {"transaction_date": {"order": "desc", "unmapped_type": "date"}}
            ],
            "_source": [
                "transaction_id", "primary_description", "merchant_name",
                "amount", "transaction_date", "searchable_text", "category_id"
            ],
            "highlight": {
                "fields": {
                    "searchable_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    },
                    "primary_description": {
                        "fragment_size": 100,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    },
                    "merchant_name": {}
                }
            }
        }
        
        return optimized_query
    
    async def test_optimized_lexical_search(self):
        """Teste les recherches lexicales optimisées sur les transactions financières."""
        if not self.bonsai_url or "your-" in self.bonsai_url:
            self.log("WARNING", "⚠️ Bonsai non configuré - tests lexicaux ignorés")
            return
        
        ssl_context = ssl.create_default_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=25)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for test_query in self.financial_test_queries:
                query_text = test_query["query"]
                description = test_query["description"]
                search_type = test_query["search_type"]
                expected_results = test_query["expected_results"]
                
                self.log("SEARCH", f"🔍 Test lexical optimisé: '{query_text}' ({description})")
                
                try:
                    # Utiliser la requête Elasticsearch optimisée
                    search_body = self.build_optimized_elasticsearch_query(query_text, self.test_user_id)
                    
                    start_time = time.time()
                    async with session.post(
                        f"{self.bonsai_url}/{self.elasticsearch_index}/_search",
                        json=search_body
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            hits = data.get("hits", {}).get("hits", [])
                            total = data.get("hits", {}).get("total", {})
                            
                            total_count = total.get("value", 0) if isinstance(total, dict) else total
                            
                            # Analyser la pertinence des résultats avec méthode améliorée
                            relevance_score = self.analyze_enhanced_relevance(hits, query_text, test_query)
                            
                            result = {
                                "query": query_text,
                                "description": description,
                                "search_type": search_type,
                                "expected_results": expected_results,
                                "success": True,
                                "results_count": len(hits),
                                "total_found": total_count,
                                "response_time": response_time,
                                "max_score": max([hit["_score"] for hit in hits]) if hits else 0,
                                "avg_score": sum([hit["_score"] for hit in hits]) / len(hits) if hits else 0,
                                "relevance_score": relevance_score,
                                "quality_assessment": self.assess_result_quality(relevance_score, len(hits), expected_results),
                                "sample_results": [
                                    {
                                        "score": hit["_score"],
                                        "description": hit["_source"].get("primary_description", ""),
                                        "merchant": hit["_source"].get("merchant_name", ""),
                                        "amount": hit["_source"].get("amount", 0),
                                        "date": hit["_source"].get("transaction_date", ""),
                                        "highlights": hit.get("highlight", {})
                                    }
                                    for hit in hits[:5]
                                ]
                            }
                            
                            self.results["lexical_results"].append(result)
                            quality = result["quality_assessment"]
                            self.log("SUCCESS", f"   ✅ {len(hits)} résultats, score max: {result['max_score']:.2f}, pertinence: {relevance_score:.1f}% ({quality}), temps: {response_time:.2f}s")
                            
                        else:
                            error_text = await response.text()
                            error_result = {
                                "query": query_text,
                                "description": description,
                                "search_type": search_type,
                                "expected_results": expected_results,
                                "success": False,
                                "error": f"HTTP {response.status} - {error_text}",
                                "response_time": response_time
                            }
                            self.results["lexical_results"].append(error_result)
                            self.log("ERROR", f"   ❌ HTTP {response.status}")
                
                except Exception as e:
                    error_result = {
                        "query": query_text,
                        "description": description,
                        "search_type": search_type,
                        "expected_results": expected_results,
                        "success": False,
                        "error": str(e),
                        "response_time": 0
                    }
                    self.results["lexical_results"].append(error_result)
                    self.log("ERROR", f"   ❌ Exception: {e}")
                
                # Délai entre les requêtes
                await asyncio.sleep(0.7)
    
    async def test_corrected_semantic_search(self):
        """Teste les recherches sémantiques corrigées sur les transactions financières."""
        if not self.qdrant_url or "your-" in self.qdrant_url:
            self.log("WARNING", "⚠️ Qdrant non configuré - tests sémantiques ignorés")
            return
        
        headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}
        timeout = aiohttp.ClientTimeout(total=35)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for test_query in self.financial_test_queries:
                query_text = test_query["query"]
                description = test_query["description"]
                search_type = test_query["search_type"]
                expected_results = test_query["expected_results"]
                
                self.log("SEARCH", f"🧠 Test sémantique corrigé: '{query_text}' ({description})")
                
                try:
                    # Générer embedding pour la requête
                    query_embedding = await self.generate_query_embedding(session, query_text)
                    
                    if not query_embedding:
                        self.log("WARNING", f"   ⚠️ Impossible de générer l'embedding pour '{query_text}'")
                        continue
                    
                    # Requête Qdrant corrigée avec test graduel
                    search_success = False
                    
                    # 1. Test simple sans filtre d'abord
                    simple_search_body = {
                        "vector": query_embedding,
                        "limit": 10,
                        "with_payload": True,
                        "with_vector": False
                    }
                    
                    start_time = time.time()
                    async with session.post(
                        f"{self.qdrant_url}/collections/{self.qdrant_collection}/points/search",
                        json=simple_search_body,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            # Test simple réussi, essayer avec filtre
                            filtered_search_body = {
                                "vector": query_embedding,
                                "filter": {
                                    "must": [
                                        {
                                            "key": "user_id",
                                            "match": {"value": self.test_user_id}
                                        }
                                    ]
                                },
                                "limit": 15,
                                "score_threshold": 0.5,  # Seuil plus permissif
                                "with_payload": True,
                                "with_vector": False
                            }
                            
                            async with session.post(
                                f"{self.qdrant_url}/collections/{self.qdrant_collection}/points/search",
                                json=filtered_search_body,
                                headers=headers
                            ) as filtered_response:
                                response_time = time.time() - start_time
                                
                                if filtered_response.status == 200:
                                    data = await filtered_response.json()
                                    points = data.get("result", [])
                                    search_success = True
                                    
                                    # Analyser la pertinence sémantique
                                    relevance_score = self.analyze_enhanced_semantic_relevance(points, query_text, test_query)
                                    
                                    result = {
                                        "query": query_text,
                                        "description": description,
                                        "search_type": search_type,
                                        "expected_results": expected_results,
                                        "success": True,
                                        "results_count": len(points),
                                        "response_time": response_time,
                                        "max_score": max([point["score"] for point in points]) if points else 0,
                                        "avg_score": sum([point["score"] for point in points]) / len(points) if points else 0,
                                        "min_score": min([point["score"] for point in points]) if points else 0,
                                        "relevance_score": relevance_score,
                                        "quality_assessment": self.assess_result_quality(relevance_score, len(points), expected_results),
                                        "sample_results": [
                                            {
                                                "score": point["score"],
                                                "description": point["payload"].get("primary_description", ""),
                                                "merchant": point["payload"].get("merchant_name", ""),
                                                "amount": point["payload"].get("amount", 0),
                                                "date": point["payload"].get("date", ""),
                                                "searchable_text": point["payload"].get("searchable_text", "")[:100] + "..."
                                            }
                                            for point in points[:5]
                                        ]
                                    }
                                    
                                    self.results["semantic_results"].append(result)
                                    quality = result["quality_assessment"]
                                    self.log("SUCCESS", f"   ✅ {len(points)} résultats, score max: {result['max_score']:.3f}, pertinence: {relevance_score:.1f}% ({quality}), temps: {response_time:.2f}s")
                                else:
                                    # Erreur avec filtre, utiliser résultats simple
                                    response_time = time.time() - start_time
                                    simple_data = await response.json()
                                    simple_points = simple_data.get("result", [])
                                    
                                    # Filtrer manuellement par user_id
                                    user_points = [
                                        point for point in simple_points 
                                        if point.get("payload", {}).get("user_id") == self.test_user_id
                                    ]
                                    
                                    if user_points:
                                        relevance_score = self.analyze_enhanced_semantic_relevance(user_points, query_text, test_query)
                                        
                                        result = {
                                            "query": query_text,
                                            "description": description,
                                            "search_type": search_type,
                                            "expected_results": expected_results,
                                            "success": True,
                                            "results_count": len(user_points),
                                            "response_time": response_time,
                                            "max_score": max([point["score"] for point in user_points]) if user_points else 0,
                                            "avg_score": sum([point["score"] for point in user_points]) / len(user_points) if user_points else 0,
                                            "relevance_score": relevance_score,
                                            "quality_assessment": self.assess_result_quality(relevance_score, len(user_points), expected_results),
                                            "note": "Filtrage manuel user_id (problème filtre Qdrant)",
                                            "sample_results": [
                                                {
                                                    "score": point["score"],
                                                    "description": point["payload"].get("primary_description", ""),
                                                    "merchant": point["payload"].get("merchant_name", ""),
                                                    "amount": point["payload"].get("amount", 0)
                                                }
                                                for point in user_points[:5]
                                            ]
                                        }
                                        
                                        self.results["semantic_results"].append(result)
                                        quality = result["quality_assessment"]
                                        self.log("SUCCESS", f"   ✅ {len(user_points)} résultats (filtrage manuel), score max: {result['max_score']:.3f}, pertinence: {relevance_score:.1f}% ({quality})")
                        
                        if not search_success:
                            response_time = time.time() - start_time
                            error_text = await response.text()
                            error_result = {
                                "query": query_text,
                                "description": description,
                                "search_type": search_type,
                                "expected_results": expected_results,
                                "success": False,
                                "error": f"HTTP {response.status} - {error_text}",
                                "response_time": response_time
                            }
                            self.results["semantic_results"].append(error_result)
                            self.log("ERROR", f"   ❌ HTTP {response.status}")
                
                except Exception as e:
                    error_result = {
                        "query": query_text,
                        "description": description,
                        "search_type": search_type,
                        "expected_results": expected_results,
                        "success": False,
                        "error": str(e),
                        "response_time": 0
                    }
                    self.results["semantic_results"].append(error_result)
                    self.log("ERROR", f"   ❌ Exception: {e}")
                
                # Délai entre les requêtes
                await asyncio.sleep(1.0)
    
    async def generate_query_embedding(self, session: aiohttp.ClientSession, query_text: str) -> Optional[List[float]]:
        """Génère un embedding pour une requête via OpenAI."""
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": query_text,
                "model": "text-embedding-3-small"
            }
            
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"][0]["embedding"]
                else:
                    self.log("ERROR", f"❌ OpenAI embedding error: HTTP {response.status}")
                    return None
        except Exception as e:
            self.log("ERROR", f"❌ Exception génération embedding: {e}")
            return None
    
    def analyze_enhanced_relevance(self, hits: List[Dict], query: str, test_query: Dict) -> float:
        """Analyse la pertinence des résultats avec méthode améliorée."""
        if not hits:
            return 0.0
        
        relevance_score = 0
        total_weight = 0
        
        query_words = set(query.lower().split())
        expected_categories = set(cat.lower() for cat in test_query.get("expected_categories", []))
        
        for i, hit in enumerate(hits[:10]):
            # Poids dégressif selon le rang (plus agressif)
            position_weight = 1.0 / (i * 0.3 + 1)
            total_weight += position_weight
            
            source = hit.get("_source", {})
            score = hit.get("_score", 0)
            
            # Récupérer tous les champs textuels
            all_text = " ".join([
                source.get("searchable_text", ""),
                source.get("primary_description", ""),
                source.get("merchant_name", ""),
                source.get("clean_description", "")
            ]).lower()
            
            # Score de correspondance textuelle amélioré
            text_matches = sum(1 for word in query_words if word in all_text)
            text_coverage = text_matches / len(query_words) if query_words else 0
            
            # Score de catégorie attendue amélioré
            category_match = any(cat in all_text for cat in expected_categories)
            category_score = 1.0 if category_match else 0
            
            # Score Elasticsearch normalisé (0-1)
            es_score_normalized = min(score / 100.0, 1.0)  # Normaliser sur base 100
            
            # Bonus pour correspondance exacte
            exact_match_bonus = 0.2 if query.lower() in all_text else 0
            
            # Score combiné pondéré
            result_score = (
                text_coverage * 0.35 +          # Correspondance mots
                category_score * 0.25 +         # Catégorie attendue
                es_score_normalized * 0.25 +    # Score Elasticsearch
                exact_match_bonus * 0.15        # Bonus correspondance exacte
            ) * position_weight
            
            relevance_score += result_score
        
        final_score = (relevance_score / total_weight * 100) if total_weight > 0 else 0
        return min(final_score, 100.0)  # Cap à 100%
    
    def analyze_enhanced_semantic_relevance(self, points: List[Dict], query: str, test_query: Dict) -> float:
        """Analyse la pertinence sémantique des résultats Qdrant avec méthode améliorée."""
        if not points:
            return 0.0
        
        relevance_score = 0
        total_weight = 0
        
        query_words = set(query.lower().split())
        expected_categories = set(cat.lower() for cat in test_query.get("expected_categories", []))
        
        for i, point in enumerate(points[:10]):
            position_weight = 1.0 / (i * 0.2 + 1)
            total_weight += position_weight
            
            payload = point.get("payload", {})
            description = payload.get("primary_description", "").lower()
            merchant = payload.get("merchant_name", "").lower()
            searchable_text = payload.get("searchable_text", "").lower()
            
            all_text = f"{description} {merchant} {searchable_text}"
            
            # Score basé sur la similarité vectorielle (plus important)
            vector_score = point.get("score", 0)
            
            # Score de correspondance conceptuelle
            concept_score = 0
            for category in expected_categories:
                if category in all_text:
                    concept_score = 1.0
                    break
            
            # Score de correspondance textuelle
            text_matches = sum(1 for word in query_words if word in all_text)
            text_score = text_matches / len(query_words) if query_words else 0
            
            # Score combiné (favorise la similarité vectorielle pour la recherche sémantique)
            result_score = (
                vector_score * 0.6 +      # Similarité vectorielle principale
                concept_score * 0.25 +    # Correspondance conceptuelle
                text_score * 0.15         # Correspondance textuelle
            ) * position_weight
            
            relevance_score += result_score
        
        final_score = (relevance_score / total_weight * 100) if total_weight > 0 else 0
        return min(final_score, 100.0)
    
    def assess_result_quality(self, relevance_score: float, result_count: int, expected_results: str) -> str:
        """Évalue la qualité des résultats."""
        if relevance_score >= 80:
            return "excellent"
        elif relevance_score >= 60:
            return "bon"
        elif relevance_score >= 40:
            return "moyen"
        elif relevance_score >= 20:
            return "faible"
        else:
            return "mauvais"
    
    async def analyze_enhanced_performance(self) -> Dict[str, Any]:
        """Analyse les performances de recherche améliorée."""
        performance = {
            "lexical_performance": {},
            "semantic_performance": {},
            "embedding_performance": {},
            "overall_metrics": {},
            "quality_distribution": {}
        }
        
        # Performances lexicales
        lexical_results = [r for r in self.results["lexical_results"] if r["success"]]
        if lexical_results:
            performance["lexical_performance"] = {
                "success_rate": len(lexical_results) / len(self.results["lexical_results"]),
                "avg_response_time": sum(r["response_time"] for r in lexical_results) / len(lexical_results),
                "avg_results_count": sum(r["results_count"] for r in lexical_results) / len(lexical_results),
                "avg_relevance": sum(r.get("relevance_score", 0) for r in lexical_results) / len(lexical_results),
                "fastest_query": min(lexical_results, key=lambda x: x["response_time"])["response_time"],
                "slowest_query": max(lexical_results, key=lambda x: x["response_time"])["response_time"],
                "quality_distribution": self._calculate_quality_distribution(lexical_results)
            }
        
        # Performances sémantiques
        semantic_results = [r for r in self.results["semantic_results"] if r["success"]]
        if semantic_results:
            performance["semantic_performance"] = {
                "success_rate": len(semantic_results) / len(self.results["semantic_results"]),
                "avg_response_time": sum(r["response_time"] for r in semantic_results) / len(semantic_results),
                "avg_results_count": sum(r["results_count"] for r in semantic_results) / len(semantic_results),
                "avg_relevance": sum(r.get("relevance_score", 0) for r in semantic_results) / len(semantic_results),
                "avg_similarity_score": sum(r.get("avg_score", 0) for r in semantic_results) / len(semantic_results),
                "fastest_query": min(semantic_results, key=lambda x: x["response_time"])["response_time"],
                "slowest_query": max(semantic_results, key=lambda x: x["response_time"])["response_time"],
                "quality_distribution": self._calculate_quality_distribution(semantic_results)
            }
        
        # Performances embeddings
        embedding_results = [r for r in self.results["embedding_tests"] if r["success"]]
        if embedding_results:
            performance["embedding_performance"] = {
                "success_rate": len(embedding_results) / len(self.results["embedding_tests"]),
                "avg_response_time": sum(r["response_time"] for r in embedding_results) / len(embedding_results),
                "avg_embedding_size": sum(r["embedding_length"] for r in embedding_results) / len(embedding_results)
            }
        
        # Métriques globales
        performance["overall_metrics"] = {
            "total_tests_run": len(self.results["lexical_results"]) + len(self.results["semantic_results"]),
            "lexical_available": len(lexical_results) > 0,
            "semantic_available": len(semantic_results) > 0,
            "hybrid_ready": len(lexical_results) > 0 and len(semantic_results) > 0,
            "overall_health": self._calculate_overall_health(performance)
        }
        
        return performance
    
    def _calculate_quality_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Calcule la distribution de qualité des résultats."""
        distribution = {"excellent": 0, "bon": 0, "moyen": 0, "faible": 0, "mauvais": 0}
        
        for result in results:
            quality = result.get("quality_assessment", "mauvais")
            if quality in distribution:
                distribution[quality] += 1
        
        return distribution
    
    def _calculate_overall_health(self, performance: Dict) -> str:
        """Calcule l'état de santé global du système de recherche."""
        lexical_perf = performance.get("lexical_performance", {})
        semantic_perf = performance.get("semantic_performance", {})
        
        lexical_health = lexical_perf.get("avg_relevance", 0) if lexical_perf else 0
        semantic_health = semantic_perf.get("avg_relevance", 0) if semantic_perf else 0
        
        overall_relevance = max(lexical_health, semantic_health)
        
        if overall_relevance >= 70:
            return "excellent"
        elif overall_relevance >= 50:
            return "bon"
        elif overall_relevance >= 30:
            return "moyen"
        else:
            return "problématique"
    
    async def compare_search_approaches_enhanced(self) -> List[Dict[str, Any]]:
        """Compare les approches de recherche avec analyse améliorée."""
        comparisons = []
        
        for test_query in self.financial_test_queries:
            query = test_query["query"]
            
            lexical_result = next((r for r in self.results["lexical_results"] if r["query"] == query), None)
            semantic_result = next((r for r in self.results["semantic_results"] if r["query"] == query), None)
            
            if lexical_result or semantic_result:
                comparison = {
                    "query": query,
                    "description": test_query["description"],
                    "search_type": test_query["search_type"],
                    "expected_categories": test_query["expected_categories"],
                    "expected_results": test_query["expected_results"],
                    "lexical": {
                        "available": lexical_result is not None and lexical_result["success"],
                        "results_count": lexical_result["results_count"] if lexical_result and lexical_result["success"] else 0,
                        "relevance": lexical_result.get("relevance_score", 0) if lexical_result and lexical_result["success"] else 0,
                        "response_time": lexical_result["response_time"] if lexical_result else 0,
                        "max_score": lexical_result.get("max_score", 0) if lexical_result and lexical_result["success"] else 0,
                        "quality": lexical_result.get("quality_assessment", "N/A") if lexical_result and lexical_result["success"] else "N/A"
                    },
                    "semantic": {
                        "available": semantic_result is not None and semantic_result["success"],
                        "results_count": semantic_result["results_count"] if semantic_result and semantic_result["success"] else 0,
                        "relevance": semantic_result.get("relevance_score", 0) if semantic_result and semantic_result["success"] else 0,
                        "response_time": semantic_result["response_time"] if semantic_result else 0,
                        "similarity_score": semantic_result.get("avg_score", 0) if semantic_result and semantic_result["success"] else 0,
                        "quality": semantic_result.get("quality_assessment", "N/A") if semantic_result and semantic_result["success"] else "N/A"
                    }
                }
                
                # Déterminer la meilleure approche
                lexical_quality_score = comparison["lexical"]["relevance"] * (1 + comparison["lexical"]["results_count"] / 20)
                semantic_quality_score = comparison["semantic"]["relevance"] * (1 + comparison["semantic"]["results_count"] / 20)
                
                if lexical_quality_score > semantic_quality_score * 1.2:
                    comparison["recommended_approach"] = "lexical"
                elif semantic_quality_score > lexical_quality_score * 1.2:
                    comparison["recommended_approach"] = "semantic"
                else:
                    comparison["recommended_approach"] = "hybrid"
                
                # Calcul score hybride potentiel
                if comparison["lexical"]["available"] and comparison["semantic"]["available"]:
                    comparison["hybrid_potential"] = (lexical_quality_score + semantic_quality_score) / 2
                else:
                    comparison["hybrid_potential"] = max(lexical_quality_score, semantic_quality_score)
                
                comparisons.append(comparison)
        
        return comparisons
    
    def generate_enhanced_recommendations(self, connectivity: Dict, collection_status: Dict, performance: Dict) -> List[str]:
        """Génère des recommandations améliorées spécifiques à Harena."""
        recommendations = []
        
        # Recommandations de connectivité
        if not connectivity["bonsai"]["connected"]:
            recommendations.append("🔧 CRITIQUE: Corriger la connectivité Bonsai Elasticsearch")
        if not connectivity["qdrant"]["connected"]:
            recommendations.append("🔧 CRITIQUE: Corriger la connectivité Qdrant")
        
        # Recommandations de collection/index
        if not collection_status.get("qdrant_ready"):
            if collection_status.get("needs_repair"):
                recommendations.append("🛠️ URGENT: Recréer la collection Qdrant avec --repair-collections")
            else:
                recommendations.append("📊 Synchroniser les données Qdrant via /api/v1/enrichment/dual/sync-user")
        
        if collection_status["elasticsearch_status"]["transaction_count"] == 0:
            recommendations.append("📊 URGENT: Synchroniser les données Elasticsearch via /api/v1/enrichment/dual/sync-user")
        
        # Recommandations de performance
        lexical_perf = performance.get("lexical_performance", {})
        semantic_perf = performance.get("semantic_performance", {})
        overall_health = performance.get("overall_metrics", {}).get("overall_health", "problématique")
        
        if lexical_perf.get("avg_response_time", 0) > 2.0:
            recommendations.append("⚡ PERFORMANCE: Optimiser les requêtes Elasticsearch (temps > 2s)")
        
        if semantic_perf.get("avg_response_time", 0) > 3.0:
            recommendations.append("⚡ PERFORMANCE: Optimiser les requêtes Qdrant ou dimensionner le cluster")
        
        # Recommandations de pertinence
        if lexical_perf.get("avg_relevance", 0) < 50:
            recommendations.append("🎯 PERTINENCE: Améliorer le mapping Elasticsearch et les requêtes")
        
        if semantic_perf.get("avg_relevance", 0) < 50:
            recommendations.append("🎯 PERTINENCE: Ajuster les seuils de similarité Qdrant")
        
        # Recommandations architecturales selon la santé globale
        if overall_health == "excellent":
            recommendations.append("🎉 EXCELLENT: Système optimal - Implémenter la recherche hybride en production")
        elif overall_health == "bon":
            recommendations.append("✅ BON: Système fonctionnel - Optimiser les réglages pour production")
        elif overall_health == "moyen":
            recommendations.append("⚠️ MOYEN: Améliorations nécessaires avant production")
        else:
            recommendations.append("🚨 PROBLÉMATIQUE: Corrections critiques requises")
        
        # Recommandations spécifiques par distribution de qualité
        lexical_quality = lexical_perf.get("quality_distribution", {})
        semantic_quality = semantic_perf.get("quality_distribution", {})
        
        if lexical_quality.get("excellent", 0) + lexical_quality.get("bon", 0) > lexical_quality.get("faible", 0) + lexical_quality.get("mauvais", 0):
            recommendations.append("✅ Recherche lexicale prête - Développer search_service avec priorité lexicale")
        
        if semantic_quality.get("excellent", 0) + semantic_quality.get("bon", 0) > semantic_quality.get("faible", 0) + semantic_quality.get("mauvais", 0):
            recommendations.append("✅ Recherche sémantique prête - Développer search_service avec support vectoriel")
        
        if not recommendations:
            recommendations.append("🎉 Configuration parfaite - Système prêt pour la production!")
        
        return recommendations
    
    def display_enhanced_report(self, report: Dict[str, Any]):
        """Affiche le rapport de validation Harena amélioré."""
        print("\n" + "=" * 85)
        print("📊 RAPPORT DE VALIDATION RECHERCHE HARENA FINANCE - VERSION CORRIGÉE")
        print("=" * 85)
        
        # Résumé exécutif
        summary = report["validation_summary"]
        connectivity = summary["connectivity"]
        collection_diagnostics = summary["collection_diagnostics"]
        performance = report["performance"]
        
        print(f"\n🏢 PLATEFORME: {summary['platform'].upper()}")
        print(f"👤 UTILISATEUR DE TEST: {summary['user_id']}")
        print(f"🔧 MODE RÉPARATION: {'Activé' if summary['repair_mode'] else 'Désactivé'}")
        print(f"📅 TIMESTAMP: {summary['timestamp']}")
        
        # Configuration et connectivité
        print(f"\n🔧 CONFIGURATION & CONNECTIVITÉ:")
        config = summary["configuration"]
        print(f"   Bonsai Elasticsearch: {'✅' if config['services']['bonsai'] else '❌'}")
        print(f"   Qdrant Vector DB: {'✅' if config['services']['qdrant'] else '❌'}")
        print(f"   OpenAI Embeddings: {'✅' if config['services']['openai'] else '❌'}")
        
        if connectivity["bonsai"]["connected"]:
            cluster_info = connectivity["bonsai"]["cluster_info"]
            response_time = connectivity["bonsai"]["response_time"]
            print(f"   Bonsai: ✅ {cluster_info['name']} v{cluster_info['version']} ({response_time:.2f}s)")
        else:
            print(f"   Bonsai: ❌ ({connectivity['bonsai']['error']})")
        
        if connectivity["qdrant"]["connected"]:
            version = connectivity["qdrant"]["version"]
            response_time = connectivity["qdrant"]["response_time"]
            print(f"   Qdrant: ✅ v{version} ({response_time:.2f}s)")
        else:
            print(f"   Qdrant: ❌ ({connectivity['qdrant']['error']})")
        
        # Statut des collections
        print(f"\n📊 STATUT DES COLLECTIONS:")
        es_status = collection_diagnostics["elasticsearch_status"]
        qdrant_status = collection_diagnostics["qdrant_status"]
        
        if es_status["exists"]:
            print(f"   Index Elasticsearch: ✅ ({es_status['transaction_count']} transactions)")
            print(f"   Mapping: {'✅' if es_status['mapping_ok'] else '⚠️'}")
        else:
            print(f"   Index Elasticsearch: ❌ (non trouvé)")
        
        if qdrant_status["exists"]:
            vector_config = qdrant_status["vector_config"]
            compatibility = "✅" if qdrant_status["compatible"] else "❌"
            print(f"   Collection Qdrant: ✅ ({qdrant_status['point_count']} points)")
            print(f"   Configuration vectorielle: {compatibility} {vector_config.get('size', 'unknown')} dims, {vector_config.get('distance', 'unknown')}")
        else:
            print(f"   Collection Qdrant: ❌ (non trouvée)")
        
        if collection_diagnostics.get("needs_repair"):
            print(f"   🛠️ RÉPARATION NÉCESSAIRE: {', '.join(collection_diagnostics.get('repair_actions', []))}")
        
        # Performances de recherche détaillées
        print(f"\n🔍 PERFORMANCES DE RECHERCHE:")
        
        lexical_perf = performance.get("lexical_performance", {})
        semantic_perf = performance.get("semantic_performance", {})
        
        if lexical_perf:
            success_rate = lexical_perf["success_rate"] * 100
            avg_time = lexical_perf["avg_response_time"]
            avg_relevance = lexical_perf["avg_relevance"]
            quality_dist = lexical_perf["quality_distribution"]
            print(f"   Recherche Lexicale: {success_rate:.1f}% succès, {avg_time:.2f}s, {avg_relevance:.1f}% pertinence")
            print(f"      Qualité: {quality_dist.get('excellent', 0)} excellent, {quality_dist.get('bon', 0)} bon, {quality_dist.get('moyen', 0)} moyen, {quality_dist.get('faible', 0)} faible, {quality_dist.get('mauvais', 0)} mauvais")
        
        if semantic_perf:
            success_rate = semantic_perf["success_rate"] * 100
            avg_time = semantic_perf["avg_response_time"]
            avg_relevance = semantic_perf["avg_relevance"]
            quality_dist = semantic_perf["quality_distribution"]
            print(f"   Recherche Sémantique: {success_rate:.1f}% succès, {avg_time:.2f}s, {avg_relevance:.1f}% pertinence")
            print(f"      Qualité: {quality_dist.get('excellent', 0)} excellent, {quality_dist.get('bon', 0)} bon, {quality_dist.get('moyen', 0)} moyen, {quality_dist.get('faible', 0)} faible, {quality_dist.get('mauvais', 0)} mauvais")
        
        # Comparaisons par type de transaction
        print(f"\n💰 ANALYSE PAR TYPE DE TRANSACTION:")
        comparisons = report["comparison"]
        
        for comp in comparisons:
            query = comp["query"]
            search_type = comp["search_type"]
            recommended = comp["recommended_approach"]
            expected = comp["expected_results"]
            
            lex_rel = comp["lexical"]["relevance"]
            lex_quality = comp["lexical"]["quality"]
            sem_rel = comp["semantic"]["relevance"]
            sem_quality = comp["semantic"]["quality"]
            
            icon = {"lexical": "🔍", "semantic": "🧠", "hybrid": "🔄"}[recommended]
            print(f"   {query} ({search_type}, attendu: {expected}): {icon} {recommended}")
            print(f"      Lexical: {lex_rel:.0f}% ({lex_quality}) | Sémantique: {sem_rel:.0f}% ({sem_quality})")
        
        # État de santé global
        overall_health = performance.get("overall_metrics", {}).get("overall_health", "problématique")
        overall_icons = {
            "excellent": "🎉 EXCELLENT",
            "bon": "✅ BON", 
            "moyen": "⚠️ MOYEN",
            "problématique": "🚨 PROBLÉMATIQUE"
        }
        
        print(f"\n🎯 ÉTAT DE SANTÉ GLOBAL: {overall_icons.get(overall_health, '❓ INCONNU')}")
        
        # Recommandations prioritaires
        print(f"\n💡 RECOMMANDATIONS PRIORITAIRES:")
        for i, rec in enumerate(report["recommendations"][:8], 1):  # Top 8 recommandations
            print(f"   {i}. {rec}")
        
        # Actions de suivi
        print(f"\n🚀 ACTIONS DE SUIVI:")
        overall = performance.get("overall_metrics", {})
        if overall.get("hybrid_ready"):
            print("   1. 🎯 Implémenter la recherche hybride en production")
            print("   2. 📊 Configurer le monitoring des performances")
            print("   3. 🧪 Tester la charge avec des volumes réels")
        elif overall.get("lexical_available") and overall.get("semantic_available"):
            print("   1. 🔧 Corriger les problèmes de pertinence identifiés")
            print("   2. ⚡ Optimiser les performances avant production")
            print("   3. 🧪 Re-tester après optimisations")
        elif overall.get("lexical_available") or overall.get("semantic_available"):
            method = "lexicale" if overall.get("lexical_available") else "sémantique"
            print(f"   1. ✅ Implémenter la recherche {method} en production")
            print("   2. 🔧 Corriger l'autre méthode de recherche")
            print("   3. 🔄 Évoluer vers recherche hybride")
        else:
            print("   1. 🚨 Corriger les problèmes de connectivité")
            print("   2. 📊 Synchroniser les données manquantes")
            print("   3. 🔧 Re-lancer la validation avec --repair-collections")
        
        print("=" * 85)


async def main():
    """Fonction principale avec options de réparation."""
    parser = argparse.ArgumentParser(description="Validateur de recherche Harena Finance corrigé")
    parser.add_argument("--user-id", type=int, default=34, help="ID de l'utilisateur de test")
    parser.add_argument("--repair-collections", action="store_true", help="Réparer automatiquement les collections problématiques")
    parser.add_argument("--save-report", action="store_true", help="Sauvegarder le rapport JSON détaillé")
    
    args = parser.parse_args()
    
    print("🚀 VALIDATION RECHERCHE HARENA FINANCE PLATFORM - VERSION CORRIGÉE")
    print("=" * 70)
    
    # Vérifier la configuration minimale
    if not any([settings.BONSAI_URL, settings.QDRANT_URL]):
        print("❌ ERREUR: Aucun service de recherche configuré")
        print("\n📋 Configurez au moins un service:")
        print("   BONSAI_URL=https://user:pass@cluster.bonsaisearch.net:443")
        print("   QDRANT_URL=https://cluster.qdrant.io")
        print("   QDRANT_API_KEY=your-api-key")
        print("   OPENAI_API_KEY=sk-...")
        return 1
    
    # Information sur le mode réparation
    if args.repair_collections:
        print("🛠️ MODE RÉPARATION ACTIVÉ - Les collections seront réparées automatiquement")
        print("⚠️ ATTENTION: Cela peut supprimer et recréer les collections Qdrant")
        
        confirm = input("Continuer? (y/N): ")
        if confirm.lower() != 'y':
            print("Validation annulée.")
            return 0
    
    # Créer le validateur corrigé
    validator = HarenaSearchValidatorFixed(
        test_user_id=args.user_id, 
        repair_mode=args.repair_collections
    )
    
    try:
        # Lancer la validation complète
        report = await validator.run_harena_search_validation()
        
        # Sauvegarder si demandé
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"harena_search_validation_fixed_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Rapport détaillé sauvegardé: {filename}")
        
        # Code de sortie selon les résultats
        performance = report.get("performance", {})
        overall_health = performance.get("overall_metrics", {}).get("overall_health", "problématique")
        
        if overall_health == "excellent":
            print("\n🎉 VALIDATION EXCELLENTE - Système prêt pour la production")
            return 0
        elif overall_health == "bon":
            print("\n✅ VALIDATION RÉUSSIE - Système fonctionnel avec optimisations possibles")
            return 0
        elif overall_health == "moyen":
            print("\n⚠️ VALIDATION PARTIELLE - Améliorations nécessaires")
            return 1
        else:
            print("\n🚨 VALIDATION PROBLÉMATIQUE - Corrections critiques requises")
            return 2
    
    except Exception as e:
        print(f"💥 ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)