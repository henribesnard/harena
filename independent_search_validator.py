#!/usr/bin/env python3
"""
Script de validation complètement indépendant des fonctionnalités de recherche.
Teste directement Bonsai Elasticsearch et Qdrant sans passer par search_service.
Valide que les recherches lexicales et sémantiques fonctionnent bien.

Usage:
    python independent_search_validator.py

Le script utilise la configuration centralisée de Harena via config_service.
"""
import asyncio
import aiohttp
import json
import time
import ssl
import hashlib
import sys
import os
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
    print("💡 Assurez-vous que le script est dans le bon répertoire du projet")
    sys.exit(1)

class IndependentSearchValidator:
    """Validateur indépendant des fonctionnalités de recherche."""
    
    def __init__(self):
        # Configuration depuis config_service centralisé
        self.bonsai_url = settings.BONSAI_URL
        self.qdrant_url = settings.QDRANT_URL
        self.qdrant_api_key = settings.QDRANT_API_KEY
        
        # Utilisateur de test (modifiez selon vos données)
        self.test_user_id = 34
        
        # Index/Collection à tester (selon votre architecture Harena)
        self.elasticsearch_index = "harena_transactions"  # Index Elasticsearch/Bonsai 
        self.qdrant_collection = "financial_transactions"  # Collection Qdrant 
        
        # Requêtes de test
        self.test_queries = [
            {
                "query": "virement",
                "description": "Terme financier exact",
                "expected_type": "lexical_perfect"
            },
            {
                "query": "restaurant",
                "description": "Terme commercial courant",
                "expected_type": "both_good"
            },
            {
                "query": "carte bancaire",
                "description": "Expression composée",
                "expected_type": "lexical_better"
            },
            {
                "query": "achat nourriture",
                "description": "Concept alimentaire",
                "expected_type": "semantic_better"
            },
            {
                "query": "transport déplacement",
                "description": "Concept mobilité",
                "expected_type": "semantic_better"
            }
        ]
        
        self.results = {
            "bonsai_lexical": [],
            "qdrant_semantic": [],
            "comparison": []
        }
    
    def log(self, level: str, message: str):
        """Log avec format coloré."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "TEST": "🧪", "SEARCH": "🔍"}
        icon = icons.get(level, "📝")
        print(f"[{timestamp}] {icon} {message}")
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Lance la validation complète des fonctionnalités de recherche."""
        self.log("INFO", "🚀 VALIDATION INDÉPENDANTE DES RECHERCHES")
        print("=" * 80)
        
        # 1. Vérifier la configuration
        self.log("INFO", "📋 Étape 1: Vérification de la configuration")
        config_ok = await self.verify_configuration()
        
        if not config_ok:
            return {"error": "Configuration incomplète", "config_issues": True}
        
        # 2. Tester la connectivité de base
        self.log("INFO", "🌐 Étape 2: Test de connectivité")
        connectivity = await self.test_basic_connectivity()
        
        if not (connectivity["bonsai"]["connected"] or connectivity["qdrant"]["connected"]):
            return {"error": "Aucun service accessible", "connectivity": connectivity}
        
        # 3. Vérifier les index/collections
        self.log("INFO", "📊 Étape 3: Vérification des données")
        data_status = await self.verify_data_availability()
        
        # 4. Tests de recherche lexicale (Bonsai)
        if connectivity["bonsai"]["connected"]:
            self.log("INFO", "🔍 Étape 4: Tests de recherche lexicale (Bonsai)")
            await self.test_bonsai_lexical_search()
        
        # 5. Tests de recherche sémantique (Qdrant)
        if connectivity["qdrant"]["connected"]:
            self.log("INFO", "🧠 Étape 5: Tests de recherche sémantique (Qdrant)")
            await self.test_qdrant_semantic_search()
        
        # 6. Comparaison des approches
        self.log("INFO", "⚖️ Étape 6: Comparaison des approches")
        comparison_results = await self.compare_search_approaches()
        
        # 7. Compilation du rapport final
        final_report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "configuration": config_ok,
                "connectivity": connectivity,
                "data_status": data_status
            },
            "search_results": {
                "bonsai_lexical": self.results["bonsai_lexical"],
                "qdrant_semantic": self.results["qdrant_semantic"],
                "comparison": comparison_results
            }
        }
        
        # 8. Analyse et recommandations
        final_report["analysis"] = self.analyze_search_performance(final_report)
        
        # 9. Affichage du rapport
        self.display_validation_report(final_report)
        
        return final_report
    
    async def verify_configuration(self) -> bool:
        """Vérifie que la configuration est complète."""
        issues = []
        
        if not self.bonsai_url or "your-" in self.bonsai_url:
            issues.append("BONSAI_URL non configurée")
            self.log("ERROR", "❌ BONSAI_URL manquante ou invalide")
        else:
            self.log("SUCCESS", f"✅ BONSAI_URL configurée: {self.mask_url(self.bonsai_url)}")
        
        if not self.qdrant_url or "your-" in self.qdrant_url:
            issues.append("QDRANT_URL non configurée")
            self.log("ERROR", "❌ QDRANT_URL manquante ou invalide")
        else:
            self.log("SUCCESS", f"✅ QDRANT_URL configurée: {self.qdrant_url}")
        
        if not self.qdrant_api_key or "your-" in self.qdrant_api_key:
            issues.append("QDRANT_API_KEY non configurée")
            self.log("ERROR", "❌ QDRANT_API_KEY manquante ou invalide")
        else:
            self.log("SUCCESS", "✅ QDRANT_API_KEY configurée")
        
        if issues:
            self.log("WARNING", f"⚠️ Problèmes de configuration: {', '.join(issues)}")
            return False
        
        return True
    
    def mask_url(self, url: str) -> str:
        """Masque les credentials dans l'URL."""
        if "@" in url:
            parts = url.split("@")
            if len(parts) == 2:
                protocol_part = parts[0]
                host_part = parts[1]
                if "://" in protocol_part:
                    protocol = protocol_part.split("://")[0]
                    return f"{protocol}://***:***@{host_part}"
        return url
    
    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Teste la connectivité de base vers les services."""
        connectivity = {
            "bonsai": {"connected": False, "response_time": 0, "error": None, "cluster_info": {}},
            "qdrant": {"connected": False, "response_time": 0, "error": None, "version": ""}
        }
        
        # Test Bonsai
        if self.bonsai_url and "your-" not in self.bonsai_url:
            try:
                ssl_context = ssl.create_default_context()
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                timeout = aiohttp.ClientTimeout(total=10)
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    start_time = time.time()
                    async with session.get(self.bonsai_url) as response:
                        connectivity["bonsai"]["response_time"] = time.time() - start_time
                        
                        if response.status == 200:
                            connectivity["bonsai"]["connected"] = True
                            data = await response.json()
                            connectivity["bonsai"]["cluster_info"] = {
                                "name": data.get('cluster_name', 'unknown'),
                                "version": data.get('version', {}).get('number', 'unknown')
                            }
                            self.log("SUCCESS", f"✅ Bonsai connecté: {data.get('cluster_name', 'unknown')}")
                        else:
                            connectivity["bonsai"]["error"] = f"HTTP {response.status}"
                            self.log("ERROR", f"❌ Bonsai: HTTP {response.status}")
            except Exception as e:
                connectivity["bonsai"]["error"] = str(e)
                self.log("ERROR", f"❌ Bonsai inaccessible: {e}")
        
        # Test Qdrant
        if self.qdrant_url and "your-" not in self.qdrant_url:
            try:
                headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}
                timeout = aiohttp.ClientTimeout(total=10)
                
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
        
        return connectivity
    
    async def verify_data_availability(self) -> Dict[str, Any]:
        """Vérifie la disponibilité des données dans les index/collections."""
        data_status = {
            "bonsai_index": {"exists": False, "document_count": 0, "error": None},
            "qdrant_collection": {"exists": False, "point_count": 0, "error": None}
        }
        
        # Vérifier l'index Bonsai (harena_transactions)
        if self.bonsai_url and "your-" not in self.bonsai_url:
            try:
                ssl_context = ssl.create_default_context()
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                timeout = aiohttp.ClientTimeout(total=10)
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    # Vérifier l'existence de l'index
                    async with session.head(f"{self.bonsai_url}/{self.elasticsearch_index}") as response:
                        if response.status == 200:
                            data_status["bonsai_index"]["exists"] = True
                            
                            # Compter les documents pour l'utilisateur de test
                            count_query = {
                                "query": {
                                    "bool": {
                                        "must": [
                                            {"term": {"user_id": self.test_user_id}}
                                        ]
                                    }
                                }
                            }
                            
                            async with session.post(
                                f"{self.bonsai_url}/{self.elasticsearch_index}/_count",
                                json=count_query
                            ) as count_response:
                                if count_response.status == 200:
                                    count_data = await count_response.json()
                                    doc_count = count_data.get("count", 0)
                                    data_status["bonsai_index"]["document_count"] = doc_count
                                    self.log("SUCCESS", f"✅ Index Bonsai ({self.elasticsearch_index}): {doc_count} documents pour user {self.test_user_id}")
                                else:
                                    self.log("WARNING", f"⚠️ Impossible de compter les documents Bonsai")
                        else:
                            data_status["bonsai_index"]["error"] = f"Index not found (HTTP {response.status})"
                            self.log("WARNING", f"⚠️ Index {self.elasticsearch_index} non trouvé")
            except Exception as e:
                data_status["bonsai_index"]["error"] = str(e)
                self.log("ERROR", f"❌ Erreur vérification index Bonsai: {e}")
        
        # Vérifier la collection Qdrant (financial_transactions)
        if self.qdrant_url and "your-" not in self.qdrant_url:
            try:
                headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}
                timeout = aiohttp.ClientTimeout(total=10)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Vérifier l'existence de la collection
                    async with session.get(
                        f"{self.qdrant_url}/collections/{self.qdrant_collection}", 
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            collection_info = await response.json()
                            data_status["qdrant_collection"]["exists"] = True
                            
                            point_count = collection_info.get("result", {}).get("points_count", 0)
                            data_status["qdrant_collection"]["point_count"] = point_count
                            
                            self.log("SUCCESS", f"✅ Collection Qdrant ({self.qdrant_collection}): {point_count} points")
                        else:
                            data_status["qdrant_collection"]["error"] = f"Collection not found (HTTP {response.status})"
                            self.log("WARNING", f"⚠️ Collection {self.qdrant_collection} non trouvée")
                            
                            # Proposer de créer la collection
                            if response.status == 404:
                                self.log("INFO", "💡 Voulez-vous créer la collection manquante ?")
                                await self.create_missing_qdrant_collection(session, headers)
            except Exception as e:
                data_status["qdrant_collection"]["error"] = str(e)
                self.log("ERROR", f"❌ Erreur vérification collection Qdrant: {e}")
        
        return data_status
    
    async def create_missing_qdrant_collection(self, session: aiohttp.ClientSession, headers: Dict[str, str]):
        """Crée la collection Qdrant manquante."""
        try:
            self.log("INFO", f"🔧 Tentative de création de la collection {self.qdrant_collection}")
            
            # Configuration de la collection (dimension standard pour OpenAI text-embedding-3-small)
            collection_config = {
                "vectors": {
                    "size": 1536,  # Dimension pour OpenAI text-embedding-3-small
                    "distance": "Cosine"
                }
            }
            
            async with session.put(
                f"{self.qdrant_url}/collections/{self.qdrant_collection}",
                json=collection_config,
                headers=headers
            ) as response:
                if response.status in [200, 201]:
                    self.log("SUCCESS", f"✅ Collection {self.qdrant_collection} créée avec succès")
                else:
                    response_text = await response.text()
                    self.log("ERROR", f"❌ Échec création collection: HTTP {response.status} - {response_text}")
        except Exception as e:
            self.log("ERROR", f"❌ Erreur lors de la création de collection: {e}")
    
    async def test_bonsai_lexical_search(self):
        """Teste les recherches lexicales sur Bonsai Elasticsearch."""
        if not self.bonsai_url or "your-" in self.bonsai_url:
            self.log("WARNING", "⚠️ Bonsai non configuré, tests lexicaux ignorés")
            return
        
        ssl_context = ssl.create_default_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=15)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for test_query in self.test_queries:
                query_text = test_query["query"]
                description = test_query["description"]
                
                self.log("SEARCH", f"🔍 Test lexical: '{query_text}' ({description})")
                
                try:
                    # Construire la requête Elasticsearch optimisée pour Bonsai
                    search_body = {
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"user_id": self.test_user_id}}
                                ],
                                "should": [
                                    {
                                        "multi_match": {
                                            "query": query_text,
                                            "fields": [
                                                "searchable_text^3",
                                                "primary_description^2", 
                                                "merchant_name^2",
                                                "category_name"
                                            ],
                                            "type": "best_fields",
                                            "operator": "or",
                                            "fuzziness": "AUTO"
                                        }
                                    },
                                    {
                                        "match": {
                                            "primary_description": {
                                                "query": query_text,
                                                "boost": 2.0
                                            }
                                        }
                                    }
                                ],
                                "minimum_should_match": 1
                            }
                        },
                        "size": 10,
                        "sort": [
                            {"_score": {"order": "desc"}},
                            {"transaction_date": {"order": "desc", "unmapped_type": "date"}}
                        ],
                        "highlight": {
                            "fields": {
                                "searchable_text": {},
                                "primary_description": {},
                                "merchant_name": {}
                            }
                        }
                    }
                    
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
                            
                            if isinstance(total, dict):
                                total_count = total.get("value", 0)
                            else:
                                total_count = total
                            
                            result = {
                                "query": query_text,
                                "description": description,
                                "success": True,
                                "results_count": len(hits),
                                "total_found": total_count,
                                "response_time": response_time,
                                "max_score": max([hit["_score"] for hit in hits]) if hits else 0,
                                "avg_score": sum([hit["_score"] for hit in hits]) / len(hits) if hits else 0,
                                "sample_results": [
                                    {
                                        "score": hit["_score"],
                                        "description": hit["_source"].get("primary_description", ""),
                                        "merchant": hit["_source"].get("merchant_name", ""),
                                        "amount": hit["_source"].get("amount", 0)
                                    }
                                    for hit in hits[:3]
                                ]
                            }
                            
                            self.results["bonsai_lexical"].append(result)
                            self.log("SUCCESS", f"   ✅ {len(hits)} résultats, score max: {result['max_score']:.2f}, temps: {response_time:.2f}s")
                            
                        else:
                            error_result = {
                                "query": query_text,
                                "description": description,
                                "success": False,
                                "error": f"HTTP {response.status}",
                                "response_time": response_time
                            }
                            self.results["bonsai_lexical"].append(error_result)
                            self.log("ERROR", f"   ❌ HTTP {response.status}")
                
                except Exception as e:
                    error_result = {
                        "query": query_text,
                        "description": description,
                        "success": False,
                        "error": str(e),
                        "response_time": 0
                    }
                    self.results["bonsai_lexical"].append(error_result)
                    self.log("ERROR", f"   ❌ Exception: {e}")
                
                # Petit délai entre les requêtes
                await asyncio.sleep(0.3)
    
    async def test_qdrant_semantic_search(self):
        """Teste les recherches sémantiques sur Qdrant."""
        if not self.qdrant_url or "your-" not in self.qdrant_url:
            self.log("WARNING", "⚠️ Qdrant non configuré, tests sémantiques ignorés")
            return
        
        headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}
        timeout = aiohttp.ClientTimeout(total=15)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for test_query in self.test_queries:
                query_text = test_query["query"]
                description = test_query["description"]
                
                self.log("SEARCH", f"🧠 Test sémantique: '{query_text}' ({description})")
                
                try:
                    # Générer un embedding simple (simulé pour le test)
                    # En production, vous utiliseriez OpenAI, Cohere, ou un autre service d'embeddings
                    fake_embedding = await self.generate_fake_embedding(query_text)
                    
                    # Construire la requête Qdrant
                    search_body = {
                        "vector": fake_embedding,
                        "filter": {
                            "must": [
                                {
                                    "key": "user_id",
                                    "match": {"value": self.test_user_id}
                                }
                            ]
                        },
                        "limit": 10,
                        "with_payload": True,
                        "with_vector": False
                    }
                    
                    start_time = time.time()
                    async with session.post(
                        f"{self.qdrant_url}/collections/{self.qdrant_collection}/points/search",
                        json=search_body,
                        headers=headers
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            points = data.get("result", [])
                            
                            result = {
                                "query": query_text,
                                "description": description,
                                "success": True,
                                "results_count": len(points),
                                "response_time": response_time,
                                "max_score": max([point["score"] for point in points]) if points else 0,
                                "avg_score": sum([point["score"] for point in points]) / len(points) if points else 0,
                                "sample_results": [
                                    {
                                        "score": point["score"],
                                        "description": point["payload"].get("primary_description", ""),
                                        "merchant": point["payload"].get("merchant_name", ""),
                                        "amount": point["payload"].get("amount", 0)
                                    }
                                    for point in points[:3]
                                ]
                            }
                            
                            self.results["qdrant_semantic"].append(result)
                            self.log("SUCCESS", f"   ✅ {len(points)} résultats, score max: {result['max_score']:.2f}, temps: {response_time:.2f}s")
                            
                        else:
                            error_result = {
                                "query": query_text,
                                "description": description,
                                "success": False,
                                "error": f"HTTP {response.status}",
                                "response_time": response_time
                            }
                            self.results["qdrant_semantic"].append(error_result)
                            self.log("ERROR", f"   ❌ HTTP {response.status}")
                
                except Exception as e:
                    error_result = {
                        "query": query_text,
                        "description": description,
                        "success": False,
                        "error": str(e),
                        "response_time": 0
                    }
                    self.results["qdrant_semantic"].append(error_result)
                    self.log("ERROR", f"   ❌ Exception: {e}")
                
                # Petit délai entre les requêtes
                await asyncio.sleep(0.3)
    
    async def generate_fake_embedding(self, text: str) -> List[float]:
        """Génère un embedding factice pour les tests (à remplacer par un vrai service)."""
        # Ceci est un embedding factice basé sur un hash simple
        # En production, utilisez OpenAI, Cohere, ou un autre service d'embeddings
        
        # Créer un hash basé sur le texte
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convertir en embedding de 384 dimensions (taille courante pour les embeddings)
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0 - 0.5  # Normaliser entre -0.5 et 0.5
            embedding.append(value)
        
        # Compléter à 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        
        return embedding[:384]
    
    async def compare_search_approaches(self) -> List[Dict[str, Any]]:
        """Compare les résultats des approches lexicale et sémantique."""
        comparisons = []
        
        for query_data in self.test_queries:
            query = query_data["query"]
            
            # Trouver les résultats correspondants
            lexical_result = next((r for r in self.results["bonsai_lexical"] if r["query"] == query), None)
            semantic_result = next((r for r in self.results["qdrant_semantic"] if r["query"] == query), None)
            
            if lexical_result or semantic_result:
                comparison = {
                    "query": query,
                    "description": query_data["description"],
                    "expected_type": query_data["expected_type"],
                    "lexical": {
                        "success": lexical_result["success"] if lexical_result else False,
                        "count": lexical_result["results_count"] if lexical_result and lexical_result["success"] else 0,
                        "max_score": lexical_result["max_score"] if lexical_result and lexical_result["success"] else 0,
                        "response_time": lexical_result["response_time"] if lexical_result else 0
                    },
                    "semantic": {
                        "success": semantic_result["success"] if semantic_result else False,
                        "count": semantic_result["results_count"] if semantic_result and semantic_result["success"] else 0,
                        "max_score": semantic_result["max_score"] if semantic_result and semantic_result["success"] else 0,
                        "response_time": semantic_result["response_time"] if semantic_result else 0
                    }
                }
                
                # Déterminer le meilleur
                lexical_score = comparison["lexical"]["count"] * comparison["lexical"]["max_score"]
                semantic_score = comparison["semantic"]["count"] * comparison["semantic"]["max_score"]
                
                if lexical_score > semantic_score:
                    comparison["winner"] = "lexical"
                elif semantic_score > lexical_score:
                    comparison["winner"] = "semantic"
                else:
                    comparison["winner"] = "tie"
                
                comparisons.append(comparison)
                
                self.log("INFO", f"🔍 '{query}': Lexical={comparison['lexical']['count']} résultats, "
                               f"Semantic={comparison['semantic']['count']} résultats → Meilleur: {comparison['winner']}")
        
        return comparisons
    
    def analyze_search_performance(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les performances de recherche."""
        analysis = {
            "lexical_performance": {},
            "semantic_performance": {},
            "overall_assessment": {},
            "recommendations": []
        }
        
        # Analyser les performances lexicales
        lexical_results = report["search_results"]["bonsai_lexical"]
        if lexical_results:
            successful_lexical = [r for r in lexical_results if r["success"]]
            analysis["lexical_performance"] = {
                "success_rate": len(successful_lexical) / len(lexical_results),
                "avg_response_time": sum(r["response_time"] for r in successful_lexical) / len(successful_lexical) if successful_lexical else 0,
                "avg_results_count": sum(r["results_count"] for r in successful_lexical) / len(successful_lexical) if successful_lexical else 0,
                "avg_max_score": sum(r["max_score"] for r in successful_lexical) / len(successful_lexical) if successful_lexical else 0
            }
        
        # Analyser les performances sémantiques
        semantic_results = report["search_results"]["qdrant_semantic"]
        if semantic_results:
            successful_semantic = [r for r in semantic_results if r["success"]]
            analysis["semantic_performance"] = {
                "success_rate": len(successful_semantic) / len(semantic_results),
                "avg_response_time": sum(r["response_time"] for r in successful_semantic) / len(successful_semantic) if successful_semantic else 0,
                "avg_results_count": sum(r["results_count"] for r in successful_semantic) / len(successful_semantic) if successful_semantic else 0,
                "avg_max_score": sum(r["max_score"] for r in successful_semantic) / len(successful_semantic) if successful_semantic else 0
            }
        
        # Évaluation globale
        lexical_ok = analysis["lexical_performance"].get("success_rate", 0) > 0.7
        semantic_ok = analysis["semantic_performance"].get("success_rate", 0) > 0.7
        
        if lexical_ok and semantic_ok:
            analysis["overall_assessment"]["status"] = "excellent"
            analysis["overall_assessment"]["message"] = "Les deux approches fonctionnent bien"
            analysis["recommendations"].append("✅ Vous pouvez implémenter une recherche hybride")
        elif lexical_ok:
            analysis["overall_assessment"]["status"] = "lexical_only"
            analysis["overall_assessment"]["message"] = "Seule la recherche lexicale fonctionne"
            analysis["recommendations"].append("🔧 Corriger la recherche sémantique (Qdrant)")
        elif semantic_ok:
            analysis["overall_assessment"]["status"] = "semantic_only"
            analysis["overall_assessment"]["message"] = "Seule la recherche sémantique fonctionne"
            analysis["recommendations"].append("🔧 Corriger la recherche lexicale (Bonsai)")
        else:
            analysis["overall_assessment"]["status"] = "failure"
            analysis["overall_assessment"]["message"] = "Aucune approche ne fonctionne"
            analysis["recommendations"].append("🚨 Vérifier les configurations et la connectivité")
        
        # Recommandations spécifiques
        if analysis["lexical_performance"].get("avg_response_time", 0) > 2.0:
            analysis["recommendations"].append("⚡ Optimiser les performances Elasticsearch")
        
        if analysis["semantic_performance"].get("avg_response_time", 0) > 3.0:
            analysis["recommendations"].append("⚡ Optimiser les performances Qdrant")
        
        return analysis
    
    def display_validation_report(self, report: Dict[str, Any]):
        """Affiche un rapport de validation détaillé."""
        print("\n" + "=" * 80)
        print("📊 RAPPORT DE VALIDATION COMPLET")
        print("=" * 80)
        
        # Résumé de la configuration
        config = report["test_summary"]["configuration"]
        connectivity = report["test_summary"]["connectivity"]
        data_status = report["test_summary"]["data_status"]
        
        print(f"\n🔧 CONFIGURATION:")
        print(f"   Configuration complète: {'✅' if config else '❌'}")
        print(f"   Bonsai accessible: {'✅' if connectivity['bonsai']['connected'] else '❌'}")
        print(f"   Qdrant accessible: {'✅' if connectivity['qdrant']['connected'] else '❌'}")
        
        # Détails de connectivité
        if connectivity["bonsai"]["connected"]:
            cluster_name = connectivity["bonsai"]["cluster_info"].get("name", "unknown")
            response_time = connectivity["bonsai"]["response_time"]
            print(f"   Bonsai cluster: {cluster_name} ({response_time:.2f}s)")
        
        if connectivity["qdrant"]["connected"]:
            version = connectivity["qdrant"]["version"]
            response_time = connectivity["qdrant"]["response_time"]
            print(f"   Qdrant version: {version} ({response_time:.2f}s)")
        
        # Statut des données
        print(f"\n📊 DONNÉES:")
        if data_status["bonsai_index"]["exists"]:
            doc_count = data_status["bonsai_index"]["document_count"]
            print(f"   Index Bonsai: ✅ ({doc_count} documents)")
        else:
            print(f"   Index Bonsai: ❌ (non trouvé)")
        
        if data_status["qdrant_collection"]["exists"]:
            point_count = data_status["qdrant_collection"]["point_count"]
            print(f"   Collection Qdrant: ✅ ({point_count} points)")
        else:
            print(f"   Collection Qdrant: ❌ (non trouvée)")
        
        # Résultats de recherche
        print(f"\n🔍 RÉSULTATS DE RECHERCHE:")
        
        lexical_results = report["search_results"]["bonsai_lexical"]
        semantic_results = report["search_results"]["qdrant_semantic"]
        
        if lexical_results:
            successful_lexical = [r for r in lexical_results if r["success"]]
            print(f"   Recherche lexicale: {len(successful_lexical)}/{len(lexical_results)} tests réussis")
            
            if successful_lexical:
                avg_time = sum(r["response_time"] for r in successful_lexical) / len(successful_lexical)
                avg_results = sum(r["results_count"] for r in successful_lexical) / len(successful_lexical)
                print(f"     Temps moyen: {avg_time:.2f}s")
                print(f"     Résultats moyens: {avg_results:.1f}")
        
        if semantic_results:
            successful_semantic = [r for r in semantic_results if r["success"]]
            print(f"   Recherche sémantique: {len(successful_semantic)}/{len(semantic_results)} tests réussis")
            
            if successful_semantic:
                avg_time = sum(r["response_time"] for r in successful_semantic) / len(successful_semantic)
                avg_results = sum(r["results_count"] for r in successful_semantic) / len(successful_semantic)
                print(f"     Temps moyen: {avg_time:.2f}s")
                print(f"     Résultats moyens: {avg_results:.1f}")
        
        # Comparaisons détaillées
        comparisons = report["search_results"]["comparison"]
        if comparisons:
            print(f"\n⚖️ COMPARAISONS PAR REQUÊTE:")
            for comp in comparisons:
                query = comp["query"]
                winner = comp["winner"]
                lex_count = comp["lexical"]["count"]
                sem_count = comp["semantic"]["count"]
                
                winner_icon = {"lexical": "🔍", "semantic": "🧠", "tie": "🤝"}[winner]
                print(f"   '{query}': {winner_icon} {winner} (Lex:{lex_count}, Sem:{sem_count})")
        
        # Analyse et recommandations
        analysis = report["analysis"]
        status = analysis["overall_assessment"]["status"]
        message = analysis["overall_assessment"]["message"]
        
        print(f"\n🎯 ÉVALUATION GLOBALE:")
        status_icons = {
            "excellent": "🎉",
            "lexical_only": "⚠️",
            "semantic_only": "⚠️", 
            "failure": "🚨"
        }
        print(f"   {status_icons.get(status, '❓')} {message}")
        
        print(f"\n💡 RECOMMANDATIONS:")
        for recommendation in analysis["recommendations"]:
            print(f"   {recommendation}")
        
        print("\n" + "=" * 80)


async def main():
    """Fonction principale pour lancer la validation."""
    print("🚀 DÉMARRAGE DE LA VALIDATION INDÉPENDANTE")
    print("=" * 50)
    
    # Afficher la configuration chargée
    print(f"📋 Configuration Harena:")
    print(f"   Environnement: {settings.ENVIRONMENT}")
    print(f"   BONSAI_URL: {'✅ Configuré' if settings.BONSAI_URL else '❌ Non configuré'}")
    print(f"   QDRANT_URL: {'✅ Configuré' if settings.QDRANT_URL else '❌ Non configuré'}")
    print(f"   QDRANT_API_KEY: {'✅ Configuré' if settings.QDRANT_API_KEY else '❌ Non configuré'}")
    print()
    
    # Vérifier qu'au moins un service est configuré
    if not any([settings.BONSAI_URL, settings.QDRANT_URL]):
        print("❌ ERREUR: Aucun service de recherche configuré")
        print("\n📋 Vérifiez votre fichier .env ou vos variables d'environnement:")
        print("   BONSAI_URL=https://user:pass@your-cluster.bonsaisearch.net:443")
        print("   QDRANT_URL=https://your-cluster.qdrant.io")
        print("   QDRANT_API_KEY=your-api-key")
        return 1
    
    # Créer le validateur
    validator = IndependentSearchValidator()
    
    try:
        # Lancer la validation complète
        report = await validator.run_complete_validation()
        
        # Sauvegarder le rapport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"search_validation_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Rapport sauvegardé: {report_file}")
        
        # Retourner le code de sortie approprié
        analysis = report.get("analysis", {})
        status = analysis.get("overall_assessment", {}).get("status", "failure")
        
        if status == "excellent":
            print("🎉 VALIDATION RÉUSSIE - Tous les services fonctionnent parfaitement")
            return 0
        elif status in ["lexical_only", "semantic_only"]:
            print("⚠️ VALIDATION PARTIELLE - Un service fonctionne")
            return 1
        else:
            print("🚨 VALIDATION ÉCHOUÉE - Problèmes critiques détectés")
            return 2
            
    except Exception as e:
        print(f"💥 ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    # Configuration pour éviter les warnings SSL
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Lancer la validation
    exit_code = asyncio.run(main())