"""
Client Bonsai compatible - Alternative pour contourner les problèmes de compatibilité.

Ce client utilise des requêtes HTTP directes pour interagir avec Bonsai
quand le client Elasticsearch officiel refuse la connexion.
"""
import logging
import time
import json
from typing import List, Dict, Any, Optional
import aiohttp
import asyncio

from config_service.config import settings

logger = logging.getLogger("search_service.bonsai")
metrics_logger = logging.getLogger("search_service.metrics.bonsai")


class BonsaiClient:
    """Client HTTP direct pour Bonsai quand le client Elasticsearch ne fonctionne pas."""
    
    def __init__(self):
        self.base_url = None
        self.session = None
        self.index_name = "harena_transactions"
        self._initialized = False
        self._connection_attempts = 0
        self._last_health_check = None
        
    async def initialize(self):
        """Initialise la connexion Bonsai avec HTTP direct."""
        logger.info("🔄 Initialisation du client Bonsai (HTTP direct)...")
        start_time = time.time()
        
        if not settings.BONSAI_URL:
            logger.error("❌ BONSAI_URL non configurée")
            return False
        
        # Préparer l'URL de base
        self.base_url = settings.BONSAI_URL.rstrip('/')
        
        # Masquer les credentials pour l'affichage
        safe_url = self._mask_credentials(self.base_url)
        logger.info(f"🔗 Connexion HTTP directe à Bonsai: {safe_url}")
        
        try:
            self._connection_attempts += 1
            logger.info(f"🔄 Tentative de connexion #{self._connection_attempts}")
            
            # Créer une session HTTP
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Test de connexion
            logger.info("⏱️ Test de connexion...")
            connection_start = time.time()
            
            async with self.session.get(self.base_url) as response:
                if response.status == 200:
                    info = await response.json()
                    connection_time = time.time() - connection_start
                    
                    logger.info(f"✅ Connexion réussie en {connection_time:.2f}s")
                    logger.info(f"📊 Version: {info.get('version', {}).get('number', 'N/A')}")
                    logger.info(f"🏷️ Cluster: {info.get('cluster_name', 'N/A')}")
                    
                    # Tester la santé
                    await self._check_cluster_health()
                    
                    # Configurer l'index
                    await self._setup_index()
                    
                    self._initialized = True
                    total_time = time.time() - start_time
                    logger.info(f"🎉 Client Bonsai initialisé avec succès en {total_time:.2f}s")
                    
                    return True
                else:
                    logger.error(f"❌ Erreur HTTP: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"💥 Erreur lors de l'initialisation Bonsai: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            return False
    
    async def _check_cluster_health(self):
        """Vérifie la santé du cluster via HTTP."""
        try:
            async with self.session.get(f"{self.base_url}/_cluster/health") as response:
                if response.status == 200:
                    health = await response.json()
                    status = health.get("status", "unknown")
                    
                    if status == "green":
                        logger.info("💚 Cluster Bonsai: Santé EXCELLENTE")
                    elif status == "yellow":
                        logger.warning("💛 Cluster Bonsai: Santé ACCEPTABLE")
                    elif status == "red":
                        logger.error("💔 Cluster Bonsai: Santé CRITIQUE")
                    
                    logger.info(f"📊 Nœuds: {health.get('number_of_nodes', 0)}")
                    logger.info(f"🗂️ Shards actifs: {health.get('active_shards', 0)}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Impossible de vérifier la santé: {e}")
    
    async def _setup_index(self):
        """Configure l'index pour les transactions."""
        try:
            # Vérifier si l'index existe
            async with self.session.head(f"{self.base_url}/{self.index_name}") as response:
                exists = response.status == 200
            
            if not exists:
                logger.info(f"📚 Création de l'index '{self.index_name}'...")
                
                # Mapping pour les transactions financières
                mapping = {
                    "mappings": {
                        "properties": {
                            "user_id": {"type": "integer"},
                            "transaction_id": {"type": "keyword"},
                            "amount": {"type": "float"},
                            "description": {
                                "type": "text",
                                "analyzer": "standard",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "merchant": {
                                "type": "text",
                                "analyzer": "standard",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "category": {"type": "keyword"},
                            "date": {"type": "date"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"}
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
                
                async with self.session.put(
                    f"{self.base_url}/{self.index_name}",
                    data=json.dumps(mapping)
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(f"✅ Index '{self.index_name}' créé avec succès")
                    else:
                        logger.error(f"❌ Erreur création index: {response.status}")
            else:
                logger.info(f"📚 Index '{self.index_name}' existe déjà")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration de l'index: {e}")
    
    def _mask_credentials(self, url: str) -> str:
        """Masque les credentials dans l'URL."""
        if "@" in url:
            parts = url.split("@")
            if len(parts) == 2:
                protocol_and_creds = parts[0]
                host_and_path = parts[1]
                
                if "://" in protocol_and_creds:
                    protocol = protocol_and_creds.split("://")[0]
                    return f"{protocol}://***:***@{host_and_path}"
        
        return url
    
    async def is_healthy(self) -> bool:
        """Vérifie si le client est sain et fonctionnel."""
        if not self.session or not self._initialized:
            return False
        
        try:
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/_cluster/health") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    health = await response.json()
                    status = health.get("status", "red")
                    is_healthy = status in ["green", "yellow"]
                    
                    self._last_health_check = {
                        "timestamp": time.time(),
                        "healthy": is_healthy,
                        "status": status,
                        "response_time": response_time
                    }
                    
                    return is_healthy
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    async def search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None,
        include_highlights: bool = True
    ) -> List[Dict[str, Any]]:
        """Recherche des transactions via HTTP direct."""
        if not self.session or not self._initialized:
            raise RuntimeError("Client Bonsai non initialisé")
        
        search_id = f"search_{int(time.time() * 1000)}"
        logger.info(f"🔍 [{search_id}] Recherche pour user {user_id}: '{query}' (limit: {limit})")
        
        start_time = time.time()
        
        try:
            # Construction de la requête de recherche
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}}
                        ],
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["description^2", "merchant", "category"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "wildcard": {
                                    "description": f"*{query.lower()}*"
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": limit,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"date": {"order": "desc"}}
                ]
            }
            
            # Ajouter les filtres si spécifiés
            if filters:
                search_body["query"]["bool"]["filter"] = []
                for field, value in filters.items():
                    if value is not None:
                        search_body["query"]["bool"]["filter"].append({"term": {field: value}})
            
            # Ajouter la mise en évidence si demandée
            if include_highlights:
                search_body["highlight"] = {
                    "fields": {
                        "description": {},
                        "merchant": {}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            
            # Exécuter la recherche
            logger.info(f"🎯 [{search_id}] Exécution recherche lexicale...")
            
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_search",
                data=json.dumps(search_body)
            ) as response:
                
                if response.status != 200:
                    logger.error(f"❌ Erreur recherche HTTP: {response.status}")
                    return []
                
                result = await response.json()
                query_time = time.time() - start_time
                
                # Analyser les résultats
                hits = result.get("hits", {}).get("hits", [])
                total_hits = result.get("hits", {}).get("total", {})
                
                if isinstance(total_hits, dict):
                    total_count = total_hits.get("value", 0)
                else:
                    total_count = total_hits
                
                # Formater les résultats
                results = []
                scores = []
                
                for hit in hits:
                    result_item = {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "source": hit["_source"]
                    }
                    
                    # Ajouter les highlights si disponibles
                    if "highlight" in hit:
                        result_item["highlights"] = hit["highlight"]
                    
                    results.append(result_item)
                    scores.append(hit["_score"])
                
                # Statistiques des scores
                if scores:
                    max_score = max(scores)
                    min_score = min(scores)
                    avg_score = sum(scores) / len(scores)
                else:
                    max_score = min_score = avg_score = 0
                
                # Logs de résultats
                logger.info(f"✅ [{search_id}] Recherche terminée en {query_time:.3f}s")
                logger.info(f"📊 [{search_id}] Résultats: {len(results)}/{total_count}")
                logger.info(f"🎯 [{search_id}] Scores: max={max_score:.3f}, min={min_score:.3f}, avg={avg_score:.3f}")
                
                # Métriques
                metrics_logger.info(
                    f"bonsai.search.success,"
                    f"user_id={user_id},"
                    f"query_time={query_time:.3f},"
                    f"results={len(results)},"
                    f"total={total_count},"
                    f"max_score={max_score:.3f}"
                )
                
                return results
                
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur recherche après {query_time:.3f}s: {e}")
            metrics_logger.error(f"bonsai.search.failed,user_id={user_id},time={query_time:.3f},error={type(e).__name__}")
            return []
    
    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Indexe un document."""
        if not self.session or not self._initialized:
            return False
        
        try:
            async with self.session.put(
                f"{self.base_url}/{self.index_name}/_doc/{doc_id}",
                data=json.dumps(document)
            ) as response:
                return response.status in [200, 201]
                
        except Exception as e:
            logger.error(f"❌ Erreur indexation: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document."""
        if not self.session or not self._initialized:
            return False
        
        try:
            async with self.session.delete(
                f"{self.base_url}/{self.index_name}/_doc/{doc_id}"
            ) as response:
                return response.status in [200, 404]  # 404 = déjà supprimé
                
        except Exception as e:
            logger.error(f"❌ Erreur suppression: {e}")
            return False
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Indexation en lot."""
        if not self.session or not self._initialized:
            return False
        
        if not documents:
            return True
        
        try:
            # Construire le payload bulk
            bulk_data = []
            for doc in documents:
                doc_id = doc.get("id") or doc.get("transaction_id")
                if not doc_id:
                    continue
                
                # Action header
                bulk_data.append(json.dumps({
                    "index": {
                        "_index": self.index_name,
                        "_id": doc_id
                    }
                }))
                
                # Document data
                bulk_data.append(json.dumps(doc))
            
            if not bulk_data:
                return True
            
            bulk_payload = "\n".join(bulk_data) + "\n"
            
            async with self.session.post(
                f"{self.base_url}/_bulk",
                data=bulk_payload,
                headers={"Content-Type": "application/x-ndjson"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    errors = result.get("errors", False)
                    
                    if not errors:
                        logger.info(f"✅ {len(documents)} documents indexés en lot")
                        return True
                    else:
                        logger.warning("⚠️ Erreurs lors de l'indexation en lot")
                        return False
                else:
                    logger.error(f"❌ Erreur bulk: {response.status}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Erreur indexation bulk: {e}")
            return False
    
    async def close(self):
        """Ferme la connexion."""
        if self.session:
            logger.info("🔒 Fermeture connexion Bonsai...")
            await self.session.close()
            self.session = None
            self._initialized = False
            logger.info("✅ Connexion Bonsai fermée")