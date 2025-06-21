"""
Client Elasticsearch hybride qui utilise le client officiel ou Bonsai HTTP selon la compatibilité.
"""
import logging
import json
import time
from typing import List, Dict, Any, Optional

from config_service.config import settings

logger = logging.getLogger("search_service.elasticsearch")
metrics_logger = logging.getLogger("search_service.metrics.elasticsearch")


class HybridElasticClient:
    """Client hybride qui choisit automatiquement entre Elasticsearch officiel et Bonsai HTTP."""
    
    def __init__(self):
        self.client = None
        self.bonsai_client = None
        self.client_type = None  # 'elasticsearch' ou 'bonsai'
        self.index_name = "harena_transactions"
        self._initialized = False
        self._connection_attempts = 0
        self._last_health_check = None
        
    async def initialize(self):
        """Initialise la connexion en essayant d'abord Elasticsearch, puis Bonsai."""
        logger.info("🔄 Initialisation du client Elasticsearch hybride...")
        start_time = time.time()
        
        if not settings.BONSAI_URL:
            logger.error("❌ BONSAI_URL non configurée")
            return False
        
        # Masquer les credentials pour l'affichage
        safe_url = self._mask_credentials(settings.BONSAI_URL)
        logger.info(f"🔗 Connexion à: {safe_url}")
        
        # Essayer d'abord le client Elasticsearch officiel
        elasticsearch_success = await self._try_elasticsearch_client()
        
        if elasticsearch_success:
            self.client_type = 'elasticsearch'
            self._initialized = True
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Elasticsearch officiel initialisé en {total_time:.2f}s")
            return True
        
        # Si échec, essayer le client Bonsai HTTP
        logger.info("🔄 Tentative avec client Bonsai HTTP...")
        bonsai_success = await self._try_bonsai_client()
        
        if bonsai_success:
            self.client_type = 'bonsai'
            self._initialized = True
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Bonsai HTTP initialisé en {total_time:.2f}s")
            return True
        
        # Les deux ont échoué
        logger.error("❌ Impossible d'initialiser un client de recherche")
        return False
    
    async def _try_elasticsearch_client(self):
        """Essaie d'initialiser le client Elasticsearch officiel."""
        try:
            from elasticsearch import AsyncElasticsearch
            
            logger.info("🔍 Tentative avec client Elasticsearch officiel...")
            
            self.client = AsyncElasticsearch(
                [settings.BONSAI_URL],
                verify_certs=True,
                ssl_show_warn=False,
                max_retries=3,
                retry_on_timeout=True,
                request_timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Test de connexion
            info = await self.client.info()
            logger.info(f"✅ Client Elasticsearch: {info['cluster_name']} v{info['version']['number']}")
            
            # Test de santé
            health = await self.client.cluster.health()
            logger.info(f"💚 Santé: {health['status']}")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Client Elasticsearch échoué: {e}")
            if "UnsupportedProductError" in str(e):
                logger.info("💡 Bonsai détecté comme incompatible avec le client standard")
            
            if self.client:
                try:
                    await self.client.close()
                except:
                    pass
                self.client = None
            
            return False
    
    async def _try_bonsai_client(self):
        """Essaie d'initialiser le client Bonsai HTTP."""
        try:
            from .bonsai_client import BonsaiClient
            
            logger.info("🌐 Tentative avec client Bonsai HTTP...")
            
            self.bonsai_client = BonsaiClient()
            success = await self.bonsai_client.initialize()
            
            if success:
                logger.info("✅ Client Bonsai HTTP opérationnel")
                return True
            else:
                logger.error("❌ Client Bonsai HTTP échoué")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur client Bonsai: {e}")
            return False
    
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
        if not self._initialized:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                health = await self.client.cluster.health()
                status = health.get("status", "red")
                is_healthy = status in ["green", "yellow"]
                
                self._last_health_check = {
                    "timestamp": time.time(),
                    "healthy": is_healthy,
                    "status": status,
                    "client_type": "elasticsearch"
                }
                
                return is_healthy
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.is_healthy()
            
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
        """Recherche des transactions via le client approprié."""
        if not self._initialized:
            raise RuntimeError("Client non initialisé")
        
        search_id = f"search_{int(time.time() * 1000)}"
        logger.info(f"🔍 [{search_id}] Recherche via {self.client_type}: '{query}'")
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                return await self._search_elasticsearch(user_id, query, limit, filters, include_highlights)
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.search(user_id, query, limit, filters, include_highlights)
            else:
                logger.error("❌ Aucun client disponible pour la recherche")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erreur de recherche: {e}")
            return []
    
    async def _search_elasticsearch(
        self,
        user_id: int,
        query: str,
        limit: int,
        filters: Dict[str, Any],
        include_highlights: bool
    ) -> List[Dict[str, Any]]:
        """Recherche avec le client Elasticsearch officiel."""
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
        
        # Ajouter les filtres
        if filters:
            search_body["query"]["bool"]["filter"] = []
            for field, value in filters.items():
                if value is not None:
                    search_body["query"]["bool"]["filter"].append({"term": {field: value}})
        
        # Ajouter la mise en évidence
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
        response = await self.client.search(index=self.index_name, body=search_body)
        
        # Formater les résultats
        hits = response.get("hits", {}).get("hits", [])
        results = []
        
        for hit in hits:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"]
            }
            
            if "highlight" in hit:
                result["highlights"] = hit["highlight"]
            
            results.append(result)
        
        return results
    
    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Indexe un document via le client approprié."""
        if not self._initialized:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                result = await self.client.index(
                    index=self.index_name,
                    id=doc_id,
                    body=document
                )
                return result.get("result") in ["created", "updated"]
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.index_document(doc_id, document)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur indexation: {e}")
            return False
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Indexation en lot via le client approprié."""
        if not self._initialized:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                from elasticsearch.helpers import async_bulk
                
                actions = []
                for doc in documents:
                    doc_id = doc.get("id") or doc.get("transaction_id")
                    if doc_id:
                        actions.append({
                            "_index": self.index_name,
                            "_id": doc_id,
                            "_source": doc
                        })
                
                if actions:
                    success, failed = await async_bulk(self.client, actions)
                    logger.info(f"📦 Bulk: {success} succès, {len(failed)} échecs")
                    return len(failed) == 0
                
                return True
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.bulk_index(documents)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur bulk indexation: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document via le client approprié."""
        if not self._initialized:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                result = await self.client.delete(index=self.index_name, id=doc_id)
                return result.get("result") == "deleted"
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.delete_document(doc_id)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression: {e}")
            return False
    
    async def count_documents(self, user_id: Optional[int] = None, filters: Dict[str, Any] = None) -> int:
        """Compte le nombre de documents dans l'index."""
        if not self._initialized:
            return 0
        
        try:
            count_body = {}
            
            if user_id is not None or filters:
                count_body["query"] = {"bool": {"must": []}}
                
                if user_id is not None:
                    count_body["query"]["bool"]["must"].append({"term": {"user_id": user_id}})
                
                if filters:
                    for field, value in filters.items():
                        if value is not None:
                            count_body["query"]["bool"]["must"].append({"term": {field: value}})
            
            if self.client_type == 'elasticsearch' and self.client:
                response = await self.client.count(index=self.index_name, body=count_body)
                return response.get("count", 0)
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                # Pour le client Bonsai, utiliser une recherche avec size=0
                search_body = count_body.copy()
                search_body["size"] = 0
                
                async with self.bonsai_client.session.post(
                    f"{self.bonsai_client.base_url}/{self.index_name}/_search",
                    data=json.dumps(search_body)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        total = result.get("hits", {}).get("total", {})
                        if isinstance(total, dict):
                            return total.get("value", 0)
                        else:
                            return total
                    return 0
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Erreur comptage documents: {e}")
            return 0
    
    async def refresh_index(self) -> bool:
        """Force le refresh de l'index pour rendre les documents visibles."""
        if not self._initialized:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                await self.client.indices.refresh(index=self.index_name)
                return True
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                async with self.bonsai_client.session.post(
                    f"{self.bonsai_client.base_url}/{self.index_name}/_refresh"
                ) as response:
                    return response.status == 200
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur refresh index: {e}")
            return False
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'index."""
        if not self._initialized:
            return {"error": "Client not initialized"}
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                # Vérifier si l'index existe
                exists = await self.client.indices.exists(index=self.index_name)
                if not exists:
                    return {"exists": False, "index_name": self.index_name}
                
                # Statistiques de l'index
                stats = await self.client.indices.stats(index=self.index_name)
                index_stats = stats.get("indices", {}).get(self.index_name, {})
                
                return {
                    "exists": True,
                    "index_name": self.index_name,
                    "client_type": self.client_type,
                    "document_count": index_stats.get("total", {}).get("docs", {}).get("count", 0),
                    "store_size_bytes": index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0),
                    "primary_shards": index_stats.get("primaries", {}).get("docs", {}).get("count", 0)
                }
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                # Pour Bonsai, utiliser l'API de stats
                async with self.bonsai_client.session.get(
                    f"{self.bonsai_client.base_url}/{self.index_name}/_stats"
                ) as response:
                    if response.status == 200:
                        stats = await response.json()
                        index_stats = stats.get("indices", {}).get(self.index_name, {})
                        
                        return {
                            "exists": True,
                            "index_name": self.index_name,
                            "client_type": self.client_type,
                            "document_count": index_stats.get("total", {}).get("docs", {}).get("count", 0),
                            "store_size_bytes": index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0)
                        }
                    elif response.status == 404:
                        return {"exists": False, "index_name": self.index_name, "client_type": self.client_type}
                    else:
                        return {"error": f"HTTP {response.status}", "client_type": self.client_type}
            
            return {"error": "No client available"}
            
        except Exception as e:
            logger.error(f"❌ Erreur info index: {e}")
            return {"error": str(e), "client_type": self.client_type}
    
    def get_client_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le client utilisé."""
        return {
            "initialized": self._initialized,
            "client_type": self.client_type,
            "elasticsearch_available": self.client is not None,
            "bonsai_available": self.bonsai_client is not None,
            "connection_attempts": self._connection_attempts,
            "last_health_check": self._last_health_check,
            "index_name": self.index_name
        }
    
    async def close(self):
        """Ferme les connexions."""
        if self.client:
            logger.info("🔒 Fermeture client Elasticsearch...")
            await self.client.close()
            self.client = None
            
        if self.bonsai_client:
            logger.info("🔒 Fermeture client Bonsai...")
            await self.bonsai_client.close()
            self.bonsai_client = None
            
        self._initialized = False
        logger.info("✅ Clients fermés")