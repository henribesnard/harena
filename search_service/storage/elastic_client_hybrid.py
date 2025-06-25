"""
Client Elasticsearch hybride qui utilise le client officiel ou Bonsai HTTP selon la compatibilité.
VERSION CORRIGÉE - Corrige le bug 'dict' object has no attribute 'lower' + gestion connexions
"""
import logging
import json
import time
import asyncio
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
        self._closed = False
        
    async def initialize(self):
        """Initialise la connexion en essayant d'abord Elasticsearch, puis Bonsai."""
        logger.info("🔄 Initialisation du client Elasticsearch hybride...")
        start_time = time.time()
        
        if not settings.BONSAI_URL:
            logger.error("❌ BONSAI_URL non configurée")
            return False
        
        # Fermer les clients existants
        await self._cleanup_existing_clients()
        
        # Masquer les credentials pour l'affichage
        safe_url = self._mask_credentials(settings.BONSAI_URL)
        logger.info(f"🔗 Connexion à: {safe_url}")
        
        # Essayer d'abord le client Elasticsearch officiel
        elasticsearch_success = await self._try_elasticsearch_client()
        
        if elasticsearch_success:
            self.client_type = 'elasticsearch'
            self._initialized = True
            self._closed = False
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Elasticsearch officiel initialisé en {total_time:.2f}s")
            return True
        
        # Si échec, essayer le client Bonsai HTTP
        logger.info("🔄 Tentative avec client Bonsai HTTP...")
        bonsai_success = await self._try_bonsai_client()
        
        if bonsai_success:
            self.client_type = 'bonsai'
            self._initialized = True
            self._closed = False
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Bonsai HTTP initialisé en {total_time:.2f}s")
            return True
        
        # Les deux ont échoué
        logger.error("❌ Impossible d'initialiser un client de recherche")
        return False
    
    async def _cleanup_existing_clients(self):
        """Nettoie les clients existants."""
        cleanup_tasks = []
        
        if self.client:
            logger.debug("🧹 Nettoyage client Elasticsearch existant...")
            try:
                cleanup_tasks.append(self.client.close())
            except Exception as e:
                logger.warning(f"⚠️ Erreur nettoyage Elasticsearch: {e}")
            finally:
                self.client = None
        
        if self.bonsai_client:
            logger.debug("🧹 Nettoyage client Bonsai existant...")
            try:
                cleanup_tasks.append(self.bonsai_client.close())
            except Exception as e:
                logger.warning(f"⚠️ Erreur nettoyage Bonsai: {e}")
            finally:
                self.bonsai_client = None
        
        # Attendre tous les nettoyages
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"⚠️ Erreur nettoyage groupé: {e}")
    
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
            
            # Test de connexion avec timeout
            try:
                info = await asyncio.wait_for(self.client.info(), timeout=15.0)
                logger.info(f"✅ Client Elasticsearch: {info['cluster_name']} v{info['version']['number']}")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout connexion Elasticsearch")
                await self._safe_close_elasticsearch()
                return False
            
            # Test de santé
            try:
                health = await asyncio.wait_for(self.client.cluster.health(), timeout=10.0)
                logger.info(f"💚 Santé: {health['status']}")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout health check Elasticsearch")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Client Elasticsearch échoué: {e}")
            if "UnsupportedProductError" in str(e):
                logger.info("💡 Bonsai détecté comme incompatible avec le client standard")
            
            await self._safe_close_elasticsearch()
            return False
    
    async def _safe_close_elasticsearch(self):
        """Ferme le client Elasticsearch de manière sécurisée."""
        if self.client:
            try:
                await asyncio.wait_for(self.client.close(), timeout=5.0)
                logger.debug("🔒 Client Elasticsearch fermé proprement")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture Elasticsearch: {e}")
            finally:
                self.client = None
    
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
        if not self._initialized or self._closed:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                health = await asyncio.wait_for(self.client.cluster.health(), timeout=5.0)
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
        if not self._initialized or self._closed:
            raise RuntimeError("Client non initialisé ou fermé")
        
        # VALIDATION CRITIQUE: S'assurer que query est une string
        if not isinstance(query, str):
            logger.error(f"❌ Query doit être une string, reçu: {type(query)} = {query}")
            query = str(query) if query is not None else ""
        
        search_id = f"search_{int(time.time() * 1000)}"
        logger.info(f"🔍 [{search_id}] Recherche via {self.client_type}: '{query}' (type: {type(query)})")
        
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
            logger.error(f"   Query type: {type(query)}")
            logger.error(f"   Query value: {query}")
            return []
    
    async def _search_elasticsearch(
        self,
        user_id: int,
        query: str,
        limit: int,
        filters: Dict[str, Any],
        include_highlights: bool
    ) -> List[Dict[str, Any]]:
        """Recherche avec le client Elasticsearch officiel - VERSION CORRIGÉE."""
        
        # VALIDATION CRITIQUE
        if not isinstance(query, str):
            logger.error(f"❌ Query doit être string dans _search_elasticsearch: {type(query)}")
            query = str(query) if query is not None else ""
        
        search_id = f"search_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Expansion des termes de recherche - SÉCURISÉE
            from search_service.utils.query_expansion import expand_query_terms
            expanded_terms = expand_query_terms(query)
            
            # Construction de la chaîne de recherche
            search_string = " ".join(expanded_terms)
            
            # VALIDATION: s'assurer que tous les éléments sont des strings
            validated_terms = []
            for term in expanded_terms:
                if isinstance(term, str):
                    validated_terms.append(term)
                else:
                    logger.warning(f"Term ignoré (pas string): {type(term)} = {term}")
                    validated_terms.append(str(term))
            
            # Construction de la requête Elasticsearch
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}}
                        ],
                        "should": [
                            {
                                "multi_match": {
                                    "query": search_string,  # String validée
                                    "fields": ["searchable_text^3", "primary_description^2", "merchant_name^2", "category_name"],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "terms": {
                                    "primary_description": validated_terms,  # Liste de strings validées
                                    "boost": 2.0
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
            
            # Log de la requête construite (pour debug)
            logger.info(f"🔍 [{search_id}] Query construite: search_string='{search_string}', terms={validated_terms}")
            
            # Ajouter les filtres
            if filters:
                if "filter" not in search_body["query"]["bool"]:
                    search_body["query"]["bool"]["filter"] = []
                
                for field, value in filters.items():
                    if value is not None:
                        search_body["query"]["bool"]["filter"].append({"term": {field: value}})
            
            # Ajouter highlighting
            if include_highlights:
                search_body["highlight"] = {
                    "fields": {
                        "searchable_text": {},
                        "primary_description": {},
                        "merchant_name": {}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            
            # Exécuter la recherche avec timeout
            logger.info(f"🎯 [{search_id}] Exécution recherche Elasticsearch...")
            
            response = await asyncio.wait_for(
                self.client.search(index=self.index_name, body=search_body),
                timeout=15.0
            )
            
            query_time = time.time() - start_time
            
            # Traiter les résultats
            hits = response.get("hits", {}).get("hits", [])
            total_hits = response.get("hits", {}).get("total", {})
            
            if isinstance(total_hits, dict):
                total_count = total_hits.get("value", 0)
            else:
                total_count = total_hits
            
            results = []
            scores = []
            
            for hit in hits:
                result_item = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                }
                
                if "highlight" in hit:
                    result_item["highlights"] = hit["highlight"]
                
                results.append(result_item)
                scores.append(hit["_score"])
            
            # Logs de résultats
            if scores:
                max_score, min_score, avg_score = max(scores), min(scores), sum(scores) / len(scores)
            else:
                max_score = min_score = avg_score = 0
            
            logger.info(f"✅ [{search_id}] Recherche terminée en {query_time:.3f}s")
            logger.info(f"📊 [{search_id}] Résultats: {len(results)}/{total_count}")
            logger.info(f"🎯 [{search_id}] Scores: max={max_score:.3f}, min={min_score:.3f}, avg={avg_score:.3f}")
            
            # Métriques
            metrics_logger.info(
                f"elasticsearch.search.success,"
                f"user_id={user_id},"
                f"query_time={query_time:.3f},"
                f"results={len(results)},"
                f"total={total_count},"
                f"max_score={max_score:.3f}"
            )
            
            return results
            
        except asyncio.TimeoutError:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Timeout Elasticsearch après {query_time:.3f}s")
            metrics_logger.error(f"elasticsearch.search.timeout,user_id={user_id},time={query_time:.3f}")
            return []
            
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur Elasticsearch après {query_time:.3f}s: {e}")
            logger.error(f"   Query type: {type(query)}")
            logger.error(f"   Query value: {query}")
            metrics_logger.error(f"elasticsearch.search.failed,user_id={user_id},time={query_time:.3f},error={type(e).__name__}")
            return []
    
    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Indexe un document."""
        if not self._initialized or self._closed:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                response = await asyncio.wait_for(
                    self.client.index(index=self.index_name, id=doc_id, body=document),
                    timeout=10.0
                )
                return response.get("result") in ["created", "updated"]
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.index_document(doc_id, document)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur indexation: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document."""
        if not self._initialized or self._closed:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                response = await asyncio.wait_for(
                    self.client.delete(index=self.index_name, id=doc_id),
                    timeout=10.0
                )
                return response.get("result") == "deleted"
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.delete_document(doc_id)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression: {e}")
            return False
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Indexation en lot."""
        if not self._initialized or self._closed or not documents:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                from elasticsearch.helpers import async_bulk
                
                actions = []
                for doc in documents:
                    action = {
                        "_index": self.index_name,
                        "_id": doc.get("id"),
                        "_source": doc
                    }
                    actions.append(action)
                
                success, failed = await async_bulk(self.client, actions)
                logger.info(f"✅ Bulk indexation: {success} succès, {len(failed)} échecs")
                return len(failed) == 0
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.bulk_index(documents)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur bulk indexation: {e}")
            return False
    
    async def count_documents(self, user_id: int = None, filters: Dict[str, Any] = None) -> int:
        """Compte le nombre de documents."""
        if not self._initialized or self._closed:
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
                response = await asyncio.wait_for(
                    self.client.count(index=self.index_name, body=count_body),
                    timeout=10.0
                )
                return response.get("count", 0)
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.count_documents(user_id, filters)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Erreur comptage documents: {e}")
            return 0
    
    async def refresh_index(self) -> bool:
        """Force le refresh de l'index pour rendre les documents visibles."""
        if not self._initialized or self._closed:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                await asyncio.wait_for(
                    self.client.indices.refresh(index=self.index_name),
                    timeout=10.0
                )
                return True
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.refresh_index()
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur refresh index: {e}")
            return False
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'index."""
        return {
            "initialized": self._initialized,
            "client_type": self.client_type,
            "elasticsearch_available": self.client is not None,
            "bonsai_available": self.bonsai_client is not None,
            "connection_attempts": self._connection_attempts,
            "last_health_check": self._last_health_check,
            "index_name": self.index_name,
            "closed": self._closed
        }
    
    async def close(self):
        """Ferme les connexions."""
        if self._closed:
            return
        
        logger.info("🔒 Fermeture des clients de recherche...")
        self._closed = True
        self._initialized = False
        
        close_tasks = []
        
        if self.client:
            logger.info("🔒 Fermeture client Elasticsearch...")
            try:
                close_tasks.append(asyncio.create_task(self._safe_close_elasticsearch()))
            except Exception as e:
                logger.warning(f"⚠️ Erreur préparation fermeture Elasticsearch: {e}")
                self.client = None
            
        if self.bonsai_client:
            logger.info("🔒 Fermeture client Bonsai...")
            try:
                close_tasks.append(asyncio.create_task(self.bonsai_client.close()))
            except Exception as e:
                logger.warning(f"⚠️ Erreur préparation fermeture Bonsai: {e}")
                self.bonsai_client = None
        
        # Attendre toutes les fermetures avec timeout
        if close_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout fermeture clients")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture groupée: {e}")
        
        logger.info("✅ Clients fermés")
    
    async def __aenter__(self):
        """Support pour 'async with'."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique lors de la sortie du contexte."""
        await self.close()