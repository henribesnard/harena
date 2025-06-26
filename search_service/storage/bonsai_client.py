"""
Client HTTP direct pour Bonsai Elasticsearch - VERSION CORRIGÉE
Résout le problème 'keepalive_timeout cannot be set if force_close is True'
"""
import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import aiohttp

from config_service.config import settings

logger = logging.getLogger("search_service.bonsai")
metrics_logger = logging.getLogger("search_service.metrics.bonsai")


class BonsaiClient:
    """Client HTTP direct pour Bonsai Elasticsearch avec configuration corrigée."""
    
    def __init__(self):
        self.session = None
        self.base_url = None
        self.auth = None
        self.index_name = "harena_transactions"
        self._initialized = False
        self._connection_attempts = 0
        self._last_health_check = None
        self._closed = False
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def initialize(self, url: str = None, verify_ssl: bool = True, timeout: float = 30.0) -> bool:
        """
        Initialise le client Bonsai avec configuration corrigée.
        
        Args:
            url: URL Bonsai (utilise settings.BONSAI_URL si None)
            verify_ssl: Vérifier les certificats SSL
            timeout: Timeout en secondes
        """
        logger.info("🌐 Initialisation du client Bonsai HTTP...")
        start_time = time.time()
        
        # Fermer les connexions existantes
        await self._cleanup_existing_session()
        
        try:
            # Utiliser l'URL fournie ou celle des settings
            bonsai_url = url or settings.BONSAI_URL
            
            if not bonsai_url:
                logger.error("❌ BONSAI_URL non configurée")
                return False
            
            self._connection_attempts += 1
            
            # Parser l'URL pour extraire les credentials
            parsed = urlparse(bonsai_url)
            
            if parsed.username and parsed.password:
                self.auth = aiohttp.BasicAuth(parsed.username, parsed.password)
                self.base_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
                logger.info(f"🔑 Authentification configurée pour {parsed.username}")
            else:
                self.base_url = bonsai_url
                logger.warning("⚠️ Pas d'authentification détectée")
            
            safe_url = self._mask_credentials(bonsai_url)
            logger.info(f"🔗 Connexion Bonsai HTTP: {safe_url}")
            
            # Configuration SSL
            ssl_context = None
            if not verify_ssl:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                logger.warning("⚠️ Vérification SSL désactivée")
            
            # CORRECTION: Configuration aiohttp compatible
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True,
                # SUPPRIMÉ: keepalive_timeout et force_close incompatibles
                # keepalive_timeout=None,  # ← CAUSE DU PROBLÈME
                # force_close=False,       # ← CAUSE DU PROBLÈME
            )
            
            self.timeout = aiohttp.ClientTimeout(total=timeout)
            
            # Créer la session avec configuration corrigée
            self.session = aiohttp.ClientSession(
                connector=connector,
                auth=self.auth,
                timeout=self.timeout,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "Harena-Search-Service/1.0"
                }
            )
            
            # Test de connexion
            logger.info("🔍 Test de connexion...")
            connection_start = time.time()
            
            async with self.session.get(self.base_url) as response:
                connection_time = time.time() - connection_start
                
                if response.status == 200:
                    data = await response.json()
                    cluster_name = data.get('cluster_name', 'unknown')
                    version = data.get('version', {}).get('number', 'unknown')
                    
                    logger.info(f"✅ Connexion Bonsai réussie en {connection_time:.3f}s")
                    logger.info(f"🏷️ Cluster: {cluster_name}")
                    logger.info(f"📈 Version: {version}")
                    
                    # Test santé cluster
                    await self._test_cluster_health()
                    
                    # Test existence de l'index
                    await self._check_index_status()
                    
                    self._initialized = True
                    self._closed = False
                    
                    total_time = time.time() - start_time
                    logger.info(f"🎉 Client Bonsai initialisé avec succès en {total_time:.2f}s")
                    
                    # Métriques
                    metrics_logger.info(f"bonsai.connection.success,time={total_time:.3f},attempt={self._connection_attempts}")
                    
                    return True
                else:
                    logger.error(f"❌ Erreur HTTP: {response.status}")
                    await self._cleanup_existing_session()
                    return False
        
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Bonsai: {e}")
            logger.error(f"📍 Type: {type(e).__name__}")
            
            # Diagnostic spécifique
            error_str = str(e).lower()
            if "keepalive_timeout" in error_str and "force_close" in error_str:
                logger.error("🔧 DIAGNOSTIC: Configuration aiohttp incompatible")
                logger.error("   - Solution: Retirer keepalive_timeout avec force_close=True")
            elif "connection" in error_str or "timeout" in error_str:
                logger.error("🔌 DIAGNOSTIC: Problème de connectivité réseau")
            elif "ssl" in error_str or "certificate" in error_str:
                logger.error("🔒 DIAGNOSTIC: Problème SSL/TLS")
            
            await self._cleanup_existing_session()
            metrics_logger.error(f"bonsai.connection.failed,type={type(e).__name__},attempt={self._connection_attempts}")
            return False
    
    async def _cleanup_existing_session(self):
        """Nettoie la session existante."""
        if self.session:
            try:
                await asyncio.wait_for(self.session.close(), timeout=5.0)
                logger.debug("🧹 Session Bonsai fermée proprement")
            except Exception as e:
                logger.warning(f"⚠️ Erreur nettoyage session: {e}")
            finally:
                self.session = None
    
    async def _test_cluster_health(self):
        """Test de santé du cluster."""
        try:
            async with self.session.get(f"{self.base_url}/_cluster/health") as response:
                if response.status == 200:
                    health = await response.json()
                    status = health.get('status', 'unknown')
                    nodes = health.get('number_of_nodes', 0)
                    
                    logger.info(f"💚 Santé cluster: {status}")
                    logger.info(f"📊 Nœuds: {nodes}")
                    
                    self._last_health_check = {
                        "timestamp": time.time(),
                        "status": status,
                        "healthy": status in ['green', 'yellow'],
                        "nodes": nodes
                    }
                else:
                    logger.warning(f"⚠️ Health check HTTP {response.status}")
        except Exception as e:
            logger.warning(f"⚠️ Test santé échoué: {e}")
    
    async def _check_index_status(self):
        """Vérifie le statut de l'index."""
        try:
            async with self.session.head(f"{self.base_url}/{self.index_name}") as response:
                if response.status == 200:
                    logger.info(f"📚 Index {self.index_name} existe")
                elif response.status == 404:
                    logger.info(f"📚 Index {self.index_name} n'existe pas encore")
                else:
                    logger.warning(f"⚠️ Status index: {response.status}")
        except Exception as e:
            logger.warning(f"⚠️ Vérification index échouée: {e}")
    
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
        """Vérifie la santé de la connexion."""
        if not self._initialized or self._closed or not self.session:
            return False
        
        try:
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/_cluster/health", timeout=5) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'red')
                    is_healthy = status in ['green', 'yellow']
                    
                    # Mettre à jour le cache de santé
                    self._last_health_check = {
                        "timestamp": time.time(),
                        "healthy": is_healthy,
                        "status": status,
                        "response_time": response_time
                    }
                    
                    return is_healthy
                
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
        """Effectue une recherche via l'API Bonsai."""
        if not self._initialized or self._closed:
            return []
        
        # Validation de la query
        if not isinstance(query, str):
            logger.error(f"❌ Query doit être string, reçu: {type(query)}")
            query = str(query) if query is not None else ""
        
        search_id = f"bonsai_search_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(f"🔍 [{search_id}] Recherche Bonsai: '{query}' (user: {user_id})")
        
        try:
            # Expansion des termes de recherche
            from search_service.utils.query_expansion import expand_query_terms
            expanded_terms = expand_query_terms(query)
            search_string = " ".join(expanded_terms)
            
            # Construire la requête de recherche
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}}
                        ],
                        "should": [
                            {
                                "multi_match": {
                                    "query": search_string,
                                    "fields": ["searchable_text^3", "primary_description^2", "merchant_name^2", "category_name"],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "terms": {
                                    "primary_description": expanded_terms,
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
            
            # Exécuter la recherche
            search_url = f"{self.base_url}/{self.index_name}/_search"
            
            async with self.session.post(search_url, json=search_body) as response:
                query_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    hits = data.get('hits', {}).get('hits', [])
                    total = data.get('hits', {}).get('total', {})
                    
                    if isinstance(total, dict):
                        total_count = total.get('value', 0)
                    else:
                        total_count = total
                    
                    # Traiter les résultats
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
                    
                    # Logs de performance
                    if scores:
                        max_score = max(scores)
                        min_score = min(scores)
                        avg_score = sum(scores) / len(scores)
                    else:
                        max_score = min_score = avg_score = 0
                    
                    logger.info(f"✅ [{search_id}] Recherche Bonsai terminée en {query_time:.3f}s")
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
                    
                elif response.status == 404:
                    logger.warning(f"⚠️ [{search_id}] Index {self.index_name} n'existe pas")
                    return []
                else:
                    logger.error(f"❌ [{search_id}] Erreur recherche: {response.status}")
                    error_text = await response.text()
                    logger.error(f"   Détails: {error_text[:200]}...")
                    return []
        
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur recherche Bonsai: {e}")
            metrics_logger.error(f"bonsai.search.failed,user_id={user_id},time={query_time:.3f},error={type(e).__name__}")
            return []
    
    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Indexe un document."""
        if not self._initialized or self._closed:
            return False
        
        try:
            index_url = f"{self.base_url}/{self.index_name}/_doc/{doc_id}"
            
            async with self.session.put(index_url, json=document) as response:
                if response.status in [200, 201]:
                    logger.debug(f"✅ Document {doc_id} indexé")
                    return True
                else:
                    logger.error(f"❌ Erreur indexation {doc_id}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur indexation document: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document."""
        if not self._initialized or self._closed:
            return False
        
        try:
            delete_url = f"{self.base_url}/{self.index_name}/_doc/{doc_id}"
            
            async with self.session.delete(delete_url) as response:
                if response.status in [200, 404]:  # 404 = déjà supprimé
                    logger.debug(f"✅ Document {doc_id} supprimé")
                    return True
                else:
                    logger.error(f"❌ Erreur suppression {doc_id}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur suppression document: {e}")
            return False
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Indexation en lot."""
        if not self._initialized or self._closed or not documents:
            return False
        
        try:
            # Construire le body bulk
            bulk_body = []
            for doc in documents:
                doc_id = doc.get("id")
                if not doc_id:
                    continue
                
                # Action d'indexation
                action = {"index": {"_index": self.index_name, "_id": doc_id}}
                bulk_body.append(json.dumps(action))
                bulk_body.append(json.dumps(doc))
            
            if not bulk_body:
                return True
            
            # Joindre avec des newlines (format bulk Elasticsearch)
            bulk_data = "\n".join(bulk_body) + "\n"
            
            bulk_url = f"{self.base_url}/_bulk"
            headers = {"Content-Type": "application/x-ndjson"}
            
            async with self.session.post(bulk_url, data=bulk_data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    errors = result.get("errors", False)
                    
                    if not errors:
                        logger.info(f"✅ Bulk indexation: {len(documents)} documents")
                        return True
                    else:
                        logger.warning(f"⚠️ Bulk avec erreurs: {len(documents)} documents")
                        return False
                else:
                    logger.error(f"❌ Erreur bulk indexation: {response.status}")
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
            
            count_url = f"{self.base_url}/{self.index_name}/_count"
            
            async with self.session.post(count_url, json=count_body) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("count", 0)
                elif response.status == 404:
                    return 0  # Index n'existe pas
                else:
                    logger.error(f"❌ Erreur count: {response.status}")
                    return 0
                    
        except Exception as e:
            logger.error(f"❌ Erreur comptage documents: {e}")
            return 0
    
    async def refresh_index(self) -> bool:
        """Force le refresh de l'index."""
        if not self._initialized or self._closed:
            return False
        
        try:
            refresh_url = f"{self.base_url}/{self.index_name}/_refresh"
            
            async with self.session.post(refresh_url) as response:
                if response.status == 200:
                    logger.debug(f"✅ Index {self.index_name} refreshed")
                    return True
                else:
                    logger.error(f"❌ Erreur refresh: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur refresh index: {e}")
            return False
    
    def get_client_info(self) -> Dict[str, Any]:
        """Retourne les informations du client."""
        return {
            "type": "bonsai_http",
            "initialized": self._initialized,
            "closed": self._closed,
            "base_url": self.base_url if self.base_url else None,
            "has_auth": self.auth is not None,
            "connection_attempts": self._connection_attempts,
            "last_health_check": self._last_health_check,
            "index_name": self.index_name
        }
    
    async def close(self):
        """Ferme la session."""
        if self._closed:
            return
        
        logger.info("🔒 Fermeture du client Bonsai...")
        self._closed = True
        self._initialized = False
        
        if self.session:
            try:
                await asyncio.wait_for(self.session.close(), timeout=5.0)
                logger.info("✅ Session Bonsai fermée proprement")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout fermeture session Bonsai")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture session Bonsai: {e}")
            finally:
                self.session = None
    
    async def __aenter__(self):
        """Support pour 'async with'."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique."""
        await self.close()