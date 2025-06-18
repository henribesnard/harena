"""
Client Elasticsearch avec logging amélioré pour le monitoring et le debugging.
"""
import logging
import time
from typing import List, Dict, Any, Optional
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError, TransportError

from config_service.config import settings

# Configuration du logger spécifique à Elasticsearch
logger = logging.getLogger("search_service.elasticsearch")
# Logger séparé pour les métriques
metrics_logger = logging.getLogger("search_service.metrics.elasticsearch")


class ElasticClient:
    """Client pour interagir avec Elasticsearch avec logging amélioré."""
    
    def __init__(self):
        self.client = None
        self.index_name = "harena_transactions"
        self._initialized = False
        self._connection_attempts = 0
        self._last_health_check = None
        
    async def initialize(self):
        """Initialise la connexion Elasticsearch avec logging détaillé."""
        logger.info("🔄 Initialisation du client Elasticsearch...")
        start_time = time.time()
        
        # Log des configurations
        if settings.SEARCHBOX_URL:
            es_url = settings.SEARCHBOX_URL
            logger.info("📡 Configuration: SearchBox Elasticsearch")
        elif settings.BONSAI_URL:
            es_url = settings.BONSAI_URL
            logger.info("📡 Configuration: Bonsai Elasticsearch")
        else:
            logger.error("❌ Aucune URL Elasticsearch configurée (SEARCHBOX_URL/BONSAI_URL)")
            raise ValueError("No Elasticsearch URL configured")
        
        # Masquer les credentials dans les logs
        safe_url = self._mask_credentials(es_url)
        logger.info(f"🔗 Connexion à: {safe_url}")
        
        try:
            self._connection_attempts += 1
            logger.info(f"🔄 Tentative de connexion #{self._connection_attempts}")
            
            # Créer le client avec configuration détaillée
            self.client = AsyncElasticsearch(
                [es_url],
                verify_certs=True,
                ssl_show_warn=False,
                max_retries=3,
                retry_on_timeout=True,
                timeout=30,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Test de connexion avec timeout
            logger.info("⏱️ Test de connexion...")
            connection_start = time.time()
            
            info = await self.client.info()
            connection_time = time.time() - connection_start
            
            # Logs détaillés de la connexion
            logger.info(f"✅ Connexion réussie en {connection_time:.2f}s")
            logger.info(f"📊 Elasticsearch version: {info['version']['number']}")
            logger.info(f"🏷️ Cluster name: {info['cluster_name']}")
            logger.info(f"🆔 Cluster UUID: {info['cluster_uuid']}")
            
            # Métriques de connexion
            metrics_logger.info(f"elasticsearch.connection.success,time={connection_time:.3f},attempt={self._connection_attempts}")
            
            # Vérifier la santé du cluster
            await self._check_cluster_health()
            
            # Créer l'index si nécessaire
            await self._setup_index()
            
            self._initialized = True
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Elasticsearch initialisé avec succès en {total_time:.2f}s")
            
            # IMPORTANT: Retourner True en cas de succès
            return True
            
        except ConnectionError as e:
            logger.error(f"🔌 Erreur de connexion Elasticsearch: {e}")
            metrics_logger.error(f"elasticsearch.connection.failed,type=connection,attempt={self._connection_attempts}")
            self._handle_connection_error(e)
            raise  # Re-lever l'exception
            
        except TransportError as e:
            logger.error(f"🚫 Erreur de transport Elasticsearch: {e}")
            logger.error(f"📍 Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            metrics_logger.error(f"elasticsearch.connection.failed,type=transport,status={getattr(e, 'status_code', 'unknown')}")
            raise  # Re-lever l'exception
            
        except Exception as e:
            logger.error(f"💥 Erreur inattendue lors de l'initialisation Elasticsearch: {type(e).__name__}: {e}")
            logger.error(f"📍 Détails: {str(e)}", exc_info=True)
            metrics_logger.error(f"elasticsearch.connection.failed,type=unexpected,error={type(e).__name__}")
            raise  # Re-lever l'exception
            
        finally:
            if not self._initialized:
                self.client = None
                logger.warning("⚠️ Client Elasticsearch non initialisé - recherche lexicale indisponible")
    
    async def _check_cluster_health(self):
        """Vérifie la santé du cluster Elasticsearch."""
        try:
            logger.debug("🩺 Vérification santé cluster Elasticsearch...")
            
            # Vérification basique avec cluster health
            health = await self.client.cluster.health()
            status = health.get('status', 'unknown')
            
            if status == 'green':
                logger.info("🟢 Cluster Elasticsearch: Excellent état")
            elif status == 'yellow':
                logger.warning("🟡 Cluster Elasticsearch: État dégradé mais fonctionnel")
            elif status == 'red':
                logger.error("🔴 Cluster Elasticsearch: État critique")
            else:
                logger.warning(f"⚪ Cluster Elasticsearch: État inconnu ({status})")
            
            metrics_logger.info(f"elasticsearch.cluster.health,status={status}")
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de vérifier la santé du cluster: {e}")
    
    async def _setup_index(self):
        """Configure l'index Elasticsearch."""
        try:
            logger.debug(f"📁 Configuration de l'index {self.index_name}...")
            
            # Vérifier si l'index existe
            exists = await self.client.indices.exists(index=self.index_name)
            
            if not exists:
                logger.info(f"📁 Création de l'index {self.index_name}...")
                
                # Configuration de l'index pour les transactions
                index_config = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "harena_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop", "snowball"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "transaction_id": {"type": "keyword"},
                            "description": {
                                "type": "text",
                                "analyzer": "harena_analyzer",
                                "fields": {
                                    "raw": {"type": "keyword"}
                                }
                            },
                            "category": {"type": "keyword"},
                            "amount": {"type": "float"},
                            "date": {"type": "date"},
                            "account_id": {"type": "keyword"},
                            "merchant": {"type": "text", "analyzer": "harena_analyzer"},
                            "location": {"type": "text"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"}
                        }
                    }
                }
                
                await self.client.indices.create(
                    index=self.index_name,
                    body=index_config
                )
                logger.info(f"✅ Index {self.index_name} créé avec succès")
                metrics_logger.info(f"elasticsearch.index.created,name={self.index_name}")
            else:
                logger.info(f"📁 Index {self.index_name} existe déjà")
                
            # Vérifier les stats de l'index
            stats = await self.client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            logger.info(f"📊 Index {self.index_name}: {doc_count} documents")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration de l'index: {e}")
            # Ne pas lever l'exception car ce n'est pas critique pour l'initialisation
    
    async def is_healthy(self) -> bool:
        """Vérifie si le client Elasticsearch est en bonne santé."""
        if not self.client:
            logger.debug("❌ Client non initialisé")
            return False
        
        try:
            logger.debug("🩺 Vérification santé client...")
            start_time = time.time()
            
            # Ping simple
            ping_result = await self.client.ping()
            ping_time = time.time() - start_time
            
            if ping_result:
                logger.debug(f"✅ Ping réussi en {ping_time:.3f}s")
                metrics_logger.info(f"elasticsearch.health.ping.success,time={ping_time:.3f}")
                
                # Vérification plus approfondie si nécessaire
                if time.time() - (self._last_health_check or 0) > 60:  # Chaque minute
                    await self._detailed_health_check()
                    self._last_health_check = time.time()
                
                return True
            else:
                logger.warning(f"⚠️ Ping échoué en {ping_time:.3f}s")
                metrics_logger.warning(f"elasticsearch.health.ping.failed,time={ping_time:.3f}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du health check: {type(e).__name__}: {e}")
            metrics_logger.error(f"elasticsearch.health.check.failed,error={type(e).__name__}")
            return False
    
    async def _detailed_health_check(self):
        """Effectue une vérification de santé approfondie."""
        try:
            logger.debug("🔍 Vérification santé détaillée...")
            
            # Vérifier l'index
            exists = await self.client.indices.exists(index=self.index_name)
            if not exists:
                logger.error(f"❌ Index {self.index_name} n'existe pas")
                metrics_logger.error("elasticsearch.health.index.missing")
                return
            
            # Statistiques de l'index
            stats = await self.client.indices.stats(index=self.index_name)
            index_stats = stats.get("indices", {}).get(self.index_name, {})
            
            total_docs = index_stats.get("total", {}).get("docs", {}).get("count", 0)
            store_size = index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0)
            
            logger.info(f"📊 Index stats: {total_docs} documents, {store_size} bytes")
            metrics_logger.info(f"elasticsearch.index.stats,docs={total_docs},size={store_size}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification détaillée: {e}")
    
    def _mask_credentials(self, url: str) -> str:
        """Masque les credentials dans l'URL pour les logs."""
        if "@" in url:
            # Format: https://user:pass@host:port/path
            parts = url.split("@")
            if len(parts) == 2:
                protocol_and_creds = parts[0]
                host_and_path = parts[1]
                
                if "://" in protocol_and_creds:
                    protocol = protocol_and_creds.split("://")[0]
                    return f"{protocol}://***:***@{host_and_path}"
        
        return url
    
    def _handle_connection_error(self, error):
        """Gère les erreurs de connexion avec logging détaillé."""
        logger.error("🔌 Diagnostic de l'erreur de connexion:")
        logger.error(f"   - Type: {type(error).__name__}")
        logger.error(f"   - Message: {str(error)}")
        
        # Suggestions de diagnostic
        logger.error("🔧 Actions de diagnostic suggérées:")
        logger.error("   - Vérifier la connectivité réseau")
        logger.error("   - Valider les credentials Elasticsearch")
        logger.error("   - Contrôler les variables d'environnement")
        logger.error("   - Tester la connection depuis un autre client")
    
    async def search_transactions(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> Dict[str, Any]:
        """Recherche des transactions avec logging des performances."""
        if not self.client or not self._initialized:
            raise RuntimeError("Client Elasticsearch non initialisé")
        
        search_id = f"search_{int(time.time() * 1000)}"
        logger.info(f"🔍 [{search_id}] Recherche: '{query}' (limit: {limit})")
        
        start_time = time.time()
        
        try:
            # Construction de la requête de recherche
            search_body = {
                "query": {
                    "bool": {
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
                ],
                "_source": True
            }
            
            # Ajouter des filtres si fournis
            if filters:
                filter_clauses = []
                for field, value in filters.items():
                    if value is not None:
                        filter_clauses.append({"term": {field: value}})
                
                if filter_clauses:
                    search_body["query"]["bool"]["filter"] = filter_clauses
            
            # Exécuter la recherche
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            search_time = time.time() - start_time
            hits = response.get('hits', {})
            total_hits = hits.get('total', {}).get('value', 0)
            
            logger.info(f"✅ [{search_id}] Trouvé {total_hits} résultats en {search_time:.3f}s")
            metrics_logger.info(f"elasticsearch.search.success,time={search_time:.3f},results={total_hits},query_length={len(query)}")
            
            # Formater les résultats
            results = []
            for hit in hits.get('hits', []):
                source = hit.get('_source', {})
                source['_score'] = hit.get('_score', 0)
                results.append(source)
            
            return {
                "query": query,
                "total": total_hits,
                "results": results,
                "search_time": search_time,
                "search_id": search_id
            }
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur recherche après {search_time:.3f}s: {type(e).__name__}: {e}")
            metrics_logger.error(f"elasticsearch.search.failed,time={search_time:.3f},error={type(e).__name__}")
            
            if isinstance(e, TransportError):
                self._handle_transport_error(e, search_id)
            
            raise
    
    def _handle_transport_error(self, error, search_id):
        """Gère les erreurs de transport avec contexte."""
        status_code = getattr(error, 'status_code', 'unknown')
        logger.error(f"🚫 [{search_id}] Erreur transport - Status: {status_code}")
        
        if status_code == 429:
            logger.error(f"🚫 [{search_id}] Rate limiting détecté")
        elif status_code >= 500:
            logger.error(f"🚫 [{search_id}] Erreur serveur Elasticsearch")
        elif status_code >= 400:
            logger.error(f"🚫 [{search_id}] Erreur client - vérifier la requête")
    
    async def close(self):
        """Ferme la connexion avec logging."""
        if self.client:
            logger.info("🔒 Fermeture connexion Elasticsearch...")
            await self.client.close()
            self._initialized = False
            logger.info("✅ Connexion Elasticsearch fermée")
        else:
            logger.debug("🔒 Aucun client à fermer")
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Récupère les informations détaillées du cluster."""
        if not self.client:
            return {}
        
        try:
            info = await self.client.info()
            cluster_stats = await self.client.cluster.stats()
            
            return {
                "cluster_name": info.get("cluster_name"),
                "version": info.get("version", {}).get("number"),
                "nodes": cluster_stats.get("nodes", {}).get("count", {}).get("total", 0),
                "indices_count": cluster_stats.get("indices", {}).get("count", 0),
                "docs_count": cluster_stats.get("indices", {}).get("docs", {}).get("count", 0),
                "store_size": cluster_stats.get("indices", {}).get("store", {}).get("size_in_bytes", 0)
            }
        except Exception as e:
            logger.warning(f"Impossible de récupérer les infos cluster: {e}")
            return {}
    
    async def get_indices_info(self) -> Dict[str, Any]:
        """Récupère les informations sur les indices."""
        if not self.client:
            return {}
        
        try:
            stats = await self.client.indices.stats()
            indices_info = {}
            
            for index_name, index_stats in stats.get("indices", {}).items():
                indices_info[index_name] = {
                    "docs": index_stats.get("total", {}).get("docs", {}),
                    "store": index_stats.get("total", {}).get("store", {}),
                }
            
            return indices_info
        except Exception as e:
            logger.warning(f"Impossible de récupérer les infos indices: {e}")
            return {}