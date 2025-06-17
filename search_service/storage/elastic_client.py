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
            return
        
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
            
        except ConnectionError as e:
            logger.error(f"🔌 Erreur de connexion Elasticsearch: {e}")
            metrics_logger.error(f"elasticsearch.connection.failed,type=connection,attempt={self._connection_attempts}")
            self._handle_connection_error(e)
            
        except TransportError as e:
            logger.error(f"🚫 Erreur de transport Elasticsearch: {e}")
            logger.error(f"📍 Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            metrics_logger.error(f"elasticsearch.connection.failed,type=transport,status={getattr(e, 'status_code', 'unknown')}")
            
        except Exception as e:
            logger.error(f"💥 Erreur inattendue lors de l'initialisation Elasticsearch: {type(e).__name__}: {e}")
            logger.error(f"📍 Détails: {str(e)}", exc_info=True)
            metrics_logger.error(f"elasticsearch.connection.failed,type=unexpected,error={type(e).__name__}")
            
        finally:
            if not self._initialized:
                self.client = None
                logger.warning("⚠️ Client Elasticsearch non initialisé - recherche lexicale indisponible")
    
    async def _check_cluster_health(self):
        """Vérifie la santé du cluster Elasticsearch."""
        try:
            logger.info("🩺 Vérification de la santé du cluster...")
            health = await self.client.cluster.health()
            
            status = health.get('status', 'unknown')
            logger.info(f"💚 Santé cluster: {status}")
            logger.info(f"📊 Nœuds: {health.get('number_of_nodes', 'unknown')}")
            logger.info(f"📊 Nœuds data: {health.get('number_of_data_nodes', 'unknown')}")
            logger.info(f"📊 Shards actifs: {health.get('active_shards', 'unknown')}")
            
            if status == 'red':
                logger.error("🚨 CLUSTER EN ÉTAT CRITIQUE (red)")
            elif status == 'yellow':
                logger.warning("⚠️ Cluster en état dégradé (yellow)")
            else:
                logger.info("✅ Cluster en bonne santé (green)")
                
            metrics_logger.info(f"elasticsearch.cluster.health,status={status},nodes={health.get('number_of_nodes', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Impossible de vérifier la santé du cluster: {e}")
    
    async def search(
        self,
        user_id: int,
        query: Dict[str, Any],
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        include_highlights: bool = False
    ) -> List[Dict[str, Any]]:
        """Effectue une recherche avec logging détaillé."""
        if not self.client:
            logger.error("❌ Client Elasticsearch non initialisé")
            return []
        
        search_id = f"search_{int(time.time()*1000)}"
        logger.info(f"🔍 [{search_id}] Début recherche pour user_id={user_id}")
        
        start_time = time.time()
        
        try:
            # Log des paramètres de recherche
            logger.debug(f"🔍 [{search_id}] Paramètres: limit={limit}, highlights={include_highlights}")
            logger.debug(f"🔍 [{search_id}] Filtres: {filters}")
            logger.debug(f"🔍 [{search_id}] Requête: {query}")
            
            # Construire la requête
            search_body = {
                "query": query,
                "size": limit,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"date": {"order": "desc"}}
                ]
            }
            
            # Ajouter les filtres
            if filters:
                logger.debug(f"🔍 [{search_id}] Application des filtres")
                # Logique de filtrage...
            
            # Ajouter le highlighting
            if include_highlights:
                logger.debug(f"🔍 [{search_id}] Highlighting activé")
                search_body["highlight"] = {
                    "fields": {
                        "searchable_text": {},
                        "primary_description": {},
                        "merchant_name": {},
                        "category_name": {}
                    },
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fragment_size": 150
                }
            
            # Exécuter la recherche
            logger.info(f"🔍 [{search_id}] Exécution de la requête...")
            query_start = time.time()
            
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            query_time = time.time() - query_start
            total_time = time.time() - start_time
            
            # Analyser les résultats
            hits = response.get("hits", {})
            total_hits = hits.get("total", {})
            
            if isinstance(total_hits, dict):
                total_count = total_hits.get("value", 0)
                relation = total_hits.get("relation", "eq")
            else:
                total_count = total_hits
                relation = "eq"
            
            results = hits.get("hits", [])
            
            # Logs de résultats
            logger.info(f"✅ [{search_id}] Recherche terminée en {total_time:.3f}s")
            logger.info(f"📊 [{search_id}] Résultats: {len(results)}/{total_count} ({relation})")
            logger.info(f"⏱️ [{search_id}] Temps requête: {query_time:.3f}s")
            
            # Métriques détaillées
            metrics_logger.info(
                f"elasticsearch.search.success,"
                f"user_id={user_id},"
                f"query_time={query_time:.3f},"
                f"total_time={total_time:.3f},"
                f"results={len(results)},"
                f"total_available={total_count}"
            )
            
            # Log des scores si en mode debug
            if logger.isEnabledFor(logging.DEBUG):
                for i, hit in enumerate(results[:5]):  # Top 5 seulement
                    score = hit.get("_score", 0)
                    source = hit.get("_source", {})
                    description = source.get("primary_description", "N/A")[:50]
                    logger.debug(f"📊 [{search_id}] #{i+1}: score={score:.3f}, desc='{description}...'")
            
            return results
            
        except NotFoundError as e:
            logger.error(f"❌ [{search_id}] Index non trouvé: {self.index_name}")
            metrics_logger.error(f"elasticsearch.search.failed,type=not_found,user_id={user_id}")
            return []
            
        except RequestError as e:
            logger.error(f"❌ [{search_id}] Erreur de requête: {e}")
            logger.error(f"📍 [{search_id}] Détails: {e.info if hasattr(e, 'info') else 'N/A'}")
            metrics_logger.error(f"elasticsearch.search.failed,type=request_error,user_id={user_id}")
            return []
            
        except TransportError as e:
            logger.error(f"🚫 [{search_id}] Erreur de transport: {e}")
            self._handle_transport_error(e, search_id)
            metrics_logger.error(f"elasticsearch.search.failed,type=transport,user_id={user_id}")
            return []
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"💥 [{search_id}] Erreur inattendue après {search_time:.3f}s: {type(e).__name__}: {e}")
            logger.error(f"📍 [{search_id}] Détails", exc_info=True)
            metrics_logger.error(f"elasticsearch.search.failed,type=unexpected,user_id={user_id},error={type(e).__name__}")
            return []
    
    async def is_healthy(self) -> bool:
        """Vérifie l'état de santé avec logging détaillé."""
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
                
                # Vérification plus approfondie
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