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
        
        # Vérifier la configuration
        if not settings.BONSAI_URL:
            logger.error("❌ BONSAI_URL non configurée")
            return False
        
        # Masquer les credentials dans les logs
        safe_url = self._mask_credentials(settings.BONSAI_URL)
        logger.info(f"🔗 Connexion à Bonsai Elasticsearch: {safe_url}")
        
        try:
            self._connection_attempts += 1
            logger.info(f"🔄 Tentative de connexion #{self._connection_attempts}")
            
            # Créer le client avec configuration adaptée à Bonsai
            self.client = AsyncElasticsearch(
                [settings.BONSAI_URL],
                verify_certs=True,
                ssl_show_warn=False,
                max_retries=3,
                retry_on_timeout=True,
                request_timeout=30.0,  # Utiliser request_timeout au lieu de timeout
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Test de connexion avec timeout
            logger.info("⏱️ Test de connexion...")
            connection_start = time.time()
            
            # Test basic de connexion
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
            
            return True
            
        except ConnectionError as e:
            logger.error(f"🔌 Erreur de connexion Elasticsearch: {e}")
            metrics_logger.error(f"elasticsearch.connection.failed,type=connection,attempt={self._connection_attempts}")
            self._handle_connection_error(e)
            return False
            
        except TransportError as e:
            logger.error(f"🚫 Erreur de transport Elasticsearch: {e}")
            logger.error(f"📍 Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            metrics_logger.error(f"elasticsearch.connection.failed,type=transport,status={getattr(e, 'status_code', 'unknown')}")
            return False
            
        except Exception as e:
            logger.error(f"💥 Erreur inattendue lors de l'initialisation Elasticsearch: {type(e).__name__}: {e}")
            logger.error(f"📍 Détails: {str(e)}", exc_info=True)
            metrics_logger.error(f"elasticsearch.connection.failed,type=unexpected,error={type(e).__name__}")
            return False
            
        finally:
            if not self._initialized:
                self.client = None
                logger.warning("⚠️ Client Elasticsearch non initialisé - recherche lexicale indisponible")
    
    async def _check_cluster_health(self):
        """Vérifie la santé du cluster Elasticsearch."""
        try:
            health = await self.client.cluster.health()
            status = health.get("status", "unknown")
            
            if status == "green":
                logger.info("💚 Cluster Elasticsearch: Santé EXCELLENTE")
            elif status == "yellow":
                logger.warning("💛 Cluster Elasticsearch: Santé ACCEPTABLE (réplicas manquants)")
            elif status == "red":
                logger.error("💔 Cluster Elasticsearch: Santé CRITIQUE")
            else:
                logger.warning(f"⚠️ Cluster Elasticsearch: Statut inconnu ({status})")
            
            logger.info(f"📊 Nœuds: {health.get('number_of_nodes', 0)}")
            logger.info(f"🗂️ Shards actifs: {health.get('active_shards', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Impossible de vérifier la santé du cluster: {e}")
    
    async def _setup_index(self):
        """Configure l'index pour les transactions."""
        try:
            # Vérifier si l'index existe
            exists = await self.client.indices.exists(index=self.index_name)
            
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
                
                await self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"✅ Index '{self.index_name}' créé avec succès")
            else:
                logger.info(f"📚 Index '{self.index_name}' existe déjà")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration de l'index: {e}")
    
    def _mask_credentials(self, url: str) -> str:
        """Masque les credentials dans l'URL pour l'affichage."""
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
    
    async def is_healthy(self) -> bool:
        """Vérifie si le client est sain et fonctionnel."""
        if not self.client or not self._initialized:
            return False
        
        try:
            # Test rapide de ping
            start_time = time.time()
            health = await self.client.cluster.health()
            response_time = time.time() - start_time
            
            status = health.get("status", "red")
            is_healthy = status in ["green", "yellow"]
            
            # Mettre à jour le cache de santé
            self._last_health_check = {
                "timestamp": time.time(),
                "healthy": is_healthy,
                "status": status,
                "response_time": response_time
            }
            
            return is_healthy
            
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
        """Recherche des transactions avec logging des performances."""
        if not self.client or not self._initialized:
            raise RuntimeError("Client Elasticsearch non initialisé")
        
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
                for field, value in filters.items():
                    if value is not None:
                        search_body["query"]["bool"]["filter"] = search_body["query"]["bool"].get("filter", [])
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
            response = await self.client.search(index=self.index_name, body=search_body)
            
            query_time = time.time() - start_time
            
            # Analyser les résultats
            hits = response.get("hits", {}).get("hits", [])
            total_hits = response.get("hits", {}).get("total", {})
            
            if isinstance(total_hits, dict):
                total_count = total_hits.get("value", 0)
            else:
                total_count = total_hits
            
            # Formater les résultats
            results = []
            scores = []
            
            for hit in hits:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                }
                
                # Ajouter les highlights si disponibles
                if "highlight" in hit:
                    result["highlights"] = hit["highlight"]
                
                results.append(result)
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
                f"elasticsearch.search.success,"
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
            metrics_logger.error(f"elasticsearch.search.failed,user_id={user_id},time={query_time:.3f},error={type(e).__name__}")
            return []
    
    async def close(self):
        """Ferme la connexion avec logging."""
        if self.client:
            logger.info("🔒 Fermeture connexion Elasticsearch...")
            await self.client.close()
            self._initialized = False
            logger.info("✅ Connexion Elasticsearch fermée")
        else:
            logger.debug("🔒 Aucun client à fermer")