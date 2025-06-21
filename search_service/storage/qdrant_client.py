"""
Client Qdrant avec logging amélioré pour la recherche sémantique.
"""
import logging
import time
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, Match, Range, PointStruct, VectorParams, Distance
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from config_service.config import settings

# Configuration des loggers
logger = logging.getLogger("search_service.qdrant")
metrics_logger = logging.getLogger("search_service.metrics.qdrant")


class QdrantClient:
    """Client Qdrant avec logging amélioré pour la recherche sémantique."""
    
    def __init__(self):
        self.client = None
        self.collection_name = "financial_transactions"
        self._initialized = False
        self._connection_attempts = 0
        self._last_health_check = None
        self._collection_info = None
        
    async def initialize(self):
        """Initialise la connexion Qdrant avec logging détaillé."""
        logger.info("🔄 Initialisation du client Qdrant...")
        start_time = time.time()
        
        if not settings.QDRANT_URL:
            logger.error("❌ QDRANT_URL non configurée")
            return False
        
        # Log de la configuration (masquer API key)
        safe_url = self._mask_api_key(settings.QDRANT_URL)
        logger.info(f"🔗 Connexion à Qdrant: {safe_url}")
        
        if settings.QDRANT_API_KEY:
            logger.info("🔑 Authentification API key configurée")
        else:
            logger.info("🔓 Connexion sans authentification")
        
        try:
            self._connection_attempts += 1
            logger.info(f"🔄 Tentative de connexion #{self._connection_attempts}")
            
            # Créer le client
            connection_start = time.time()
            
            if settings.QDRANT_API_KEY:
                self.client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=30.0
                )
            else:
                self.client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    timeout=30.0
                )
            
            # Test de connexion
            logger.info("⏱️ Test de connexion Qdrant...")
            collections = await self.client.get_collections()
            
            connection_time = time.time() - connection_start
            total_time = time.time() - start_time
            
            # Vérification des collections
            collection_names = [col.name for col in collections.collections]
            logger.info(f"✅ Connexion Qdrant réussie en {connection_time:.3f}s")
            logger.info(f"📚 Collections disponibles: {len(collection_names)}")
            
            for col_name in collection_names:
                logger.info(f"   - {col_name}")
            
            # Vérifier si notre collection existe
            if self.collection_name in collection_names:
                logger.info(f"✅ Collection '{self.collection_name}' trouvée")
                # Obtenir les infos de la collection
                try:
                    collection_info = await self.client.get_collection(self.collection_name)
                    self._collection_info = {
                        "vectors_count": collection_info.vectors_count,
                        "indexed_vectors_count": collection_info.indexed_vectors_count,
                        "points_count": collection_info.points_count,
                        "segments_count": collection_info.segments_count,
                        "status": collection_info.status
                    }
                    logger.info(f"📊 Collection '{self.collection_name}': {collection_info.points_count} points")
                except Exception as collection_error:
                    logger.warning(f"⚠️ Erreur récupération infos collection: {collection_error}")
            else:
                logger.warning(f"⚠️ Collection '{self.collection_name}' non trouvée")
                logger.info("💡 La collection sera créée lors du premier enrichissement")
            
            # Marquer comme initialisé
            self._initialized = True
            logger.info(f"🎉 Client Qdrant prêt en {total_time:.3f}s")
            
            return True
            
        except Exception as e:
            connection_time = time.time() - start_time
            self._handle_connection_error(e, connection_time)
            return False
    
    def _mask_api_key(self, url: str) -> str:
        """Masque l'API key dans l'URL pour les logs."""
        # Qdrant URL est généralement juste l'endpoint, pas de credentials dans l'URL
        return url
    
    def _handle_connection_error(self, error, connection_time):
        """Gère les erreurs de connexion avec diagnostic détaillé."""
        error_type = error.__class__.__name__
        error_msg = str(error)
        
        logger.error(f"💥 Erreur Qdrant après {connection_time:.3f}s:")
        logger.error(f"   Type: {error_type}")
        logger.error(f"   Message: {error_msg}")
        
        # Diagnostic spécifique des erreurs
        error_str = error_msg.lower()
        if "connection" in error_str or "timeout" in error_str:
            logger.error("🔌 DIAGNOSTIC: Problème de connectivité réseau")
            logger.error("   - Vérifiez l'URL Qdrant")
            logger.error("   - Vérifiez la connectivité réseau")
        elif "401" in error_msg or "403" in error_msg or "auth" in error_str:
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez QDRANT_API_KEY")
        elif "ssl" in error_str or "certificate" in error_str:
            logger.error("🔒 DIAGNOSTIC: Problème SSL/TLS")
        
        # Suggestions de diagnostic
        logger.error("🔧 Actions de diagnostic suggérées:")
        logger.error("   - Vérifier la connectivité réseau vers Qdrant")
        logger.error("   - Valider QDRANT_URL et QDRANT_API_KEY")
        logger.error("   - Contrôler l'état du service Qdrant")
    
    def _handle_response_error(self, error):
        """Gère les erreurs de réponse avec diagnostic."""
        status_code = getattr(error, 'status_code', 'unknown')
        error_type = error.__class__.__name__
        error_msg = str(error)
        
        logger.error("🚫 Diagnostic de l'erreur Qdrant:")
        logger.error(f"   - Type: {error_type}")
        logger.error(f"   - Status: {status_code}")
        logger.error(f"   - Message: {error_msg}")
        
        # Diagnostic selon le code d'erreur
        if status_code == 401:
            logger.error("🔑 Erreur d'authentification - vérifier QDRANT_API_KEY")
        elif status_code == 403:
            logger.error("🚫 Accès refusé - vérifier les permissions")
        elif status_code == 404:
            logger.error("❌ Ressource non trouvée - vérifier le nom de la collection")
        elif status_code == 429:
            logger.error("🐌 Rate limiting - réduire la fréquence des requêtes")
        elif status_code >= 500:
            logger.error("🚨 Erreur serveur Qdrant - problème côté serveur")
        
        # Suggestions de diagnostic
        logger.error("🔧 Actions de diagnostic suggérées:")
        logger.error("   - Vérifier la connectivité réseau vers Qdrant")
        logger.error("   - Valider QDRANT_URL et QDRANT_API_KEY")
        logger.error("   - Contrôler l'état du service Qdrant")
        logger.error("   - Vérifier que la collection existe")
    
    async def is_healthy(self) -> bool:
        """Vérifie si le client est sain et fonctionnel."""
        if not self.client or not self._initialized:
            return False
        
        try:
            # Test rapide de connexion
            start_time = time.time()
            collections = await self.client.get_collections()
            response_time = time.time() - start_time
            
            is_healthy = len(collections.collections) >= 0  # Au moins le service répond
            
            # Mettre à jour le cache de santé
            self._last_health_check = {
                "timestamp": time.time(),
                "healthy": is_healthy,
                "response_time": response_time,
                "collections_count": len(collections.collections)
            }
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    async def search(
        self,
        query_vector: List[float],
        user_id: int,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Effectue une recherche vectorielle avec logging des performances."""
        if not self.client or not self._initialized:
            raise RuntimeError("Client Qdrant non initialisé")
        
        search_id = f"search_{int(time.time() * 1000)}"
        logger.info(f"🎯 [{search_id}] Recherche vectorielle pour user {user_id} (limit: {limit})")
        
        start_time = time.time()
        
        try:
            # Construire le filtre pour l'utilisateur
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=Match(value=user_id)
                    )
                ]
            )
            
            # Ajouter des filtres supplémentaires si spécifiés
            if filters:
                for field, value in filters.items():
                    if value is not None:
                        if isinstance(value, (int, float)):
                            # Filtre numérique
                            search_filter.must.append(
                                FieldCondition(
                                    key=field,
                                    match=Match(value=value)
                                )
                            )
                        elif isinstance(value, str):
                            # Filtre texte
                            search_filter.must.append(
                                FieldCondition(
                                    key=field,
                                    match=Match(value=value)
                                )
                            )
                        elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                            # Filtre de plage
                            search_filter.must.append(
                                FieldCondition(
                                    key=field,
                                    range=Range(
                                        gte=value.get("gte"),
                                        lte=value.get("lte")
                                    )
                                )
                            )
            
            # Exécuter la recherche
            logger.info(f"🎯 [{search_id}] Exécution recherche vectorielle...")
            query_start = time.time()
            
            # Utiliser query_points au lieu de search (deprecated)
            try:
                search_result = await self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False  # Pas besoin des vecteurs en retour
                )
            except AttributeError:
                # Fallback pour les versions plus anciennes
                search_result = await self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
            
            query_time = time.time() - query_start
            total_time = time.time() - start_time
            
            # Analyser les résultats
            results = []
            scores = []
            
            for point in search_result.points if hasattr(search_result, 'points') else search_result:
                result_data = {
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": point.payload
                }
                results.append(result_data)
                scores.append(point.score)
            
            # Métriques de performance
            avg_score = sum(scores) / len(scores) if scores else 0
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 0
            
            # Logging des résultats
            logger.info(f"✅ [{search_id}] Recherche terminée:")
            logger.info(f"   - Temps total: {total_time:.3f}s")
            logger.info(f"   - Temps requête: {query_time:.3f}s")
            logger.info(f"   - Résultats: {len(results)}/{limit}")
            logger.info(f"   - Score moyen: {avg_score:.3f}")
            
            if len(results) > 0:
                logger.info(f"   - Score min/max: {min_score:.3f}/{max_score:.3f}")
            
            # Métriques pour monitoring
            metrics_logger.info("qdrant_search", extra={
                "search_id": search_id,
                "user_id": user_id,
                "query_time": query_time,
                "total_time": total_time,
                "results_count": len(results),
                "requested_limit": limit,
                "avg_score": avg_score,
                "filters_count": len(filters) if filters else 0
            })
            
            return results
            
        except ResponseHandlingException as e:
            search_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur Qdrant après {search_time:.3f}s:")
            self._handle_response_error(e)
            raise
        except Exception as e:
            search_time = time.time() - start_time
            error_type = e.__class__.__name__
            logger.error(f"❌ [{search_id}] Erreur générale après {search_time:.3f}s:")
            logger.error(f"   Type: {error_type}")
            logger.error(f"   Message: {str(e)}")
            raise
    
    async def get_collections(self):
        """Retourne la liste des collections disponibles."""
        if not self.client or not self._initialized:
            raise RuntimeError("Client Qdrant non initialisé")
        
        try:
            collections = await self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"❌ Erreur récupération collections: {e}")
            raise
    
    async def collection_exists(self, collection_name: str = None) -> bool:
        """Vérifie si une collection existe."""
        collection_name = collection_name or self.collection_name
        
        try:
            collections = await self.get_collections()
            return collection_name in collections
        except Exception:
            return False
    
    async def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Retourne les informations détaillées d'une collection."""
        collection_name = collection_name or self.collection_name
        
        if not self.client or not self._initialized:
            raise RuntimeError("Client Qdrant non initialisé")
        
        try:
            collection_info = await self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status),
                "optimizer_status": collection_info.optimizer_status.value if hasattr(collection_info.optimizer_status, 'value') else str(collection_info.optimizer_status)
            }
        except Exception as e:
            logger.error(f"❌ Erreur récupération infos collection '{collection_name}': {e}")
            raise
    
    async def close(self):
        """Ferme la connexion Qdrant."""
        if self.client:
            try:
                await self.client.close()
                logger.info("✅ Connexion Qdrant fermée proprement")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture Qdrant: {e}")
            finally:
                self.client = None
                self._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du client."""
        return {
            "initialized": self._initialized,
            "connection_attempts": self._connection_attempts,
            "last_health_check": self._last_health_check,
            "collection_name": self.collection_name,
            "collection_info": self._collection_info
        }