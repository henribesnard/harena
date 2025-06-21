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
            
            logger.info(f"✅ Connexion Qdrant réussie en {connection_time:.2f}s")
            
            # Log des collections disponibles
            collection_names = [col.name for col in collections.collections]
            logger.info(f"📚 Collections disponibles: {collection_names}")
            
            # Vérifier notre collection
            if self.collection_name in collection_names:
                logger.info(f"✅ Collection '{self.collection_name}' trouvée")
                await self._analyze_collection()
                self._initialized = True
            else:
                logger.warning(f"⚠️ Collection '{self.collection_name}' non trouvée")
                logger.info("🔧 La collection sera créée automatiquement si nécessaire")
                # On peut tout de même marquer comme initialisé pour permettre la création
                self._initialized = True
            
            # Métriques de connexion
            metrics_logger.info(
                f"qdrant.connection.success,"
                f"time={connection_time:.3f},"
                f"attempt={self._connection_attempts},"
                f"collections={len(collection_names)}"
            )
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Qdrant initialisé avec succès en {total_time:.2f}s")
            return True
                
        except ResponseHandlingException as e:
            logger.error(f"🚫 Erreur de réponse Qdrant: {e}")
            logger.error(f"📍 Status code: {getattr(e, 'status_code', 'unknown')}")
            metrics_logger.error(f"qdrant.connection.failed,type=response_error,attempt={self._connection_attempts}")
            self._handle_response_error(e)
            return False
            
        except UnexpectedResponse as e:
            logger.error(f"🚫 Réponse inattendue de Qdrant: {e}")
            metrics_logger.error(f"qdrant.connection.failed,type=unexpected_response,attempt={self._connection_attempts}")
            return False
            
        except Exception as e:
            logger.error(f"💥 Erreur inattendue lors de l'initialisation Qdrant: {type(e).__name__}: {e}")
            logger.error(f"📍 Détails", exc_info=True)
            metrics_logger.error(f"qdrant.connection.failed,type=unexpected,error={type(e).__name__}")
            return False
            
        finally:
            if not self._initialized:
                self.client = None
                logger.warning("⚠️ Client Qdrant non initialisé - recherche sémantique indisponible")
    
    async def _analyze_collection(self):
        """Analyse la collection pour obtenir des métriques."""
        try:
            # Obtenir les informations de la collection
            collection_info = await self.client.get_collection(self.collection_name)
            
            points_count = collection_info.points_count
            vectors_count = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else points_count
            
            logger.info(f"📊 Collection '{self.collection_name}':")
            logger.info(f"   📍 Points: {points_count}")
            logger.info(f"   🎯 Vecteurs: {vectors_count}")
            
            # Stocker les informations pour le monitoring
            self._collection_info = {
                "points_count": points_count,
                "vectors_count": vectors_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status.dict() if hasattr(collection_info, 'optimizer_status') else {}
            }
            
            # Log des détails de configuration
            if hasattr(collection_info, 'config'):
                config = collection_info.config
                if hasattr(config, 'params'):
                    logger.info(f"   🔧 Configuration: {config.params.vectors}")
                
        except Exception as e:
            logger.warning(f"⚠️ Impossible d'analyser la collection: {e}")
            self._collection_info = {"error": str(e)}
    
    def _mask_api_key(self, url: str) -> str:
        """Masque l'API key dans l'URL pour l'affichage."""
        # Qdrant URL est généralement juste l'endpoint, pas de credentials dans l'URL
        return url
    
    def _handle_response_error(self, error):
        """Gère les erreurs de réponse avec diagnostic."""
        status_code = getattr(error, 'status_code', 'unknown')
        
        logger.error("🚫 Diagnostic de l'erreur Qdrant:")
        logger.error(f"   - Type: {type(error).__name__}")
        logger.error(f"   - Status: {status_code}")
        logger.error(f"   - Message: {str(error)}")
        
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
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                results.append(result)
                scores.append(point.score)
            
            # Statistiques des scores
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
            else:
                max_score = min_score = avg_score = 0
            
            # Logs de résultats
            logger.info(f"✅ [{search_id}] Recherche terminée en {total_time:.3f}s")
            logger.info(f"📊 [{search_id}] Résultats: {len(results)}")
            logger.info(f"⏱️ [{search_id}] Temps requête: {query_time:.3f}s")
            logger.info(f"🎯 [{search_id}] Scores: max={max_score:.3f}, min={min_score:.3f}, avg={avg_score:.3f}")
            
            # Métriques détaillées
            metrics_logger.info(
                f"qdrant.search.success,"
                f"user_id={user_id},"
                f"query_time={query_time:.3f},"
                f"total_time={total_time:.3f},"
                f"results={len(results)},"
                f"max_score={max_score:.3f}"
            )
            
            return results
            
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur recherche vectorielle après {query_time:.3f}s: {e}")
            metrics_logger.error(f"qdrant.search.failed,user_id={user_id},time={query_time:.3f},error={type(e).__name__}")
            return []
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Retourne les informations de la collection."""
        if not self.client or not self._initialized:
            return {"error": "Client not initialized"}
        
        try:
            if not self._collection_info:
                await self._analyze_collection()
            
            return self._collection_info or {"error": "Collection info not available"}
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des infos collection: {e}")
            return {"error": str(e)}
    
    async def create_collection_if_not_exists(
        self,
        vector_size: int = 1536,
        distance_metric: str = "Cosine"
    ) -> bool:
        """Crée la collection si elle n'existe pas."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return False
        
        try:
            # Vérifier si la collection existe
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"✅ Collection '{self.collection_name}' existe déjà")
                return True
            
            # Créer la collection
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            logger.info(f"🔧 Création de la collection '{self.collection_name}'...")
            
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance_metric, Distance.COSINE)
                )
            )
            
            logger.info(f"✅ Collection '{self.collection_name}' créée avec succès")
            await self._analyze_collection()
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de la collection: {e}")
            return False
    
    async def upsert_points(
        self,
        points: List[Dict[str, Any]]
    ) -> bool:
        """Insère ou met à jour des points dans la collection."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return False
        
        if not points:
            logger.warning("⚠️ Aucun point à insérer")
            return True
        
        try:
            # Convertir les points au format Qdrant
            qdrant_points = []
            for point in points:
                qdrant_point = PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {})
                )
                qdrant_points.append(qdrant_point)
            
            logger.info(f"💾 Insertion de {len(qdrant_points)} points dans '{self.collection_name}'...")
            
            # Insérer les points
            operation_info = await self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )
            
            logger.info(f"✅ {len(qdrant_points)} points insérés avec succès")
            logger.info(f"📊 Opération: {operation_info.status}")
            
            # Mettre à jour les informations de la collection
            await self._analyze_collection()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'insertion des points: {e}")
            return False
    
    async def delete_points(
        self,
        point_ids: List[str]
    ) -> bool:
        """Supprime des points de la collection."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return False
        
        if not point_ids:
            logger.warning("⚠️ Aucun point à supprimer")
            return True
        
        try:
            logger.info(f"🗑️ Suppression de {len(point_ids)} points de '{self.collection_name}'...")
            
            operation_info = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            logger.info(f"✅ {len(point_ids)} points supprimés avec succès")
            logger.info(f"📊 Opération: {operation_info.status}")
            
            # Mettre à jour les informations de la collection
            await self._analyze_collection()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression des points: {e}")
            return False
    
    async def scroll_points(
        self,
        user_id: Optional[int] = None,
        limit: int = 100,
        offset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parcourt les points de la collection avec pagination."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return {"points": [], "next_page_offset": None}
        
        try:
            scroll_filter = None
            if user_id is not None:
                scroll_filter = Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=Match(value=user_id)
                        )
                    ]
                )
            
            logger.info(f"📖 Parcours des points (user_id: {user_id}, limit: {limit})")
            
            result = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = []
            for point in result[0]:  # result[0] contient les points
                points.append({
                    "id": point.id,
                    "payload": point.payload
                })
            
            next_offset = result[1]  # result[1] contient le next_page_offset
            
            logger.info(f"📊 {len(points)} points récupérés")
            
            return {
                "points": points,
                "next_page_offset": next_offset
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du parcours des points: {e}")
            return {"points": [], "next_page_offset": None}
    
    async def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un point spécifique par son ID."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return None
        
        try:
            logger.info(f"🔍 Récupération du point {point_id}")
            
            result = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            
            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "payload": point.payload
                }
            else:
                logger.warning(f"⚠️ Point {point_id} non trouvé")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération du point {point_id}: {e}")
            return None
    
    async def count_points(self, user_id: Optional[int] = None) -> int:
        """Compte le nombre de points dans la collection."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return 0
        
        try:
            count_filter = None
            if user_id is not None:
                count_filter = Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=Match(value=user_id)
                        )
                    ]
                )
            
            result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True
            )
            
            count = result.count
            logger.info(f"📊 Nombre de points (user_id: {user_id}): {count}")
            
            return count
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du comptage des points: {e}")
            return 0
    
    async def get_collections_list(self) -> List[str]:
        """Retourne la liste des collections disponibles."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return []
        
        try:
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            logger.info(f"📚 Collections disponibles: {collection_names}")
            return collection_names
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des collections: {e}")
            return []
    
    async def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Vérifie si une collection existe."""
        if not self.client or not self._initialized:
            return False
        
        target_collection = collection_name or self.collection_name
        
        try:
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            exists = target_collection in collection_names
            logger.info(f"🔍 Collection '{target_collection}': {'Existe' if exists else 'N\'existe pas'}")
            
            return exists
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de la collection: {e}")
            return False
    
    async def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Supprime une collection."""
        if not self.client or not self._initialized:
            logger.error("❌ Client Qdrant non initialisé")
            return False
        
        target_collection = collection_name or self.collection_name
        
        try:
            logger.warning(f"🗑️ Suppression de la collection '{target_collection}'...")
            
            await self.client.delete_collection(collection_name=target_collection)
            
            logger.info(f"✅ Collection '{target_collection}' supprimée avec succès")
            
            # Réinitialiser les infos si c'est notre collection principale
            if target_collection == self.collection_name:
                self._collection_info = None
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression de la collection: {e}")
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le cluster Qdrant."""
        if not self.client or not self._initialized:
            return {"error": "Client not initialized"}
        
        try:
            # Obtenir les informations du cluster
            cluster_info = await self.client.get_cluster_info()
            
            return {
                "peer_id": cluster_info.peer_id if hasattr(cluster_info, 'peer_id') else "unknown",
                "raft_info": cluster_info.raft_info.dict() if hasattr(cluster_info, 'raft_info') else {},
                "status": "healthy",
                "collections_count": len(await self.get_collections_list())
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des infos cluster: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Ferme la connexion avec logging."""
        if self.client:
            logger.info("🔒 Fermeture connexion Qdrant...")
            await self.client.close()
            self._initialized = False
            logger.info("✅ Connexion Qdrant fermée")
        else:
            logger.debug("🔒 Aucun client à fermer")