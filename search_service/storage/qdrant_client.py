"""
Client Qdrant avec logging amélioré pour la recherche sémantique.
"""
import logging
import time
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, Match, Range
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
            return
        
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
                logger.error(f"❌ Collection '{self.collection_name}' non trouvée")
                logger.error("🔧 La collection doit être créée par l'enrichment_service")
                self._initialized = False
            
            # Métriques de connexion
            metrics_logger.info(
                f"qdrant.connection.success,"
                f"time={connection_time:.3f},"
                f"attempt={self._connection_attempts},"
                f"collections={len(collection_names)}"
            )
            
            total_time = time.time() - start_time
            if self._initialized:
                logger.info(f"🎉 Client Qdrant initialisé avec succès en {total_time:.2f}s")
            else:
                logger.warning(f"⚠️ Client Qdrant partiellement initialisé en {total_time:.2f}s")
                
        except ResponseHandlingException as e:
            logger.error(f"🚫 Erreur de réponse Qdrant: {e}")
            logger.error(f"📍 Status code: {getattr(e, 'status_code', 'unknown')}")
            metrics_logger.error(f"qdrant.connection.failed,type=response_error,attempt={self._connection_attempts}")
            self._handle_response_error(e)
            
        except UnexpectedResponse as e:
            logger.error(f"🚫 Réponse inattendue de Qdrant: {e}")
            metrics_logger.error(f"qdrant.connection.failed,type=unexpected_response,attempt={self._connection_attempts}")
            
        except Exception as e:
            logger.error(f"💥 Erreur inattendue lors de l'initialisation Qdrant: {type(e).__name__}: {e}")
            logger.error(f"📍 Détails", exc_info=True)
            metrics_logger.error(f"qdrant.connection.failed,type=unexpected,error={type(e).__name__}")
            
        finally:
            if not self._initialized:
                self.client = None
                logger.warning("⚠️ Client Qdrant non initialisé - recherche sémantique indisponible")
    
    async def _analyze_collection(self):
        """Analyse la collection pour obtenir des métriques."""
        try:
            logger.info(f"🔍 Analyse de la collection '{self.collection_name}'...")
            
            # Informations sur la collection
            collection_info = await self.client.get_collection(self.collection_name)
            self._collection_info = collection_info
            
            # Métriques de la collection
            points_count = collection_info.points_count
            vectors_count = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 'unknown'
            config = collection_info.config
            
            logger.info(f"📊 Points dans la collection: {points_count}")
            logger.info(f"📊 Vecteurs: {vectors_count}")
            
            if config:
                logger.info(f"📊 Dimension des vecteurs: {config.params.vectors.size}")
                logger.info(f"📊 Distance: {config.params.vectors.distance}")
            
            # Métriques pour monitoring
            metrics_logger.info(
                f"qdrant.collection.info,"
                f"name={self.collection_name},"
                f"points={points_count},"
                f"vectors={vectors_count}"
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse de la collection: {e}")
    
    async def search(
        self,
        query_vector: List[float],
        user_id: int,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Effectue une recherche vectorielle avec logging détaillé."""
        if not self.client:
            logger.error("❌ Client Qdrant non initialisé")
            return []
        
        if not self._initialized:
            logger.error("❌ Collection non disponible")
            return []
        
        search_id = f"vector_search_{int(time.time()*1000)}"
        logger.info(f"🎯 [{search_id}] Début recherche vectorielle pour user_id={user_id}")
        
        start_time = time.time()
        
        try:
            # Log des paramètres
            vector_dim = len(query_vector) if query_vector else 0
            logger.debug(f"🎯 [{search_id}] Vecteur dimension: {vector_dim}")
            logger.debug(f"🎯 [{search_id}] Limite: {limit}")
            logger.debug(f"🎯 [{search_id}] Filtres: {filters}")
            
            # Validation du vecteur
            if not query_vector:
                logger.error(f"❌ [{search_id}] Vecteur de requête vide")
                return []
            
            if vector_dim != 1536:  # Dimension OpenAI text-embedding-3-small
                logger.warning(f"⚠️ [{search_id}] Dimension inattendue: {vector_dim} (attendu: 1536)")
            
            # Construire les filtres Qdrant
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=Match(value=user_id)
                    ),
                    FieldCondition(
                        key="is_deleted",
                        match=Match(value=False)
                    )
                ]
            )
            
            # Ajouter des filtres additionnels
            filter_count = 2  # user_id + is_deleted
            if filters:
                logger.debug(f"🎯 [{search_id}] Application des filtres personnalisés...")
                
                if "amount_min" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="amount_abs",
                            range=Range(gte=filters["amount_min"])
                        )
                    )
                    filter_count += 1
                
                if "amount_max" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="amount_abs",
                            range=Range(lte=filters["amount_max"])
                        )
                    )
                    filter_count += 1
                
                if "transaction_type" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="transaction_type",
                            match=Match(value=filters["transaction_type"])
                        )
                    )
                    filter_count += 1
                
                if "date_from" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="timestamp",
                            range=Range(gte=filters["date_from"])
                        )
                    )
                    filter_count += 1
                
                if "date_to" in filters:
                    search_filter.must.append(
                        FieldCondition(
                            key="timestamp",
                            range=Range(lte=filters["date_to"])
                        )
                    )
                    filter_count += 1
            
            logger.debug(f"🎯 [{search_id}] Total filtres: {filter_count}")
            
            # Exécuter la recherche
            logger.info(f"🎯 [{search_id}] Exécution recherche vectorielle...")
            query_start = time.time()
            
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Pas besoin des vecteurs en retour
            )
            
            query_time = time.time() - query_start
            total_time = time.time() - start_time
            
            # Analyser les résultats
            results = []
            scores = []
            
            for point in search_result:
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
                f"filters={filter_count},"
                f"max_score={max_score:.3f},"
                f"avg_score={avg_score:.3f}"
            )
            
            # Log détaillé des top résultats en mode debug
            if logger.isEnabledFor(logging.DEBUG):
                for i, result in enumerate(results[:3]):  # Top 3 seulement
                    payload = result["payload"]
                    description = payload.get("primary_description", "N/A")[:50]
                    amount = payload.get("amount", "N/A")
                    logger.debug(
                        f"🎯 [{search_id}] #{i+1}: "
                        f"score={result['score']:.3f}, "
                        f"amount={amount}, "
                        f"desc='{description}...'"
                    )
            
            return results
            
        except ResponseHandlingException as e:
            logger.error(f"🚫 [{search_id}] Erreur de réponse Qdrant: {e}")
            metrics_logger.error(f"qdrant.search.failed,type=response_error,user_id={user_id}")
            self._handle_response_error(e)
            return []
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"💥 [{search_id}] Erreur inattendue après {search_time:.3f}s: {type(e).__name__}: {e}")
            logger.error(f"📍 [{search_id}] Détails", exc_info=True)
            metrics_logger.error(f"qdrant.search.failed,type=unexpected,user_id={user_id},error={type(e).__name__}")
            return []
    
    async def is_healthy(self) -> bool:
        """Vérifie l'état de santé avec logging détaillé."""
        if not self.client:
            logger.debug("❌ Client non initialisé")
            return False
        
        try:
            logger.debug("🩺 Vérification santé Qdrant...")
            start_time = time.time()
            
            # Test simple de connexion
            collections = await self.client.get_collections()
            ping_time = time.time() - start_time
            
            collection_names = [col.name for col in collections.collections]
            collection_exists = self.collection_name in collection_names
            
            if collection_exists:
                logger.debug(f"✅ Ping Qdrant réussi en {ping_time:.3f}s")
                metrics_logger.info(f"qdrant.health.ping.success,time={ping_time:.3f}")
                
                # Vérification détaillée périodique
                if time.time() - (self._last_health_check or 0) > 60:  # Chaque minute
                    await self._detailed_health_check()
                    self._last_health_check = time.time()
                
                return True
            else:
                logger.warning(f"⚠️ Collection {self.collection_name} non trouvée")
                metrics_logger.warning("qdrant.health.collection.missing")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du health check: {type(e).__name__}: {e}")
            metrics_logger.error(f"qdrant.health.check.failed,error={type(e).__name__}")
            return False
    
    async def _detailed_health_check(self):
        """Effectue une vérification de santé approfondie."""
        try:
            logger.debug("🔍 Vérification santé Qdrant détaillée...")
            
            # Statistiques de la collection
            collection_info = await self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            
            # Vérifier si le nombre de points a changé
            if self._collection_info:
                previous_count = self._collection_info.points_count
                if points_count != previous_count:
                    change = points_count - previous_count
                    logger.info(f"📊 Collection mise à jour: {change:+d} points ({points_count} total)")
            
            self._collection_info = collection_info
            
            # Test de recherche simple (vecteur zéro pour test)
            test_vector = [0.0] * 1536  # Dimension standard
            test_start = time.time()
            
            await self.client.search(
                collection_name=self.collection_name,
                query_vector=test_vector,
                limit=1,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=Match(value=1))  # Test user
                    ]
                )
            )
            
            test_time = time.time() - test_start
            logger.debug(f"✅ Test recherche réussi en {test_time:.3f}s")
            
            metrics_logger.info(
                f"qdrant.health.detailed,"
                f"points={points_count},"
                f"test_search_time={test_time:.3f}"
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification détaillée: {e}")
    
    def _mask_api_key(self, url: str) -> str:
        """Masque l'API key dans l'URL pour les logs."""
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
    
    async def close(self):
        """Ferme la connexion avec logging."""
        if self.client:
            logger.info("🔒 Fermeture connexion Qdrant...")
            await self.client.close()
            self._initialized = False
            logger.info("✅ Connexion Qdrant fermée")
        else:
            logger.debug("🔒 Aucun client à fermer")