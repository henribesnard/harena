"""
Client Qdrant pour le service de recherche.

Ce module fournit une interface optimisée pour interagir avec
Qdrant pour les recherches sémantiques de transactions financières.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import aiohttp

from search_service.clients.base_client import BaseClient, RetryConfig, CircuitBreakerConfig, HealthCheckConfig

logger = logging.getLogger(__name__)


class QdrantClient(BaseClient):
    """
    Client pour Qdrant optimisé pour les recherches sémantiques.
    
    Responsabilités:
    - Recherches vectorielles par similarité
    - Gestion des filtres par métadonnées
    - Optimisation des seuils de similarité
    - Monitoring des performances vectorielles
    """
    
    def __init__(
        self,
        qdrant_url: str,
        api_key: Optional[str] = None,
        collection_name: str = "financial_transactions",
        timeout: float = 8.0,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        headers = {}
        if api_key:
            headers["api-key"] = api_key
        
        health_check_config = HealthCheckConfig(
            enabled=True,
            interval_seconds=30.0,
            timeout_seconds=5.0,
            endpoint="/"
        )
        
        super().__init__(
            base_url=qdrant_url,
            service_name="qdrant",
            timeout=timeout,
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            health_check_config=health_check_config,
            headers=headers
        )
        
        self.collection_name = collection_name
        self.api_key = api_key
        
        # Cache des requêtes vectorielles récentes
        self._vector_cache: Dict[str, List[float]] = {}
        
        logger.info(f"Qdrant client initialized for collection: {collection_name}")
    
    async def test_connection(self) -> bool:
        """Teste la connectivité de base à Qdrant."""
        try:
            async def _test():
                async with self.session.get(self.base_url) as response:
                    return response.status == 200
            
            return await self.execute_with_retry(_test, "connection_test")
        except Exception as e:
            logger.error(f"Qdrant connection test failed: {e}")
            return False
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Effectue une vérification de santé spécifique à Qdrant."""
        try:
            # Vérifier le service principal
            async with self.session.get(self.base_url) as response:
                if response.status == 200:
                    service_info = await response.json()
                    
                    # Vérifier l'existence de la collection
                    collection_exists, collection_info = await self._check_collection_status()
                    
                    return {
                        "version": service_info.get("version", "unknown"),
                        "collection_exists": collection_exists,
                        "collection_info": collection_info,
                        "status": "healthy" if collection_exists else "degraded"
                    }
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_collection_status(self) -> tuple[bool, Dict[str, Any]]:
        """Vérifie le statut de la collection."""
        try:
            async with self.session.get(
                f"{self.base_url}/collections/{self.collection_name}"
            ) as response:
                if response.status == 200:
                    collection_data = await response.json()
                    result = collection_data.get("result", {})
                    
                    return True, {
                        "points_count": result.get("points_count", 0),
                        "vectors_config": result.get("config", {}).get("params", {}).get("vectors", {}),
                        "status": result.get("status", "unknown")
                    }
                else:
                    return False, {"error": f"HTTP {response.status}"}
        except Exception as e:
            return False, {"error": str(e)}
    
    # ============================================================================
    # MÉTHODES MANQUANTES AJOUTÉES POUR COMPATIBILITÉ AVEC LES MOTEURS
    # ============================================================================
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_filter: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche vectorielle dans Qdrant (méthode générique).
        
        Cette méthode est utilisée par le semantic_engine.py et doit être
        compatible avec l'interface attendue.
        
        Args:
            collection_name: Nom de la collection
            query_vector: Vecteur de requête
            query_filter: Filtres à appliquer
            limit: Nombre de résultats
            score_threshold: Seuil de score minimum
            with_payload: Inclure les métadonnées
            with_vectors: Inclure les vecteurs
            
        Returns:
            Liste des résultats de recherche
        """
        search_request = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vectors
        }
        
        if query_filter:
            search_request["filter"] = query_filter
        
        if score_threshold is not None:
            search_request["score_threshold"] = score_threshold
        
        async def _search():
            async with self.session.post(
                f"{self.base_url}/collections/{collection_name}/points/search",
                json=search_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Search failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_search, "search")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé du service Qdrant (méthode générique).
        
        Cette méthode est utilisée par les health checks et doit retourner
        un format standard.
        
        Returns:
            Statut de santé du service
        """
        return await self._perform_health_check()
    
    async def count(
        self,
        collection_name: str,
        count_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compte les points dans une collection (méthode générique).
        
        Cette méthode est utilisée par les moteurs pour compter les points.
        
        Args:
            collection_name: Nom de la collection
            count_filter: Filtres à appliquer
            
        Returns:
            Résultat du comptage
        """
        count_request = {}
        if count_filter:
            count_request["filter"] = count_filter
        
        async def _count():
            async with self.session.post(
                f"{self.base_url}/collections/{collection_name}/points/count",
                json=count_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    raise Exception(f"Count failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_count, "count")
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Obtient les informations d'une collection (méthode générique).
        
        Cette méthode est utilisée par les moteurs pour obtenir les infos de collection.
        
        Args:
            collection_name: Nom de la collection
            
        Returns:
            Informations de la collection
        """
        async def _get_info():
            async with self.session.get(
                f"{self.base_url}/collections/{collection_name}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    raise Exception(f"Get collection info failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_get_info, "get_collection_info")
    
    async def get_vector(
        self,
        collection_name: str,
        point_id: Union[int, str]
    ) -> Optional[List[float]]:
        """
        Récupère le vecteur d'un point spécifique (méthode générique).
        
        Cette méthode est utilisée par les moteurs pour récupérer des vecteurs.
        
        Args:
            collection_name: Nom de la collection
            point_id: ID du point
            
        Returns:
            Vecteur du point ou None si non trouvé
        """
        async def _get_vector():
            async with self.session.get(
                f"{self.base_url}/collections/{collection_name}/points/{point_id}",
                params={"with_vector": "true"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    point_data = result.get("result", {})
                    return point_data.get("vector")
                elif response.status == 404:
                    return None
                else:
                    error_text = await response.text()
                    raise Exception(f"Get vector failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_get_vector, "get_vector")
    
    # ============================================================================
    # MÉTHODES SPÉCIALISÉES EXISTANTES
    # ============================================================================
    
    async def search_similar_transactions(
        self,
        query_vector: List[float],
        user_id: int,
        limit: int = 15,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vector: bool = False
    ) -> Dict[str, Any]:
        """
        Recherche de transactions similaires par vecteur.
        
        Args:
            query_vector: Vecteur de la requête (embedding)
            user_id: ID de l'utilisateur
            limit: Nombre de résultats
            score_threshold: Seuil de similarité minimum
            filters: Filtres additionnels par métadonnées
            with_payload: Inclure les métadonnées
            with_vector: Inclure les vecteurs (généralement False)
            
        Returns:
            Résultats de recherche Qdrant
        """
        # Construction de la requête avec stratégie corrigée basée sur le validateur
        search_body = {
            "vector": query_vector,
            "limit": limit,
            "score_threshold": score_threshold,
            "with_payload": with_payload,
            "with_vector": with_vector
        }
        
        # Gestion des filtres - Le validateur montre des problèmes avec le filtrage Qdrant
        # On essaie d'abord avec filtre, puis sans filtre si échec
        
        # Essayer d'abord avec le filtre utilisateur
        if user_id:
            user_filter = {
                "must": [
                    {
                        "key": "user_id",
                        "match": {"value": user_id}
                    }
                ]
            }
            
            # Ajouter des filtres additionnels si fournis
            if filters:
                additional_filters = self._build_qdrant_filters(filters)
                if additional_filters:
                    user_filter["must"].extend(additional_filters)
            
            search_body["filter"] = user_filter
        
        async def _search_with_filter():
            async with self.session.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=search_body
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Search failed: HTTP {response.status} - {error_text}")
        
        try:
            # Essayer avec le filtre
            results = await self.execute_with_retry(_search_with_filter, "vector_search_filtered")
            
            # Si on a des résultats, les retourner
            if results:
                return {"result": results, "filtered": True}
                
        except Exception as e:
            logger.warning(f"Filtered search failed, trying unfiltered: {e}")
        
        # Fallback: recherche sans filtre avec filtrage manuel
        # Basé sur les observations du validateur qui montre des problèmes de filtre Qdrant
        search_body_unfiltered = {
            "vector": query_vector,
            "limit": limit * 3,  # Chercher plus pour avoir assez après filtrage
            "score_threshold": max(score_threshold - 0.1, 0.3),  # Seuil plus permissif
            "with_payload": with_payload,
            "with_vector": with_vector
        }
        
        async def _search_unfiltered():
            async with self.session.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=search_body_unfiltered
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    all_results = data.get("result", [])
                    
                    # Filtrage manuel par user_id
                    user_results = [
                        point for point in all_results
                        if point.get("payload", {}).get("user_id") == user_id
                    ]
                    
                    # Appliquer les filtres additionnels manuellement
                    if filters:
                        user_results = self._apply_manual_filters(user_results, filters)
                    
                    # Limiter au nombre demandé
                    return user_results[:limit]
                else:
                    error_text = await response.text()
                    raise Exception(f"Unfiltered search failed: HTTP {response.status} - {error_text}")
        
        try:
            results = await self.execute_with_retry(_search_unfiltered, "vector_search_unfiltered")
            return {"result": results, "filtered": False, "manual_filtering": True}
        except Exception as e:
            logger.error(f"All search strategies failed: {e}")
            return {"result": [], "error": str(e)}
    
    def _build_qdrant_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construit les filtres Qdrant à partir des paramètres."""
        qdrant_filters = []
        
        # Filtre de montant
        if "amount_min" in filters:
            qdrant_filters.append({
                "key": "amount",
                "range": {"gte": filters["amount_min"]}
            })
        
        if "amount_max" in filters:
            qdrant_filters.append({
                "key": "amount", 
                "range": {"lte": filters["amount_max"]}
            })
        
        # Filtre de date
        if "date_from" in filters:
            qdrant_filters.append({
                "key": "date",
                "range": {"gte": filters["date_from"]}
            })
        
        if "date_to" in filters:
            qdrant_filters.append({
                "key": "date",
                "range": {"lte": filters["date_to"]}
            })
        
        # Filtre de catégories
        if "category_ids" in filters and filters["category_ids"]:
            qdrant_filters.append({
                "key": "category_id",
                "match": {"any": filters["category_ids"]}
            })
        
        # Filtre de comptes
        if "account_ids" in filters and filters["account_ids"]:
            qdrant_filters.append({
                "key": "account_id",
                "match": {"any": filters["account_ids"]}
            })
        
        # Filtre de type de transaction
        if "transaction_type" in filters and filters["transaction_type"] != "all":
            qdrant_filters.append({
                "key": "transaction_type",
                "match": {"value": filters["transaction_type"]}
            })
        
        return qdrant_filters
    
    def _apply_manual_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Applique les filtres manuellement sur les résultats."""
        filtered_results = []
        
        for result in results:
            payload = result.get("payload", {})
            
            # Vérifier tous les filtres
            if self._matches_filters(payload, filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def _matches_filters(self, payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Vérifie si un payload correspond aux filtres."""
        # Filtre de montant
        amount = payload.get("amount", 0)
        if "amount_min" in filters and amount < filters["amount_min"]:
            return False
        if "amount_max" in filters and amount > filters["amount_max"]:
            return False
        
        # Filtre de date
        date_str = payload.get("date", "")
        if "date_from" in filters and date_str < filters["date_from"]:
            return False
        if "date_to" in filters and date_str > filters["date_to"]:
            return False
        
        # Filtre de catégories
        category_id = payload.get("category_id")
        if "category_ids" in filters and filters["category_ids"]:
            if category_id not in filters["category_ids"]:
                return False
        
        # Filtre de comptes
        account_id = payload.get("account_id")
        if "account_ids" in filters and filters["account_ids"]:
            if account_id not in filters["account_ids"]:
                return False
        
        # Filtre de type de transaction
        transaction_type = payload.get("transaction_type", "")
        if "transaction_type" in filters and filters["transaction_type"] != "all":
            if transaction_type != filters["transaction_type"]:
                return False
        
        return True
    
    async def get_similar_by_id(
        self,
        transaction_id: int,
        user_id: int,
        limit: int = 10,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Trouve des transactions similaires à une transaction donnée.
        
        Args:
            transaction_id: ID de la transaction de référence
            user_id: ID de l'utilisateur
            limit: Nombre de résultats
            score_threshold: Seuil de similarité
            
        Returns:
            Liste des transactions similaires
        """
        # D'abord récupérer le vecteur de la transaction de référence
        try:
            reference_point = await self._get_point_by_id(transaction_id)
            if not reference_point or "vector" not in reference_point:
                return []
            
            reference_vector = reference_point["vector"]
            
            # Chercher des transactions similaires
            results = await self.search_similar_transactions(
                query_vector=reference_vector,
                user_id=user_id,
                limit=limit + 1,  # +1 pour exclure la transaction elle-même
                score_threshold=score_threshold
            )
            
            # Filtrer la transaction de référence
            similar_transactions = [
                point for point in results.get("result", [])
                if point.get("payload", {}).get("transaction_id") != transaction_id
            ]
            
            return similar_transactions[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar transactions: {e}")
            return []
    
    async def _get_point_by_id(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """Récupère un point par son ID de transaction."""
        try:
            # Rechercher par transaction_id dans les métadonnées
            search_body = {
                "filter": {
                    "must": [
                        {
                            "key": "transaction_id",
                            "match": {"value": transaction_id}
                        }
                    ]
                },
                "limit": 1,
                "with_payload": True,
                "with_vector": True
            }
            
            async with self.session.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=search_body
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("result", [])
                    return results[0] if results else None
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting point by ID: {e}")
            return None
    
    async def count_points(
        self,
        user_id: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Compte les points dans la collection.
        
        Args:
            user_id: ID de l'utilisateur
            filters: Filtres additionnels
            
        Returns:
            Nombre de points
        """
        try:
            count_body = {
                "filter": {
                    "must": [
                        {
                            "key": "user_id",
                            "match": {"value": user_id}
                        }
                    ]
                }
            }
            
            # Ajouter des filtres additionnels
            if filters:
                additional_filters = self._build_qdrant_filters(filters)
                count_body["filter"]["must"].extend(additional_filters)
            
            async with self.session.post(
                f"{self.base_url}/collections/{self.collection_name}/points/count",
                json=count_body
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result", {}).get("count", 0)
                else:
                    return 0
                    
        except Exception as e:
            logger.error(f"Error counting points: {e}")
            return 0
    
    # Version surchargée pour compatibilité (utilise collection_name par défaut)
    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les informations de la collection.
        
        Args:
            collection_name: Nom de la collection (optionnel, utilise self.collection_name par défaut)
        
        Returns:
            Informations détaillées de la collection
        """
        target_collection = collection_name or self.collection_name
        
        async def _get_info():
            async with self.session.get(
                f"{self.base_url}/collections/{target_collection}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    raise Exception(f"Collection info failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_get_info, "collection_info")
    
    async def scroll_points(
        self,
        user_id: int,
        limit: int = 100,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vector: bool = False
    ) -> Dict[str, Any]:
        """
        Parcourt les points de la collection (pagination).
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre de points par page
            offset: Offset pour la pagination (point ID)
            with_payload: Inclure les métadonnées
            with_vector: Inclure les vecteurs
            
        Returns:
            Points avec pagination
        """
        scroll_body = {
            "filter": {
                "must": [
                    {
                        "key": "user_id",
                        "match": {"value": user_id}
                    }
                ]
            },
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector
        }
        
        if offset:
            scroll_body["offset"] = offset
        
        async def _scroll():
            async with self.session.post(
                f"{self.base_url}/collections/{self.collection_name}/points/scroll",
                json=scroll_body
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Scroll failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_scroll, "scroll_points")
    
    async def recommend_transactions(
        self,
        positive_ids: List[int],
        user_id: int,
        negative_ids: Optional[List[int]] = None,
        limit: int = 10,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Recommande des transactions basées sur des exemples positifs/négatifs.
        
        Args:
            positive_ids: IDs de transactions appréciées
            negative_ids: IDs de transactions non appréciées
            user_id: ID de l'utilisateur
            limit: Nombre de recommandations
            score_threshold: Seuil de similarité
            
        Returns:
            Transactions recommandées
        """
        try:
            # Récupérer les vecteurs des exemples positifs
            positive_vectors = []
            for tid in positive_ids:
                point = await self._get_point_by_id(tid)
                if point and "vector" in point:
                    positive_vectors.append(point["vector"])
            
            if not positive_vectors:
                return []
            
            # Récupérer les vecteurs des exemples négatifs
            negative_vectors = []
            if negative_ids:
                for tid in negative_ids:
                    point = await self._get_point_by_id(tid)
                    if point and "vector" in point:
                        negative_vectors.append(point["vector"])
            
            # Construire la requête de recommandation
            recommend_body = {
                "positive": positive_vectors,
                "filter": {
                    "must": [
                        {
                            "key": "user_id",
                            "match": {"value": user_id}
                        }
                    ]
                },
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
                "with_vector": False
            }
            
            if negative_vectors:
                recommend_body["negative"] = negative_vectors
            
            async with self.session.post(
                f"{self.base_url}/collections/{self.collection_name}/points/recommend",
                json=recommend_body
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result", [])
                else:
                    error_text = await response.text()
                    logger.error(f"Recommendation failed: HTTP {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du cluster Qdrant.
        
        Returns:
            Informations du cluster
        """
        async def _get_cluster_info():
            async with self.session.get(f"{self.base_url}/cluster") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    # Fallback pour les instances sans cluster
                    return {"cluster_enabled": False, "status": "single_node"}
        
        try:
            return await self.execute_with_retry(_get_cluster_info, "cluster_info")
        except Exception as e:
            logger.warning(f"Cluster info not available: {e}")
            return {"cluster_enabled": False, "error": str(e)}
    
    def clear_cache(self):
        """Vide le cache des vecteurs."""
        self._vector_cache.clear()
        logger.info("Qdrant vector cache cleared")