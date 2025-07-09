"""
Performance Optimizer - Optimisations et cache pour les performances

Responsabilité : Optimise les performances du search service via cache intelligent,
optimisations de requêtes, et monitoring des métriques de performance.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import hashlib
import json

from ..models.service_contracts import SearchServiceQuery, SearchServiceResponse
from ..utils.cache import LRUCache
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Optimiseur de performance haute efficacité pour le search service.
    
    Responsabilités:
    - Cache LRU intelligent multi-niveaux
    - Optimisation automatique des requêtes
    - Monitoring et métriques de performance
    - Détection patterns d'usage et adaptation
    - Circuit breaker pour résilience
    - Batch processing et parallélisation
    """
    
    def __init__(self, cache_size: int = 1000):
        self.settings = get_settings()
        
        # Cache multi-niveaux
        self.response_cache = LRUCache(cache_size)
        self.query_optimization_cache = LRUCache(cache_size // 2)
        self.aggregation_cache = LRUCache(cache_size // 4)
        
        # Métriques de performance
        self.performance_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time_ms": 0.0,
            "slow_queries": 0,
            "optimization_count": 0,
            "total_response_time": 0.0
        }
        
        # Patterns d'optimisation détectés
        self.optimization_patterns = defaultdict(int)
        self.slow_query_patterns = []
        
        # Circuit breaker
        self.circuit_breaker = {
            "is_open": False,
            "failure_count": 0,
            "last_failure_time": None,
            "threshold": 5,
            "timeout": 60  # secondes
        }
        
        # Configuration des seuils
        self.slow_query_threshold_ms = 1000
        self.cache_ttl_seconds = 300  # 5 minutes
        self.optimization_threshold = 10  # Nombre de requêtes similaires pour optimiser
        
        logger.info("PerformanceOptimizer initialisé avec cache de taille %d", cache_size)
    
    async def optimize_and_cache_search(
        self,
        query_contract: SearchServiceQuery,
        search_executor_func,
        *args, **kwargs
    ) -> Tuple[SearchServiceResponse, Dict[str, Any]]:
        """
        Point d'entrée principal pour optimisation + cache + exécution.
        
        Args:
            query_contract: Contrat de requête à optimiser
            search_executor_func: Fonction d'exécution de la recherche
            *args, **kwargs: Arguments pour la fonction d'exécution
            
        Returns:
            Tuple[SearchServiceResponse, Dict]: (Réponse, Métadonnées performance)
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(query_contract)
        
        try:
            # 1. Vérification du circuit breaker
            if self._is_circuit_breaker_open():
                raise Exception("Circuit breaker ouvert - service temporairement indisponible")
            
            # 2. Tentative de récupération depuis le cache
            cached_response = await self._get_from_cache(cache_key)
            if cached_response:
                self.performance_metrics["cache_hits"] += 1
                execution_time = (time.time() - start_time) * 1000
                
                return cached_response, {
                    "cache_hit": True,
                    "execution_time_ms": execution_time,
                    "optimization_applied": ["cache_hit"]
                }
            
            # 3. Cache miss - optimisation de la requête
            optimized_contract = await self._optimize_query_contract(query_contract)
            optimizations_applied = self._get_applied_optimizations(
                query_contract, optimized_contract
            )
            
            # 4. Exécution de la recherche optimisée
            response = await search_executor_func(optimized_contract, *args, **kwargs)
            
            # 5. Post-traitement et mise en cache
            await self._cache_response(cache_key, response)
            execution_time = (time.time() - start_time) * 1000
            
            # 6. Mise à jour des métriques
            self._update_performance_metrics(execution_time, False)
            self._record_optimization_pattern(query_contract, optimizations_applied)
            
            # 7. Détection de requêtes lentes
            if execution_time > self.slow_query_threshold_ms:
                await self._handle_slow_query(query_contract, execution_time)
            
            self.performance_metrics["cache_misses"] += 1
            
            return response, {
                "cache_hit": False,
                "execution_time_ms": execution_time,
                "optimization_applied": optimizations_applied,
                "query_optimized": optimized_contract != query_contract
            }
            
        except Exception as e:
            self._handle_circuit_breaker_failure()
            execution_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time, True)
            logger.error(f"Erreur lors de l'optimisation/exécution: {str(e)}")
            raise
    
    async def batch_optimize_searches(
        self,
        query_contracts: List[SearchServiceQuery],
        search_executor_func,
        max_parallel: int = 5
    ) -> List[Tuple[SearchServiceResponse, Dict[str, Any]]]:
        """
        Optimise et exécute plusieurs recherches en batch avec parallélisation.
        
        Args:
            query_contracts: Liste des contrats à traiter
            search_executor_func: Fonction d'exécution
            max_parallel: Nombre maximum de requêtes parallèles
            
        Returns:
            List: Résultats optimisés et métriques
        """
        if not query_contracts:
            return []
        
        # Groupement par similarité pour optimisations batch
        groups = self._group_similar_queries(query_contracts)
        
        results = []
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_query(query_contract):
            async with semaphore:
                return await self.optimize_and_cache_search(
                    query_contract, search_executor_func
                )
        
        # Traitement par groupes avec optimisations spécifiques
        for group in groups:
            if len(group) > 1:
                # Optimisation spéciale pour requêtes similaires
                optimized_group = await self._optimize_similar_queries(group)
                group_tasks = [process_query(contract) for contract in optimized_group]
            else:
                group_tasks = [process_query(group[0])]
            
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            
            for result in group_results:
                if isinstance(result, Exception):
                    logger.error(f"Erreur dans le batch: {str(result)}")
                    # Ajouter une réponse d'erreur par défaut
                    results.append((None, {"error": str(result)}))
                else:
                    results.append(result)
        
        logger.info(f"Batch de {len(query_contracts)} requêtes traité")
        return results
    
    async def _optimize_query_contract(
        self,
        query_contract: SearchServiceQuery
    ) -> SearchServiceQuery:
        """
        Optimise un contrat de requête basé sur les patterns détectés.
        """
        # Vérification du cache d'optimisation
        opt_cache_key = self._generate_optimization_cache_key(query_contract)
        cached_optimization = self.query_optimization_cache.get(opt_cache_key)
        
        if cached_optimization:
            return self._apply_cached_optimization(query_contract, cached_optimization)
        
        # Optimisations basées sur l'intention et les patterns
        optimized_contract = query_contract.model_copy(deep=True)
        optimizations = []
        
        # 1. Optimisation des filtres
        optimized_contract = self._optimize_filters(optimized_contract)
        optimizations.append("filter_optimization")
        
        # 2. Optimisation des champs demandés
        optimized_contract = self._optimize_fields(optimized_contract)
        optimizations.append("field_optimization")
        
        # 3. Optimisation des agrégations
        if optimized_contract.aggregations and optimized_contract.aggregations.enabled:
            optimized_contract = self._optimize_aggregations(optimized_contract)
            optimizations.append("aggregation_optimization")
        
        # 4. Optimisation de la pagination
        optimized_contract = self._optimize_pagination(optimized_contract)
        optimizations.append("pagination_optimization")
        
        # 5. Optimisation du timeout
        optimized_contract = self._optimize_timeout(optimized_contract)
        optimizations.append("timeout_optimization")
        
        # Cache de l'optimisation
        self.query_optimization_cache.set(opt_cache_key, {
            "optimizations": optimizations,
            "optimized_fields": self._extract_optimization_diff(query_contract, optimized_contract)
        })
        
        return optimized_contract
    
    def _optimize_filters(self, contract: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise l'ordre et la structure des filtres pour de meilleures performances.
        """
        # Réorganiser les filtres par sélectivité (plus sélectifs en premier)
        if contract.filters.required:
            # user_id toujours en premier (plus sélectif)
            user_filters = [f for f in contract.filters.required if f.field == "user_id"]
            other_filters = [f for f in contract.filters.required if f.field != "user_id"]
            
            # Trier les autres filtres par sélectivité estimée
            other_filters.sort(key=self._estimate_filter_selectivity)
            
            contract.filters.required = user_filters + other_filters
        
        return contract
    
    def _optimize_fields(self, contract: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise les champs demandés pour réduire la taille de transfert.
        """
        # Si aucun champ spécifique demandé, utiliser un set optimal par défaut
        if not contract.search_parameters.fields:
            # Champs essentiels pour la plupart des cas d'usage
            essential_fields = [
                "transaction_id", "user_id", "amount", "amount_abs", 
                "transaction_type", "date", "primary_description",
                "merchant_name", "category_name", "currency_code"
            ]
            contract.search_parameters.fields = essential_fields
        
        # Éviter les champs lourds non nécessaires pour certaines intentions
        intent = contract.query_metadata.intent_type
        
        if intent in ["COUNT_OPERATIONS", "TEMPORAL_ANALYSIS"]:
            # Pour le comptage et l'analyse temporelle, limiter aux champs nécessaires
            minimal_fields = ["transaction_id", "user_id", "date", "amount", "amount_abs"]
            contract.search_parameters.fields = minimal_fields
        
        elif intent == "SEARCH_BY_CATEGORY":
            # Ajouter category_name si pas présent
            if "category_name" not in contract.search_parameters.fields:
                contract.search_parameters.fields.append("category_name")
        
        return contract
    
    def _optimize_aggregations(self, contract: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise les agrégations pour de meilleures performances.
        """
        if not contract.aggregations or not contract.aggregations.enabled:
            return contract
        
        # Si on fait uniquement des agrégations, réduire les résultats retournés
        if contract.aggregations.types and not contract.filters.text_search:
            # Pour les agrégations pures, on n'a pas besoin des documents
            if contract.search_parameters.limit > 10:
                contract.search_parameters.limit = 0  # Pas de documents
        
        # Optimiser l'ordre des groupements (plus sélectifs en premier)
        if contract.aggregations.group_by:
            selectivity_order = {
                "user_id": 1,
                "category_name": 2,
                "month_year": 3,
                "merchant_name": 4,
                "weekday": 5
            }
            
            contract.aggregations.group_by.sort(
                key=lambda field: selectivity_order.get(field, 999)
            )
        
        return contract
    
    def _optimize_pagination(self, contract: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise la pagination pour éviter les deep pagination.
        """
        # Limiter l'offset pour éviter les performances dégradées
        max_offset = self.settings.MAX_SEARCH_OFFSET or 10000
        if contract.search_parameters.offset > max_offset:
            logger.warning(f"Offset trop élevé: {contract.search_parameters.offset}, limité à {max_offset}")
            contract.search_parameters.offset = max_offset
        
        # Optimiser la taille de page selon le type de requête
        if contract.query_metadata.intent_type in ["COUNT_OPERATIONS", "TEMPORAL_ANALYSIS"]:
            # Pour les analyses, réduire la taille par défaut
            if contract.search_parameters.limit > 50:
                contract.search_parameters.limit = 50
        
        elif contract.aggregations and contract.aggregations.enabled:
            # Pour les agrégations, souvent pas besoin de beaucoup de documents
            if contract.search_parameters.limit > 20:
                contract.search_parameters.limit = 20
        
        return contract
    
    def _optimize_timeout(self, contract: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise le timeout selon la complexité de la requête.
        """
        base_timeout = contract.search_parameters.timeout_ms or self.settings.ELASTICSEARCH_TIMEOUT_MS
        
        # Ajuster selon la complexité
        complexity_factors = {
            "aggregations": 1.5 if contract.aggregations and contract.aggregations.enabled else 1.0,
            "text_search": 1.3 if contract.filters.text_search else 1.0,
            "large_result_set": 1.2 if contract.search_parameters.limit > 100 else 1.0,
            "multiple_filters": 1.1 if len(contract.filters.required) > 3 else 1.0
        }
        
        total_factor = 1.0
        for factor in complexity_factors.values():
            total_factor *= factor
        
        optimized_timeout = min(int(base_timeout * total_factor), 10000)  # Max 10s
        contract.search_parameters.timeout_ms = optimized_timeout
        
        return contract
    
    def _estimate_filter_selectivity(self, filter_item) -> int:
        """
        Estime la sélectivité d'un filtre (plus bas = plus sélectif).
        """
        selectivity_map = {
            "user_id": 1,          # Très sélectif
            "transaction_id": 2,   # Très sélectif
            "account_id": 3,       # Sélectif
            "category_name": 4,    # Moyennement sélectif
            "merchant_name": 5,    # Moyennement sélectif
            "transaction_type": 6, # Peu sélectif (debit/credit)
            "currency_code": 7,    # Peu sélectif
            "operation_type": 8    # Peu sélectif
        }
        
        return selectivity_map.get(filter_item.field, 999)
    
    def _group_similar_queries(
        self, 
        query_contracts: List[SearchServiceQuery]
    ) -> List[List[SearchServiceQuery]]:
        """
        Groupe les requêtes similaires pour optimisations batch.
        """
        groups = defaultdict(list)
        
        for contract in query_contracts:
            # Clé de groupement basée sur l'intention et les filtres principaux
            group_key = (
                contract.query_metadata.intent_type,
                contract.search_parameters.query_type,
                len(contract.filters.required),
                bool(contract.filters.text_search),
                bool(contract.aggregations and contract.aggregations.enabled)
            )
            groups[group_key].append(contract)
        
        return list(groups.values())
    
    async def _optimize_similar_queries(
        self, 
        similar_queries: List[SearchServiceQuery]
    ) -> List[SearchServiceQuery]:
        """
        Applique des optimisations spécifiques aux requêtes similaires.
        """
        if len(similar_queries) <= 1:
            return similar_queries
        
        optimized = []
        
        # Optimisation commune: harmoniser les timeouts
        max_timeout = max(
            q.search_parameters.timeout_ms or 5000 for q in similar_queries
        )
        
        # Optimisation commune: harmoniser les tailles de résultats
        max_limit = max(q.search_parameters.limit for q in similar_queries)
        
        for query in similar_queries:
            opt_query = query.model_copy(deep=True)
            
            # Appliquer les optimisations communes
            opt_query.search_parameters.timeout_ms = max_timeout
            
            # Pour les requêtes similaires, utiliser la même limite
            if abs(opt_query.search_parameters.limit - max_limit) <= 10:
                opt_query.search_parameters.limit = max_limit
            
            optimized.append(opt_query)
        
        return optimized
    
    async def _get_from_cache(self, cache_key: str) -> Optional[SearchServiceResponse]:
        """
        Récupère une réponse depuis le cache avec validation TTL.
        """
        cached_item = self.response_cache.get(cache_key)
        if not cached_item:
            return None
        
        response, timestamp = cached_item
        
        # Vérification TTL
        if time.time() - timestamp > self.cache_ttl_seconds:
            self.response_cache.delete(cache_key)
            return None
        
        # Mettre à jour les métadonnées pour indiquer le cache hit
        response.response_metadata.cache_hit = True
        return response
    
    async def _cache_response(
        self, 
        cache_key: str, 
        response: SearchServiceResponse
    ) -> None:
        """
        Met en cache une réponse avec timestamp.
        """
        # Ne pas cacher les réponses d'erreur ou vides
        if (not response.results and 
            not (response.aggregations and response.aggregations.transaction_count > 0)):
            return
        
        # Copier la réponse pour éviter les modifications
        cached_response = response.model_copy(deep=True)
        cached_response.response_metadata.cache_hit = False  # Reset pour le cache
        
        self.response_cache.set(cache_key, (cached_response, time.time()))
    
    def _generate_cache_key(self, query_contract: SearchServiceQuery) -> str:
        """
        Génère une clé de cache déterministe pour la requête.
        """
        # Extraire les éléments pertinents pour le cache
        cache_data = {
            "intent_type": query_contract.query_metadata.intent_type,
            "user_id": query_contract.query_metadata.user_id,
            "query_type": query_contract.search_parameters.query_type,
            "filters": self._serialize_filters(query_contract.filters),
            "fields": sorted(query_contract.search_parameters.fields or []),
            "limit": query_contract.search_parameters.limit,
            "offset": query_contract.search_parameters.offset,
            "aggregations": self._serialize_aggregations(query_contract.aggregations)
        }
        
        # Sérialisation déterministe
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _generate_optimization_cache_key(self, query_contract: SearchServiceQuery) -> str:
        """
        Génère une clé de cache pour les optimisations.
        """
        opt_data = {
            "intent_type": query_contract.query_metadata.intent_type,
            "query_type": query_contract.search_parameters.query_type,
            "filter_count": len(query_contract.filters.required),
            "has_text_search": bool(query_contract.filters.text_search),
            "has_aggregations": bool(query_contract.aggregations and query_contract.aggregations.enabled)
        }
        
        opt_str = json.dumps(opt_data, sort_keys=True)
        return hashlib.md5(opt_str.encode()).hexdigest()
    
    def _serialize_filters(self, filters) -> Dict[str, Any]:
        """
        Sérialise les filtres pour le cache.
        """
        if not filters:
            return {}
        
        return {
            "required": [(f.field, f.operator, str(f.value)) for f in filters.required],
            "optional": [(f.field, f.operator, str(f.value)) for f in filters.optional],
            "ranges": [(f.field, f.operator, str(f.value)) for f in filters.ranges],
            "text_search": dict(filters.text_search) if filters.text_search else None
        }
    
    def _serialize_aggregations(self, aggregations) -> Dict[str, Any]:
        """
        Sérialise les agrégations pour le cache.
        """
        if not aggregations or not aggregations.enabled:
            return {}
        
        return {
            "enabled": aggregations.enabled,
            "types": sorted(aggregations.types or []),
            "group_by": sorted(aggregations.group_by or []),
            "metrics": sorted(aggregations.metrics or [])
        }
    
    def _apply_cached_optimization(
        self, 
        original_contract: SearchServiceQuery, 
        cached_opt: Dict[str, Any]
    ) -> SearchServiceQuery:
        """
        Applique une optimisation mise en cache.
        """
        # Pour l'instant, retourner le contrat original
        # Dans une implémentation plus avancée, appliquer les optimizations stockées
        return original_contract
    
    def _extract_optimization_diff(
        self, 
        original: SearchServiceQuery, 
        optimized: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Extrait les différences entre la requête originale et optimisée.
        """
        diff = {}
        
        if original.search_parameters.fields != optimized.search_parameters.fields:
            diff["fields_changed"] = True
        
        if original.search_parameters.limit != optimized.search_parameters.limit:
            diff["limit_changed"] = True
        
        if original.search_parameters.timeout_ms != optimized.search_parameters.timeout_ms:
            diff["timeout_changed"] = True
        
        return diff
    
    def _get_applied_optimizations(
        self, 
        original: SearchServiceQuery, 
        optimized: SearchServiceQuery
    ) -> List[str]:
        """
        Détermine quelles optimisations ont été appliquées.
        """
        optimizations = []
        
        if original.search_parameters.fields != optimized.search_parameters.fields:
            optimizations.append("field_optimization")
        
        if original.search_parameters.timeout_ms != optimized.search_parameters.timeout_ms:
            optimizations.append("timeout_optimization")
        
        if original.search_parameters.limit != optimized.search_parameters.limit:
            optimizations.append("pagination_optimization")
        
        # Toujours inclure ces optimisations de base
        optimizations.extend(["filter_reordering", "query_structure_optimization"])
        
        return optimizations
    
    def _record_optimization_pattern(
        self, 
        query_contract: SearchServiceQuery, 
        optimizations: List[str]
    ) -> None:
        """
        Enregistre les patterns d'optimisation pour apprentissage.
        """
        pattern_key = (
            query_contract.query_metadata.intent_type,
            tuple(sorted(optimizations))
        )
        self.optimization_patterns[pattern_key] += 1
        
        # Déclencher des optimisations automatiques après un seuil
        if self.optimization_patterns[pattern_key] == self.optimization_threshold:
            logger.info(f"Pattern d'optimisation détecté: {pattern_key}")
    
    async def _handle_slow_query(
        self, 
        query_contract: SearchServiceQuery, 
        execution_time_ms: float
    ) -> None:
        """
        Gère les requêtes lentes pour amélioration future.
        """
        slow_query_info = {
            "intent_type": query_contract.query_metadata.intent_type,
            "execution_time_ms": execution_time_ms,
            "query_type": query_contract.search_parameters.query_type,
            "has_aggregations": bool(query_contract.aggregations and query_contract.aggregations.enabled),
            "filter_count": len(query_contract.filters.required),
            "result_limit": query_contract.search_parameters.limit,
            "timestamp": datetime.now()
        }
        
        self.slow_query_patterns.append(slow_query_info)
        self.performance_metrics["slow_queries"] += 1
        
        # Garder seulement les 100 dernières requêtes lentes
        if len(self.slow_query_patterns) > 100:
            self.slow_query_patterns = self.slow_query_patterns[-100:]
        
        logger.warning(
            f"Requête lente détectée: {execution_time_ms:.2f}ms, "
            f"intent: {query_contract.query_metadata.intent_type}"
        )
    
    def _is_circuit_breaker_open(self) -> bool:
        """
        Vérifie si le circuit breaker est ouvert.
        """
        if not self.circuit_breaker["is_open"]:
            return False
        
        # Vérifier si le timeout est dépassé
        if self.circuit_breaker["last_failure_time"]:
            time_since_failure = time.time() - self.circuit_breaker["last_failure_time"]
            if time_since_failure > self.circuit_breaker["timeout"]:
                # Réinitialiser le circuit breaker
                self.circuit_breaker["is_open"] = False
                self.circuit_breaker["failure_count"] = 0
                logger.info("Circuit breaker réinitialisé")
                return False
        
        return True
    
    def _handle_circuit_breaker_failure(self) -> None:
        """
        Gère les échecs pour le circuit breaker.
        """
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure_time"] = time.time()
        
        if self.circuit_breaker["failure_count"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["is_open"] = True
            logger.error("Circuit breaker ouvert après %d échecs", self.circuit_breaker["failure_count"])
    
    def _update_performance_metrics(self, execution_time_ms: float, is_error: bool) -> None:
        """
        Met à jour les métriques de performance.
        """
        self.performance_metrics["total_requests"] += 1
        
        if not is_error:
            self.performance_metrics["total_response_time"] += execution_time_ms
            
            # Calcul de la moyenne mobile
            total_requests = self.performance_metrics["total_requests"]
            total_time = self.performance_metrics["total_response_time"]
            self.performance_metrics["avg_response_time_ms"] = total_time / total_requests
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de performance actuelles.
        """
        metrics = self.performance_metrics.copy()
        
        # Calcul du hit rate du cache
        total_cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
        if total_cache_requests > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_requests
        else:
            metrics["cache_hit_rate"] = 0.0
        
        # Informations sur les patterns d'optimisation
        metrics["optimization_patterns_count"] = len(self.optimization_patterns)
        metrics["slow_queries_recent"] = len([
            q for q in self.slow_query_patterns 
            if (datetime.now() - q["timestamp"]).seconds < 3600  # Dernière heure
        ])
        
        # État du circuit breaker
        metrics["circuit_breaker_status"] = {
            "is_open": self.circuit_breaker["is_open"],
            "failure_count": self.circuit_breaker["failure_count"]
        }
        
        return metrics
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées du cache.
        """
        return {
            "response_cache": {
                "size": len(self.response_cache),
                "max_size": self.response_cache.maxsize,
                "hit_rate": self.response_cache.hit_rate
            },
            "optimization_cache": {
                "size": len(self.query_optimization_cache),
                "max_size": self.query_optimization_cache.maxsize,
                "hit_rate": self.query_optimization_cache.hit_rate
            },
            "aggregation_cache": {
                "size": len(self.aggregation_cache),
                "max_size": self.aggregation_cache.maxsize,
                "hit_rate": self.aggregation_cache.hit_rate
            }
        }
    
    def clear_all_caches(self) -> None:
        """
        Vide tous les caches.
        """
        self.response_cache.clear()
        self.query_optimization_cache.clear()
        self.aggregation_cache.clear()
        logger.info("Tous les caches vidés")
    
    def reset_performance_metrics(self) -> None:
        """
        Remet à zéro les métriques de performance.
        """
        self.performance_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time_ms": 0.0,
            "slow_queries": 0,
            "optimization_count": 0,
            "total_response_time": 0.0
        }
        self.optimization_patterns.clear()
        self.slow_query_patterns.clear()
        logger.info("Métriques de performance réinitialisées")