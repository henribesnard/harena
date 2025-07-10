"""
Optimiseur de performance - Composant Core #3
Optimise les performances des requêtes et du traitement
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import time
from dataclasses import dataclass
from enum import Enum

from search_service.models.service_contracts import SearchServiceQuery
from search_service.utils.cache import LRUCache

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Niveaux d'optimisation disponibles"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass
class QueryPerformanceMetrics:
    """Métriques de performance d'une requête"""
    query_hash: str
    execution_time: float
    result_count: int
    cache_hit: bool
    optimization_applied: str
    timestamp: datetime


@dataclass
class OptimizationRule:
    """Règle d'optimisation"""
    name: str
    condition: Callable[[SearchServiceQuery], bool]
    optimization: Callable[[SearchServiceQuery], SearchServiceQuery]
    priority: int = 1
    enabled: bool = True


class PerformanceOptimizer:
    """
    Optimiseur de performance avancé
    
    Responsabilités:
    - Cache intelligent des requêtes fréquentes
    - Optimisation automatique des requêtes
    - Monitoring des performances
    - Parallélisation des opérations
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        
        # Cache des requêtes
        cache_size = self._get_cache_size_for_level(optimization_level)
        self.query_cache = LRUCache(max_size=cache_size, name="query_cache")
        
        # Métriques de performance
        self.performance_metrics: deque = deque(maxlen=1000)
        self.query_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Règles d'optimisation
        self.optimization_rules: List[OptimizationRule] = []
        self._initialize_optimization_rules()
        
        # Configuration des timeouts
        self.query_timeout = self._get_timeout_for_level(optimization_level)
        self.batch_size = self._get_batch_size_for_level(optimization_level)
        
        # Cache warming
        self.warm_cache_enabled = optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]
        
        logger.info(f"✅ PerformanceOptimizer initialisé - Niveau: {optimization_level.value}")
    
    async def optimize_query(self, query: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise une requête avant exécution
        
        Args:
            query: Requête à optimiser
            
        Returns:
            Requête optimisée
        """
        try:
            start_time = time.time()
            optimized_query = query
            applied_optimizations = []
            
            # Application des règles d'optimisation
            for rule in sorted(self.optimization_rules, key=lambda r: r.priority, reverse=True):
                if rule.enabled and rule.condition(optimized_query):
                    optimized_query = rule.optimization(optimized_query)
                    applied_optimizations.append(rule.name)
            
            # Optimisations spécifiques selon le niveau
            if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                optimized_query = await self._apply_aggressive_optimizations(optimized_query)
                applied_optimizations.append("aggressive_mode")
            
            optimization_time = time.time() - start_time
            
            if applied_optimizations:
                logger.debug(f"🚀 Optimisations appliquées: {', '.join(applied_optimizations)} ({optimization_time:.3f}s)")
            
            return optimized_query
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'optimisation de la requête: {e}")
            return query
    
    async def execute_with_cache(
        self, 
        query: SearchServiceQuery, 
        executor_func: Callable,
        cache_ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Exécute une requête avec mise en cache intelligente
        
        Args:
            query: Requête à exécuter
            executor_func: Fonction d'exécution
            cache_ttl: TTL du cache en secondes
            
        Returns:
            Résultats (depuis le cache ou fraîchement calculés)
        """
        try:
            start_time = time.time()
            
            # Génération de la clé de cache
            cache_key = self._generate_cache_key(query)
            
            # Vérification du cache
            cached_result = await self.query_cache.get(cache_key)
            if cached_result is not None:
                execution_time = time.time() - start_time
                self._record_performance_metric(
                    query, execution_time, len(cached_result.get("hits", {}).get("hits", [])), 
                    cache_hit=True, optimization="cache_hit"
                )
                logger.debug(f"💾 Cache hit pour la requête: {cache_key[:16]}...")
                return cached_result
            
            # Exécution de la requête
            result = await executor_func(query)
            
            # Mise en cache du résultat
            ttl = cache_ttl or self._get_cache_ttl_for_query(query)
            await self.query_cache.set(cache_key, result, ttl=ttl)
            
            execution_time = time.time() - start_time
            self._record_performance_metric(
                query, execution_time, len(result.get("hits", {}).get("hits", [])), 
                cache_hit=False, optimization="executed"
            )
            
            logger.debug(f"✅ Requête exécutée et mise en cache: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution avec cache: {e}")
            raise
    
    async def execute_batch_optimized(
        self, 
        queries: List[SearchServiceQuery], 
        executor_func: Callable
    ) -> List[Dict[str, Any]]:
        """
        Exécute un batch de requêtes avec optimisations
        
        Args:
            queries: Liste des requêtes
            executor_func: Fonction d'exécution
            
        Returns:
            Liste des résultats
        """
        try:
            start_time = time.time()
            
            # Groupement des requêtes par type pour optimisation
            query_groups = self._group_queries_for_optimization(queries)
            
            # Exécution par groupes avec parallélisation
            all_results = []
            for group in query_groups:
                # Limitation de la concurrence
                batch_size = min(len(group), self.batch_size)
                
                # Exécution en parallèle des requêtes du groupe
                group_results = await self._execute_parallel_batch(group, executor_func, batch_size)
                all_results.extend(group_results)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Batch de {len(queries)} requêtes exécuté en {execution_time:.3f}s")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution du batch: {e}")
            raise
    
    async def _execute_parallel_batch(
        self, 
        queries: List[SearchServiceQuery], 
        executor_func: Callable, 
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """
        Exécute un batch de requêtes en parallèle
        
        Args:
            queries: Requêtes à exécuter
            executor_func: Fonction d'exécution
            batch_size: Taille du batch parallèle
            
        Returns:
            Résultats des requêtes
        """
        try:
            results = []
            
            # Exécution par chunks pour contrôler la concurrence
            for i in range(0, len(queries), batch_size):
                chunk = queries[i:i + batch_size]
                
                # Création des tâches parallèles
                tasks = []
                for query in chunk:
                    task = asyncio.create_task(
                        self.execute_with_cache(query, executor_func)
                    )
                    tasks.append(task)
                
                # Exécution parallèle avec timeout
                chunk_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.query_timeout
                )
                
                # Traitement des résultats
                for result in chunk_results:
                    if isinstance(result, Exception):
                        logger.error(f"❌ Erreur dans l'exécution parallèle: {result}")
                        results.append({})  # Résultat vide en cas d'erreur
                    else:
                        results.append(result)
            
            return results
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout lors de l'exécution du batch parallèle")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution parallèle: {e}")
            raise
    
    def _initialize_optimization_rules(self):
        """
        Initialise les règles d'optimisation
        """
        try:
            # Règle 1: Limitation de la taille des résultats
            self.optimization_rules.append(OptimizationRule(
                name="limit_result_size",
                condition=lambda q: q.size is None or q.size > 100,
                optimization=lambda q: self._limit_result_size(q),
                priority=10
            ))
            
            # Règle 2: Optimisation des requêtes vides
            self.optimization_rules.append(OptimizationRule(
                name="optimize_empty_query",
                condition=lambda q: not q.query or len(q.query.strip()) == 0,
                optimization=lambda q: self._optimize_empty_query(q),
                priority=9
            ))
            
            # Règle 3: Optimisation des requêtes courtes
            self.optimization_rules.append(OptimizationRule(
                name="optimize_short_query",
                condition=lambda q: q.query and len(q.query.strip()) < 3,
                optimization=lambda q: self._optimize_short_query(q),
                priority=8
            ))
            
            # Règle 4: Optimisation des filtres redondants
            self.optimization_rules.append(OptimizationRule(
                name="remove_redundant_filters",
                condition=lambda q: q.filters and len(q.filters) > 1,
                optimization=lambda q: self._remove_redundant_filters(q),
                priority=7
            ))
            
            # Règle 5: Optimisation des agrégations coûteuses
            self.optimization_rules.append(OptimizationRule(
                name="optimize_expensive_aggregations",
                condition=lambda q: q.aggregations and len(q.aggregations) > 3,
                optimization=lambda q: self._optimize_aggregations(q),
                priority=6
            ))
            
            logger.info(f"✅ {len(self.optimization_rules)} règles d'optimisation initialisées")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des règles: {e}")
    
    def _limit_result_size(self, query: SearchServiceQuery) -> SearchServiceQuery:
        """
        Limite la taille des résultats
        
        Args:
            query: Requête à optimiser
            
        Returns:
            Requête avec taille limitée
        """
        # Utilisation de model_copy pour Pydantic v2
        optimized = query.model_copy(deep=True)
        max_size = 50 if self.optimization_level == OptimizationLevel.AGGRESSIVE else 100
        
        if optimized.size is None or optimized.size > max_size:
            optimized.size = max_size
            logger.debug(f"🔧 Taille des résultats limitée à {max_size}")
        
        return optimized
    
    def _optimize_empty_query(self, query: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise les requêtes vides
        
        Args:
            query: Requête vide
            
        Returns:
            Requête optimisée
        """
        optimized = query.model_copy(deep=True)
        
        # Pour les requêtes vides, utiliser match_all avec limitations
        optimized.search_type = "match_all"
        optimized.size = min(optimized.size or 20, 20)
        
        # Ajouter un tri par date pour les requêtes vides
        if not optimized.sort_by:
            optimized.sort_by = "created_at"
            optimized.sort_order = "desc"
        
        logger.debug("🔧 Requête vide optimisée avec match_all")
        
        return optimized
    
    def _optimize_short_query(self, query: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise les requêtes courtes
        
        Args:
            query: Requête courte
            
        Returns:
            Requête optimisée
        """
        optimized = query.model_copy(deep=True)
        
        # Pour les requêtes courtes, utiliser la recherche floue
        if optimized.search_type != "fuzzy":
            optimized.search_type = "fuzzy"
            logger.debug("🔧 Requête courte optimisée avec recherche floue")
        
        return optimized
    
    def _remove_redundant_filters(self, query: SearchServiceQuery) -> SearchServiceQuery:
        """
        Supprime les filtres redondants
        
        Args:
            query: Requête avec filtres
            
        Returns:
            Requête avec filtres optimisés
        """
        optimized = query.model_copy(deep=True)
        
        if not optimized.filters:
            return optimized
        
        # Suppression des doublons
        unique_filters = []
        seen_filters = set()
        
        for filter_item in optimized.filters:
            filter_key = f"{filter_item.field}:{filter_item.operator}:{filter_item.value}"
            if filter_key not in seen_filters:
                seen_filters.add(filter_key)
                unique_filters.append(filter_item)
        
        if len(unique_filters) < len(optimized.filters):
            optimized.filters = unique_filters
            logger.debug(f"🔧 Filtres redondants supprimés: {len(optimized.filters)} → {len(unique_filters)}")
        
        return optimized
    
    def _optimize_aggregations(self, query: SearchServiceQuery) -> SearchServiceQuery:
        """
        Optimise les agrégations coûteuses
        
        Args:
            query: Requête avec agrégations
            
        Returns:
            Requête avec agrégations optimisées
        """
        optimized = query.model_copy(deep=True)
        
        if not optimized.aggregations:
            return optimized
        
        # Limitation du nombre d'agrégations selon le niveau
        max_aggs = 2 if self.optimization_level == OptimizationLevel.AGGRESSIVE else 5
        
        if len(optimized.aggregations) > max_aggs:
            optimized.aggregations = optimized.aggregations[:max_aggs]
            logger.debug(f"🔧 Agrégations limitées à {max_aggs}")
        
        # Optimisation de la taille des agrégations terms
        for agg in optimized.aggregations:
            if agg.type == "terms" and (agg.size is None or agg.size > 20):
                agg.size = 10 if self.optimization_level == OptimizationLevel.AGGRESSIVE else 20
        
        return optimized
    
    async def _apply_aggressive_optimizations(self, query: SearchServiceQuery) -> SearchServiceQuery:
        """
        Applique les optimisations agressives
        
        Args:
            query: Requête à optimiser
            
        Returns:
            Requête avec optimisations agressives
        """
        optimized = query.model_copy(deep=True)
        
        # Réduction drastique des timeouts
        optimized.timeout = "10s"
        
        # Limitation des champs source
        if not optimized.source_fields:
            optimized.source_fields = ["title", "summary", "document_type", "created_at"]
        
        # Désactivation des highlights pour les grandes requêtes
        if optimized.size and optimized.size > 20:
            optimized.highlight = False
        
        logger.debug("🚀 Optimisations agressives appliquées")
        
        return optimized
    
    def _generate_cache_key(self, query: SearchServiceQuery) -> str:
        """
        Génère une clé de cache pour une requête
        
        Args:
            query: Requête
            
        Returns:
            Clé de cache unique
        """
        try:
            # Création d'un hash basé sur les éléments importants de la requête
            key_elements = [
                query.query or "",
                query.index or "",
                str(query.size or 10),
                str(query.from_ or 0),
                query.search_type or "standard",
                query.sort_by or "",
                query.sort_order or "",
            ]
            
            # Ajout des filtres
            if query.filters:
                filter_strings = []
                for filter_item in query.filters:
                    filter_str = f"{filter_item.field}:{filter_item.operator}:{filter_item.value}"
                    filter_strings.append(filter_str)
                key_elements.append("|".join(sorted(filter_strings)))
            
            # Ajout des agrégations
            if query.aggregations:
                agg_strings = []
                for agg in query.aggregations:
                    agg_str = f"{agg.name}:{agg.type}:{agg.field}"
                    agg_strings.append(agg_str)
                key_elements.append("|".join(sorted(agg_strings)))
            
            # Génération du hash
            key_string = "|".join(key_elements)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
            
            return cache_key
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de la clé de cache: {e}")
            return str(hash(str(query)))
    
    def _get_cache_ttl_for_query(self, query: SearchServiceQuery) -> int:
        """
        Détermine le TTL du cache pour une requête
        
        Args:
            query: Requête
            
        Returns:
            TTL en secondes
        """
        try:
            # TTL basé sur le type de requête
            if query.search_type == "aggregation":
                return 600  # 10 minutes pour les agrégations
            elif query.filters and len(query.filters) > 0:
                return 300  # 5 minutes pour les requêtes filtrées
            else:
                return 180  # 3 minutes pour les requêtes standard
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul du TTL: {e}")
            return 300  # TTL par défaut
    
    def _group_queries_for_optimization(self, queries: List[SearchServiceQuery]) -> List[List[SearchServiceQuery]]:
        """
        Groupe les requêtes pour optimisation
        
        Args:
            queries: Liste des requêtes
            
        Returns:
            Groupes de requêtes
        """
        try:
            # Groupement par type de recherche
            groups = defaultdict(list)
            
            for query in queries:
                group_key = f"{query.search_type or 'standard'}_{query.index or 'default'}"
                groups[group_key].append(query)
            
            return list(groups.values())
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du groupement des requêtes: {e}")
            return [queries]  # Retour d'un seul groupe en cas d'erreur
    
    def _record_performance_metric(
        self, 
        query: SearchServiceQuery, 
        execution_time: float, 
        result_count: int, 
        cache_hit: bool, 
        optimization: str
    ):
        """
        Enregistre une métrique de performance
        
        Args:
            query: Requête exécutée
            execution_time: Temps d'exécution
            result_count: Nombre de résultats
            cache_hit: Si c'était un hit de cache
            optimization: Type d'optimisation appliquée
        """
        try:
            query_hash = self._generate_cache_key(query)
            
            metric = QueryPerformanceMetrics(
                query_hash=query_hash,
                execution_time=execution_time,
                result_count=result_count,
                cache_hit=cache_hit,
                optimization_applied=optimization,
                timestamp=datetime.now()
            )
            
            self.performance_metrics.append(metric)
            self.query_stats[query_hash].append(execution_time)
            
            # Nettoyage des anciennes métriques
            self._cleanup_old_metrics()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement des métriques: {e}")
    
    def _cleanup_old_metrics(self):
        """
        Nettoie les anciennes métriques
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Nettoyage des métriques détaillées
            self.performance_metrics = deque(
                (m for m in self.performance_metrics if m.timestamp > cutoff_time),
                maxlen=1000
            )
            
            # Nettoyage des stats par requête (garder seulement les 100 dernières)
            for query_hash in list(self.query_stats.keys()):
                if len(self.query_stats[query_hash]) > 100:
                    self.query_stats[query_hash] = self.query_stats[query_hash][-100:]
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du nettoyage des métriques: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de performance
        
        Returns:
            Statistiques détaillées
        """
        try:
            if not self.performance_metrics:
                return {"message": "Aucune métrique disponible"}
            
            # Calculs statistiques
            total_queries = len(self.performance_metrics)
            cache_hits = sum(1 for m in self.performance_metrics if m.cache_hit)
            cache_hit_rate = (cache_hits / total_queries) * 100 if total_queries > 0 else 0
            
            execution_times = [m.execution_time for m in self.performance_metrics if not m.cache_hit]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            result_counts = [m.result_count for m in self.performance_metrics]
            avg_result_count = sum(result_counts) / len(result_counts) if result_counts else 0
            
            # Requêtes les plus lentes
            slowest_queries = sorted(
                [m for m in self.performance_metrics if not m.cache_hit],
                key=lambda m: m.execution_time,
                reverse=True
            )[:5]
            
            # Requêtes les plus fréquentes
            query_frequency = defaultdict(int)
            for metric in self.performance_metrics:
                query_frequency[metric.query_hash] += 1
            
            most_frequent = sorted(
                query_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "total_queries": total_queries,
                "cache_hit_rate": round(cache_hit_rate, 2),
                "average_execution_time": round(avg_execution_time, 3),
                "average_result_count": round(avg_result_count, 1),
                "optimization_level": self.optimization_level.value,
                "cache_size": len(self.query_cache),
                "active_rules": len([r for r in self.optimization_rules if r.enabled]),
                "slowest_queries": [
                    {
                        "query_hash": m.query_hash[:16] + "...",
                        "execution_time": round(m.execution_time, 3),
                        "result_count": m.result_count,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in slowest_queries
                ],
                "most_frequent_queries": [
                    {
                        "query_hash": query_hash[:16] + "...",
                        "frequency": frequency
                    }
                    for query_hash, frequency in most_frequent
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des statistiques: {e}")
            return {"error": str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache
        
        Returns:
            Statistiques du cache
        """
        try:
            cache_stats = self.query_cache.get_stats()
            return {
                "cache_size": cache_stats.size,
                "cache_hits": cache_stats.hits,
                "cache_misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate,
                "max_size": cache_stats.max_size,
                "memory_usage_bytes": cache_stats.memory_usage_bytes
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des stats cache: {e}")
            return {
                "cache_size": len(self.query_cache),
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0.0,
                "max_size": self.query_cache.max_size if hasattr(self.query_cache, 'max_size') else 0
            }
    
    def warm_cache(self, common_queries: List[SearchServiceQuery], executor_func: Callable):
        """
        Préchauffe le cache avec des requêtes communes
        
        Args:
            common_queries: Requêtes communes à préchauffer
            executor_func: Fonction d'exécution
        """
        if not self.warm_cache_enabled:
            return
        
        async def _warm_cache():
            try:
                logger.info(f"🔥 Préchauffage du cache avec {len(common_queries)} requêtes")
                
                for query in common_queries:
                    try:
                        await self.execute_with_cache(query, executor_func, cache_ttl=3600)
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur lors du préchauffage: {e}")
                
                logger.info("✅ Cache préchauffé")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors du préchauffage du cache: {e}")
        
        # Exécution asynchrone du préchauffage
        asyncio.create_task(_warm_cache())
    
    async def clear_cache(self):
        """
        Vide le cache
        """
        try:
            await self.query_cache.clear()
            logger.info("🧹 Cache vidé")
        except Exception as e:
            logger.error(f"❌ Erreur lors du vidage du cache: {e}")
    
    def enable_rule(self, rule_name: str):
        """
        Active une règle d'optimisation
        
        Args:
            rule_name: Nom de la règle
        """
        for rule in self.optimization_rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"✅ Règle activée: {rule_name}")
                return
        
        logger.warning(f"⚠️ Règle non trouvée: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """
        Désactive une règle d'optimisation
        
        Args:
            rule_name: Nom de la règle
        """
        for rule in self.optimization_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"❌ Règle désactivée: {rule_name}")
                return
        
        logger.warning(f"⚠️ Règle non trouvée: {rule_name}")
    
    def add_custom_rule(self, rule: OptimizationRule):
        """
        Ajoute une règle d'optimisation personnalisée
        
        Args:
            rule: Règle à ajouter
        """
        self.optimization_rules.append(rule)
        self.optimization_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"➕ Règle personnalisée ajoutée: {rule.name}")
    
    @staticmethod
    def _get_cache_size_for_level(level: OptimizationLevel) -> int:
        """
        Retourne la taille du cache selon le niveau d'optimisation
        
        Args:
            level: Niveau d'optimisation
            
        Returns:
            Taille du cache
        """
        sizes = {
            OptimizationLevel.BASIC: 100,
            OptimizationLevel.STANDARD: 500,
            OptimizationLevel.AGGRESSIVE: 1000
        }
        return sizes.get(level, 500)
    
    @staticmethod
    def _get_timeout_for_level(level: OptimizationLevel) -> int:
        """
        Retourne le timeout selon le niveau d'optimisation
        
        Args:
            level: Niveau d'optimisation
            
        Returns:
            Timeout en secondes
        """
        timeouts = {
            OptimizationLevel.BASIC: 60,
            OptimizationLevel.STANDARD: 30,
            OptimizationLevel.AGGRESSIVE: 15
        }
        return timeouts.get(level, 30)
    
    @staticmethod
    def _get_batch_size_for_level(level: OptimizationLevel) -> int:
        """
        Retourne la taille de batch selon le niveau d'optimisation
        
        Args:
            level: Niveau d'optimisation
            
        Returns:
            Taille de batch
        """
        sizes = {
            OptimizationLevel.BASIC: 3,
            OptimizationLevel.STANDARD: 5,
            OptimizationLevel.AGGRESSIVE: 10
        }
        return sizes.get(level, 5)


# ==================== OPTIMISATION AVANCÉE ====================

class AdvancedPerformanceOptimizer(PerformanceOptimizer):
    """
    Version avancée de l'optimiseur avec fonctionnalités supplémentaires
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        super().__init__(optimization_level)
        
        # Métriques avancées
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.optimization_effectiveness: Dict[str, float] = defaultdict(float)
        
        # Apprentissage automatique des optimisations
        self.auto_learning_enabled = True
        self.learning_threshold = 10  # Nombre d'exécutions avant apprentissage
        
        logger.info("✅ AdvancedPerformanceOptimizer initialisé avec apprentissage automatique")
    
    async def learn_from_query_patterns(self):
        """
        Apprend des patterns de requêtes pour optimiser automatiquement
        """
        try:
            if not self.auto_learning_enabled:
                return
            
            # Analyse des patterns de requêtes
            for query_hash, executions in self.query_stats.items():
                if len(executions) >= self.learning_threshold:
                    avg_time = sum(executions) / len(executions)
                    
                    # Si la requête est lente, créer une règle d'optimisation
                    if avg_time > 2.0:  # Plus de 2 secondes
                        await self._create_learned_optimization_rule(query_hash, avg_time)
            
            logger.info("🧠 Apprentissage automatique des optimisations terminé")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'apprentissage: {e}")
    
    async def _create_learned_optimization_rule(self, query_hash: str, avg_time: float):
        """
        Crée une règle d'optimisation basée sur l'apprentissage
        
        Args:
            query_hash: Hash de la requête
            avg_time: Temps d'exécution moyen
        """
        try:
            rule_name = f"learned_optimization_{query_hash[:8]}"
            
            # Vérifier si la règle existe déjà
            if any(rule.name == rule_name for rule in self.optimization_rules):
                return
            
            # Créer une règle spécifique pour ce pattern
            def condition(query: SearchServiceQuery) -> bool:
                return self._generate_cache_key(query) == query_hash
            
            def optimization(query: SearchServiceQuery) -> SearchServiceQuery:
                optimized = query.model_copy(deep=True)
                # Optimisations spécifiques apprises
                optimized.size = min(optimized.size or 10, 10)  # Limiter drastiquement
                optimized.timeout = "15s"  # Timeout réduit
                return optimized
            
            learned_rule = OptimizationRule(
                name=rule_name,
                condition=condition,
                optimization=optimization,
                priority=5,
                enabled=True
            )
            
            self.add_custom_rule(learned_rule)
            logger.info(f"🧠 Règle apprise créée: {rule_name} (temps moyen: {avg_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de la règle apprise: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'apprentissage
        
        Returns:
            Statistiques d'apprentissage
        """
        try:
            learned_rules = [rule for rule in self.optimization_rules if rule.name.startswith("learned_")]
            
            return {
                "auto_learning_enabled": self.auto_learning_enabled,
                "learning_threshold": self.learning_threshold,
                "learned_rules_count": len(learned_rules),
                "query_patterns_analyzed": len(self.query_patterns),
                "learned_rules": [
                    {
                        "name": rule.name,
                        "enabled": rule.enabled,
                        "priority": rule.priority
                    }
                    for rule in learned_rules
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des stats d'apprentissage: {e}")
            return {"error": str(e)}


# ==================== FACTORY ET UTILITAIRES ====================

class PerformanceOptimizerFactory:
    """
    Factory pour créer des optimiseurs de performance
    """
    
    @staticmethod
    def create_optimizer(
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        advanced: bool = False
    ) -> PerformanceOptimizer:
        """
        Crée un optimiseur de performance
        
        Args:
            optimization_level: Niveau d'optimisation
            advanced: Utiliser la version avancée
            
        Returns:
            Optimiseur configuré
        """
        if advanced:
            return AdvancedPerformanceOptimizer(optimization_level)
        else:
            return PerformanceOptimizer(optimization_level)
    
    @staticmethod
    def create_development_optimizer() -> PerformanceOptimizer:
        """
        Crée un optimiseur pour le développement
        
        Returns:
            Optimiseur configuré pour le développement
        """
        return PerformanceOptimizer(OptimizationLevel.BASIC)
    
    @staticmethod
    def create_production_optimizer() -> AdvancedPerformanceOptimizer:
        """
        Crée un optimiseur pour la production
        
        Returns:
            Optimiseur avancé configuré pour la production
        """
        return AdvancedPerformanceOptimizer(OptimizationLevel.AGGRESSIVE)


def create_custom_optimization_rule(
    name: str,
    condition_func: Callable[[SearchServiceQuery], bool],
    optimization_func: Callable[[SearchServiceQuery], SearchServiceQuery],
    priority: int = 1
) -> OptimizationRule:
    """
    Crée une règle d'optimisation personnalisée
    
    Args:
        name: Nom de la règle
        condition_func: Fonction de condition
        optimization_func: Fonction d'optimisation
        priority: Priorité de la règle
        
    Returns:
        Règle d'optimisation
    """
    return OptimizationRule(
        name=name,
        condition=condition_func,
        optimization=optimization_func,
        priority=priority,
        enabled=True
    )


# ==================== MÉTRIQUES SPÉCIALISÉES ====================

class PerformanceMetricsCollector:
    """
    Collecteur de métriques de performance spécialisé
    """
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)
        self.aggregated_metrics: Dict[str, Any] = {}
        
    def collect_execution_metrics(
        self,
        query: SearchServiceQuery,
        execution_time: float,
        result_count: int,
        cache_hit: bool,
        optimization_applied: str
    ):
        """
        Collecte les métriques d'exécution
        """
        try:
            metric = {
                "timestamp": datetime.now(),
                "query_type": query.search_type or "standard",
                "query_length": len(query.query) if query.query else 0,
                "execution_time": execution_time,
                "result_count": result_count,
                "cache_hit": cache_hit,
                "optimization_applied": optimization_applied,
                "has_filters": bool(query.filters),
                "has_aggregations": bool(query.aggregations),
                "size_requested": query.size or 10
            }
            
            self.metrics_history.append(metric)
            self._update_aggregated_metrics(metric)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la collecte des métriques: {e}")
    
    def _update_aggregated_metrics(self, metric: Dict[str, Any]):
        """
        Met à jour les métriques agrégées
        """
        try:
            # Initialisation si première métrique
            if not self.aggregated_metrics:
                self.aggregated_metrics = {
                    "total_queries": 0,
                    "total_execution_time": 0.0,
                    "cache_hits": 0,
                    "by_query_type": defaultdict(int),
                    "by_optimization": defaultdict(int),
                    "avg_execution_time": 0.0,
                    "avg_result_count": 0.0
                }
            
            # Mise à jour des compteurs
            self.aggregated_metrics["total_queries"] += 1
            self.aggregated_metrics["total_execution_time"] += metric["execution_time"]
            
            if metric["cache_hit"]:
                self.aggregated_metrics["cache_hits"] += 1
            
            self.aggregated_metrics["by_query_type"][metric["query_type"]] += 1
            self.aggregated_metrics["by_optimization"][metric["optimization_applied"]] += 1
            
            # Calcul des moyennes
            total = self.aggregated_metrics["total_queries"]
            self.aggregated_metrics["avg_execution_time"] = (
                self.aggregated_metrics["total_execution_time"] / total
            )
            
            # Moyenne des résultats (des dernières métriques)
            recent_metrics = list(self.metrics_history)[-100:]  # 100 dernières
            if recent_metrics:
                avg_results = sum(m["result_count"] for m in recent_metrics) / len(recent_metrics)
                self.aggregated_metrics["avg_result_count"] = avg_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour des métriques agrégées: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Génère un rapport de performance complet
        
        Returns:
            Rapport de performance
        """
        try:
            if not self.metrics_history:
                return {"message": "Aucune métrique disponible"}
            
            # Calculs sur les métriques récentes
            recent_metrics = list(self.metrics_history)[-1000:]  # 1000 dernières
            
            # Performance par type de requête
            by_type = defaultdict(list)
            for metric in recent_metrics:
                by_type[metric["query_type"]].append(metric["execution_time"])
            
            type_performance = {}
            for query_type, times in by_type.items():
                type_performance[query_type] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "max_time": max(times),
                    "min_time": min(times)
                }
            
            # Tendances temporelles (par heure)
            hourly_performance = defaultdict(list)
            for metric in recent_metrics:
                hour = metric["timestamp"].hour
                hourly_performance[hour].append(metric["execution_time"])
            
            hourly_stats = {}
            for hour, times in hourly_performance.items():
                if times:
                    hourly_stats[hour] = {
                        "count": len(times),
                        "avg_time": sum(times) / len(times)
                    }
            
            return {
                "summary": self.aggregated_metrics,
                "performance_by_type": type_performance,
                "hourly_performance": hourly_stats,
                "cache_hit_rate": (
                    self.aggregated_metrics["cache_hits"] / 
                    self.aggregated_metrics["total_queries"] * 100
                ) if self.aggregated_metrics["total_queries"] > 0 else 0,
                "metrics_collected": len(self.metrics_history),
                "report_generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du rapport: {e}")
            return {"error": str(e)}


# ==================== EXPORTS ====================

__all__ = [
    # Classes principales
    "PerformanceOptimizer",
    "AdvancedPerformanceOptimizer",
    "PerformanceOptimizerFactory",
    "PerformanceMetricsCollector",
    
    # Enums et dataclasses
    "OptimizationLevel",
    "QueryPerformanceMetrics",
    "OptimizationRule",
    
    # Fonctions utilitaires
    "create_custom_optimization_rule"
]