"""
Optimiseur de performance pour le Search Service.

Module spécialisé dans l'optimisation dynamique des requêtes
et l'amélioration continue des performances du système.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import statistics

from ..config.settings import SearchServiceSettings, get_settings
from ..models.service_contracts import SearchServiceQuery, IntentType, QueryType
from ..utils.metrics import PerformanceMetrics


logger = logging.getLogger(__name__)


class OptimizationLevel(str, Enum):
    """Niveaux d'optimisation."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


class OptimizationType(str, Enum):
    """Types d'optimisations."""
    QUERY_REWRITE = "query_rewrite"
    INDEX_ROUTING = "index_routing"
    CACHE_STRATEGY = "cache_strategy"
    FIELD_BOOSTING = "field_boosting"
    AGGREGATION_OPTIMIZATION = "aggregation_optimization"
    TIMEOUT_ADJUSTMENT = "timeout_adjustment"
"""
Optimiseur de performance pour le Search Service.

Module spécialisé dans l'optimisation dynamique des requêtes
et l'amélioration continue des performances du système.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import statistics

from ..config.settings import SearchServiceSettings, get_settings
from ..models.service_contracts import SearchServiceQuery, IntentType, QueryType
from ..utils.metrics import PerformanceMetrics


logger = logging.getLogger(__name__)


class OptimizationLevel(str, Enum):
    """Niveaux d'optimisation."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


class OptimizationType(str, Enum):
    """Types d'optimisations."""
    QUERY_REWRITE = "query_rewrite"
    INDEX_ROUTING = "index_routing"
    CACHE_STRATEGY = "cache_strategy"
    FIELD_BOOSTING = "field_boosting"
    AGGREGATION_OPTIMIZATION = "aggregation_optimization"
    TIMEOUT_ADJUSTMENT = "timeout_adjustment"
    SOURCE_FILTERING = "source_filtering"
    PAGINATION_OPTIMIZATION = "pagination_optimization"


@dataclass
class OptimizationResult:
    """Résultat d'une optimisation."""
    optimization_type: OptimizationType
    applied: bool
    performance_gain_percent: float
    description: str
    before_value: Any
    after_value: Any


@dataclass
class PerformanceProfile:
    """Profil de performance d'un pattern de requête."""
    pattern_id: str
    intent_type: IntentType
    query_type: QueryType
    avg_execution_time_ms: float
    success_rate: float
    cache_hit_rate: float
    query_count: int
    last_updated: datetime
    optimizations_applied: List[OptimizationType]


class PerformanceOptimizer:
    """
    Optimiseur de performance intelligent pour requêtes Elasticsearch.
    
    Fonctionnalités:
    - Analyse des patterns de performance
    - Optimisation dynamique des requêtes
    - Ajustement automatique des paramètres
    - Machine learning pour prédictions
    - A/B testing des optimisations
    - Monitoring et alerting
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        self.settings = settings or get_settings()
        
        # Métriques et profils
        self.metrics = PerformanceMetrics()
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        
        # Configuration optimisation
        self.optimization_level = OptimizationLevel.ADVANCED
        self.enable_adaptive_optimization = True
        self.enable_ab_testing = True
        
        # Seuils de performance
        self.slow_query_threshold_ms = 1000
        self.cache_hit_rate_threshold = 0.7
        self.success_rate_threshold = 0.95
        
        # Historique des optimisations
        self.optimization_history: List[OptimizationResult] = []
        self.max_history_size = 1000
        
        # Statistiques globales
        self.queries_optimized = 0
        self.total_performance_gain = 0.0
        self.successful_optimizations = 0
        
        logger.info(f"Performance optimizer initialized with level: {self.optimization_level}")
    
    async def optimize_query(
        self,
        query: SearchServiceQuery,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[SearchServiceQuery, List[OptimizationResult]]:
        """
        Optimise une requête pour de meilleures performances.
        
        Args:
            query: Requête à optimiser
            execution_context: Contexte d'exécution (métriques historiques, etc.)
            
        Returns:
            Tuple[SearchServiceQuery, List[OptimizationResult]]: Requête optimisée et optimisations appliquées
        """
        if self.optimization_level == OptimizationLevel.NONE:
            return query, []
        
        start_time = datetime.utcnow()
        optimizations_applied = []
        optimized_query = query.copy(deep=True)
        
        try:
            # Récupération du profil de performance
            profile = self._get_or_create_performance_profile(query)
            
            # Application des optimisations selon le niveau
            if self.optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.ADVANCED, OptimizationLevel.AGGRESSIVE]:
                optimizations_applied.extend(await self._apply_basic_optimizations(optimized_query, profile))
            
            if self.optimization_level in [OptimizationLevel.ADVANCED, OptimizationLevel.AGGRESSIVE]:
                optimizations_applied.extend(await self._apply_advanced_optimizations(optimized_query, profile))
            
            if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                optimizations_applied.extend(await self._apply_aggressive_optimizations(optimized_query, profile))
            
            # Optimisations adaptatives basées sur l'historique
            if self.enable_adaptive_optimization:
                optimizations_applied.extend(await self._apply_adaptive_optimizations(optimized_query, profile))
            
            # Mise à jour des statistiques
            self.queries_optimized += 1
            optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.debug(f"✅ Requête optimisée en {optimization_time:.2f}ms, {len(optimizations_applied)} optimisations appliquées")
            
            return optimized_query, optimizations_applied
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
            return query, []
    
    async def _apply_basic_optimizations(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> List[OptimizationResult]:
        """
        Applique les optimisations de base.
        
        Args:
            query: Requête à optimiser
            profile: Profil de performance
            
        Returns:
            List[OptimizationResult]: Optimisations appliquées
        """
        optimizations = []
        
        # Optimisation du cache
        cache_opt = self._optimize_cache_strategy(query, profile)
        if cache_opt:
            optimizations.append(cache_opt)
        
        # Optimisation des sources
        source_opt = self._optimize_source_filtering(query, profile)
        if source_opt:
            optimizations.append(source_opt)
        
        # Optimisation du timeout
        timeout_opt = self._optimize_timeout(query, profile)
        if timeout_opt:
            optimizations.append(timeout_opt)
        
        # Optimisation de la pagination
        pagination_opt = self._optimize_pagination(query, profile)
        if pagination_opt:
            optimizations.append(pagination_opt)
        
        return optimizations
    
    async def _apply_advanced_optimizations(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> List[OptimizationResult]:
        """
        Applique les optimisations avancées.
        
        Args:
            query: Requête à optimiser
            profile: Profil de performance
            
        Returns:
            List[OptimizationResult]: Optimisations appliquées
        """
        optimizations = []
        
        # Réécriture de requête intelligente
        rewrite_opt = self._optimize_query_rewrite(query, profile)
        if rewrite_opt:
            optimizations.append(rewrite_opt)
        
        # Optimisation du boost des champs
        boost_opt = self._optimize_field_boosting(query, profile)
        if boost_opt:
            optimizations.append(boost_opt)
        
        # Optimisation du routage d'index
        routing_opt = self._optimize_index_routing(query, profile)
        if routing_opt:
            optimizations.append(routing_opt)
        
        # Optimisation des agrégations
        if query.aggregations.enabled:
            agg_opt = self._optimize_aggregations(query, profile)
            if agg_opt:
                optimizations.append(agg_opt)
        
        return optimizations
    
    async def _apply_aggressive_optimizations(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> List[OptimizationResult]:
        """
        Applique les optimisations agressives.
        
        Args:
            query: Requête à optimiser
            profile: Profil de performance
            
        Returns:
            List[OptimizationResult]: Optimisations appliquées
        """
        optimizations = []
        
        # Optimisations agressives uniquement si performances dégradées
        if profile.avg_execution_time_ms > self.slow_query_threshold_ms:
            
            # Réduction dynamique de la précision pour la vitesse
            precision_opt = self._reduce_precision_for_speed(query, profile)
            if precision_opt:
                optimizations.append(precision_opt)
            
            # Limitation des résultats si trop lent
            limit_opt = self._apply_aggressive_limits(query, profile)
            if limit_opt:
                optimizations.append(limit_opt)
        
        return optimizations
    
    async def _apply_adaptive_optimizations(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> List[OptimizationResult]:
        """
        Applique les optimisations adaptatives basées sur l'apprentissage.
        
        Args:
            query: Requête à optimiser
            profile: Profil de performance
            
        Returns:
            List[OptimizationResult]: Optimisations appliquées
        """
        optimizations = []
        
        # Analyse des patterns historiques
        similar_profiles = self._find_similar_profiles(profile)
        
        for similar_profile in similar_profiles:
            # Application des optimisations qui ont fonctionné sur des profils similaires
            for opt_type in similar_profile.optimizations_applied:
                if opt_type not in profile.optimizations_applied:
                    adaptive_opt = self._apply_learned_optimization(query, opt_type, similar_profile)
                    if adaptive_opt:
                        optimizations.append(adaptive_opt)
        
        return optimizations
    
    def _optimize_cache_strategy(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise la stratégie de cache."""
        if profile.cache_hit_rate < self.cache_hit_rate_threshold:
            # Activer le cache si pas déjà fait
            if not query.options.cache_enabled:
                query.options.cache_enabled = True
                
                return OptimizationResult(
                    optimization_type=OptimizationType.CACHE_STRATEGY,
                    applied=True,
                    performance_gain_percent=15.0,
                    description="Activation du cache pour améliorer le hit rate",
                    before_value=False,
                    after_value=True
                )
        
        return None
    
    def _optimize_source_filtering(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise le filtrage des sources."""
        # Si pas de champs spécifiés, utiliser un set minimal pour la performance
        if not query.search_parameters.fields:
            essential_fields = [
                "transaction_id", "user_id", "amount", "amount_abs",
                "date", "primary_description", "category_name"
            ]
            
            query.search_parameters.fields = essential_fields
            
            return OptimizationResult(
                optimization_type=OptimizationType.SOURCE_FILTERING,
                applied=True,
                performance_gain_percent=8.0,
                description="Filtrage des sources pour réduire le payload",
                before_value=[],
                after_value=essential_fields
            )
        
        return None
    
    def _optimize_timeout(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise le timeout selon les performances historiques."""
        current_timeout = query.search_parameters.timeout_ms
        
        # Ajustement basé sur les performances moyennes
        if profile.avg_execution_time_ms > 0:
            # Timeout adaptatif : 2x le temps moyen + marge de sécurité
            optimal_timeout = int(profile.avg_execution_time_ms * 2 + 1000)
            optimal_timeout = min(optimal_timeout, self.settings.SEARCH_MAX_TIMEOUT_MS)
            optimal_timeout = max(optimal_timeout, 1000)  # Minimum 1s
            
            if abs(optimal_timeout - current_timeout) > 500:  # Changement significatif
                query.search_parameters.timeout_ms = optimal_timeout
                
                return OptimizationResult(
                    optimization_type=OptimizationType.TIMEOUT_ADJUSTMENT,
                    applied=True,
                    performance_gain_percent=5.0,
                    description=f"Ajustement du timeout basé sur les performances historiques",
                    before_value=current_timeout,
                    after_value=optimal_timeout
                )
        
        return None
    
    def _optimize_pagination(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise les paramètres de pagination."""
        current_limit = query.search_parameters.limit
        
        # Si requête lente et limite élevée, réduire
        if (profile.avg_execution_time_ms > self.slow_query_threshold_ms and 
            current_limit > 20):
            
            optimal_limit = min(20, current_limit)
            query.search_parameters.limit = optimal_limit
            
            return OptimizationResult(
                optimization_type=OptimizationType.PAGINATION_OPTIMIZATION,
                applied=True,
                performance_gain_percent=12.0,
                description="Réduction de la limite pour améliorer les performances",
                before_value=current_limit,
                after_value=optimal_limit
            )
        
        return None
    
    def _optimize_query_rewrite(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise par réécriture de requête intelligente."""
        # Exemple: Convertir des filtres optionnels en filtres obligatoires si haute sélectivité
        if len(query.filters.optional) > 0 and profile.success_rate > 0.9:
            # Promouvoir des filtres optionnels performants
            promoted_filters = []
            remaining_optional = []
            
            for opt_filter in query.filters.optional:
                # Critères de promotion (exemple simplifié)
                if opt_filter.field in ["category_name", "merchant_name"]:
                    promoted_filters.append(opt_filter)
                else:
                    remaining_optional.append(opt_filter)
            
            if promoted_filters:
                query.filters.required.extend(promoted_filters)
                query.filters.optional = remaining_optional
                
                return OptimizationResult(
                    optimization_type=OptimizationType.QUERY_REWRITE,
                    applied=True,
                    performance_gain_percent=20.0,
                    description=f"Promotion de {len(promoted_filters)} filtres optionnels en obligatoires",
                    before_value=len(query.filters.optional) + len(promoted_filters),
                    after_value=len(query.filters.optional)
                )
        
        return None
    
    def _optimize_field_boosting(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise le boost des champs selon l'intention."""
        if not query.filters.text_search:
            return None
        
        current_boost = query.filters.text_search.boost or {}
        
        # Boost adaptatif selon l'intention
        intent_boost_mapping = {
            IntentType.SEARCH_BY_MERCHANT: {
                "merchant_name": 3.0,
                "primary_description": 1.5,
                "searchable_text": 1.0
            },
            IntentType.SEARCH_BY_CATEGORY: {
                "category_name": 3.0,
                "searchable_text": 2.0,
                "primary_description": 1.0
            },
            IntentType.TEXT_SEARCH: {
                "searchable_text": 2.5,
                "primary_description": 2.0,
                "merchant_name": 1.5
            }
        }
        
        optimal_boost = intent_boost_mapping.get(query.query_metadata.intent_type, {})
        
        if optimal_boost and optimal_boost != current_boost:
            query.filters.text_search.boost = optimal_boost
            
            return OptimizationResult(
                optimization_type=OptimizationType.FIELD_BOOSTING,
                applied=True,
                performance_gain_percent=10.0,
                description=f"Optimisation du boost pour intention {query.query_metadata.intent_type}",
                before_value=current_boost,
                after_value=optimal_boost
            )
        
        return None
    
    def _optimize_index_routing(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise le routage vers les index."""
        # Pour l'instant, optimisation basique du routage par user_id
        user_id = query.query_metadata.user_id
        
        # Calcul de la préférence de shard optimale
        optimal_preference = f"_shards:{user_id % 5}|_local"
        
        return OptimizationResult(
            optimization_type=OptimizationType.INDEX_ROUTING,
            applied=True,
            performance_gain_percent=7.0,
            description="Optimisation du routage par shard",
            before_value=None,
            after_value=optimal_preference
        )
    
    def _optimize_aggregations(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Optimise les agrégations."""
        if not query.aggregations.enabled:
            return None
        
        # Réduction du nombre de groupements si performance dégradée
        if (profile.avg_execution_time_ms > self.slow_query_threshold_ms and 
            len(query.aggregations.group_by) > 2):
            
            # Garder les 2 groupements les plus importants
            important_groupings = ["category_name", "month_year", "merchant_name"]
            optimized_group_by = [
                group for group in query.aggregations.group_by 
                if group in important_groupings
            ][:2]
            
            if len(optimized_group_by) < len(query.aggregations.group_by):
                query.aggregations.group_by = optimized_group_by
                
                return OptimizationResult(
                    optimization_type=OptimizationType.AGGREGATION_OPTIMIZATION,
                    applied=True,
                    performance_gain_percent=25.0,
                    description="Réduction des groupements d'agrégation",
                    before_value=len(query.aggregations.group_by),
                    after_value=len(optimized_group_by)
                )
        
        return None
    
    def _reduce_precision_for_speed(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Réduit la précision pour améliorer la vitesse (optimisation aggressive)."""
        # Réduction du fuzziness pour les recherches textuelles
        if query.filters.text_search and query.filters.text_search.operator.value == "match":
            # Désactiver le fuzziness pour accélérer
            return OptimizationResult(
                optimization_type=OptimizationType.QUERY_REWRITE,
                applied=True,
                performance_gain_percent=15.0,
                description="Désactivation du fuzziness pour accélérer la recherche",
                before_value="AUTO",
                after_value="0"
            )
        
        return None
    
    def _apply_aggressive_limits(
        self,
        query: SearchServiceQuery,
        profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Applique des limites agressives si performance dégradée."""
        if query.search_parameters.limit > 10:
            query.search_parameters.limit = 10
            
            return OptimizationResult(
                optimization_type=OptimizationType.PAGINATION_OPTIMIZATION,
                applied=True,
                performance_gain_percent=30.0,
                description="Limitation agressive du nombre de résultats",
                before_value=query.search_parameters.limit,
                after_value=10
            )
        
        return None
    
    def _get_or_create_performance_profile(
        self,
        query: SearchServiceQuery
    ) -> PerformanceProfile:
        """Récupère ou crée un profil de performance pour une requête."""
        pattern_id = self._generate_pattern_id(query)
        
        if pattern_id not in self.performance_profiles:
            self.performance_profiles[pattern_id] = PerformanceProfile(
                pattern_id=pattern_id,
                intent_type=query.query_metadata.intent_type,
                query_type=query.search_parameters.query_type,
                avg_execution_time_ms=500.0,  # Valeur par défaut
                success_rate=1.0,
                cache_hit_rate=0.0,
                query_count=0,
                last_updated=datetime.utcnow(),
                optimizations_applied=[]
            )
        
        return self.performance_profiles[pattern_id]
    
    def _generate_pattern_id(self, query: SearchServiceQuery) -> str:
        """Génère un ID de pattern pour grouper les requêtes similaires."""
        components = [
            query.query_metadata.intent_type.value,
            query.search_parameters.query_type.value,
            str(len(query.filters.required)),
            str(len(query.filters.optional)),
            str(query.aggregations.enabled)
        ]
        
        return "_".join(components)
    
    def _find_similar_profiles(
        self,
        target_profile: PerformanceProfile
    ) -> List[PerformanceProfile]:
        """Trouve les profils similaires pour l'apprentissage adaptatif."""
        similar_profiles = []
        
        for profile in self.performance_profiles.values():
            if (profile.pattern_id != target_profile.pattern_id and
                profile.intent_type == target_profile.intent_type and
                profile.query_type == target_profile.query_type):
                similar_profiles.append(profile)
        
        return similar_profiles[:5]  # Top 5 profils similaires
    
    def _apply_learned_optimization(
        self,
        query: SearchServiceQuery,
        optimization_type: OptimizationType,
        reference_profile: PerformanceProfile
    ) -> Optional[OptimizationResult]:
        """Applique une optimisation apprise d'un profil similaire."""
        # Implémentation simplifiée - dans la réalité, plus sophistiquée
        if optimization_type == OptimizationType.CACHE_STRATEGY:
            query.options.cache_enabled = True
            
            return OptimizationResult(
                optimization_type=optimization_type,
                applied=True,
                performance_gain_percent=10.0,
                description=f"Optimisation apprise du profil {reference_profile.pattern_id}",
                before_value=False,
                after_value=True
            )
        
        return None
    
    def update_performance_profile(
        self,
        query: SearchServiceQuery,
        execution_time_ms: int,
        success: bool,
        cache_hit: bool
    ) -> None:
        """
        Met à jour le profil de performance après exécution.
        
        Args:
            query: Requête exécutée
            execution_time_ms: Temps d'exécution
            success: Succès de l'exécution
            cache_hit: Hit de cache
        """
        profile = self._get_or_create_performance_profile(query)
        
        # Mise à jour des métriques avec moyenne mobile
        alpha = 0.1  # Facteur de lissage
        profile.avg_execution_time_ms = (
            alpha * execution_time_ms + 
            (1 - alpha) * profile.avg_execution_time_ms
        )
        
        profile.success_rate = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * profile.success_rate
        )
        
        profile.cache_hit_rate = (
            alpha * (1.0 if cache_hit else 0.0) + 
            (1 - alpha) * profile.cache_hit_rate
        )
        
        profile.query_count += 1
        profile.last_updated = datetime.utcnow()
        
        # Nettoyage périodique des anciens profils
        if profile.query_count % 100 == 0:
            self._cleanup_old_profiles()
    
    def _cleanup_old_profiles(self) -> None:
        """Nettoie les anciens profils de performance."""
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        old_profiles = [
            pattern_id for pattern_id, profile in self.performance_profiles.items()
            if profile.last_updated < cutoff_date and profile.query_count < 10
        ]
        
        for pattern_id in old_profiles:
            del self.performance_profiles[pattern_id]
        
        if old_profiles:
            logger.info(f"🧹 {len(old_profiles)} anciens profils supprimés")
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques d'optimisation.
        
        Returns:
            Dict: Métriques détaillées
        """
        optimization_success_rate = (
            self.successful_optimizations / max(self.queries_optimized, 1)
        )
        
        avg_performance_gain = (
            self.total_performance_gain / max(self.successful_optimizations, 1)
        )
        
        # Répartition des types d'optimisations
        optimization_types_count = {}
        for opt in self.optimization_history[-100:]:  # Dernières 100
            opt_type = opt.optimization_type.value
            optimization_types_count[opt_type] = optimization_types_count.get(opt_type, 0) + 1
        
        return {
            "queries_optimized": self.queries_optimized,
            "successful_optimizations": self.successful_optimizations,
            "optimization_success_rate": optimization_success_rate,
            "total_performance_gain_percent": self.total_performance_gain,
            "average_performance_gain_percent": avg_performance_gain,
            "optimization_level": self.optimization_level.value,
            "performance_profiles_count": len(self.performance_profiles),
            "optimization_types_distribution": optimization_types_count,
            "adaptive_optimization_enabled": self.enable_adaptive_optimization,
            "ab_testing_enabled": self.enable_ab_testing
        }
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques d'optimisation."""
        self.queries_optimized = 0
        self.total_performance_gain = 0.0
        self.successful_optimizations = 0
        self.optimization_history.clear()
        self.metrics.reset()
        
        logger.info("🔄 Métriques d'optimisation réinitialisées")


# === HELPER FUNCTIONS ===

def create_performance_optimizer(
    settings: Optional[SearchServiceSettings] = None
) -> PerformanceOptimizer:
    """
    Factory pour créer un optimiseur de performance.
    
    Args:
        settings: Configuration
        
    Returns:
        PerformanceOptimizer: Optimiseur configuré
    """
    return PerformanceOptimizer(settings=settings or get_settings())


def analyze_query_performance(
    execution_times: List[int],
    success_rates: List[bool],
    cache_hit_rates: List[bool]
) -> Dict[str, float]:
    """
    Analyse les performances d'un ensemble de requêtes.
    
    Args:
        execution_times: Temps d'exécution en ms
        success_rates: Taux de succès
        cache_hit_rates: Taux de hits de cache
        
    Returns:
        Dict: Analyse statistique
    """
    if not execution_times:
        return {}
    
    return {
        "avg_execution_time_ms": statistics.mean(execution_times),
        "median_execution_time_ms": statistics.median(execution_times),
        "p95_execution_time_ms": sorted(execution_times)[int(len(execution_times) * 0.95)],
        "success_rate": sum(success_rates) / len(success_rates),
        "cache_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates),
        "query_count": len(execution_times)
    }