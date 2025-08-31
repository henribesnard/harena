"""
Optimiseur de requêtes search_service
Applique des optimisations automatiques pour améliorer les performances
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import copy

from conversation_service.models.contracts.search_service import (
    DEFAULT_PAGE_SIZE, MAX_AGGREGATION_BUCKETS, ESSENTIAL_FIELDS
)

logger = logging.getLogger("conversation_service.query_optimizer")


class QueryOptimizer:
    """Optimiseur de requêtes avec règles de performance automatiques"""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
        self.performance_cache = {}
    
    def optimize_query(self, query: Dict[str, Any], intent_type: str = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Optimise une requête search_service selon les meilleures pratiques
        
        Args:
            query: Requête à optimiser
            intent_type: Type d'intention pour optimisations spécialisées
            
        Returns:
            Tuple[Dict, List]: Requête optimisée et liste des optimisations appliquées
        """
        optimized_query = copy.deepcopy(query)
        applied_optimizations = []
        
        try:
            # 1. Optimisations structurelles de base
            struct_opts = self._apply_structural_optimizations(optimized_query)
            applied_optimizations.extend(struct_opts)
            
            # 2. Optimisations des filtres
            filter_opts = self._optimize_filters(optimized_query)
            applied_optimizations.extend(filter_opts)
            
            # 3. Optimisations des agrégations
            agg_opts = self._optimize_aggregations(optimized_query)
            applied_optimizations.extend(agg_opts)
            
            # 4. Optimisations du tri
            sort_opts = self._optimize_sorting(optimized_query)
            applied_optimizations.extend(sort_opts)
            
            # 5. Optimisations spécifiques à l'intention
            if intent_type:
                intent_opts = self._apply_intent_optimizations(optimized_query, intent_type)
                applied_optimizations.extend(intent_opts)
            
            # 6. Optimisations de performance finale
            perf_opts = self._apply_performance_optimizations(optimized_query)
            applied_optimizations.extend(perf_opts)
            
            logger.info(f"Requête optimisée avec {len(applied_optimizations)} améliorations")
            return optimized_query, applied_optimizations
            
        except Exception as e:
            logger.error(f"Erreur optimisation requête: {str(e)}")
            return query, ["Erreur lors de l'optimisation, requête originale conservée"]
    
    def _apply_structural_optimizations(self, query: Dict[str, Any]) -> List[str]:
        """Optimisations structurelles de base"""
        optimizations = []
        
        # user_id obligatoire au niveau racine
        if "user_id" not in query and "filters" in query:
            user_id = query["filters"].get("user_id")
            if user_id:
                query["user_id"] = user_id
                optimizations.append("Promotion user_id au niveau racine")
        
        # user_id obligatoire dans les filtres si pas présent
        if "filters" not in query:
            query["filters"] = {}
        
        if "user_id" not in query["filters"] and "user_id" in query:
            query["filters"]["user_id"] = query["user_id"]
            optimizations.append("Ajout user_id dans filtres pour performance")
        
        # page_size par défaut
        if "page_size" not in query and not query.get("aggregation_only", False):
            query["page_size"] = DEFAULT_PAGE_SIZE
            optimizations.append(f"Configuration page_size par défaut: {DEFAULT_PAGE_SIZE}")
        
        # include_fields par défaut si pas aggregation_only
        if ("include_fields" not in query and 
            not query.get("aggregation_only", False) and
            "exclude_fields" not in query):
            query["include_fields"] = ESSENTIAL_FIELDS
            optimizations.append("Limitation aux champs essentiels pour performance")
        
        # Nettoyage aggregation_only
        if query.get("aggregation_only", False):
            if "include_fields" in query:
                del query["include_fields"]
                optimizations.append("Suppression include_fields en mode aggregation_only")
            if query.get("page_size", 0) > 0:
                query["page_size"] = 0
                optimizations.append("page_size=0 en mode aggregation_only")
        
        return optimizations
    
    def _optimize_filters(self, query: Dict[str, Any]) -> List[str]:
        """Optimisation des filtres"""
        optimizations = []
        filters = query.get("filters", {})
        
        # Promotion filtres sélectifs vers le début
        selective_filters = ["user_id", "account_id", "transaction_id"]
        reordered_filters = {}
        
        # Filtres sélectifs en premier
        for filter_name in selective_filters:
            if filter_name in filters:
                reordered_filters[filter_name] = filters[filter_name]
        
        # Autres filtres ensuite
        for filter_name, filter_value in filters.items():
            if filter_name not in selective_filters:
                reordered_filters[filter_name] = filter_value
        
        if reordered_filters != filters:
            query["filters"] = reordered_filters
            optimizations.append("Réorganisation filtres par sélectivité")
        
        # Optimisation filtres textuels
        for field_name, filter_value in filters.items():
            if isinstance(filter_value, dict):
                # Conversion match vers term pour valeurs exactes courtes
                if "match" in filter_value:
                    match_value = filter_value["match"]
                    if isinstance(match_value, str) and len(match_value) < 10 and " " not in match_value:
                        filter_value["term"] = match_value
                        del filter_value["match"]
                        optimizations.append(f"Optimisation filtre '{field_name}': match → term")
        
        # Consolidation plages de dates/montants
        date_filters = ["date", "created_at", "updated_at"]
        for date_field in date_filters:
            if date_field in filters and isinstance(filters[date_field], dict):
                date_filter = filters[date_field]
                consolidated = self._consolidate_range_filter(date_filter)
                if consolidated != date_filter:
                    filters[date_field] = consolidated
                    optimizations.append(f"Consolidation filtre de plage '{date_field}'")
        
        return optimizations
    
    def _optimize_aggregations(self, query: Dict[str, Any]) -> List[str]:
        """Optimisation des agrégations"""
        optimizations = []
        aggregations = query.get("aggregations", {})
        
        if not aggregations:
            return optimizations
        
        # Limitation du nombre de buckets
        for agg_name, agg_config in aggregations.items():
            if isinstance(agg_config, dict):
                # Agrégations terms
                if "terms" in agg_config:
                    terms_config = agg_config["terms"]
                    current_size = terms_config.get("size", 10)
                    
                    if current_size > MAX_AGGREGATION_BUCKETS:
                        terms_config["size"] = MAX_AGGREGATION_BUCKETS
                        optimizations.append(f"Limitation buckets '{agg_name}' de {current_size} à {MAX_AGGREGATION_BUCKETS}")
                    elif "size" not in terms_config:
                        terms_config["size"] = min(10, MAX_AGGREGATION_BUCKETS)
                        optimizations.append(f"Définition size par défaut pour '{agg_name}': 10")
                
                # Optimisation agrégations imbriquées
                if "aggs" in agg_config:
                    nested_opts = self._optimize_nested_aggregations(agg_config["aggs"], agg_name)
                    optimizations.extend(nested_opts)
        
        # Réorganisation agrégations par coût
        expensive_agg_types = ["cardinality", "percentiles", "significant_terms"]
        cheap_agg_types = ["value_count", "sum", "avg", "max", "min"]
        
        reordered_aggs = {}
        
        # Agrégations peu coûteuses en premier
        for agg_name, agg_config in aggregations.items():
            if any(cheap_type in agg_config for cheap_type in cheap_agg_types):
                reordered_aggs[agg_name] = agg_config
        
        # Agrégations moyennes
        for agg_name, agg_config in aggregations.items():
            if (agg_name not in reordered_aggs and 
                not any(exp_type in agg_config for exp_type in expensive_agg_types)):
                reordered_aggs[agg_name] = agg_config
        
        # Agrégations coûteuses en dernier
        for agg_name, agg_config in aggregations.items():
            if agg_name not in reordered_aggs:
                reordered_aggs[agg_name] = agg_config
        
        if list(reordered_aggs.keys()) != list(aggregations.keys()):
            query["aggregations"] = reordered_aggs
            optimizations.append("Réorganisation agrégations par coût de calcul")
        
        return optimizations
    
    def _optimize_nested_aggregations(self, nested_aggs: Dict[str, Any], parent_name: str) -> List[str]:
        """Optimisation agrégations imbriquées"""
        optimizations = []
        
        for nested_name, nested_config in nested_aggs.items():
            if isinstance(nested_config, dict) and "terms" in nested_config:
                terms_config = nested_config["terms"]
                current_size = terms_config.get("size", 10)
                
                # Limitation plus stricte pour agrégations imbriquées
                max_nested_size = min(5, MAX_AGGREGATION_BUCKETS // 2)
                if current_size > max_nested_size:
                    terms_config["size"] = max_nested_size
                    optimizations.append(f"Limitation buckets imbriqués '{parent_name}.{nested_name}': {max_nested_size}")
        
        return optimizations
    
    def _optimize_sorting(self, query: Dict[str, Any]) -> List[str]:
        """Optimisation du tri"""
        optimizations = []
        
        sort_config = query.get("sort", [])
        if not sort_config:
            # Tri par défaut selon type de requête
            if query.get("aggregation_only", False):
                # Pas de tri nécessaire pour aggregation_only
                pass
            elif "filters" in query and "date" in query["filters"]:
                # Tri par date si filtre date présent
                query["sort"] = [{"date": {"order": "desc"}}]
                optimizations.append("Ajout tri par date pour performance")
            else:
                # Tri générique par date
                query["sort"] = [{"date": {"order": "desc"}}]
                optimizations.append("Ajout tri par défaut: date desc")
        
        # Optimisation tri multiple
        elif len(sort_config) > 3:
            # Limitation à 3 critères de tri maximum
            query["sort"] = sort_config[:3]
            optimizations.append("Limitation tri à 3 critères maximum")
        
        return optimizations
    
    def _apply_intent_optimizations(self, query: Dict[str, Any], intent_type: str) -> List[str]:
        """Optimisations spécifiques par type d'intention"""
        optimizations = []
        
        intent_configs = {
            "BALANCE_INQUIRY": {
                "force_aggregation_only": True,
                "essential_aggregations": ["balance_by_account", "total_balance"],
                "remove_sort": True
            },
            "SPENDING_ANALYSIS": {
                "force_aggregation_only": True,
                "force_debit_filter": True,
                "essential_aggregations": ["category_breakdown", "total_spending"]
            },
            "COUNT_TRANSACTIONS": {
                "force_aggregation_only": True,
                "essential_aggregations": ["transaction_count"],
                "remove_sort": True
            },
            "SEARCH_BY_MERCHANT": {
                "add_merchant_aggregations": True,
                "limit_page_size": 30
            },
            "SEARCH_BY_AMOUNT": {
                "sort_by_amount": True,
                "limit_page_size": 50
            }
        }
        
        config = intent_configs.get(intent_type, {})
        
        # Force aggregation_only
        if config.get("force_aggregation_only", False) and not query.get("aggregation_only", False):
            query["aggregation_only"] = True
            query["page_size"] = 0
            if "include_fields" in query:
                del query["include_fields"]
            optimizations.append(f"Mode aggregation_only forcé pour {intent_type}")
        
        # Force filtre debit pour analyses dépenses
        if config.get("force_debit_filter", False):
            filters = query.setdefault("filters", {})
            if "transaction_type" not in filters:
                filters["transaction_type"] = "debit"
                optimizations.append("Ajout filtre transaction_type=debit pour analyse dépenses")
        
        # Suppression tri inutile
        if config.get("remove_sort", False) and "sort" in query:
            del query["sort"]
            optimizations.append("Suppression tri inutile pour requête d'agrégation")
        
        # Tri spécialisé par montant
        if config.get("sort_by_amount", False):
            query["sort"] = [{"amount_abs": {"order": "desc"}}]
            optimizations.append("Tri optimisé par montant pour recherche par montant")
        
        # Limitation page_size spécialisée
        page_limit = config.get("limit_page_size")
        if page_limit and query.get("page_size", DEFAULT_PAGE_SIZE) > page_limit:
            query["page_size"] = page_limit
            optimizations.append(f"Limitation page_size à {page_limit} pour {intent_type}")
        
        # Ajout agrégations marchands automatique
        if config.get("add_merchant_aggregations", False):
            aggregations = query.setdefault("aggregations", {})
            if "merchant_stats" not in aggregations:
                aggregations["merchant_stats"] = {
                    "terms": {"field": "merchant_name.keyword", "size": 10},
                    "aggs": {
                        "total_spent": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
                optimizations.append("Ajout agrégations marchands automatique")
        
        return optimizations
    
    def _apply_performance_optimizations(self, query: Dict[str, Any]) -> List[str]:
        """Optimisations finales de performance"""
        optimizations = []
        
        # Détection requêtes potentiellement lentes
        performance_score = self._calculate_performance_score(query)
        
        if performance_score < 60:  # Requête potentiellement lente
            # Réduction automatique des buckets
            aggregations = query.get("aggregations", {})
            for agg_name, agg_config in aggregations.items():
                if isinstance(agg_config, dict) and "terms" in agg_config:
                    current_size = agg_config["terms"].get("size", 10)
                    if current_size > 10:
                        agg_config["terms"]["size"] = 10
                        optimizations.append(f"Réduction buckets '{agg_name}' pour performance")
            
            # Limitation page_size pour requêtes lentes
            if not query.get("aggregation_only", False) and query.get("page_size", 0) > 20:
                query["page_size"] = 20
                optimizations.append("Réduction page_size pour requête complexe")
        
        # Cache hint pour requêtes répétitives
        if self._is_cacheable_query(query):
            query["_cache_hint"] = True
            optimizations.append("Activation cache pour requête répétitive")
        
        # Timeout adaptatif
        complexity = self._estimate_query_complexity(query)
        if complexity > 50:
            query["_timeout"] = "30s"
            optimizations.append("Timeout étendu pour requête complexe")
        elif complexity < 20:
            query["_timeout"] = "10s"
            optimizations.append("Timeout réduit pour requête simple")
        
        return optimizations
    
    def _consolidate_range_filter(self, range_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidation des filtres de plage"""
        # Conversion de multiples filtres en range unique si bénéfique
        has_gte = "gte" in range_filter
        has_lte = "lte" in range_filter
        has_gt = "gt" in range_filter
        has_lt = "lt" in range_filter
        
        if (has_gte or has_gt) and (has_lte or has_lt):
            # Déjà optimisé en plage
            return range_filter
        
        return range_filter
    
    def _calculate_performance_score(self, query: Dict[str, Any]) -> int:
        """Calcul score de performance (0-100, 100 = optimal)"""
        score = 100
        
        # Pénalités
        aggregations = query.get("aggregations", {})
        
        # Recherche textuelle libre
        if "query" in query and query["query"]:
            score -= 20
        
        # Filtres textuels floues
        filters = query.get("filters", {})
        for filter_value in filters.values():
            if isinstance(filter_value, dict) and "match" in filter_value:
                score -= 10
        
        # Nombre d'agrégations
        agg_count = len(aggregations)
        if agg_count > 5:
            score -= 15
        elif agg_count > 3:
            score -= 10
        
        # Buckets agrégations
        for agg_config in aggregations.values():
            if isinstance(agg_config, dict) and "terms" in agg_config:
                size = agg_config["terms"].get("size", 10)
                if size > 20:
                    score -= 10
                elif size > 50:
                    score -= 20
        
        # Page size
        page_size = query.get("page_size", DEFAULT_PAGE_SIZE)
        if page_size > 100:
            score -= 15
        elif page_size > 50:
            score -= 10
        
        return max(0, score)
    
    def _estimate_query_complexity(self, query: Dict[str, Any]) -> int:
        """Estimation complexité requête (0-100)"""
        complexity = 0
        
        # Filtres
        filters = query.get("filters", {})
        complexity += len(filters) * 5
        
        # Agrégations
        aggregations = query.get("aggregations", {})
        complexity += len(aggregations) * 10
        
        # Agrégations imbriquées
        for agg_config in aggregations.values():
            if isinstance(agg_config, dict) and "aggs" in agg_config:
                complexity += len(agg_config["aggs"]) * 15
        
        # Tri multiple
        sort_config = query.get("sort", [])
        if len(sort_config) > 1:
            complexity += (len(sort_config) - 1) * 5
        
        # Recherche textuelle
        if "query" in query:
            complexity += 20
        
        return min(100, complexity)
    
    def _is_cacheable_query(self, query: Dict[str, Any]) -> bool:
        """Détermine si la requête est cacheable"""
        # Pas de timestamps relatifs, pas de recherche libre complexe
        has_relative_dates = False
        filters = query.get("filters", {})
        
        if "date" in filters:
            date_filter = filters["date"]
            if isinstance(date_filter, dict):
                # Vérification dates relatives (aujourd'hui, etc.)
                for date_value in date_filter.values():
                    if isinstance(date_value, str) and "now" in date_value.lower():
                        has_relative_dates = True
                        break
        
        return not has_relative_dates and not query.get("query", "").strip()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Chargement règles d'optimisation personnalisables"""
        return {
            "max_aggregation_buckets": MAX_AGGREGATION_BUCKETS,
            "default_page_size": DEFAULT_PAGE_SIZE,
            "performance_threshold": 60,
            "complexity_threshold": 50,
            "enable_auto_cache": True,
            "enable_timeout_adaptation": True,
            "max_sort_criteria": 3,
            "max_nested_agg_buckets": 5
        }