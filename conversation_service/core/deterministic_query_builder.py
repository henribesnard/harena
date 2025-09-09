"""
QueryBuilder déterministe sans LLM
Construit des requêtes Elasticsearch basées sur les intentions et entités
Remplace query_builder.py LLM par une logique pure
"""
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone

from conversation_service.models.contracts.search_service import SearchQuery, SearchFilters
from conversation_service.models.responses.conversation_responses import IntentClassificationResult
from conversation_service.utils.metrics_collector import metrics_collector

logger = logging.getLogger("conversation_service.deterministic_query_builder")


class DeterministicQueryBuilder:
    """
    QueryBuilder déterministe pour construire des requêtes Elasticsearch
    sans utilisation de LLM, basé uniquement sur la logique
    """
    
    def __init__(self):
        self.total_queries_built = 0
        self.successful_queries = 0
        self.failed_queries = 0
        
        logger.info("DeterministicQueryBuilder initialisé")
    
    def build_query(
        self,
        intent_result: IntentClassificationResult,
        entity_result: Dict[str, Any],
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> SearchQuery:
        """
        Construction déterministe de la requête Elasticsearch
        
        Args:
            intent_result: Résultat classification intention
            entity_result: Résultat extraction entités
            user_id: ID utilisateur
            context: Contexte additionnel
            
        Returns:
            SearchQuery: Requête optimisée pour search_service
        """
        try:
            self.total_queries_built += 1
            
            intent_type = intent_result.intent_type.value
            entities = entity_result.get("entities", {})
            
            logger.debug(f"Construction requête pour intention: {intent_type}")
            
            # Construction des filtres basée sur les entités
            filter_dict = self._build_base_filters(user_id, entities)
            
            # Construction de la requête selon l'intention
            query_config = self._get_query_config_for_intent(intent_type, entities)
            
            # Application configuration spécifique à l'intention
            filter_dict.update(query_config["filters"])
            
            # Création de l'objet SearchFilters à partir du dictionnaire
            # Enlever user_id qui ne fait pas partie de SearchFilters
            search_filter_dict = {k: v for k, v in filter_dict.items() if k != "user_id"}
            search_filters = SearchFilters(**search_filter_dict) if search_filter_dict else None
            
            # Construction des agrégations
            aggregations = self._build_aggregations_for_intent(intent_type, entities)
            
            # Configuration tri et pagination
            sort_config = query_config.get("sort", [{"date": {"order": "desc"}}])
            page_size = query_config.get("page_size", 20)
            
            # Pour les requêtes vagues, utiliser aggregation_only avec page_size=1
            is_vague = self._is_vague_query(entities)
            if is_vague and aggregations:
                page_size = 1
                aggregation_only = True
                logger.info(f"Requête vague détectée: utilisation aggregation_only=True avec page_size=1")
            else:
                aggregation_only = False
            
            # Construction SearchQuery
            search_query = SearchQuery(
                user_id=user_id,
                filters=search_filters,
                sort=sort_config,
                page_size=page_size,
                aggregations=aggregations if aggregations else None,
                aggregation_only=aggregation_only
            )
            
            self.successful_queries += 1
            metrics_collector.increment_counter("deterministic_query_builder.success")
            
            logger.info(f"Requête construite avec succès pour {intent_type}: {len(filter_dict)} filtres, page_size={page_size}")
            
            return search_query
            
        except Exception as e:
            self.failed_queries += 1
            metrics_collector.increment_counter("deterministic_query_builder.error")
            logger.error(f"Erreur construction requête: {str(e)}")
            
            # Fallback: requête simple
            return self._create_fallback_query(user_id)
    
    def _build_base_filters(self, user_id: int, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Construction des filtres de base"""
        filters = {
            "user_id": user_id
        }
        
        # Filtres montants
        amounts = entities.get("amounts", [])
        if amounts:
            amount_filters = self._build_amount_filters(amounts)
            if amount_filters:
                filters.update(amount_filters)
        
        # Filtres dates
        dates = entities.get("dates", [])
        if dates:
            date_filters = self._build_date_filters(dates)
            if date_filters:
                filters.update(date_filters)
        
        # Filtres marchands
        merchants = entities.get("merchants", [])
        if merchants:
            merchant_filters = self._build_merchant_filters(merchants)
            if merchant_filters:
                filters.update(merchant_filters)
        
        # Filtres catégories
        categories = entities.get("categories", [])
        if categories:
            category_filters = self._build_category_filters(categories)
            if category_filters:
                filters.update(category_filters)
        
        # Filtres types d'opération
        operation_types = entities.get("operation_types", [])
        if operation_types:
            op_type_filters = self._build_operation_type_filters(operation_types)
            if op_type_filters:
                filters.update(op_type_filters)
        
        # Filtres recherche textuelle
        text_searches = entities.get("text_search", [])
        if text_searches:
            text_filters = self._build_text_search_filters(text_searches)
            if text_filters:
                filters.update(text_filters)
        
        return filters
    
    def _build_amount_filters(self, amounts: List[Dict]) -> Dict[str, Any]:
        """Construction filtres montants"""
        if not amounts:
            return {}
        
        filters = {}
        for amount_data in amounts:
            if not isinstance(amount_data, dict):
                continue
            
            value = amount_data.get("value")
            operator = amount_data.get("operator", "eq")
            
            if value is not None:
                if operator == "eq":
                    filters["amount"] = value
                elif operator == "gte" or operator == "gt":
                    filters["amount_gte"] = value
                elif operator == "lte" or operator == "lt":
                    filters["amount_lte"] = value
                elif operator == "range":
                    # Si range, on prend le premier montant trouvé
                    min_amount = amount_data.get("min_value", value)
                    max_amount = amount_data.get("max_value", value)
                    filters["amount_gte"] = min_amount
                    filters["amount_lte"] = max_amount
                
                # On prend le premier montant valide
                break
        
        return filters
    
    def _build_date_filters(self, dates: List[Dict]) -> Dict[str, Any]:
        """Construction filtres dates"""
        if not dates:
            return {}
        
        filters = {}
        for date_data in dates:
            if not isinstance(date_data, dict):
                continue
            
            date_type = date_data.get("type", "specific")
            value = date_data.get("value")
            
            if not value:
                continue
            
            if date_type == "specific":
                # Date spécifique (YYYY-MM-DD)
                filters["date"] = value
            elif date_type == "period":
                # Période (YYYY-MM)
                if len(value) == 7 and value[4] == "-":  # Format YYYY-MM
                    filters["date_gte"] = f"{value}-01"
                    # Calculer dernier jour du mois
                    year, month = value.split("-")
                    year, month = int(year), int(month)
                    if month == 12:
                        next_year, next_month = year + 1, 1
                    else:
                        next_year, next_month = year, month + 1
                    
                    import calendar
                    last_day = calendar.monthrange(year, month)[1]
                    filters["date_lte"] = f"{value}-{last_day:02d}"
            elif date_type == "relative":
                # Date relative calculée
                try:
                    if value.startswith("last_"):
                        # last_7_days, last_30_days, etc.
                        days = int(value.split("_")[1])
                        from datetime import datetime, timedelta
                        end_date = datetime.now().date()
                        start_date = end_date - timedelta(days=days)
                        filters["date_gte"] = start_date.isoformat()
                        filters["date_lte"] = end_date.isoformat()
                except (ValueError, IndexError):
                    logger.warning(f"Date relative invalide: {value}")
            
            # On prend la première date valide
            break
        
        return filters
    
    def _build_merchant_filters(self, merchants: List[Dict]) -> Dict[str, Any]:
        """Construction filtres marchands"""
        if not merchants:
            return {}
        
        merchant_names = []
        for merchant_data in merchants:
            if isinstance(merchant_data, dict):
                name = merchant_data.get("name") or merchant_data.get("normalized")
                if name:
                    merchant_names.append(name)
            elif isinstance(merchant_data, str):
                merchant_names.append(merchant_data)
        
        if merchant_names:
            if len(merchant_names) == 1:
                return {"merchant": merchant_names[0]}
            else:
                return {"merchant_in": merchant_names}
        
        return {}
    
    def _build_category_filters(self, categories: List[Dict]) -> Dict[str, Any]:
        """Construction filtres catégories"""
        if not categories:
            return {}
        
        category_names = []
        for category_data in categories:
            if isinstance(category_data, dict):
                name = category_data.get("name")
                if name:
                    category_names.append(name)
            elif isinstance(category_data, str):
                category_names.append(category_data)
        
        if category_names:
            if len(category_names) == 1:
                return {"category": category_names[0]}
            else:
                return {"category_in": category_names}
        
        return {}
    
    def _build_operation_type_filters(self, operation_types: List[Dict]) -> Dict[str, Any]:
        """Construction filtres types d'opération"""
        if not operation_types:
            return {}
        
        op_types = []
        for op_data in operation_types:
            if isinstance(op_data, dict):
                op_type = op_data.get("type")
                if op_type:
                    op_types.append(op_type)
            elif isinstance(op_data, str):
                op_types.append(op_data)
        
        if op_types:
            if len(op_types) == 1:
                return {"operation_type": op_types[0]}
            else:
                return {"operation_type_in": op_types}
        
        return {}
    
    def _build_text_search_filters(self, text_searches: List[Dict]) -> Dict[str, Any]:
        """Construction filtres recherche textuelle"""
        if not text_searches:
            return {}
        
        # Prendre la première recherche textuelle
        for search_data in text_searches:
            if isinstance(search_data, dict):
                query = search_data.get("query")
                field = search_data.get("field", "description")  # Par défaut description
                
                if query:
                    if field == "description":
                        return {"description_search": query}
                    elif field == "merchant":
                        return {"merchant_search": query}
                    else:
                        return {"text_search": query}
        
        return {}
    
    def _get_query_config_for_intent(self, intent_type: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Configuration de requête spécifique selon l'intention"""
        
        # Déterminer si la requête est vague (sans critères spécifiques)
        is_vague_query = self._is_vague_query(entities)
        
        configs = {
            "TRANSACTION_HISTORY": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 20  # Toujours limité pour l'historique général
            },
            "TRANSACTION_SEARCH": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 20 if is_vague_query else 50  # Limite pour requêtes vagues
            },
            "SEARCH_BY_MERCHANT": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 20 if is_vague_query else 50  # Plus de résultats pour recherche spécifique
            },
            "SEARCH_BY_AMOUNT": {
                "filters": {},
                "sort": [{"amount": {"order": "desc"}}, {"date": {"order": "desc"}}],
                "page_size": 30
            },
            "SEARCH_BY_DATE": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 100  # Plus pour recherche par date
            },
            "SEARCH_BY_CATEGORY": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 50
            },
            "SEARCH_BY_OPERATION_TYPE": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 50
            },
            "SPENDING_ANALYSIS": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 200  # Plus pour analyses
            },
            "COUNT_TRANSACTIONS": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 1000  # Maximum pour comptage
            },
            "BALANCE_INQUIRY": {
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 1  # Juste besoin du solde actuel
            }
        }
        
        # Configuration par défaut
        default_config = {
            "filters": {},
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 20  # Limite par défaut pour requêtes vagues
        }
        
        return configs.get(intent_type, default_config)
    
    def _is_vague_query(self, entities: Dict[str, Any]) -> bool:
        """
        Détermine si une requête est vague (sans critères spécifiques)
        
        Une requête est considérée comme vague si elle n'a pas de:
        - Marchand spécifique
        - Montant spécifique
        - Période courte (moins d'un mois)
        - Catégorie spécifique
        """
        if not entities:
            return True
        
        # Vérifier la présence de critères spécifiques
        has_merchant = bool(entities.get("merchants"))
        has_amount = bool(entities.get("amounts"))
        has_category = bool(entities.get("categories"))
        has_operation_type = bool(entities.get("operation_types"))
        has_text_search = bool(entities.get("text_search"))
        
        # Vérifier si la période est spécifique
        has_specific_period = False
        dates = entities.get("dates", [])
        if dates:
            for date_info in dates:
                if isinstance(date_info, dict):
                    date_type = date_info.get("type")
                    if date_type == "specific":
                        has_specific_period = True
                        break
                    elif date_type == "period":
                        value = date_info.get("value", "")
                        # Période spécifique si c'est un mois précis
                        if len(value) == 7 and "-" in value:  # Format YYYY-MM
                            has_specific_period = True
                            break
        
        # La requête n'est pas vague si elle a au moins un critère spécifique
        return not (has_merchant or has_amount or has_category or 
                   has_operation_type or has_text_search or has_specific_period)
    
    def _build_aggregations_for_intent(self, intent_type: str, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Construction agrégations selon l'intention"""
        
        # Déterminer si on utilise des agrégations étendues pour gros volumes
        page_size = self._get_query_config_for_intent(intent_type, entities).get("page_size", 20)
        use_extended_aggregations = page_size > 30
        
        # Agrégations selon intention
        aggregation_configs = {
            "SPENDING_ANALYSIS": {
                "total_spending": {
                    "sum": {"field": "amount", "script": "Math.abs(_value)"}
                },
                "spending_by_category": {
                    "terms": {"field": "category", "size": 10},
                    "aggs": {
                        "total_amount": {
                            "sum": {"field": "amount", "script": "Math.abs(_value)"}
                        }
                    }
                },
                "spending_by_month": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "month"
                    },
                    "aggs": {
                        "monthly_spending": {
                            "sum": {"field": "amount", "script": "Math.abs(_value)"}
                        }
                    }
                }
            },
            "COUNT_TRANSACTIONS": {
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                }
            },
            "BALANCE_INQUIRY": {
                "current_balance": {
                    "sum": {"field": "amount"}
                }
            }
        }
        
        # Agrégations par défaut pour recherches spécifiques
        default_aggregations = {
            "total_amount": {
                "sum": {"field": "amount"}
            },
            "transaction_count": {
                "value_count": {"field": "transaction_id"}
            }
        }
        
        # Agrégations pour requêtes vagues avec aggregation_only
        vague_query_aggregations = {
            "recent_transactions": {
                "top_hits": {
                    "size": 20,
                    "sort": [{"date": {"order": "desc"}}],
                    "_source": ["amount", "merchant_name", "date", "category_name", "transaction_type", "primary_description", "amount_abs"]
                }
            },
            "weekly_summary": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "week",
                    "format": "yyyy-MM-dd",
                    "order": {"_key": "desc"}
                },
                "aggs": {
                    "debit_total": {
                        "filter": {"term": {"transaction_type": "debit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    },
                    "credit_total": {
                        "filter": {"term": {"transaction_type": "credit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    }
                }
            },
            "monthly_summary": {
                "date_histogram": {
                    "field": "date", 
                    "calendar_interval": "month",
                    "format": "yyyy-MM",
                    "order": {"_key": "desc"}
                },
                "aggs": {
                    "debit_total": {
                        "filter": {"term": {"transaction_type": "debit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    },
                    "credit_total": {
                        "filter": {"term": {"transaction_type": "credit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    }
                }
            },
            "category_breakdown": {
                "terms": {"field": "category_name.keyword", "size": 10},
                "aggs": {
                    "debit_total": {
                        "filter": {"term": {"transaction_type": "debit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    },
                    "credit_total": {
                        "filter": {"term": {"transaction_type": "credit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    }
                }
            },
            "merchant_breakdown": {
                "terms": {"field": "merchant_name.keyword", "size": 10},
                "aggs": {
                    "debit_total": {
                        "filter": {"term": {"transaction_type": "debit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    },
                    "credit_total": {
                        "filter": {"term": {"transaction_type": "credit"}},
                        "aggs": {"amount": {"sum": {"field": "amount_abs"}}}
                    }
                }
            }
        }
        
        # Agrégations étendues pour gros volumes (> 30 transactions attendues)
        extended_aggregations = {
            "total_amount": {
                "sum": {"field": "amount"}
            },
            "transaction_count": {
                "value_count": {"field": "transaction_id"}
            },
            "by_transaction_type": {
                "terms": {"field": "transaction_type", "size": 10},
                "aggs": {
                    "total_amount": {"sum": {"field": "amount"}}
                }
            },
            "by_operation_type": {
                "terms": {"field": "operation_type", "size": 10},
                "aggs": {
                    "total_amount": {"sum": {"field": "amount"}}
                }
            },
            "by_category": {
                "terms": {"field": "category", "size": 15},
                "aggs": {
                    "total_amount": {"sum": {"field": "amount"}}
                }
            },
            "by_period": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "week"
                },
                "aggs": {
                    "total_amount": {"sum": {"field": "amount"}}
                }
            },
            "top_merchants": {
                "terms": {
                    "field": "merchant_name", 
                    "size": 10,
                    "order": {"total_amount": "desc"}
                },
                "aggs": {
                    "total_amount": {
                        "sum": {"field": "amount", "script": "Math.abs(_value)"}
                    }
                }
            }
        }
        
        # Retourner agrégation appropriée
        if intent_type in aggregation_configs:
            return aggregation_configs[intent_type]
        elif intent_type.startswith("SEARCH_BY") or intent_type in ["TRANSACTION_SEARCH", "TRANSACTION_HISTORY"]:
            # Vérifier si c'est une requête vague pour utiliser les agrégations spécialisées
            is_vague = self._is_vague_query(entities)
            if is_vague:
                logger.debug(f"Utilisation des agrégations pour requête vague: {intent_type}")
                return vague_query_aggregations
            elif use_extended_aggregations:
                return extended_aggregations
            else:
                return default_aggregations
        
        # Pour les autres intentions vagues, utiliser aussi les agrégations spécialisées
        is_vague = self._is_vague_query(entities)
        if is_vague:
            return vague_query_aggregations
        
        return None
    
    def _create_fallback_query(self, user_id: int) -> SearchQuery:
        """Création requête fallback simple"""
        logger.warning(f"Utilisation requête fallback pour user_id: {user_id}")
        
        return SearchQuery(
            user_id=user_id,
            filters={"user_id": user_id},
            sort=[{"date": {"order": "desc"}}],
            page_size=20,
            aggregations=None
        )
    
    def validate_query(self, query: SearchQuery) -> bool:
        """Validation de la requête construite"""
        try:
            # Validation user_id
            if not query.user_id or query.user_id <= 0:
                logger.error("user_id manquant ou invalide")
                return False
            
            # Validation page_size
            if query.page_size and (query.page_size < 1 or query.page_size > 1000):
                logger.error(f"page_size invalide: {query.page_size}")
                return False
            
            # Validation filters
            if query.filters is not None and not isinstance(query.filters, SearchFilters):
                logger.error(f"filters doit être un objet SearchFilters, reçu: {type(query.filters)}")
                return False
            
            # Validation sort
            if query.sort and not isinstance(query.sort, list):
                logger.error("sort doit être une liste")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation requête: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du query builder"""
        return {
            "total_queries_built": self.total_queries_built,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": (
                self.successful_queries / self.total_queries_built
                if self.total_queries_built > 0 else 0.0
            )
        }