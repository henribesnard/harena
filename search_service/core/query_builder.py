import logging
from typing import Dict, Any, List, Optional, Set, Union
from search_service.models.request import SearchRequest

logger = logging.getLogger(__name__)

class QueryBuilder:
    """
    Constructeur de requêtes Elasticsearch - Version Multi-Index
    
    Support 2 index :
    - harena_transactions (transactions nettoyées)
    - harena_accounts (comptes avec soldes)
    
    Corrections appliquées:
    - ✅ Support complet filtre EXISTS (true/false/{})
    - ✅ Détection pipeline aggregations dans order
    - ✅ Validation robuste des champs
    - ✅ Gestion d'erreurs améliorée
    - ✅ Routage automatique multi-index
    """
    
    def __init__(self):
        # Configuration des champs de recherche pour TRANSACTIONS
        self.transaction_search_fields = [
            "primary_description^1.5",   # Description transaction principale
            "merchant_name^1.8",         # Nom marchand avec boost élevé
            "category_name^1.0",         # Catégorie standard
        ]
        
        # Configuration des champs de recherche pour ACCOUNTS
        self.account_search_fields = [
            "account_name^2.0",          # Nom du compte avec boost élevé
            "account_type^1.0",          # Type de compte
        ]
        
        # Champs nécessitant .keyword pour filtres exacts (TRANSACTIONS)
        self.transaction_keyword_fields: Set[str] = {
            'category_name', 'merchant_name', 'operation_type',
            'currency_code', 'transaction_type', 'weekday'
        }
        
        # Champs nécessitant .keyword pour filtres exacts (ACCOUNTS)  
        # Note: account_type est déjà mappé comme keyword dans ES, donc pas besoin de .keyword
        self.account_keyword_fields: Set[str] = {
            'account_name', 'account_currency'
        }
        
        # Configuration des index
        self.transactions_index = "harena_transactions"
        self.accounts_index = "harena_accounts"
        
        # Champs autorisés pour les agrégations TRANSACTIONS
        self.transaction_aggregation_fields: Set[str] = {
            "amount", "amount_abs", "date", "transaction_id", "user_id", "account_id",
            "currency_code", "transaction_type", "operation_type", "category_name",
            "merchant_name", "primary_description", "month_year", "weekday"
        }
        
        # Champs autorisés pour les agrégations ACCOUNTS
        self.account_aggregation_fields: Set[str] = {
            "user_id", "account_id", "account_name", "account_type", 
            "account_balance", "account_currency", "is_active", "created_at", "updated_at"
        }

        # Unions globales pour validations génériques
        self.allowed_aggregation_fields: Set[str] = (
            self.transaction_aggregation_fields | self.account_aggregation_fields
        )
        self.global_keyword_fields: Set[str] = (
            self.transaction_keyword_fields | self.account_keyword_fields
        )
        
        # Types d'agrégation Elasticsearch valides
        self.elasticsearch_agg_types: Set[str] = {
            # Métriques simples
            "sum", "avg", "min", "max", "value_count", "cardinality",
            "stats", "extended_stats", "percentiles", "percentile_ranks",
            
            # Buckets
            "terms", "date_histogram", "histogram", "range", "date_range",
            "filters", "filter", "missing", "nested", "reverse_nested",
            "global", "sampler", "diversified_sampler",
            
            # Métrique spéciale
            "top_hits",
            
            # Pipeline aggregations
            "derivative", "moving_avg", "cumulative_sum", "bucket_script",
            "bucket_selector", "bucket_sort", "serial_diff"
        }
        
        # Pipeline aggregations qui ne peuvent pas être utilisées pour l'ordre
        self.pipeline_agg_types: Set[str] = {
            "derivative", "moving_avg", "cumulative_sum", "bucket_script",
            "bucket_selector", "bucket_sort", "serial_diff"
        }

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """Vérifie si la chaîne fournie représente un nombre."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _detect_search_type(self, request: SearchRequest) -> str:
        """
        Détecte automatiquement si on cherche des comptes ou des transactions.
        
        Returns:
            str: "accounts" ou "transactions"
        """
        # 🏦 Détection ACCOUNTS : si on cherche spécifiquement des champs de comptes
        account_specific_fields = {
            'account_balance', 'account_name', 'account_type', 'account_currency', 'is_active'
        }
        
        # Vérifier dans les filtres
        if hasattr(request, 'filters') and request.filters:
            for field in request.filters.keys():
                if field in account_specific_fields:
                    logger.debug(f"🏦 Détection ACCOUNTS via filtre: {field}")
                    return "accounts"
        
        # Vérifier dans les agrégations
        if hasattr(request, 'aggregations') and request.aggregations:
            for agg_name, agg_def in request.aggregations.items():
                if self._aggregation_references_accounts(agg_def):
                    logger.debug(f"🏦 Détection ACCOUNTS via agrégation: {agg_name}")
                    return "accounts"
        
        # Vérifier dans source (liste de champs demandés par le client)
        if hasattr(request, 'source') and request.source:
            for field in request.source:
                if field in account_specific_fields:
                    logger.debug(f"🏦 Détection ACCOUNTS via source: {field}")
                    return "accounts"
        
        # 📄 Par défaut : TRANSACTIONS
        logger.debug("📄 Détection TRANSACTIONS (par défaut)")
        return "transactions"
    
    def _aggregation_references_accounts(self, agg_def: Dict[str, Any]) -> bool:
        """Vérifie récursivement si une agrégation référence des champs de comptes."""
        account_fields = {
            'account_balance', 'account_name', 'account_type', 'account_currency', 'is_active'
        }
        
        def check_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "field" and value in account_fields:
                        return True
                    if isinstance(value, (dict, list)):
                        if check_recursive(value):
                            return True
            elif isinstance(obj, list):
                for item in obj:
                    if check_recursive(item):
                        return True
            return False
        
        return check_recursive(agg_def)

    def build_query(self, request: SearchRequest) -> Dict[str, Any]:
        """
        Construction d'une requête Elasticsearch complète avec détection multi-index.
        """
        # 🔍 Détection automatique du type de recherche
        search_type = self._detect_search_type(request)
        target_index = self.accounts_index if search_type == "accounts" else self.transactions_index
        
        # Configuration adaptée au type de recherche
        if search_type == "accounts":
            search_fields = self.account_search_fields
            keyword_fields = self.account_keyword_fields
            allowed_agg_fields = self.account_aggregation_fields
        else:
            search_fields = self.transaction_search_fields
            keyword_fields = self.transaction_keyword_fields
            allowed_agg_fields = self.transaction_aggregation_fields
        
        logger.debug(f"🎯 Query builder: {search_type} sur index {target_index}")
        
        # Séparation critique : query (scoring) vs filters (performance)
        text_query_part = None
        must_filters = []
        
        # 1. Filtres obligatoires (sécurité critique)
        must_filters.append({"term": {"user_id": request.user_id}})
        
        # 2. Filtre document_type pour éviter la pollution des index
        if search_type == "transactions":
            # Pour les transactions : exclure les documents account qui sont mélangés dans harena_transactions
            must_filters.append({"bool": {"must_not": [{"term": {"document_type.keyword": "account"}}]}})
        
        # 3. Requête textuelle → QUERY avec scoring
        if request.query and request.query.strip():
            cleaned_query = request.query.strip()
            if self._is_numeric(cleaned_query) and search_type == "accounts":
                # Requête numérique sur accounts → filtre sur account_balance
                value = float(cleaned_query)
                must_filters.append({"range": {"account_balance": {"gte": value, "lte": value}}})
            else:
                # Requête textuelle → multi_match avec scoring
                text_query_part = self._build_text_query(cleaned_query, search_fields)
        
        # 4. Filtres additionnels (exclure user_id déjà ajouté pour sécurité)
        additional_filters_dict = {k: v for k, v in request.filters.items() if k != "user_id"}
        additional_filters = self._build_additional_filters(additional_filters_dict, keyword_fields)
        must_filters.extend(additional_filters)
        
        # 5. Construction bool query optimisée
        bool_query = {}
        
        if text_query_part:
            # Mode scoring: query dans must, filtres dans filter
            bool_query["must"] = [text_query_part]
            if must_filters:
                bool_query["filter"] = must_filters
        else:
            # Mode filtre uniquement: tout dans must
            bool_query["must"] = must_filters
        
        # 6. Tri intelligent adapté au type
        sort_criteria = request.sort if request.sort is not None else self._build_sort_criteria(request, search_type)
        
        # 7. Pagination
        page = getattr(request, "page", 1)
        offset = (page - 1) * request.page_size
        
        # 8. Construction requête finale
        query = {
            "query": {"bool": bool_query},
            "sort": sort_criteria,
            "_source": self._get_source_fields(search_type),
            "size": request.page_size,
            "from": offset
        }

        # 9. Track scores si nécessaire
        if self._needs_score_tracking(sort_criteria):
            query["track_scores"] = True

        # 10. Highlighting si demandé
        if request.highlight:
            query["highlight"] = request.highlight
        
        # 🎯 Retour avec métadonnées d'index
        return {
            "query": query,
            "target_index": target_index,
            "search_type": search_type
        }

    def _build_text_query(self, query_text: str, search_fields: List[str]) -> Dict[str, Any]:
        """Construction de la requête textuelle multi_match optimisée."""
        terms_count = len(query_text.split())
        minimum_should_match = "50%" if terms_count >= 2 else "100%"
        
        return {
            "multi_match": {
                "query": query_text,
                "fields": search_fields,  # 🔧 Utilise les champs dynamiques
                "type": "best_fields",
                "fuzziness": "AUTO",
                "minimum_should_match": minimum_should_match
            }
        }

    def _build_sort_criteria(self, request: SearchRequest, search_type: str = "transactions") -> List[Dict[str, Any]]:
        """Construction des critères de tri intelligents adaptés au type d'index."""
        sort_criteria = []
        
        # Si recherche textuelle non-numérique → tri par score d'abord
        if (request.query and request.query.strip() and 
            not self._is_numeric(request.query.strip())):
            sort_criteria.append({"_score": {"order": "desc"}})
        
        # Tri par défaut selon le type d'index
        if search_type == "accounts":
            # Pour les comptes : tri par account_id ou account_name
            sort_criteria.append({"account_id": {"order": "asc"}})
        else:
            # Pour les transactions : tri par date (historique)
            sort_criteria.append({"date": {"order": "desc"}})
        
        return sort_criteria

    def _build_additional_filters(self, filters: Dict[str, Any], keyword_fields: Set[str]) -> List[Dict[str, Any]]:
        """
        Construction des filtres additionnels avec support complet des types Elasticsearch.
        
        CORRECTIONS MAJEURES:
        - ✅ Support complet filtre EXISTS (true/false/{})
        - ✅ Gestion robuste de tous les types de filtres
        - ✅ Validation et sécurité renforcées
        - ✅ Support multi-index avec keyword_fields dynamiques
        """
        filter_list: List[Dict[str, Any]] = []

        for field, value in filters.items():
            if value is None:
                continue

            try:
                # DICTIONNAIRE: différents types de filtres complexes
                if isinstance(value, dict):
                    filter_result = self._process_dict_filter(field, value, keyword_fields)
                    if filter_result:
                        filter_list.append(filter_result)

                # LISTE: terms filter
                elif isinstance(value, list):
                    field_name = self._get_filter_field_name(field, keyword_fields)
                    filter_list.append({"terms": {field_name: value}})

                # VALEUR SIMPLE: term filter
                else:
                    field_name = self._get_filter_field_name(field, keyword_fields)
                    filter_list.append({"term": {field_name: value}})
                    
            except ValueError as e:
                logger.error(f"Error processing filter {field}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing filter {field}: {e}")
                # Continue avec les autres filtres plutôt que de planter
                continue

        return filter_list

    def _process_dict_filter(self, field: str, value: Dict[str, Any], keyword_fields: Set[str]) -> Optional[Dict[str, Any]]:
        """
        Traite un filtre de type dictionnaire (filtres complexes).
        
        CORRECTION MAJEURE: Support complet du filtre EXISTS + multi-index
        """
        
        # 1. RANGE FILTER: opérateurs numériques/temporels
        if {"gt", "gte", "lt", "lte"} & set(value.keys()):
            return {"range": {field: value}}

        # 2. EXISTS FILTER: support complet true/false/{}
        elif "exists" in value:
            exists_value = value["exists"]
            
            # Support multiple formats: true, {}, false
            if exists_value is True or (isinstance(exists_value, dict) and len(exists_value) == 0):
                return {"exists": {"field": field}}
            elif exists_value is False:
                return {"bool": {"must_not": [{"exists": {"field": field}}]}}
            else:
                logger.warning(f"Invalid exists value: {exists_value}")
                return None

        # 3. WILDCARD FILTER: patterns
        elif "wildcard" in value:
            field_name = self._get_filter_field_name(field, keyword_fields)
            pattern = value["wildcard"]
            return {"wildcard": {field_name: {"value": pattern}}}

        # 4. PREFIX FILTER: préfixes
        elif "prefix" in value:
            field_name = self._get_filter_field_name(field, keyword_fields)
            prefix = value["prefix"]
            return {"prefix": {field_name: prefix}}

        # 5. REGEXP FILTER: expressions régulières
        elif "regexp" in value:
            field_name = self._get_filter_field_name(field, keyword_fields)
            regex = value["regexp"]
            return {"regexp": {field_name: regex}}

        # 6. MATCH / MATCH_PHRASE FILTERS: support textuel et fallback keyword
        elif "match" in value:
            match_value = value["match"]
            # Pour les champs textuels connus comme keyword_fields, on convertit en term sur .keyword
            if field in keyword_fields:
                field_name = self._get_filter_field_name(field, keyword_fields)
                if isinstance(match_value, dict):
                    query_val = match_value.get("query")
                else:
                    query_val = match_value
                if query_val is None:
                    return None
                return {"term": {field_name: query_val}}
            else:
                # Champ analysé: utiliser match natif
                if isinstance(match_value, dict):
                    return {"match": {field: match_value}}
                else:
                    return {"match": {field: {"query": match_value, "operator": "and"}}}

        elif "match_phrase" in value:
            phrase_value = value["match_phrase"]
            if field in keyword_fields:
                # Sur un keyword, match_phrase n'a pas de sens → fallback term
                field_name = self._get_filter_field_name(field, keyword_fields)
                if isinstance(phrase_value, dict):
                    query_val = phrase_value.get("query")
                else:
                    query_val = phrase_value
                if query_val is None:
                    return None
                return {"term": {field_name: query_val}}
            else:
                if isinstance(phrase_value, dict):
                    return {"match_phrase": {field: phrase_value}}
                else:
                    return {"match_phrase": {field: {"query": phrase_value}}}

        # 7. TERM/TERMS EXPLICIT
        elif "term" in value:
            field_name = self._get_filter_field_name(field, keyword_fields)
            return {"term": {field_name: value["term"]}}

        elif "terms" in value:
            field_name = self._get_filter_field_name(field, keyword_fields)
            return {"terms": {field_name: value["terms"]}}

        # 8. FILTRE INCONNU
        else:
            msg = f"Unsupported filter type for field '{field}': {value}"
            logger.error(msg)
            raise ValueError(msg)

    def _get_filter_field_name(self, field: str, keyword_fields: Set[str]) -> str:
        """
        Détermine le nom de champ correct pour les filtres.
        Ajoute .keyword pour les champs textuels si nécessaire.
        """
        return f"{field}.keyword" if field in keyword_fields else field
    
    def _get_source_fields(self, search_type: str = "transactions") -> List[str]:
        """Définit les champs à retourner selon le type d'index."""
        if search_type == "accounts":
            # Champs pour l'index accounts
            return [
                "user_id", "account_id", "account_name", "account_type", 
                "account_balance", "account_currency", "is_active",
                "created_at", "updated_at", "last_sync_timestamp"
            ]
        else:
            # Champs pour l'index transactions (nettoyé)
            return [
                "transaction_id", "user_id", "account_id",  # account_id comme lien
                "amount", "amount_abs", "currency_code", "transaction_type",
                "date", "month_year", "weekday",
                "primary_description", "merchant_name", "category_name", "operation_type"
            ]

    def _needs_score_tracking(self, sort_criteria: List[Dict[str, Any]]) -> bool:
        """Détermine si le tracking des scores est nécessaire."""
        has_score = any("_score" in str(criterion) for criterion in sort_criteria)
        has_other_sort = len(sort_criteria) > 1
        return has_score and has_other_sort

    def build_aggregation_query(
        self, request: SearchRequest, aggregation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Construction d'une requête avec agrégations Elasticsearch natives.
        
        CORRECTIONS MAJEURES:
        - ✅ Détection pipeline aggregations dans order
        - ✅ Validation sécurisée des agrégations
        - ✅ Support format Elasticsearch natif et legacy
        """
        base_query = self.build_query(request)

        if not aggregation:
            return base_query

        aggregations: Dict[str, Any] = {}

        # Détection du format d'agrégation
        if self._is_elasticsearch_native_format(aggregation):
            # Validation et correction des agrégations
            validated_aggs = self._validate_elasticsearch_aggregations(aggregation)
            aggregations = validated_aggs
        else:
            aggregations = self._build_legacy_aggregations(aggregation)

        # Ajouter les agrégations à la requête finale
        if aggregations:
            base_query["query"]["aggs"] = aggregations
            
            # Mode aggregation-only
            if getattr(request, "aggregation_only", False):
                base_query["query"]["size"] = 0
            else:
                base_query["query"]["size"] = request.page_size

        return base_query

    def _build_legacy_aggregations(self, aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les agrégations au format legacy (compatibilité arrière)."""
        aggregations: Dict[str, Any] = {}
        
        group_by = aggregation.get("group_by") or []
        metrics = aggregation.get("metrics") or []

        if group_by:
            for field in group_by:
                field_name = self._get_filter_field_name(field, self.global_keyword_fields)
                agg_def: Dict[str, Any] = {
                    "terms": {"field": field_name, "size": 10}
                }
                sub_aggs: Dict[str, Any] = {}
                if "sum" in metrics:
                    sub_aggs["amount_sum"] = {"sum": {"field": "amount"}}
                if sub_aggs:
                    agg_def["aggs"] = sub_aggs
                aggregations[f"{field}_terms"] = agg_def
        elif "sum" in metrics:
            aggregations["amount_sum"] = {"sum": {"field": "amount"}}
            
        return aggregations

    def _is_elasticsearch_native_format(self, aggregation: Dict[str, Any]) -> bool:
        """Détecte si les agrégations sont au format Elasticsearch natif."""
        # Si contient les clés abstraites, c'est l'ancien format
        abstract_keys = {"group_by", "metrics", "types"}
        if any(key in aggregation for key in abstract_keys):
            return False
        
        # Vérifier que c'est bien du format Elasticsearch natif
        for agg_name, agg_def in aggregation.items():
            if not isinstance(agg_def, dict):
                continue
                
            # Chercher un type d'agrégation ES valide au niveau racine
            agg_types_found = set(agg_def.keys()) & self.elasticsearch_agg_types
            if agg_types_found:
                return True
                
            # Vérifier les sous-agrégations (aggs)
            if "aggs" in agg_def and isinstance(agg_def["aggs"], dict):
                return True
        
        return False

    def _validate_elasticsearch_aggregations(self, aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et corrige les agrégations Elasticsearch natives.
        
        CORRECTION MAJEURE: Détection et correction pipeline aggregations dans order
        """
        validated = {}
        
        for agg_name, agg_def in aggregation.items():
            # Validation nom agrégation
            if not isinstance(agg_name, str) or len(agg_name) > 100:
                logger.warning(f"Invalid aggregation name: {agg_name}")
                continue
                
            if not isinstance(agg_def, dict):
                logger.warning(f"Invalid aggregation definition for {agg_name}")
                continue
                
            # Validation et correction récursive
            validated_def = self._validate_aggregation_definition(agg_def, self.allowed_aggregation_fields)
            if validated_def:
                validated[agg_name] = validated_def
                logger.info(f"✅ Agrégation validée: {agg_name}")
            else:
                logger.warning(f"❌ Agrégation rejetée: {agg_name} - Définition: {agg_def}")
        
        return validated

    def _validate_aggregation_definition(self, agg_def: Dict[str, Any], allowed_fields: Set[str]) -> Optional[Dict[str, Any]]:
        """
        Valide récursivement une définition d'agrégation.
        
        CORRECTION MAJEURE: Fix pipeline aggregations dans order
        """
        validated: Dict[str, Any] = {}

        # 1) Traiter les définitions de types d'agrégations (terms, date_histogram, sum, ...)
        agg_type_keys = set(agg_def.keys()) & self.elasticsearch_agg_types
        for agg_type in agg_type_keys:
            sub_def = agg_def.get(agg_type, {})
            if not isinstance(sub_def, dict):
                continue
            validated_sub: Dict[str, Any] = {}
            for k, v in sub_def.items():
                if k == "field" and isinstance(v, str):
                    base_field = v.replace(".keyword", "")
                    if base_field not in allowed_fields:
                        logger.warning(f"Field not allowed in aggregation: {v}")
                        return None
                    # Pour les agrégations sur des champs textuels, forcer .keyword si applicable
                    if agg_type in {"terms", "cardinality", "value_count"} and base_field in self.global_keyword_fields:
                        validated_sub[k] = f"{base_field}.keyword"
                    else:
                        validated_sub[k] = v
                else:
                    # Validation simple des autres valeurs
                    if self._is_safe_aggregation_value(k, v):
                        validated_sub[k] = v
                    else:
                        logger.warning(f"Invalid aggregation value for key {k}: {type(v)}")
                        return None
            # Politique métier: éviter 'global' (ignore la query).
            # Remplacer tout 'global' par un bucket 'filter' neutre qui respecte le contexte de la query.
            if agg_type == "global":
                validated["filter"] = {"match_all": {}}
            elif validated_sub:
                validated[agg_type] = validated_sub

        # 2) Traiter la clé 'order' au même niveau (ex: terms + order)
        if "order" in agg_def and isinstance(agg_def["order"], dict):
            corrected_order = self._fix_pipeline_aggregation_order(agg_def["order"], agg_def)
            if corrected_order:
                validated["order"] = corrected_order
                if corrected_order != agg_def["order"]:
                    logger.info(f"Order corrected from {agg_def['order']} to {corrected_order}")
            else:
                logger.warning(f"Invalid order removed: {agg_def['order']}")

        # 3) Traiter récursivement les sous-agrégations
        if "aggs" in agg_def and isinstance(agg_def["aggs"], dict):
            sub_aggs: Dict[str, Any] = {}
            for sub_name, sub_def in agg_def["aggs"].items():
                validated_sub = self._validate_aggregation_definition(sub_def, allowed_fields)
                if validated_sub:
                    sub_aggs[sub_name] = validated_sub
            if sub_aggs:
                validated["aggs"] = sub_aggs

            # Si aucune définition d'agrégation top-level n'est présente
            # mais qu'il y a des sous-agrégations, envelopper dans un 'filter' neutre
            if not agg_type_keys and "filter" not in validated:
                validated["filter"] = {"match_all": {}}

        # 4) Conserver les autres paires clé/valeur sûres au niveau courant
        for key, value in agg_def.items():
            if key in validated:
                continue
            if key in agg_type_keys or key in {"aggs", "order"}:
                continue
            if self._is_safe_aggregation_value(key, value):
                validated[key] = value
            else:
                logger.warning(f"Invalid aggregation value for key {key}: {type(value)}")
                return None

        return validated if validated else None

    def _fix_pipeline_aggregation_order(self, order_def: Dict[str, Any], parent_agg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Corrige les problèmes d'ordre avec les pipeline aggregations.
        
        CORRECTION CRITIQUE: Les pipeline aggregations (bucket_script, derivative, etc.) 
        ne peuvent pas être utilisées dans order. On les remplace par une métrique simple.
        """
        if not isinstance(order_def, dict):
            return order_def
            
        corrected_order = {}
        aggs_section = parent_agg.get("aggs", {})
        
        for order_field, order_direction in order_def.items():
            # Vérifier si le champ d'ordre est une pipeline aggregation
            if order_field in aggs_section:
                agg_definition = aggs_section[order_field]
                
                # Si c'est une pipeline aggregation
                if self._is_pipeline_aggregation(agg_definition):
                    # Trouver une alternative non-pipeline
                    alternative = self._find_alternative_order_field(aggs_section)
                    if alternative:
                        corrected_order[alternative] = order_direction
                        logger.info(f"Pipeline aggregation order fixed: {order_field} → {alternative}")
                    else:
                        # Fallback sur _count
                        corrected_order["_count"] = order_direction
                        logger.warning(f"No alternative found for pipeline agg order: {order_field}, using _count")
                else:
                    # Garder l'ordre original si ce n'est pas une pipeline agg
                    corrected_order[order_field] = order_direction
            else:
                # Champ d'ordre non trouvé dans aggs, garder tel quel
                corrected_order[order_field] = order_direction
        
        return corrected_order if corrected_order else None

    def _is_pipeline_aggregation(self, agg_def: Dict[str, Any]) -> bool:
        """Détermine si une agrégation est une pipeline aggregation."""
        if not isinstance(agg_def, dict):
            return False
            
        agg_types = set(agg_def.keys()) & self.pipeline_agg_types
        return len(agg_types) > 0

    def _find_alternative_order_field(self, aggs_section: Dict[str, Any]) -> Optional[str]:
        """Trouve une alternative non-pipeline pour l'ordre."""
        # Chercher des agrégations de métrique simple (non-pipeline)
        simple_metrics = ["sum", "avg", "min", "max", "value_count"]
        
        for agg_name, agg_def in aggs_section.items():
            if isinstance(agg_def, dict):
                agg_types = set(agg_def.keys()) & set(simple_metrics)
                if agg_types:
                    return agg_name
        
        return None

    def _is_safe_aggregation_value(self, key: str, value: Any) -> bool:
        """Vérifie si une valeur d'agrégation est sûre."""
        # Valeurs numériques limitées
        if isinstance(value, (int, float)):
            return -1000000 <= value <= 1000000
        
        # Strings courtes
        elif isinstance(value, str):
            return len(value) <= 100
        
        # Dictionnaires et listes - passage direct
        elif isinstance(value, (dict, list)):
            return True
            
        # Booléens
        elif isinstance(value, bool):
            return True
        
        return False
