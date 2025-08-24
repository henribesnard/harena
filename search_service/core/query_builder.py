import logging
from typing import Dict, Any, List, Optional
from search_service.models.request import SearchRequest

logger = logging.getLogger(__name__)

# 🔥 DEBUG FORCING - Cette ligne DOIT apparaître dans les logs
logger.critical("🔥🔥🔥 QUERY BUILDER VERSION FINALE CORRIGÉE CHARGÉE 🔥🔥🔥")

class QueryBuilder:
    """Constructeur de requêtes Elasticsearch - 🔥 VERSION FINALE CORRIGÉE AVEC DEBUG FORCING"""
    
    def __init__(self):
        logger.critical("🔥 QueryBuilder.__init__() - VERSION FINALE CORRIGÉE")
        # ✅ CORRECTION : Configuration des champs de recherche basée sur mapping réel
        self.search_fields = [
            "primary_description^1.5",   # Description transaction (existe)
            "merchant_name^1.8",         # Nom marchand (existe) 
            "category_name^1.0",         # Catégorie (existe)
            # ❌ SUPPRIMÉ : "searchable_text^2.0" car potentiellement inexistant
        ]

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """Vérifie si la chaîne fournie représente un nombre."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def build_query(self, request: SearchRequest) -> Dict[str, Any]:
        """
        🔥 VERSION FINALE CORRIGÉE avec séparation query/filter + debug forcing complet
        """
        logger.critical(f"🔥 build_query() APPELÉE - query='{request.query}', highlight={bool(request.highlight)}")
        
        # ✅ CORRECTION CRITIQUE : Séparation query et filters
        text_query_part = None
        must_filters = []
        
        # Filtre obligatoire user_id (toujours en filter)
        must_filters.append({"term": {"user_id": request.user_id}})
        logger.critical(f"🔥 Ajout filtre user_id: {request.user_id}")
        
        # ✅ CORRECTION : Requête textuelle devient une QUERY (pas filter) 
        if request.query and request.query.strip():
            cleaned_query = request.query.strip()
            logger.critical(f"🔥 REQUÊTE TEXTUELLE DÉTECTÉE: '{cleaned_query}'")
            
            if self._is_numeric(cleaned_query):
                value = float(cleaned_query)
                must_filters.append({"range": {"account_balance": {"gte": value, "lte": value}}})
                logger.critical(f"🔥 Ajout filtre numérique account_balance: {value}")
            else:
                text_query_part = self._build_text_query(cleaned_query)
                logger.critical(f"🔥 REQUÊTE DE SCORING CRÉÉE: {text_query_part}")
        else:
            logger.critical("🔥 Pas de requête textuelle fournie")
        
        # Filtres additionnels
        additional_filters = self._build_additional_filters(request.filters)
        must_filters.extend(additional_filters)
        logger.critical(f"🔥 Total filtres ajoutés: {len(additional_filters)} additionnels")
        
        # ✅ CORRECTION CRITIQUE : Construction requête avec séparation query/filter
        bool_query = {}
        
        if text_query_part:
            bool_query["must"] = [text_query_part]
            logger.critical("🔥 SCORING QUERY AJOUTÉE DANS 'must' - va générer _score")
        
        if must_filters:
            if text_query_part:
                bool_query["filter"] = must_filters
                logger.critical(f"🔥 {len(must_filters)} FILTRES AJOUTÉS DANS 'filter' (pas de scoring)")
            else:
                bool_query["must"] = must_filters
                logger.critical(f"🔥 {len(must_filters)} FILTRES AJOUTÉS DANS 'must' (pas de query textuelle)")
        
        # Pagination : calcul de l'offset basé sur page/page_size
        page = getattr(request, "page", 1)
        page_size = request.page_size
        offset = (page - 1) * page_size

        # Construction requête finale
        query = {
            "query": {"bool": bool_query},
            "sort": self._build_sort_criteria(request),
            "_source": self._get_source_fields(),
            "size": request.page_size,
            "from": request.offset
        }

        # ✅ CORRECTION HIGHLIGHTING : Toujours ajouté si demandé
        if request.highlight:
            query["highlight"] = request.highlight
            logger.critical(f"🔥 HIGHLIGHTING AJOUTÉ À LA REQUÊTE: {request.highlight}")
        else:
            logger.critical("🔥 Pas de highlighting demandé")

        logger.critical(f"🔥 PAGINATION: page={page}, page_size={page_size}, offset={offset}")
        logger.critical(f"🔥 REQUÊTE FINALE GÉNÉRÉE: {query}")
        
        return query

    def _build_text_query(self, query_text: str) -> Dict[str, Any]:
        """
        Construction de la requête textuelle optimisée
        🔥 VERSION FINALE CORRIGÉE - Requête qui génère _score
        """
        logger.critical(f"🔥 _build_text_query('{query_text}')")
        
        terms_count = len(query_text.split())
        minimum_should_match = "50%" if terms_count >= 2 else "100%"
        
        text_query = {
            "multi_match": {
                "query": query_text,
                "fields": self.search_fields,
                "type": "best_fields",
                "fuzziness": "AUTO",
                "minimum_should_match": minimum_should_match
            }
        }
        
        logger.critical(f"🔥 TEXT QUERY GÉNÉRÉE: {text_query}")
        logger.critical(f"🔥 Champs de recherche utilisés: {self.search_fields}")
        return text_query

    def _build_sort_criteria(self, request: SearchRequest) -> List[Dict[str, Any]]:
        """
        Construction des critères de tri
        🔥 VERSION FINALE CORRIGÉE - Sort par score uniquement si query textuelle
        """
        sort_criteria = []
        
        # ✅ CORRECTION : Si c'est une recherche textuelle NON-NUMÉRIQUE, trier par score d'abord
        if (request.query and request.query.strip() and 
            not self._is_numeric(request.query.strip())):
            sort_criteria.append({"_score": {"order": "desc"}})
            logger.critical("🔥 TRI PAR _score AJOUTÉ (recherche textuelle)")
        else:
            logger.critical("🔥 Pas de tri par _score (pas de recherche textuelle ou requête numérique)")
        
        # Toujours trier par date décroissante en second
        sort_criteria.append({"date": {"order": "desc"}})
        
        logger.critical(f"🔥 CRITÈRES DE TRI FINAUX: {sort_criteria}")
        return sort_criteria

    def _build_additional_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construction des filtres additionnels - 🔥 VERSION FINALE AVEC DEBUG"""
        logger.critical(f"🔥 _build_additional_filters() - {len(filters)} filtres à traiter")
        
        filter_list: List[Dict[str, Any]] = []

        for field, value in filters.items():
            if value is None:
                continue
            
            logger.critical(f"🔥 Traitement filtre: {field} = {value}")

            # --- Dictionnaire : différents types de filtres ---
            if isinstance(value, dict):
                # 1) Range : détection des opérateurs numériques
                if {"gt", "gte", "lt", "lte"} & set(value.keys()):
                    filter_list.append({"range": {field: value}})
                    logger.critical(f"🔥 Ajout range filter sur {field}: {value}")

                # 2) Exists
                elif value.get("exists"):
                    filter_list.append({"exists": {"field": field}})
                    logger.critical(f"🔥 Ajout exists filter sur {field}")

                # 3) Wildcard
                elif "wildcard" in value:
                    field_name = self._get_filter_field_name(field)
                    pattern = value["wildcard"]
                    filter_list.append({"wildcard": {field_name: {"value": pattern}}})
                    logger.critical(f"🔥 Ajout wildcard filter sur {field_name}: {pattern}")

                # 4) Prefix
                elif "prefix" in value:
                    field_name = self._get_filter_field_name(field)
                    prefix = value["prefix"]
                    filter_list.append({"prefix": {field_name: prefix}})
                    logger.critical(f"🔥 Ajout prefix filter sur {field_name}: {prefix}")

                # 5) Regexp
                elif "regexp" in value:
                    field_name = self._get_filter_field_name(field)
                    regex = value["regexp"]
                    filter_list.append({"regexp": {field_name: regex}})
                    logger.critical(f"🔥 Ajout regexp filter sur {field_name}: {regex}")

                # 6) Term / Terms explicit
                elif "term" in value:
                    field_name = self._get_filter_field_name(field)
                    filter_list.append({"term": {field_name: value["term"]}})
                    logger.critical(f"🔥 Ajout explicit term filter sur {field_name}: {value['term']}")

                elif "terms" in value:
                    field_name = self._get_filter_field_name(field)
                    filter_list.append({"terms": {field_name: value["terms"]}})
                    logger.critical(f"🔥 Ajout explicit terms filter sur {field_name}: {len(value['terms'])} values")

                else:
                    # Filtre inconnu => erreur claire
                    msg = f"Unsupported filter type for field '{field}': {value}"
                    logger.error(f"🔥 ERREUR: {msg}")
                    raise ValueError(msg)

            # --- Liste : terms ---
            elif isinstance(value, list):
                field_name = self._get_filter_field_name(field)
                filter_list.append({"terms": {field_name: value}})
                logger.critical(f"🔥 Ajout terms filter sur {field_name}: {len(value)} values")

            # --- Valeur simple : term ---
            else:
                field_name = self._get_filter_field_name(field)
                filter_list.append({"term": {field_name: value}})
                logger.critical(f"🔥 Ajout term filter sur {field_name}: {value}")

        logger.critical(f"🔥 FILTRES FINAUX GÉNÉRÉS: {len(filter_list)} filtres")
        return filter_list
    
    def _get_filter_field_name(self, field: str) -> str:
        """
        Détermine le nom de champ correct pour les filtres
        Ajoute .keyword pour les champs textuels si nécessaire
        """
        # Champs qui nécessitent .keyword pour les filtres exacts
        keyword_fields = {
            'category_name', 'merchant_name', 'operation_type',
            'currency_code', 'transaction_type', 'weekday',
            'account_name', 'account_type', 'account_currency'
        }
        
        field_name = f"{field}.keyword" if field in keyword_fields else field
        logger.critical(f"🔥 Nom de champ pour filtre: {field} -> {field_name}")
        return field_name
    
    def _get_source_fields(self) -> List[str]:
        """Définit les champs à retourner dans les résultats"""
        source_fields = [
            "transaction_id", "user_id", "account_id",
            "account_name", "account_type", "account_balance", "account_currency",
            "amount", "amount_abs", "currency_code", "transaction_type",
            "date", "month_year", "weekday",
            "primary_description", "merchant_name", "category_name", "operation_type"
        ]
        logger.critical(f"🔥 Source fields définis: {len(source_fields)} champs")
        return source_fields
    
    def build_aggregation_query(
        self, request: SearchRequest, aggregation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Construction d'une requête avec agrégations Elasticsearch natives.
        🔥 VERSION FINALE CORRIGÉE - Support des deux formats avec debug
        """
        logger.critical(f"🔥 build_aggregation_query() APPELÉE")
        
        base_query = self.build_query(request)

        if not aggregation:
            logger.critical("🔥 Pas d'agrégations fournies")
            return base_query

        aggregations: Dict[str, Any] = {}

        # ✅ NOUVEAU : Détection et traitement du format Elasticsearch natif
        if self._is_elasticsearch_native_format(aggregation):
            logger.critical("🔥 Format d'agrégation Elasticsearch natif détecté")
            # Validation et passage direct des agrégations
            validated_aggs = self._validate_elasticsearch_aggregations(aggregation)
            aggregations = validated_aggs
        
        # ✅ ANCIEN : Support format abstrait pour compatibilité arrière
        else:
            logger.critical("🔥 Format d'agrégation abstrait legacy détecté")
            group_by = aggregation.get("group_by") or []
            metrics = aggregation.get("metrics") or []

            if group_by:
                for field in group_by:
                    field_name = self._get_filter_field_name(field)
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

        # Ajouter les agrégations à la requête finale
        if aggregations:
            base_query["aggs"] = aggregations
            logger.critical(f"🔥 Ajout de {len(aggregations)} agrégations: {list(aggregations.keys())}")
            if getattr(request, "aggregation_only", False):
                base_query["size"] = 0
                logger.critical("🔥 Requête aggregation-only: size = 0")
            else:
                base_query["size"] = request.page_size
                logger.critical(f"🔥 Requête avec agrégations + résultats: size = {request.page_size}")
        else:
            logger.critical("🔥 Aucune agrégation valide générée")

        return base_query

    def _is_elasticsearch_native_format(self, aggregation: Dict[str, Any]) -> bool:
        """
        Détecte si les agrégations sont au format Elasticsearch natif.
        """
        # Si contient les clés abstraites, c'est l'ancien format
        abstract_keys = {"group_by", "metrics", "types"}
        if any(key in aggregation for key in abstract_keys):
            return False
        
        # Vérifier que c'est bien du format Elasticsearch natif
        elasticsearch_agg_types = {
            # Métriques simples
            "sum", "avg", "min", "max", "value_count", "cardinality",
            "stats", "extended_stats", "percentiles", "percentile_ranks",
            
            # Buckets
            "terms", "date_histogram", "histogram", "range", "date_range",
            "filters", "filter", "missing", "nested", "reverse_nested",
            "global", "sampler", "diversified_sampler"
        }
        
        # Vérifier chaque agrégation
        for agg_name, agg_def in aggregation.items():
            if not isinstance(agg_def, dict):
                continue
                
            # Chercher un type d'agrégation ES valide au niveau racine
            agg_types_found = set(agg_def.keys()) & elasticsearch_agg_types
            if agg_types_found:
                return True
                
            # Vérifier les sous-agrégations (aggs)
            if "aggs" in agg_def and isinstance(agg_def["aggs"], dict):
                return True
        
        return False

    def _validate_elasticsearch_aggregations(self, aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et nettoie les agrégations Elasticsearch natives.
        """
        # Champs autorisés pour les agrégations
        allowed_fields = {
            "amount", "amount_abs", "date", "transaction_id", "user_id", "account_id",
            "currency_code", "transaction_type", "operation_type", "category_name",
            "merchant_name", "primary_description", "month_year", "weekday",
            "account_name", "account_type", "account_balance", "account_currency"
        }
        
        validated = {}
        
        for agg_name, agg_def in aggregation.items():
            # Validation nom agrégation (pas d'injection)
            if not isinstance(agg_name, str) or len(agg_name) > 100:
                logger.warning(f"Invalid aggregation name: {agg_name}")
                continue
                
            if not isinstance(agg_def, dict):
                logger.warning(f"Invalid aggregation definition for {agg_name}")
                continue
                
            # Validation récursive de la définition
            validated_def = self._validate_aggregation_definition(agg_def, allowed_fields)
            if validated_def:
                validated[agg_name] = validated_def
        
        logger.critical(f"🔥 Agrégations validées: {list(validated.keys())}")
        return validated

    def _validate_aggregation_definition(self, agg_def: Dict[str, Any], allowed_fields: set) -> Optional[Dict[str, Any]]:
        """Valide récursivement une définition d'agrégation."""
        validated = {}
        
        for key, value in agg_def.items():
            # Clé 'field' - valider que le champ existe
            if key == "field" and isinstance(value, str):
                # Retirer .keyword pour la validation
                field_name = value.replace(".keyword", "")
                if field_name in allowed_fields:
                    validated[key] = value
                else:
                    logger.warning(f"Field not allowed in aggregation: {value}")
                    return None  # Rejeter toute l'agrégation
            
            # Clé 'aggs' - validation récursive
            elif key == "aggs" and isinstance(value, dict):
                sub_aggs = {}
                for sub_name, sub_def in value.items():
                    validated_sub = self._validate_aggregation_definition(sub_def, allowed_fields)
                    if validated_sub:
                        sub_aggs[sub_name] = validated_sub
                if sub_aggs:
                    validated[key] = sub_aggs
            
            # Autres clés - passage direct avec validation basique
            else:
                # Limiter les valeurs numériques
                if isinstance(value, (int, float)):
                    if -1000000 <= value <= 1000000:  # Limites raisonnables
                        validated[key] = value
                    else:
                        logger.warning(f"Numeric value out of range: {value}")
                        return None
                
                # Strings courtes seulement
                elif isinstance(value, str) and len(value) <= 100:
                    validated[key] = value
                
                # Dictionnaires et listes - passage direct
                elif isinstance(value, (dict, list)):
                    validated[key] = value
                
                else:
                    logger.warning(f"Invalid value type for key {key}: {type(value)}")
                    return None
        
        return validated if validated else None