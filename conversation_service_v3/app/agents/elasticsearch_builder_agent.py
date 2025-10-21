"""
Elasticsearch Builder Agent - Construit et corrige les queries Elasticsearch
Utilise LangChain avec capacité d'auto-correction
"""
import logging
import json
import os
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from ..models import (
    QueryAnalysis, ElasticsearchQuery, QueryValidationResult,
    AgentResponse, AgentRole
)
from ..schemas.elasticsearch_schema import (
    ELASTICSEARCH_SCHEMA, get_schema_description, get_query_template
)
from .function_definitions import (
    SEARCH_TRANSACTIONS_FUNCTION,
    AGGREGATION_TEMPLATES,
    get_all_templates_description
)

logger = logging.getLogger(__name__)


class ElasticsearchBuilderAgent:
    """
    Agent de construction de query Elasticsearch avec auto-correction

    Responsabilités:
    - Traduire l'analyse en query Elasticsearch valide
    - Ajouter les agrégations appropriées
    - Valider la query générée
    - Se corriger automatiquement si la query échoue
    """

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.0):
        # ChatOpenAI charge automatiquement OPENAI_API_KEY depuis l'environnement
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature  # Température basse pour plus de cohérence
        )

        self.schema_description = get_schema_description()

        # Utiliser la définition complète de la fonction depuis function_definitions
        # Ajuster le nom pour correspondre à l'usage interne
        self.search_query_function = SEARCH_TRANSACTIONS_FUNCTION.copy()
        self.search_query_function["name"] = "generate_search_query"

        # Obtenir la description des templates disponibles
        templates_description = get_all_templates_description()

        # Prompt enrichi pour function calling avec templates et exemples
        self.build_prompt_text = """Génère une requête de recherche pour: {intent}

Contexte:
- User ID: {user_id}
- Date actuelle: {current_date}
- Filtres détectés: {filters}
- Agrégations demandées: {aggregations}
- Période: {time_range}

TEMPLATES D'AGRÉGATIONS DISPONIBLES:

""" + templates_description + """

RÈGLES CRITIQUES:
1. "plus de X euros" = amount_abs: {{"gt": X}} → EXCLUT X (strictement supérieur)
2. "au moins X euros" = amount_abs: {{"gte": X}} → INCLUT X (supérieur ou égal)
3. "dépenses" = transaction_type: "debit"
4. "revenus" = transaction_type: "credit"
5. TOUJOURS inclure sort: [{{"date": {{"order": "desc"}}}}] (OBLIGATOIRE)
6. Agrégations sur montants: utiliser "amount_abs", JAMAIS "amount"
7. Pour les filtres merchant_name et category_name, utiliser {{"match": "valeur"}} pour recherche floue
8. TOUJOURS ajouter des agrégations de base (total, count) pour donner des statistiques utiles
9. Si l'intent indique "statistics" ou "analyze", utiliser les templates d'agrégations disponibles
10. page_size doit être >= 1 (JAMAIS 0). Pour les queries d'agrégations, utiliser page_size: 10

EXEMPLES D'UTILISATION:

Exemple 1 - "Mes dépenses de plus de 100 euros":
{{
    "user_id": {user_id},
    "filters": {{
        "transaction_type": "debit",
        "amount_abs": {{"gt": 100}}
    }},
    "sort": [{{"date": {{"order": "desc"}}}}],
    "page_size": 50,
    "aggregations": {{
        "transaction_count": {{"value_count": {{"field": "transaction_id"}}}},
        "total_amount": {{"sum": {{"field": "amount_abs"}}}}
    }}
}}

Exemple 2 - "Combien j'ai dépensé en restaurants ce mois?":
{{
    "user_id": {user_id},
    "filters": {{
        "category_name": {{"match": "restaurant"}},
        "transaction_type": "debit",
        "date": {{"gte": "2025-10-01T00:00:00Z", "lte": "2025-10-31T23:59:59Z"}}
    }},
    "sort": [{{"date": {{"order": "desc"}}}}],
    "page_size": 10,
    "aggregations": {{
        "total_spent": {{"sum": {{"field": "amount_abs"}}}},
        "transaction_count": {{"value_count": {{"field": "transaction_id"}}}},
        "avg_transaction": {{"avg": {{"field": "amount_abs"}}}}
    }}
}}

Exemple 3 - "Évolution mensuelle de mes dépenses sur 6 mois":
{{
    "user_id": {user_id},
    "filters": {{
        "transaction_type": "debit",
        "date": {{"gte": "2025-04-01T00:00:00Z", "lte": "2025-10-31T23:59:59Z"}}
    }},
    "sort": [{{"date": {{"order": "desc"}}}}],
    "page_size": 10,
    "aggregations": {{
        "monthly_breakdown": {{
            "date_histogram": {{
                "field": "date",
                "calendar_interval": "month",
                "format": "yyyy-MM"
            }},
            "aggs": {{
                "total_spent": {{"sum": {{"field": "amount_abs"}}}},
                "transaction_count": {{"value_count": {{"field": "transaction_id"}}}}
            }}
        }}
    }}
}}

Exemple 4 - "Répartition de mes dépenses par catégorie":
{{
    "user_id": {user_id},
    "filters": {{
        "transaction_type": "debit"
    }},
    "sort": [{{"date": {{"order": "desc"}}}}],
    "page_size": 10,
    "aggregations": {{
        "by_category": {{
            "terms": {{
                "field": "category_name.keyword",
                "size": 20,
                "order": {{"total_amount": "desc"}}
            }},
            "aggs": {{
                "total_amount": {{"sum": {{"field": "amount_abs"}}}},
                "transaction_count": {{"value_count": {{"field": "transaction_id"}}}},
                "avg_transaction": {{"avg": {{"field": "amount_abs"}}}}
            }}
        }}
    }}
}}"""

        # Prompt pour l'auto-correction
        self.correction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert Elasticsearch qui corrige des queries défaillantes.

Schéma Elasticsearch:
{schema}

Ta mission: Analyser l'erreur et proposer une query corrigée.

Erreurs courantes à corriger:
- Champs inexistants → utiliser les bons noms de champs
- Syntaxe invalide → corriger la structure JSON
- Agrégations mal formées → utiliser la bonne syntaxe
- Filtres manquants → ajouter user_id

Retourne UNIQUEMENT le JSON de la query corrigée."""),
            ("user", """Query qui a échoué:
{failed_query}

Erreur rencontrée:
{error_message}

Contexte original:
Intent: {intent}
Filtres demandés: {filters}
User ID: {user_id}

Corrige cette query pour qu'elle fonctionne.""")
        ])

        # Pas de chain pour function calling, on appelle directement le LLM
        self.correction_chain = self.correction_prompt | self.llm | JsonOutputParser()

        self.stats = {
            "queries_built": 0,
            "corrections_attempted": 0,
            "successful_corrections": 0
        }

        logger.info(f"ElasticsearchBuilderAgent initialized with model {llm_model}")

    async def build_query(
        self,
        query_analysis: QueryAnalysis,
        user_id: int,
        current_date: str = "2025-01-15"
    ) -> AgentResponse:
        """
        Construit une query Elasticsearch à partir de l'analyse
        UTILISE FUNCTION CALLING pour garantir le bon format

        Args:
            query_analysis: Analyse de la requête utilisateur
            user_id: ID de l'utilisateur
            current_date: Date actuelle pour les calculs de période

        Returns:
            AgentResponse contenant ElasticsearchQuery
        """
        try:
            logger.info(f"Building Elasticsearch query for intent: {query_analysis.intent}")

            # Préparer le message pour function calling
            prompt_message = self.build_prompt_text.format(
                intent=query_analysis.intent,
                filters=json.dumps(query_analysis.filters, ensure_ascii=False),
                aggregations=json.dumps(query_analysis.aggregations_needed, ensure_ascii=False),
                time_range=json.dumps(query_analysis.time_range, ensure_ascii=False) if query_analysis.time_range else "Non spécifié",
                user_id=user_id,
                current_date=current_date
            )

            # Appeler le LLM avec function calling (utiliser predict_messages pour function calling)
            from langchain.schema import HumanMessage
            response = await self.llm.apredict_messages(
                [HumanMessage(content=prompt_message)],
                functions=[self.search_query_function],
                function_call={"name": "generate_search_query"}
            )

            # Extraire le résultat de la function call
            # apredict_messages retourne un AIMessage directement
            function_call = response.additional_kwargs.get("function_call")
            if not function_call:
                raise ValueError("No function call in LLM response")

            result = json.loads(function_call["arguments"])

            # S'assurer que user_id est présent
            if "user_id" not in result:
                result["user_id"] = user_id

            # Nettoyer les agrégations (enlever les champs "name" invalides ajoutés par le LLM)
            if "aggregations" in result and result["aggregations"]:
                result["aggregations"] = self._clean_aggregations(result["aggregations"])

            # Construire l'objet ElasticsearchQuery
            # Le résultat contient: {user_id, filters, sort, page_size, aggregations}
            es_query = ElasticsearchQuery(
                query=result,  # Format search_service complet
                aggregations=result.get("aggregations"),
                size=result.get("page_size", 50),
                sort=result.get("sort")
            )

            # Validation basique
            validation = self._validate_query(es_query, user_id)

            if not validation.is_valid:
                logger.warning(f"Query validation failed: {validation.errors}")

            self.stats["queries_built"] += 1

            logger.info(f"Query built successfully with function calling. Size: {es_query.size}, Has aggs: {es_query.aggregations is not None}")

            return AgentResponse(
                success=True,
                data=es_query,
                agent_role=AgentRole.ELASTICSEARCH_BUILDER,
                metadata={
                    "validation": validation,
                    "intent": query_analysis.intent,
                    "used_function_calling": True
                }
            )

        except Exception as e:
            logger.error(f"Error building query with function calling: {str(e)}")
            return AgentResponse(
                success=False,
                data=None,
                agent_role=AgentRole.ELASTICSEARCH_BUILDER,
                error=str(e)
            )

    async def correct_query(
        self,
        failed_query: ElasticsearchQuery,
        error_message: str,
        original_analysis: QueryAnalysis,
        user_id: int
    ) -> AgentResponse:
        """
        Tente de corriger une query qui a échoué

        Args:
            failed_query: Query qui a échoué
            error_message: Message d'erreur d'Elasticsearch
            original_analysis: Analyse originale de la requête
            user_id: ID utilisateur

        Returns:
            AgentResponse avec query corrigée
        """
        try:
            logger.info(f"Attempting to correct failed query. Error: {error_message[:100]}")

            self.stats["corrections_attempted"] += 1

            # Construire la query complète pour le contexte
            full_query = {
                "query": failed_query.query,
                "size": failed_query.size
            }
            if failed_query.aggregations:
                full_query["aggs"] = failed_query.aggregations
            if failed_query.sort:
                full_query["sort"] = failed_query.sort

            # Invoquer le LLM pour corriger
            result = await self.correction_chain.ainvoke({
                "schema": self.schema_description,
                "failed_query": json.dumps(full_query, indent=2, ensure_ascii=False),
                "error_message": error_message,
                "intent": original_analysis.intent,
                "filters": json.dumps(original_analysis.filters, ensure_ascii=False),
                "user_id": user_id
            })

            # Construire la query corrigée
            corrected_query = ElasticsearchQuery(
                query=result.get("query", {}),
                aggregations=result.get("aggs"),
                size=result.get("size", 50),
                sort=result.get("sort")
            )

            # Validation
            validation = self._validate_query(corrected_query, user_id)

            if validation.is_valid:
                self.stats["successful_corrections"] += 1
                logger.info("Query corrected successfully")
            else:
                logger.warning(f"Corrected query still has issues: {validation.errors}")

            return AgentResponse(
                success=True,
                data=corrected_query,
                agent_role=AgentRole.ELASTICSEARCH_BUILDER,
                metadata={
                    "validation": validation,
                    "correction_attempted": True,
                    "original_error": error_message
                }
            )

        except Exception as e:
            logger.error(f"Error correcting query: {str(e)}")
            return AgentResponse(
                success=False,
                data=None,
                agent_role=AgentRole.ELASTICSEARCH_BUILDER,
                error=f"Correction failed: {str(e)}"
            )

    def _validate_query(self, query: ElasticsearchQuery, user_id: int) -> QueryValidationResult:
        """
        Valide une query au format search_service

        Args:
            query: Query à valider
            user_id: ID utilisateur attendu

        Returns:
            QueryValidationResult
        """
        errors = []
        warnings = []

        # Vérifier que la query existe
        if not query.query:
            errors.append("Query is empty")
            return QueryValidationResult(is_valid=False, errors=errors)

        # Vérifier le format search_service (doit avoir user_id, filters, etc.)
        if "bool" in query.query or "must" in query.query or "filter" in query.query:
            errors.append("Query is in Elasticsearch DSL format instead of search_service format")
            errors.append("Expected format: {user_id, filters, sort, page_size, aggregations}")

        # Vérifier le user_id au niveau racine
        if "user_id" not in query.query:
            errors.append("Missing user_id at root level - SECURITY ISSUE")
        elif query.query.get("user_id") != user_id:
            errors.append(f"Wrong user_id in query: expected {user_id}, got {query.query.get('user_id')}")

        # Vérifier que filters existe (peut être vide)
        if "filters" not in query.query:
            warnings.append("Missing 'filters' field in search_service format")

        # Vérifier les champs dans les filtres
        valid_fields = set(ELASTICSEARCH_SCHEMA["fields"].keys())
        if "filters" in query.query and isinstance(query.query["filters"], dict):
            for field_name in query.query["filters"].keys():
                if field_name not in valid_fields:
                    errors.append(f"Unknown field '{field_name}' in filters")

        # Vérifier les agrégations si présentes
        if query.aggregations:
            self._validate_aggregations(query.aggregations, valid_fields, errors, warnings)

        is_valid = len(errors) == 0

        return QueryValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )

    def _check_fields_in_dict(
        self,
        obj: Any,
        valid_fields: set,
        errors: List[str],
        warnings: List[str],
        path: str = ""
    ):
        """Vérifie récursivement les champs utilisés"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Vérifier si c'est un nom de champ
                if key in ["term", "terms", "range", "match", "wildcard"]:
                    if isinstance(value, dict):
                        for field_name in value.keys():
                            if field_name not in valid_fields:
                                errors.append(f"Unknown field '{field_name}' at {current_path}")

                # Récursion
                self._check_fields_in_dict(value, valid_fields, errors, warnings, current_path)

        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                self._check_fields_in_dict(item, valid_fields, errors, warnings, f"{path}[{idx}]")

    def _clean_aggregations(self, aggs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nettoie les agrégations en enlevant les champs invalides ajoutés par le LLM

        Problèmes corrigés:
        1. Champs "name" invalides qui causent "Expected [START_OBJECT] under [name]"
        2. Multiples types d'agrégation dans le même objet (ex: sum + value_count)

        Args:
            aggs: Agrégations à nettoyer

        Returns:
            Agrégations nettoyées
        """
        if not isinstance(aggs, dict):
            return aggs

        AGG_TYPES = ["sum", "avg", "min", "max", "stats", "value_count", "terms", "date_histogram", "cardinality"]

        cleaned = {}
        for agg_name, agg_def in aggs.items():
            if isinstance(agg_def, dict):
                # Trouver tous les types d'agrégation dans cet objet
                agg_types_found = [key for key in agg_def.keys() if key in AGG_TYPES]

                if len(agg_types_found) > 1:
                    # PROBLÈME: Multiple types dans une seule agrégation
                    # Solution: Garder le premier type, créer des agrégations séparées pour les autres
                    logger.warning(f"Multiple aggregation types in '{agg_name}': {agg_types_found}. Splitting...")

                    # Garder le premier type dans l'agrégation principale
                    main_type = agg_types_found[0]
                    cleaned[agg_name] = {main_type: agg_def[main_type]}

                    # Créer de nouvelles agrégations pour les autres types
                    for idx, other_type in enumerate(agg_types_found[1:], 1):
                        new_agg_name = f"{agg_name}_{other_type}"
                        cleaned[new_agg_name] = {other_type: agg_def[other_type]}
                        logger.info(f"Created separate aggregation: {new_agg_name}")

                    # Gérer les sous-agrégations (aggs) s'il y en a
                    if "aggs" in agg_def:
                        cleaned[agg_name]["aggs"] = self._clean_aggregations(agg_def["aggs"])

                else:
                    # Une seule agrégation : copier en nettoyant
                    cleaned_agg = {}
                    for key, value in agg_def.items():
                        if key == "name":
                            # Ignorer le champ "name" invalide
                            logger.debug(f"Removed invalid 'name' field from aggregation '{agg_name}'")
                            continue
                        elif key == "aggs" and isinstance(value, dict):
                            # Nettoyer récursivement les sous-agrégations
                            cleaned_agg[key] = self._clean_aggregations(value)
                        else:
                            cleaned_agg[key] = value
                    cleaned[agg_name] = cleaned_agg
            else:
                cleaned[agg_name] = agg_def

        return cleaned

    def _validate_aggregations(
        self,
        aggs: Dict[str, Any],
        valid_fields: set,
        errors: List[str],
        warnings: List[str]
    ):
        """Valide les agrégations"""
        for agg_name, agg_def in aggs.items():
            # Vérifier les champs dans les agrégations
            if isinstance(agg_def, dict):
                for agg_type, agg_config in agg_def.items():
                    if agg_type in ["sum", "avg", "min", "max", "stats", "value_count"]:
                        if isinstance(agg_config, dict) and "field" in agg_config:
                            field = agg_config["field"]
                            if field not in valid_fields:
                                errors.append(f"Unknown field '{field}' in aggregation '{agg_name}'")

                    elif agg_type in ["terms", "date_histogram"]:
                        if isinstance(agg_config, dict) and "field" in agg_config:
                            field = agg_config["field"]
                            if field not in valid_fields:
                                errors.append(f"Unknown field '{field}' in aggregation '{agg_name}'")

                    # Agrégations imbriquées
                    if "aggs" in agg_def:
                        self._validate_aggregations(
                            agg_def["aggs"],
                            valid_fields,
                            errors,
                            warnings
                        )

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        success_rate = 0.0
        if self.stats["corrections_attempted"] > 0:
            success_rate = self.stats["successful_corrections"] / self.stats["corrections_attempted"]

        return {
            "agent": "elasticsearch_builder",
            "model": self.llm.model_name,
            **self.stats,
            "correction_success_rate": success_rate
        }
