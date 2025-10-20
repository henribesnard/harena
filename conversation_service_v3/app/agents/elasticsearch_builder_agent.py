"""
Elasticsearch Builder Agent - Construit et corrige les queries Elasticsearch
Utilise LangChain avec capacité d'auto-correction
"""
import logging
import json
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
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature  # Température basse pour plus de cohérence
        )

        self.schema_description = get_schema_description()
        self.parser = JsonOutputParser()

        # Prompt pour la construction initiale
        self.build_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert Elasticsearch spécialisé dans la construction de queries pour des données de transactions financières.

Schéma Elasticsearch:
{schema}

Règles IMPORTANTES:
1. TOUJOURS inclure le filtre user_id dans la query
2. Pour les agrégations, utiliser EXACTEMENT la syntaxe Elasticsearch
3. Les dates doivent être au format YYYY-MM-DD
4. Les montants sont en float
5. Pour "ce mois-ci", utiliser la date actuelle: {current_date}
6. TOUJOURS limiter les résultats (size: 50 par défaut)
7. Pour les agrégations, inclure les sous-agrégations nécessaires

Structure de réponse attendue:
{{
  "query": {{
    "bool": {{
      "must": [...],
      "filter": [
        {{"term": {{"user_id": USER_ID}}}}
      ]
    }}
  }},
  "aggs": {{...}},  // Optionnel
  "size": 50,
  "sort": [...]  // Optionnel
}}

Exemples d'agrégations courantes:
- Total par catégorie:
  "aggs": {{
    "by_category": {{
      "terms": {{"field": "category_name", "size": 20}},
      "aggs": {{
        "total_amount": {{"sum": {{"field": "amount"}}}}
      }}
    }}
  }}

- Statistiques sur les montants:
  "aggs": {{
    "amount_stats": {{"stats": {{"field": "amount"}}}}
  }}

Retourne UNIQUEMENT le JSON de la query, sans texte additionnel."""),
            ("user", """Analyse de la requête utilisateur:
Intent: {intent}
Filtres: {filters}
Agrégations demandées: {aggregations}
Plage temporelle: {time_range}

User ID: {user_id}
Date actuelle: {current_date}

Construis la query Elasticsearch correspondante.""")
        ])

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

        self.build_chain = self.build_prompt | self.llm | self.parser
        self.correction_chain = self.correction_prompt | self.llm | self.parser

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

        Args:
            query_analysis: Analyse de la requête utilisateur
            user_id: ID de l'utilisateur
            current_date: Date actuelle pour les calculs de période

        Returns:
            AgentResponse contenant ElasticsearchQuery
        """
        try:
            logger.info(f"Building Elasticsearch query for intent: {query_analysis.intent}")

            # Invoquer le LLM pour construire la query
            result = await self.build_chain.ainvoke({
                "schema": self.schema_description,
                "intent": query_analysis.intent,
                "filters": json.dumps(query_analysis.filters, ensure_ascii=False),
                "aggregations": json.dumps(query_analysis.aggregations_needed, ensure_ascii=False),
                "time_range": json.dumps(query_analysis.time_range, ensure_ascii=False) if query_analysis.time_range else "Non spécifié",
                "user_id": user_id,
                "current_date": current_date
            })

            # Construire l'objet ElasticsearchQuery
            es_query = ElasticsearchQuery(
                query=result.get("query", {}),
                aggregations=result.get("aggs"),
                size=result.get("size", 50),
                sort=result.get("sort")
            )

            # Validation basique
            validation = self._validate_query(es_query, user_id)

            if not validation.is_valid:
                logger.warning(f"Query validation failed: {validation.errors}")
                # Ne pas retourner d'erreur, juste logger les warnings

            self.stats["queries_built"] += 1

            logger.info(f"Query built successfully. Size: {es_query.size}, Has aggs: {es_query.aggregations is not None}")

            return AgentResponse(
                success=True,
                data=es_query,
                agent_role=AgentRole.ELASTICSEARCH_BUILDER,
                metadata={
                    "validation": validation,
                    "intent": query_analysis.intent
                }
            )

        except Exception as e:
            logger.error(f"Error building query: {str(e)}")
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
        Valide une query Elasticsearch

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

        # Vérifier la structure bool
        if "bool" not in query.query:
            warnings.append("Query should use bool structure for better performance")

        # Vérifier le filtre user_id
        user_id_found = False
        if "bool" in query.query:
            for clause_type in ["must", "filter"]:
                if clause_type in query.query["bool"]:
                    for clause in query.query["bool"][clause_type]:
                        if "term" in clause and "user_id" in clause["term"]:
                            user_id_found = True
                            # Vérifier la valeur
                            if clause["term"]["user_id"] != user_id:
                                errors.append(f"Wrong user_id in query: expected {user_id}")

        if not user_id_found:
            errors.append("Missing user_id filter - SECURITY ISSUE")

        # Vérifier les champs dans les filtres
        valid_fields = set(ELASTICSEARCH_SCHEMA["fields"].keys())
        self._check_fields_in_dict(query.query, valid_fields, errors, warnings)

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
