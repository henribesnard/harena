"""
Query Analyzer Agent - Analyse la requête utilisateur
Utilise LangChain pour comprendre l'intention et extraire les entités
"""
import logging
import json
import os
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from ..models import UserQuery, QueryAnalysis, AgentResponse, AgentRole
from ..schemas.elasticsearch_schema import get_schema_description
from ..services.metadata_service import metadata_service

logger = logging.getLogger(__name__)


class QueryAnalyzerAgent:
    """
    Agent d'analyse de requête utilisateur

    Responsabilités:
    - Comprendre l'intention de l'utilisateur
    - Extraire les entités (dates, montants, catégories, marchands)
    - Identifier les agrégations nécessaires
    - Détecter les plages temporelles
    """

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.1):
        # ChatOpenAI charge automatiquement OPENAI_API_KEY depuis l'environnement
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )

        self.schema_description = get_schema_description()
        self.parser = JsonOutputParser()

        # Charger les métadonnées dynamiques (catégories, operation_types)
        try:
            self.metadata_prompt = metadata_service.get_full_metadata_prompt()
            logger.info("Metadata loaded successfully for QueryAnalyzerAgent")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata_prompt = ""

        # Prompt template pour l'analyse
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en analyse de requêtes financières.
Ton rôle est d'analyser la question de l'utilisateur et d'extraire les informations structurées nécessaires pour construire une requête Elasticsearch.

**CONTEXTE TEMPOREL IMPORTANT:**
Date actuelle: {current_date}
Utilise cette date pour interpréter les expressions temporelles relatives comme:
- "ce mois" → mois de la date actuelle
- "aujourd'hui" → date actuelle
- "cette année" → année de la date actuelle
- "la semaine dernière" → semaine précédant la date actuelle

Schéma Elasticsearch disponible:
{schema}

{metadata}

Tu dois retourner un objet JSON avec:
- intent: L'intention ("search", "aggregate", "compare", "analyze", "stats")
- entities: Dictionnaire des entités extraites (dates, montants, catégories, marchands, etc.)
- filters: Dictionnaire des filtres à appliquer
- aggregations_needed: Liste des types d'agrégations nécessaires (ex: ["by_category", "total_amount"])
- time_range: Plage temporelle si mentionnée (format: {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}})
- confidence: Score de confiance de 0 à 1

Exemples (avec date actuelle = {current_date}):

Question: "Combien j'ai dépensé en courses ce mois-ci ?"
Réponse:
{{
  "intent": "aggregate",
  "entities": {{"category": "Alimentation", "transaction_type": "debit", "period": "current_month"}},
  "filters": {{"category_name": "Alimentation", "transaction_type": "debit"}},
  "aggregations_needed": ["total_amount"],
  "time_range": {{"period": "current_month", "reference_date": "{current_date}"}},
  "confidence": 0.95
}}

Question: "Montre-moi mes transactions chez Carrefour supérieures à 50€"
Réponse:
{{
  "intent": "search",
  "entities": {{"merchant": "Carrefour", "amount_min": 50}},
  "filters": {{"merchant_name": "Carrefour", "amount": {{"gte": 50}}}},
  "aggregations_needed": ["statistics"],
  "time_range": null,
  "confidence": 0.98
}}

Question: "Quelle est ma plus grosse dépense en loisirs ?"
Réponse:
{{
  "intent": "search",
  "entities": {{"category": "Loisirs", "transaction_type": "debit"}},
  "filters": {{"category_name": "Loisirs", "transaction_type": "debit"}},
  "aggregations_needed": ["statistics"],
  "time_range": null,
  "confidence": 0.92
}}

Question: "Mes dépenses de la semaine dernière"
Réponse:
{{
  "intent": "search",
  "entities": {{"transaction_type": "debit", "period": "last_week"}},
  "filters": {{"transaction_type": "debit"}},
  "aggregations_needed": ["total_amount", "statistics"],
  "time_range": {{"period": "last_week", "reference_date": "{current_date}"}},
  "confidence": 0.93
}}

IMPORTANT: Utilise TOUJOURS la date actuelle ({current_date}) comme référence pour les expressions temporelles relatives.

**RÈGLE SÉMANTIQUE IMPORTANTE - TERME "ACHATS":**
Lorsque l'utilisateur utilise le terme "achats" ou "mes achats" SANS préciser de marchand:
- NE PAS filtrer uniquement sur "transaction_type": "debit" (trop large, inclut virements, loyer, etc.)
- FILTRER sur les catégories d'achat typiques disponibles dans les métadonnées:
  * Alimentation / Courses
  * Loisirs
  * Shopping / Commerce
  * Santé (pharmacie, médecin)
  * Transport (carburant, péage)
  * Restaurants / Bars
  * Services (coiffeur, pressing, etc.)
- EXCLURE explicitement les catégories qui ne sont PAS des achats:
  * Virements sortants
  * Prélèvements automatiques
  * Loyer / Charges
  * Impôts
  * Épargne
  * Assurances

Exemple:
Question: "Mes achats de ce mois-ci"
→ Filtrer sur category_name IN ["Alimentation", "Loisirs", "Shopping", "Restaurants", "Santé", "Transport"]
→ NE PAS utiliser uniquement transaction_type: "debit"

Question: "Mes achats chez Carrefour"
→ Filtrer sur merchant_name: "Carrefour" (le marchand est précisé, pas besoin de filtre catégorie)

Retourne UNIQUEMENT le JSON, sans texte additionnel."""),
            ("user", "Question utilisateur: {user_message}\n\nContexte conversation (optionnel): {context}")
        ])

        self.chain = self.prompt | self.llm | self.parser

        logger.info(f"QueryAnalyzerAgent initialized with model {llm_model}")

    async def analyze(self, user_query: UserQuery, current_date: str = None) -> AgentResponse:
        """
        Analyse la requête utilisateur

        Args:
            user_query: Requête utilisateur à analyser
            current_date: Date actuelle au format YYYY-MM-DD (pour interpréter expressions temporelles)

        Returns:
            AgentResponse contenant QueryAnalysis
        """
        try:
            # Utiliser la date actuelle si non fournie
            if not current_date:
                from datetime import datetime
                current_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Analyzing query: {user_query.message[:100]} (current_date={current_date})")

            # Préparer le contexte
            context_str = ""
            if user_query.context:
                context_str = "\n".join([
                    f"{turn['role']}: {turn['content']}"
                    for turn in user_query.context[-3:]  # 3 derniers tours
                ])

            # Invoquer le LLM avec la date actuelle et les métadonnées
            result = await self.chain.ainvoke({
                "schema": self.schema_description,
                "metadata": self.metadata_prompt,
                "user_message": user_query.message,
                "context": context_str or "Aucun contexte",
                "current_date": current_date
            })

            # Parser le résultat
            analysis = QueryAnalysis(
                intent=result.get("intent", "search"),
                entities=result.get("entities", {}),
                filters=result.get("filters", {}),
                aggregations_needed=result.get("aggregations_needed", []),
                time_range=result.get("time_range"),
                confidence=result.get("confidence", 0.0)
            )

            logger.info(f"Query analysis completed: intent={analysis.intent}, confidence={analysis.confidence:.2f}")

            return AgentResponse(
                success=True,
                data=analysis,
                agent_role=AgentRole.QUERY_ANALYZER,
                metadata={
                    "raw_result": result,
                    "user_id": user_query.user_id
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return AgentResponse(
                success=False,
                data=None,
                agent_role=AgentRole.QUERY_ANALYZER,
                error=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        return {
            "agent": "query_analyzer",
            "model": self.llm.model_name,
            "temperature": self.llm.temperature
        }
