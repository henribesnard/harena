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

from ..models import UserQuery, QueryAnalysis, AgentResponse, AgentRole
from ..schemas.elasticsearch_schema import get_schema_description
from ..services.metadata_service import metadata_service
from ..core.llm_factory import create_llm_from_settings
from ..config.settings import settings

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

    def __init__(self, llm_model: str = None, temperature: float = 0.1):
        # Use factory to create LLM (supports both OpenAI and DeepSeek)
        self.llm = create_llm_from_settings(
            settings,
            model=llm_model,  # If None, uses settings.LLM_MODEL
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

Question: "Mes achats entre 50€ et 150€"
Réponse:
{{
  "intent": "search",
  "entities": {{"transaction_type": "debit", "amount_range": {{"min": 50, "max": 150}}}},
  "filters": {{"category_name": ["Alimentation", "Restaurant", "Transport", "achats en ligne", "Santé/pharmacie", "Loisirs"], "transaction_type": "debit", "amount_abs": {{"gte": 50, "lte": 150}}}},
  "aggregations_needed": ["statistics"],
  "time_range": null,
  "confidence": 0.90
}}

Question: "Mes achats de ce mois"
Réponse:
{{
  "intent": "search",
  "entities": {{"transaction_type": "debit", "period": "current_month"}},
  "filters": {{"category_name": ["Alimentation", "Restaurant", "Transport", "achats en ligne", "Santé/pharmacie", "Loisirs"], "transaction_type": "debit"}},
  "aggregations_needed": ["total_amount", "statistics"],
  "time_range": {{"period": "current_month", "reference_date": "{current_date}"}},
  "confidence": 0.95
}}

Question: "Mes dépenses en loisirs ce mois"
Réponse:
{{
  "intent": "search",
  "entities": {{"category": "Loisirs", "transaction_type": "debit", "period": "current_month"}},
  "filters": {{"category_name": "Loisirs", "transaction_type": "debit"}},
  "aggregations_needed": ["total_amount", "statistics"],
  "time_range": {{"period": "current_month", "reference_date": "{current_date}"}},
  "confidence": 0.95
}}

Question: "Mes abonnements"
Réponse:
{{
  "intent": "search",
  "entities": {{"category": "Abonnements", "transaction_type": "debit"}},
  "filters": {{"category_name": "Abonnements", "transaction_type": "debit"}},
  "aggregations_needed": ["statistics"],
  "time_range": null,
  "confidence": 0.95
}}

**NOUVEAUX EXEMPLES - QUESTIONS ANALYTIQUES:**

Question: "Compare mes dépenses de janvier à celles de février"
Réponse:
{{
  "intent": "compare",
  "entities": {{"comparison_type": "period_vs_period", "period1": "janvier", "period2": "février", "transaction_type": "debit"}},
  "filters": {{"transaction_type": "debit"}},
  "aggregations_needed": ["total_amount", "by_category"],
  "time_range": {{"type": "comparison", "period1": {{"start": "2025-01-01", "end": "2025-01-31"}}, "period2": {{"start": "2025-02-01", "end": "2025-02-28"}}}},
  "confidence": 0.95
}}

Question: "Quelle est la tendance de mes revenus cette année"
Réponse:
{{
  "intent": "trend_analysis",
  "entities": {{"metric": "revenus", "transaction_type": "credit", "period": "current_year"}},
  "filters": {{"transaction_type": "credit"}},
  "aggregations_needed": ["monthly_trend", "total_amount"],
  "time_range": {{"period": "current_year", "reference_date": "{current_date}"}},
  "confidence": 0.93
}}

Question: "Prévois mon budget pour le mois prochain"
Réponse:
{{
  "intent": "forecast",
  "entities": {{"forecast_target": "next_month", "metric": "total_expenses"}},
  "filters": {{"transaction_type": "debit"}},
  "aggregations_needed": ["monthly_trend"],
  "time_range": {{"period": "last_12_months", "reference_date": "{current_date}"}},
  "confidence": 0.88
}}

Question: "Quel type de dépenses je peux réduire pour augmenter mon taux d'épargne"
Réponse:
{{
  "intent": "optimization",
  "entities": {{"goal": "increase_savings", "target_metric": "discretionary_spending"}},
  "filters": {{"transaction_type": "debit"}},
  "aggregations_needed": ["by_category", "savings_rate"],
  "time_range": {{"period": "current_month", "reference_date": "{current_date}"}},
  "confidence": 0.90
}}

Question: "Compare mes charges fixes à mes charges variables"
Réponse:
{{
  "intent": "compare",
  "entities": {{"comparison_type": "category_classification", "type1": "fixed", "type2": "variable"}},
  "filters": {{"transaction_type": "debit"}},
  "aggregations_needed": ["fixed_vs_variable", "by_category"],
  "time_range": {{"period": "current_month", "reference_date": "{current_date}"}},
  "confidence": 0.95
}}

Question: "Analyse mon budget des 3 derniers mois"
Réponse:
{{
  "intent": "budget_analysis",
  "entities": {{"period": "last_3_months", "analysis_type": "comprehensive"}},
  "filters": {{}},
  "aggregations_needed": ["monthly_trend", "by_category", "savings_rate"],
  "time_range": {{"period": "last_3_months", "reference_date": "{current_date}"}},
  "confidence": 0.92
}}

Question: "Quel est mon taux d'épargne ce mois"
Réponse:
{{
  "intent": "savings_rate",
  "entities": {{"metric": "savings_rate", "period": "current_month"}},
  "filters": {{}},
  "aggregations_needed": ["savings_rate_components"],
  "time_range": {{"period": "current_month", "reference_date": "{current_date}"}},
  "confidence": 0.95
}}

IMPORTANT: Utilise TOUJOURS la date actuelle ({current_date}) comme référence pour les expressions temporelles relatives.

⚠️ RAPPEL CRITIQUE: Le terme "achats" = UNIQUEMENT [Alimentation, Restaurant, Transport, achats en ligne, Santé/pharmacie, Loisirs]
NE JAMAIS inclure: Abonnements, Téléphones/internet, Électricité/eau, Garde d'enfants, Frais scolarité, Jeux d'argent, Loterie, Paris sportif, Amendes, Autre paiement, Autres, Entretien maison

**RÈGLE SÉMANTIQUE CRITIQUE - TERME "ACHATS":**
Lorsque l'utilisateur utilise le terme "achats" ou "mes achats" SANS préciser de marchand:

⚠️ NE JAMAIS filtrer uniquement sur "transaction_type": "debit" (trop large, inclut virements, loyer, impôts, épargne, etc.)

✅ À LA PLACE, tu dois filtrer sur les catégories d'ACHAT (biens et services de consommation).

**Catégories qui sont des ACHATS (à INCLURE):**
Un achat est l'acquisition ponctuelle d'un bien ou service de consommation. UNIQUEMENT ces catégories :
- Alimentation (courses, supermarché)
- Restaurant (repas au restaurant, fast-food)
- Transport (tickets, carburant, taxi, parking)
- Carburant (essence, diesel si catégorie séparée)
- achats en ligne (e-commerce)
- Santé/pharmacie (médicaments, consultations)
- Loisirs (sorties, cinéma, spectacles, hobbies)
- Vêtements (habillement si disponible)
- Shopping (si disponible comme catégorie)

**Catégories qui ne sont PAS des achats (à EXCLURE):**
Toutes les autres catégories représentent des paiements récurrents, obligations, ou flux financiers :
- Virement sortants, Virement entrants
- Salaire
- Impôts, ImpÃ´ts (toutes variations)
- Assurances
- Frais bancaires
- Remboursements
- CAF, PAJE, Aide
- Pension alimentaire
- Chèques émis
- Retrait especes
- Abonnements (Netflix, Spotify, etc. - récurrents)
- Téléphones/internet (factures mensuelles)
- Éléctricité/eau, EÃ©lectricitÃ©/eau (factures)
- Entretien maison (travaux)
- Garde d'enfants (service régulier)
- Frais scolarité, Frais scolaritÃ© (frais fixes)
- Jeux d'argent, Loterie, Paris sportif (paris, pas achats)
- Amendes (sanctions)
- Garage (réparations lourdes)
- Autre paiement, Autres (trop vague)

**Processus pour "achats":**
1. Examine la liste des catégories disponibles dans les métadonnées
2. Exclut les catégories non-achats listées ci-dessus
3. Génère un filtre avec TOUTES les catégories restantes (= achats)
4. Ajoute transaction_type: "debit"

Exemples:
Question: "Mes achats de ce mois-ci"
→ Filtrer: category_name IN [toutes catégories SAUF non-achats] AND transaction_type: "debit"

Question: "Mes achats chez Carrefour"
→ Le marchand est précisé, filtrer directement sur merchant_name: "Carrefour"

Question: "Achats entre 50€ et 150€"
→ Filtrer: category_name IN [toutes catégories SAUF non-achats] AND transaction_type: "debit" AND amount between 50-150

**RÈGLE SÉMANTIQUE CRITIQUE - DISTINCTION TRANSACTIONS / DÉPENSES / REVENUS:**

⚠️ IMPORTANT: Ces trois termes ont des significations différentes et doivent générer des filtres différents:

1. **"TRANSACTIONS"** = Flux financiers dans les DEUX sens (débits ET crédits)
   → NE PAS ajouter de filtre "transaction_type"
   → Exemples: "mes transactions", "toutes mes transactions", "transactions de plus de X€"

2. **"DÉPENSES"** / **"DÉBITS"** = Flux financiers SORTANTS uniquement
   → TOUJOURS ajouter: "transaction_type": "debit"
   → Exemples: "mes dépenses", "combien j'ai dépensé", "mes débits"

3. **"REVENUS"** / **"CRÉDITS"** / **"ENTRÉES"** = Flux financiers ENTRANTS uniquement
   → TOUJOURS ajouter: "transaction_type": "credit"
   → Exemples: "mes revenus", "mes crédits", "combien j'ai reçu", "mes entrées d'argent"

**Exemples pour bien comprendre:**

Question: "Analyse mes transactions de plus de 2500 euros"
Réponse:
{{
  "intent": "search",
  "entities": {{"amount_min": 2500}},
  "filters": {{"amount_abs": {{"gt": 2500}}}},
  "aggregations_needed": ["statistics"],
  "time_range": null,
  "confidence": 0.95
}}
Note: PAS de filtre transaction_type car "transactions" inclut débits ET crédits

Question: "Analyse mes dépenses de plus de 2500 euros"
Réponse:
{{
  "intent": "search",
  "entities": {{"transaction_type": "debit", "amount_min": 2500}},
  "filters": {{"transaction_type": "debit", "amount_abs": {{"gt": 2500}}}},
  "aggregations_needed": ["statistics"],
  "time_range": null,
  "confidence": 0.95
}}
Note: Avec transaction_type: "debit" car "dépenses" = sorties d'argent uniquement

Question: "Analyse mes revenus de plus de 2500 euros"
Réponse:
{{
  "intent": "search",
  "entities": {{"transaction_type": "credit", "amount_min": 2500}},
  "filters": {{"transaction_type": "credit", "amount_abs": {{"gt": 2500}}}},
  "aggregations_needed": ["statistics"],
  "time_range": null,
  "confidence": 0.95
}}
Note: Avec transaction_type: "credit" car "revenus" = entrées d'argent uniquement

Question: "Montre-moi toutes mes transactions ce mois"
Réponse:
{{
  "intent": "search",
  "entities": {{"period": "current_month"}},
  "filters": {{}},
  "aggregations_needed": ["statistics"],
  "time_range": {{"period": "current_month", "reference_date": "{current_date}"}},
  "confidence": 0.95
}}
Note: PAS de filtre transaction_type car on veut TOUT voir (débits + crédits)

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

            # Log du provider LLM utilisé
            provider_info = f"{getattr(self.llm, 'primary_provider', 'unknown')} (fallback: {getattr(self.llm, 'fallback_provider', 'none')})"
            logger.info(f"🤖 [QUERY_ANALYZER] Using LLM: {provider_info}")

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

            # Log si fallback a été utilisé
            fallback_used = getattr(self.llm, 'fallback_used', False)
            if fallback_used:
                actual_provider = getattr(self.llm, 'fallback_provider', 'unknown')
                logger.warning(f"⚠️ [QUERY_ANALYZER] Fallback used: {actual_provider}")
            else:
                logger.debug(f"✅ [QUERY_ANALYZER] Primary LLM succeeded")

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
