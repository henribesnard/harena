"""
Analytics Agent - Orchestre les analyses financières complexes

Responsabilités:
- Analyser les demandes analytiques (comparaisons, tendances, prévisions)
- Planifier les requêtes Elasticsearch nécessaires
- Coordonner avec AnalyticsService pour calculs post-agrégation
- Générer des insights actionnables

Author: Claude Code
Date: 2025-10-23
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from ..models import AgentResponse, AgentRole
from ..models.intent import IntentCategory

logger = logging.getLogger(__name__)


class AnalyticsAgent:
    """
    Agent d'analyse financière avancée

    Coordonne les analyses complexes nécessitant:
    - Plusieurs requêtes Elasticsearch
    - Calculs post-agrégation
    - Comparaisons temporelles
    - Détection de tendances
    - Prévisions
    - Recommandations
    """

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.parser = JsonOutputParser()

        # Prompt pour analyser les demandes analytiques
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en analyse financière.

Ta mission est de planifier l'exécution d'analyses financières complexes.

Date actuelle: {current_date}

Types d'analyses supportées:

1. COMPARATIVE_ANALYSIS (Comparaisons)
   - Comparer deux périodes (ex: janvier vs février)
   - Comparer catégories (ex: charges fixes vs variables)

2. TREND_ANALYSIS (Tendances)
   - Analyser l'évolution sur plusieurs mois
   - Détecter patterns et variations

3. PREDICTIVE_ANALYSIS (Prévisions)
   - Prévoir le budget du mois suivant
   - Estimer dépenses futures

4. OPTIMIZATION_RECOMMENDATION (Recommandations)
   - Identifier catégories à réduire
   - Suggérer optimisations budgétaires

5. BUDGET_ANALYSIS (Analyse budgétaire)
   - Vue d'ensemble multi-périodes
   - Analyse globale revenus/dépenses

**IMPORTANT - Distinction transaction_type:**

1. **"Analyse mes DÉPENSES"** → filters: {{"transaction_type": "debit"}}
   - Exemples: "analyse mes dépenses", "combien j'ai dépensé"

2. **"Analyse mes REVENUS"** → filters: {{"transaction_type": "credit"}}
   - Exemples: "analyse mes revenus", "mes entrées d'argent"

3. **"Analyse mon BUDGET" ou "mes TRANSACTIONS"** → SANS filtre transaction_type
   - Ajouter aggregation: "by_transaction_type" pour distinguer crédit/débit
   - Exemples: "analyse mon budget", "vue d'ensemble", "toutes mes transactions"

Pour chaque demande, tu dois retourner un plan d'exécution JSON:

{{
  "analysis_type": "COMPARATIVE_ANALYSIS",
  "queries": [
    {{
      "period": "2025-01",
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["total_amount", "by_category"]
    }},
    {{
      "period": "2025-02",
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["total_amount", "by_category"]
    }}
  ],
  "analytics_operations": ["compare_periods", "detect_variations"],
  "output_format": "comparison_report"
}}

Exemples:

Question: "compare mes dépenses de janvier à celles de février"
{{
  "analysis_type": "COMPARATIVE_ANALYSIS",
  "comparison_config": {{
    "periods": [
      {{"start": "2025-01-01", "end": "2025-01-31", "label": "janvier"}},
      {{"start": "2025-02-01", "end": "2025-02-28", "label": "février"}}
    ]
  }},
  "queries": [
    {{
      "time_range": {{"start": "2025-01-01", "end": "2025-01-31"}},
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["total_amount", "by_category", "statistics"]
    }},
    {{
      "time_range": {{"start": "2025-02-01", "end": "2025-02-28"}},
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["total_amount", "by_category", "statistics"]
    }}
  ],
  "analytics_operations": ["compare_periods"],
  "metrics": ["total_amount", "by_category"]
}}

Question: "quelle est la tendance de mes revenus cette année"
{{
  "analysis_type": "TREND_ANALYSIS",
  "time_range": {{"start": "2025-01-01", "end": "2025-10-23"}},
  "queries": [
    {{
      "time_range": {{"start": "2025-01-01", "end": "2025-10-23"}},
      "filters": {{"transaction_type": "credit"}},
      "aggregations": ["monthly_trend", "total_amount"]
    }}
  ],
  "analytics_operations": ["detect_trend"],
  "metrics": ["total_amount"]
}}

Question: "quel est mon taux d'épargne"
{{
  "analysis_type": "BUDGET_ANALYSIS",
  "queries": [
    {{
      "filters": {{"transaction_type": "credit"}},
      "aggregations": ["total_amount"]
    }},
    {{
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["total_amount"]
    }}
  ],
  "analytics_operations": ["calculate_savings_rate"],
  "metrics": ["total_amount"]
}}

Question: "compare mes charges fixes à mes charges variables"
{{
  "analysis_type": "COMPARATIVE_ANALYSIS",
  "queries": [
    {{
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["fixed_vs_variable", "by_category"]
    }}
  ],
  "analytics_operations": ["classify_fixed_vs_variable"],
  "metrics": ["by_category"]
}}

Question: "prévois mon budget pour le mois prochain"
{{
  "analysis_type": "PREDICTIVE_ANALYSIS",
  "queries": [
    {{
      "time_range": {{"start": "2024-10-01", "end": "2025-10-23"}},
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["monthly_trend"]
    }}
  ],
  "analytics_operations": ["forecast_next_period"],
  "forecast_config": {{"method": "moving_average", "periods": 3}}
}}

Question: "quel type de dépenses je peux réduire pour augmenter mon taux d'épargne"
{{
  "analysis_type": "OPTIMIZATION_RECOMMENDATION",
  "queries": [
    {{
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["by_category"]
    }},
    {{
      "filters": {{"transaction_type": "credit"}},
      "aggregations": ["total_amount"]
    }}
  ],
  "analytics_operations": ["recommend_savings_opportunities", "calculate_savings_rate"],
  "target_improvement": 10
}}

Question: "analyse mes dépenses des 3 derniers mois"
{{
  "analysis_type": "BUDGET_ANALYSIS",
  "queries": [
    {{
      "time_range": {{"start": "2025-07-01", "end": "2025-07-31"}},
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["by_category", "total_amount"],
      "period_label": "2025-07"
    }},
    {{
      "time_range": {{"start": "2025-08-01", "end": "2025-08-31"}},
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["by_category", "total_amount"],
      "period_label": "2025-08"
    }},
    {{
      "time_range": {{"start": "2025-09-01", "end": "2025-09-30"}},
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["by_category", "total_amount"],
      "period_label": "2025-09"
    }}
  ],
  "analytics_operations": ["analyze_multi_period_budget"],
  "metrics": ["total_amount", "by_category"]
}}

Question: "analyse mon budget des 3 derniers mois"
{{
  "analysis_type": "BUDGET_ANALYSIS",
  "queries": [
    {{
      "time_range": {{"start": "2025-07-01", "end": "2025-07-31"}},
      "filters": {{}},
      "aggregations": ["by_category", "by_transaction_type", "total_amount"],
      "period_label": "2025-07"
    }},
    {{
      "time_range": {{"start": "2025-08-01", "end": "2025-08-31"}},
      "filters": {{}},
      "aggregations": ["by_category", "by_transaction_type", "total_amount"],
      "period_label": "2025-08"
    }},
    {{
      "time_range": {{"start": "2025-09-01", "end": "2025-09-30"}},
      "filters": {{}},
      "aggregations": ["by_category", "by_transaction_type", "total_amount"],
      "period_label": "2025-09"
    }}
  ],
  "analytics_operations": ["analyze_multi_period_budget"],
  "metrics": ["total_amount", "by_category", "by_transaction_type"]
}}

Question: "mon taux d'épargne cette année" ou "quel est mon taux d'épargne cette année"
{{
  "analysis_type": "BUDGET_ANALYSIS",
  "queries": [
    {{
      "time_range": {{"start": "2025-01-01", "end": "2025-12-31"}},
      "filters": {{"transaction_type": "credit"}},
      "aggregations": ["total_amount"]
    }},
    {{
      "time_range": {{"start": "2025-01-01", "end": "2025-12-31"}},
      "filters": {{"transaction_type": "debit"}},
      "aggregations": ["total_amount"]
    }}
  ],
  "analytics_operations": ["calculate_savings_rate"],
  "metrics": ["total_amount"]
}}
Note: Pour "cette année", extraire l'année actuelle de {current_date} et filtrer du 01-01 au 31-12.

Retourne UNIQUEMENT le JSON, sans texte additionnel."""),
            ("user", "Question: {user_message}\nDate actuelle: {current_date}")
        ])

        self.chain = self.analysis_prompt | self.llm | self.parser

        self.stats = {
            "analyses_planned": 0,
            "comparisons": 0,
            "trends": 0,
            "forecasts": 0,
            "recommendations": 0
        }

        logger.info(f"AnalyticsAgent initialized with model {llm_model}")

    async def plan_analysis(
        self,
        user_message: str,
        intent: IntentCategory,
        current_date: str = None
    ) -> AgentResponse:
        """
        Planifie l'exécution d'une analyse complexe

        Args:
            user_message: Message utilisateur
            intent: Intention détectée (COMPARATIVE_ANALYSIS, etc.)
            current_date: Date actuelle pour calculs temporels

        Returns:
            AgentResponse avec plan d'exécution
        """
        try:
            if not current_date:
                current_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Planning {intent.value} analysis for: {user_message[:100]}")

            # Invoquer le LLM pour générer le plan
            result = await self.chain.ainvoke({
                "user_message": user_message,
                "current_date": current_date
            })

            # Valider et enrichir le plan
            analysis_plan = self._validate_and_enrich_plan(result, intent, current_date)

            # Mise à jour stats
            self.stats["analyses_planned"] += 1
            analysis_type = analysis_plan.get("analysis_type", "")

            if "COMPARATIVE" in analysis_type:
                self.stats["comparisons"] += 1
            elif "TREND" in analysis_type:
                self.stats["trends"] += 1
            elif "PREDICTIVE" in analysis_type:
                self.stats["forecasts"] += 1
            elif "OPTIMIZATION" in analysis_type:
                self.stats["recommendations"] += 1

            logger.info(f"Analysis plan generated: {len(analysis_plan.get('queries', []))} queries planned")

            return AgentResponse(
                success=True,
                data=analysis_plan,
                agent_role=AgentRole.QUERY_ANALYZER,
                metadata={
                    "analysis_type": analysis_plan.get("analysis_type"),
                    "num_queries": len(analysis_plan.get("queries", [])),
                    "operations": analysis_plan.get("analytics_operations", [])
                }
            )

        except Exception as e:
            logger.error(f"Error planning analysis: {str(e)}", exc_info=True)
            return AgentResponse(
                success=False,
                data=None,
                agent_role=AgentRole.QUERY_ANALYZER,
                error=str(e)
            )

    def _validate_and_enrich_plan(
        self,
        plan: Dict[str, Any],
        intent: IntentCategory,
        current_date: str
    ) -> Dict[str, Any]:
        """
        Valide et enrichit le plan d'analyse

        Args:
            plan: Plan brut du LLM
            intent: Intention détectée
            current_date: Date actuelle

        Returns:
            Plan validé et enrichi
        """
        # S'assurer que le plan contient les champs requis
        if "queries" not in plan:
            plan["queries"] = []

        if "analytics_operations" not in plan:
            plan["analytics_operations"] = self._infer_operations(intent)

        if "analysis_type" not in plan:
            plan["analysis_type"] = intent.value

        # Enrichir les queries avec user_id placeholder (sera remplacé par l'orchestrator)
        for query in plan["queries"]:
            if "user_id" not in query:
                query["user_id"] = None  # Sera rempli par l'orchestrator

            # S'assurer que les filtres existent
            if "filters" not in query:
                query["filters"] = {}

        return plan

    def _infer_operations(self, intent: IntentCategory) -> List[str]:
        """Infère les opérations analytiques selon l'intention"""
        mapping = {
            IntentCategory.COMPARATIVE_ANALYSIS: ["compare_periods"],
            IntentCategory.TREND_ANALYSIS: ["detect_trend"],
            IntentCategory.PREDICTIVE_ANALYSIS: ["forecast_next_period"],
            IntentCategory.OPTIMIZATION_RECOMMENDATION: ["recommend_savings_opportunities"],
            IntentCategory.BUDGET_ANALYSIS: ["analyze_multi_period_budget"]
        }
        return mapping.get(intent, [])

    def interpret_comparison_periods(
        self,
        user_message: str,
        current_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Interprète les périodes mentionnées dans une comparaison

        Ex: "janvier vs février" → {"period1": "2025-01", "period2": "2025-02"}

        Args:
            user_message: Message utilisateur
            current_date: Date actuelle

        Returns:
            Dict avec périodes détectées
        """
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        message_lower = user_message.lower()

        # Détecter mois mentionnés
        months = {
            "janvier": 1, "février": 2, "fevrier": 2, "mars": 3,
            "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
            "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
            "novembre": 11, "décembre": 12, "decembre": 12
        }

        detected_months = []
        for month_name, month_num in months.items():
            if month_name in message_lower:
                detected_months.append(month_num)

        if len(detected_months) >= 2:
            # Année courante par défaut
            year = current_dt.year

            period1_start = f"{year}-{detected_months[0]:02d}-01"
            period1_end = self._get_month_end(year, detected_months[0])

            period2_start = f"{year}-{detected_months[1]:02d}-01"
            period2_end = self._get_month_end(year, detected_months[1])

            return {
                "period1": {
                    "start": period1_start,
                    "end": period1_end,
                    "label": list(months.keys())[list(months.values()).index(detected_months[0])]
                },
                "period2": {
                    "start": period2_start,
                    "end": period2_end,
                    "label": list(months.keys())[list(months.values()).index(detected_months[1])]
                }
            }

        return None

    def _get_month_end(self, year: int, month: int) -> str:
        """Retourne le dernier jour du mois"""
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)

        last_day = next_month - timedelta(days=1)
        return last_day.strftime("%Y-%m-%d")

    def parse_relative_period(self, period_expression: str, current_date: str) -> Dict[str, str]:
        """
        Parse une expression de période relative et retourne dates de début/fin

        Exemples:
        - "ce mois" → dates du mois courant
        - "le mois dernier" → dates du mois précédent
        - "les 3 derniers mois" → dates des 3 derniers mois
        - "cette année" → dates de l'année courante
        - "le trimestre dernier" → dates du trimestre précédent

        Args:
            period_expression: Expression temporelle
            current_date: Date de référence (YYYY-MM-DD)

        Returns:
            Dict avec start et end dates
        """
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        period_lower = period_expression.lower()

        # Ce mois / mois courant / mois en cours
        if any(x in period_lower for x in ["ce mois", "mois courant", "mois en cours", "current_month"]):
            start = current_dt.replace(day=1)
            end = current_dt
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Le mois dernier / mois précédent
        if any(x in period_lower for x in ["mois dernier", "mois précédent", "last_month"]):
            first_of_current = current_dt.replace(day=1)
            last_day_previous = first_of_current - timedelta(days=1)
            start = last_day_previous.replace(day=1)
            end = last_day_previous
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Les N derniers mois
        import re
        match_months = re.search(r'(\d+)\s*derniers?\s*mois', period_lower)
        if match_months:
            num_months = int(match_months.group(1))
            end = current_dt
            start = current_dt - relativedelta(months=num_months-1)
            start = start.replace(day=1)
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Cette année / année courante
        if any(x in period_lower for x in ["cette année", "année courante", "current_year"]):
            start = current_dt.replace(month=1, day=1)
            end = current_dt
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # L'année dernière
        if any(x in period_lower for x in ["année dernière", "année précédente", "last_year"]):
            start = datetime(current_dt.year - 1, 1, 1)
            end = datetime(current_dt.year - 1, 12, 31)
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Ce trimestre
        if any(x in period_lower for x in ["ce trimestre", "trimestre courant", "current_quarter"]):
            quarter = (current_dt.month - 1) // 3 + 1
            start_month = (quarter - 1) * 3 + 1
            start = current_dt.replace(month=start_month, day=1)
            end = current_dt
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Trimestre dernier
        if any(x in period_lower for x in ["trimestre dernier", "trimestre précédent", "last_quarter"]):
            current_quarter = (current_dt.month - 1) // 3 + 1
            if current_quarter == 1:
                # Q4 de l'année précédente
                start = datetime(current_dt.year - 1, 10, 1)
                end = datetime(current_dt.year - 1, 12, 31)
            else:
                prev_quarter = current_quarter - 1
                start_month = (prev_quarter - 1) * 3 + 1
                start = datetime(current_dt.year, start_month, 1)
                end_month = start_month + 2
                end = datetime(current_dt.year, end_month + 1, 1) - timedelta(days=1)

            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Semaine courante
        if any(x in period_lower for x in ["cette semaine", "semaine courante", "current_week"]):
            start = current_dt - timedelta(days=current_dt.weekday())
            end = current_dt
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Semaine dernière
        if any(x in period_lower for x in ["semaine dernière", "semaine précédente", "last_week"]):
            start = current_dt - timedelta(days=current_dt.weekday() + 7)
            end = start + timedelta(days=6)
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}

        # Défaut: retourne le mois courant
        logger.warning(f"Unable to parse period expression: {period_expression}, defaulting to current month")
        start = current_dt.replace(day=1)
        return {"start": start.strftime("%Y-%m-%d"), "end": current_dt.strftime("%Y-%m-%d")}

    def generate_multi_period_queries(
        self,
        num_periods: int,
        period_type: str,
        current_date: str,
        filters: Dict[str, Any],
        aggregations: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Génère plusieurs queries pour analyse multi-périodes

        Args:
            num_periods: Nombre de périodes à analyser
            period_type: Type de période (month, week, quarter, year)
            current_date: Date de référence
            filters: Filtres communs à toutes les queries
            aggregations: Agrégations à appliquer

        Returns:
            Liste de query specs
        """
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        queries = []

        for i in range(num_periods):
            if period_type == "month":
                period_start = current_dt - relativedelta(months=i)
                period_start = period_start.replace(day=1)

                # Dernier jour du mois
                if period_start.month == 12:
                    period_end = datetime(period_start.year + 1, 1, 1) - timedelta(days=1)
                else:
                    period_end = datetime(period_start.year, period_start.month + 1, 1) - timedelta(days=1)

                # Si c'est le mois courant, ne pas dépasser aujourd'hui
                if i == 0:
                    period_end = min(period_end, current_dt)

            elif period_type == "week":
                # Début de la semaine courante
                week_start = current_dt - timedelta(days=current_dt.weekday())
                period_start = week_start - timedelta(weeks=i)
                period_end = period_start + timedelta(days=6)

                if i == 0:
                    period_end = current_dt

            elif period_type == "quarter":
                # Calculer le trimestre
                quarters_ago = i
                quarter_start_dt = current_dt - relativedelta(months=quarters_ago * 3)
                quarter = (quarter_start_dt.month - 1) // 3 + 1
                start_month = (quarter - 1) * 3 + 1
                period_start = quarter_start_dt.replace(month=start_month, day=1)

                end_month = start_month + 2
                if end_month == 12:
                    period_end = datetime(period_start.year + 1, 1, 1) - timedelta(days=1)
                else:
                    period_end = datetime(period_start.year, end_month + 1, 1) - timedelta(days=1)

                if i == 0:
                    period_end = current_dt

            else:  # year
                period_start = datetime(current_dt.year - i, 1, 1)
                period_end = datetime(current_dt.year - i, 12, 31)

                if i == 0:
                    period_end = current_dt

            queries.append({
                "time_range": {
                    "start": period_start.strftime("%Y-%m-%d"),
                    "end": period_end.strftime("%Y-%m-%d")
                },
                "filters": filters,
                "aggregations": aggregations,
                "period_label": f"{period_start.strftime('%Y-%m')}"
            })

        return queries

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        return {
            "agent": "analytics",
            "model": self.llm.model_name,
            **self.stats
        }
