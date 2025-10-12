"""
Response Generator - Agent LLM Phase 4
Architecture v2.0 - Composant IA

Responsabilite : Generation de reponses avec insights automatiques
- Streaming en temps reel pour UX fluide
- Insights automatiques bases sur donnees search_service
- Templates de reponse par type d'intention
- Personnalisation selon profil utilisateur
- Gestion des erreurs avec reponses de fallback
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .llm_provider import LLMProviderManager, LLMRequest, LLMResponse

# 🆕 Import Analytics Agent (Sprint 1.1)
try:
    from conversation_service.agents.analytics.analytics_agent import (
        AnalyticsAgent,
        TimeSeriesMetrics,
        AnomalyDetectionResult,
        TrendAnalysis
    )
    ANALYTICS_AGENT_AVAILABLE = True
except ImportError:
    ANALYTICS_AGENT_AVAILABLE = False
    logger.warning("Analytics Agent not available - fallback to basic insights")

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Types de reponse supportes"""
    DIRECT_ANSWER = "direct_answer"
    DATA_PRESENTATION = "data_presentation"
    INSIGHTS_ANALYSIS = "insights_analysis"
    ERROR_HANDLING = "error_handling"
    CONVERSATION = "conversation"

class InsightType(Enum):
    """Types d'insights automatiques"""
    SPENDING_PATTERN = "spending_pattern"
    INCOME_PATTERN = "income_pattern"
    UNUSUAL_TRANSACTION = "unusual_transaction"
    BUDGET_ALERT = "budget_alert"
    TREND_ANALYSIS = "trend_analysis"
    RECOMMENDATION = "recommendation"
    FINANCIAL_SUMMARY = "financial_summary"

@dataclass
class ResponseGenerationRequest:
    """Requete de generation de reponse"""
    intent_group: str
    intent_subtype: Optional[str]
    user_message: str
    search_results: List[Dict[str, Any]]
    conversation_context: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    user_id: int
    conversation_id: Optional[str] = None
    generate_insights: bool = True
    stream_response: bool = False
    search_aggregations: Optional[Dict[str, Any]] = None

@dataclass
class GeneratedInsight:
    """Insight automatique genere"""
    type: InsightType
    title: str
    description: str
    confidence: float
    data_support: Dict[str, Any]
    actionable: bool = False
    priority: int = 1  # 1=high, 3=low

@dataclass
class ResponseGenerationResult:
    """Resultat de generation de reponse"""
    success: bool
    response_text: str
    response_type: ResponseType
    insights: List[GeneratedInsight]
    data_visualizations: List[Dict[str, Any]]
    processing_time_ms: int
    model_used: str
    tokens_used: int
    stream_completed: bool = False
    error_message: Optional[str] = None

class ResponseGenerator:
    """
    Agent LLM pour generation de reponses avec insights automatiques
    
    Utilise templates et analyse automatique des donnees
    """
    
    def __init__(
        self,
        llm_manager: LLMProviderManager,
        response_templates_path: Optional[str] = None,
        model: str = "deepseek-chat",
        max_tokens: int = 8000,
        temperature: float = 0.7,
        enable_analytics: bool = True  # 🆕 Feature flag Analytics Agent
    ):
        self.llm_manager = llm_manager
        self.response_templates_path = response_templates_path
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_analytics = enable_analytics

        # Templates de reponse par intention
        self.response_templates = {}
        self._templates_loaded = False

        # 🆕 Initialisation Analytics Agent (Sprint 1.1)
        self.analytics_agent = None
        if self.enable_analytics and ANALYTICS_AGENT_AVAILABLE:
            try:
                self.analytics_agent = AnalyticsAgent(
                    zscore_threshold=2.0,
                    iqr_multiplier=1.5,
                    stable_threshold_pct=5.0
                )
                logger.info("Analytics Agent initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Analytics Agent: {e}. Fallback to basic insights.")
                self.analytics_agent = None
        elif not ANALYTICS_AGENT_AVAILABLE:
            logger.info("Analytics Agent module not available")
        else:
            logger.info("Analytics Agent disabled (feature flag off)")

        # Configuration insights automatiques
        self.insight_generators = {
            InsightType.SPENDING_PATTERN: self._generate_spending_pattern_insight,
            InsightType.INCOME_PATTERN: self._generate_spending_pattern_insight,  # Même méthode, gère les deux
            InsightType.UNUSUAL_TRANSACTION: self._generate_unusual_transaction_insight,
            InsightType.BUDGET_ALERT: self._generate_budget_alert_insight,
            InsightType.TREND_ANALYSIS: self._generate_trend_analysis_insight,
            InsightType.RECOMMENDATION: self._generate_recommendation_insight,
            InsightType.FINANCIAL_SUMMARY: self._generate_financial_summary_insight
        }

        # Statistiques
        self.stats = {
            "responses_generated": 0,
            "streaming_responses": 0,
            "insights_generated": 0,
            "fallbacks_used": 0,
            "avg_processing_time_ms": 0,
            "total_tokens_used": 0,
            "analytics_insights_generated": 0,  # 🆕 Statistique Analytics
            "analytics_fallbacks": 0  # 🆕 Statistique fallbacks Analytics
        }

        logger.info(
            f"ResponseGenerator initialise (model={self.model}, max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}, analytics={self.enable_analytics and self.analytics_agent is not None})"
        )
    
    async def initialize(self) -> bool:
        """Initialise le generateur de reponses"""
        try:
            # Charger templates de reponse
            await self._load_response_templates()
            
            logger.info("ResponseGenerator initialise avec succes")
            return True
            
        except Exception as e:
            logger.error(f"Erreur initialisation ResponseGenerator: {str(e)}")
            return False
    
    async def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResult:
        """
        Genere une reponse complete avec insights automatiques
        
        Args:
            request: Requete avec intention, donnees et contexte
            
        Returns:
            ResponseGenerationResult avec reponse et insights
        """
        start_time = datetime.now()
        
        try:
            # 1. Selection du template de reponse
            response_template = self._select_response_template(request)
            
            # 2. Generation insights automatiques
            insights = []
            if request.generate_insights and request.search_results:
                insights = await self._generate_automatic_insights(request)
            
            # 3. Preparation du prompt avec donnees et insights
            system_prompt = self._build_system_prompt(request, response_template)
            user_prompt = self._build_user_prompt(request, insights)
            
            # 4. Generation LLM
            if request.stream_response:
                return await self._generate_streaming_response(
                    request, system_prompt, user_prompt, insights, start_time
                )
            else:
                return await self._generate_complete_response(
                    request, system_prompt, user_prompt, insights, start_time
                )
                
        except Exception as e:
            logger.error(f"Erreur generation reponse: {str(e)}")
            return await self._fallback_response(request, start_time, str(e))
    
    async def generate_streaming_response(
        self, 
        request: ResponseGenerationRequest
    ) -> AsyncIterator[str]:
        """
        Genere une reponse en streaming pour feedback temps reel
        """
        try:
            # Preparer prompts
            response_template = self._select_response_template(request)
            system_prompt = self._build_system_prompt(request, response_template)
            user_prompt = self._build_user_prompt(request, [])
            
            # Streaming LLM
            llm_request = LLMRequest(
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }],
                model=self.model,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                user_id=request.user_id,
                conversation_id=request.conversation_id
            )
            
            self.stats["streaming_responses"] += 1
            
            async for chunk in self.llm_manager.generate_stream(llm_request):
                yield chunk
                
        except Exception as e:
            yield f"Erreur streaming: {str(e)}"
    
    async def _generate_complete_response(
        self,
        request: ResponseGenerationRequest,
        system_prompt: str,
        user_prompt: str,
        insights: List[GeneratedInsight],
        start_time: datetime
    ) -> ResponseGenerationResult:
        """Genere une reponse complete (non-streaming)"""
        
        # Requete LLM
        llm_request = LLMRequest(
            messages=[{
                "role": "user",
                "content": user_prompt
            }],
            model=self.model,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        llm_response = await self.llm_manager.generate(llm_request)
        
        if llm_response.error:
            return await self._fallback_response(request, start_time, llm_response.error)
        
        # Determiner type de reponse
        response_type = self._determine_response_type(request, llm_response.content)
        
        # Preparation visualisations donnees si applicable
        data_visualizations = []
        if request.search_results and response_type == ResponseType.DATA_PRESENTATION:
            data_visualizations = self._prepare_data_visualizations(request)
        
        # Statistiques
        self._update_stats(llm_response, self._get_processing_time(start_time))
        
        return ResponseGenerationResult(
            success=True,
            response_text=llm_response.content,
            response_type=response_type,
            insights=insights,
            data_visualizations=data_visualizations,
            processing_time_ms=self._get_processing_time(start_time),
            model_used=llm_response.model_used,
            tokens_used=llm_response.usage.get("total_tokens", 0)
        )
    
    async def _generate_streaming_response(
        self,
        request: ResponseGenerationRequest,
        system_prompt: str,
        user_prompt: str,
        insights: List[GeneratedInsight],
        start_time: datetime
    ) -> ResponseGenerationResult:
        """Genere une reponse streaming (collecte le resultat final)"""
        
        response_chunks = []
        
        # Streaming LLM
        llm_request = LLMRequest(
            messages=[{
                "role": "user",
                "content": user_prompt
            }],
            model=self.model,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        try:
            async for chunk in self.llm_manager.generate_stream(llm_request):
                response_chunks.append(chunk)
            
            # Reconstituer reponse complete
            full_response = "".join(response_chunks)
            
            response_type = self._determine_response_type(request, full_response)
            data_visualizations = []
            if request.search_results and response_type == ResponseType.DATA_PRESENTATION:
                data_visualizations = self._prepare_data_visualizations(request)
            
            self.stats["streaming_responses"] += 1
            self._update_stats_basic(len(full_response) // 4, self._get_processing_time(start_time))
            
            return ResponseGenerationResult(
                success=True,
                response_text=full_response,
                response_type=response_type,
                insights=insights,
                data_visualizations=data_visualizations,
                processing_time_ms=self._get_processing_time(start_time),
                model_used="streaming",
                tokens_used=len(full_response) // 4,  # Estimation
                stream_completed=True
            )
            
        except Exception as e:
            return await self._fallback_response(request, start_time, f"Streaming error: {str(e)}")
    
    async def _generate_automatic_insights(
        self, 
        request: ResponseGenerationRequest
    ) -> List[GeneratedInsight]:
        """Genere des insights automatiques bases sur les donnees"""
        
        insights = []
        
        # Stocker temporairement request pour que les générateurs puissent accéder aux agrégations
        self.current_request = request
        
        # Generation parallele des differents types d'insights
        insight_tasks = []
        
        for insight_type, generator_func in self.insight_generators.items():
            task = asyncio.create_task(generator_func(request.search_results, request.user_profile))
            insight_tasks.append(task)
        
        # Attendre tous les insights
        try:
            insight_results = await asyncio.gather(*insight_tasks, return_exceptions=True)
            
            for result in insight_results:
                if isinstance(result, GeneratedInsight):
                    insights.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Erreur generation insight: {str(result)}")
            
        except Exception as e:
            logger.error(f"Erreur generation insights automatiques: {str(e)}")
        
        # Trier par priorite et confiance
        insights.sort(key=lambda x: (x.priority, -x.confidence))
        
        # Limiter e 3 insights max
        insights = insights[:3]
        self.stats["insights_generated"] += len(insights)
        
        # Nettoyer la référence temporaire
        if hasattr(self, 'current_request'):
            delattr(self, 'current_request')
        
        return insights
    
    async def _generate_spending_pattern_insight(
        self, 
        search_results: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere insight sur patterns de depenses et revenus selon le type de transaction"""
        
        if not search_results:
            return None
        
        try:
            # Catégoriser par type de transaction
            debits = [tx for tx in search_results if tx.get("transaction_type") == "debit"]
            credits = [tx for tx in search_results if tx.get("transaction_type") == "credit"]
            
            # Traiter les dépenses (debits)
            if debits:
                insight = await self._analyze_expense_pattern(debits, user_profile)
                if insight:
                    return insight
            
            # Traiter les revenus (credits) 
            if credits:
                insight = await self._analyze_income_pattern(credits, user_profile)
                if insight:
                    return insight
                    
        except Exception as e:
            logger.warning(f"Erreur calcul spending pattern: {str(e)}")
        
        return None
    
    async def _analyze_expense_pattern(
        self, 
        debit_transactions: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Analyse les patterns de dépenses (debits)"""
        
        amounts = [abs(float(tx.get("amount", 0))) for tx in debit_transactions]
        abs_total_amount = sum(amounts)
        transaction_count = len(amounts)
        
        # Comparaison avec moyenne utilisateur pour les dépenses
        abs_user_avg = abs(user_profile.get("avg_monthly_spending", abs_total_amount))
        
        if abs_total_amount > abs_user_avg * 1.2:  # Dépenses élevées: 20% au-dessus
            percentage_increase = ((abs_total_amount - abs_user_avg) / abs_user_avg) * 100
            
            return GeneratedInsight(
                type=InsightType.SPENDING_PATTERN,
                title="Dépenses élevées détectées",
                description=f"Vos dépenses récentes ({abs_total_amount:.2f}€) sont {percentage_increase:.0f}% au-dessus de votre moyenne habituelle ({abs_user_avg:.2f}€)",
                confidence=0.75,
                data_support={
                    "abs_total_amount": abs_total_amount, 
                    "abs_user_avg": abs_user_avg, 
                    "percentage_increase": percentage_increase,
                    "transaction_type": "debit",
                    "transaction_count": transaction_count
                },
                actionable=True,
                priority=1
            )
        elif abs_total_amount < abs_user_avg * 0.5:  # Dépenses faibles: 50% en dessous
            percentage_decrease = ((abs_user_avg - abs_total_amount) / abs_user_avg) * 100
            
            return GeneratedInsight(
                type=InsightType.SPENDING_PATTERN,
                title="Dépenses réduites détectées",
                description=f"Vos dépenses récentes ({abs_total_amount:.2f}€) sont {percentage_decrease:.0f}% en dessous de votre moyenne habituelle ({abs_user_avg:.2f}€)",
                confidence=0.70,
                data_support={
                    "abs_total_amount": abs_total_amount, 
                    "abs_user_avg": abs_user_avg, 
                    "percentage_decrease": percentage_decrease,
                    "transaction_type": "debit",
                    "transaction_count": transaction_count
                },
                actionable=False,
                priority=2
            )
        
        return None
    
    async def _analyze_income_pattern(
        self, 
        credit_transactions: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Analyse les patterns de revenus (credits)"""
        
        amounts = [abs(float(tx.get("amount", 0))) for tx in credit_transactions]
        abs_total_amount = sum(amounts)
        transaction_count = len(amounts)
        
        # Comparaison avec moyenne utilisateur pour les revenus
        abs_user_avg = abs(user_profile.get("avg_monthly_income", abs_total_amount))
        
        if abs_total_amount > abs_user_avg * 1.5:  # Revenus élevés: seuil plus haut (50%)
            percentage_increase = ((abs_total_amount - abs_user_avg) / abs_user_avg) * 100
            
            return GeneratedInsight(
                type=InsightType.INCOME_PATTERN,
                title="Revenus exceptionnels détectés",
                description=f"Vos revenus récents ({abs_total_amount:.2f}€) sont {percentage_increase:.0f}% au-dessus de votre moyenne habituelle ({abs_user_avg:.2f}€)",
                confidence=0.80,
                data_support={
                    "abs_total_amount": abs_total_amount, 
                    "abs_user_avg": abs_user_avg, 
                    "percentage_increase": percentage_increase,
                    "transaction_type": "credit",
                    "transaction_count": transaction_count
                },
                actionable=False,
                priority=1
            )
        elif abs_total_amount < abs_user_avg * 0.3:  # Revenus faibles: seuil plus bas (70%)
            percentage_decrease = ((abs_user_avg - abs_total_amount) / abs_user_avg) * 100
            
            return GeneratedInsight(
                type=InsightType.INCOME_PATTERN,
                title="Revenus réduits détectés",
                description=f"Vos revenus récents ({abs_total_amount:.2f}€) sont {percentage_decrease:.0f}% en dessous de votre moyenne habituelle ({abs_user_avg:.2f}€)",
                confidence=0.75,
                data_support={
                    "abs_total_amount": abs_total_amount, 
                    "abs_user_avg": abs_user_avg, 
                    "percentage_decrease": percentage_decrease,
                    "transaction_type": "credit",
                    "transaction_count": transaction_count
                },
                actionable=True,
                priority=1
            )
        
        return None
    
    async def _generate_unusual_transaction_insight(
        self,
        search_results: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere insight sur transactions inhabituelles selon le type (debit/credit)

        🆕 Sprint 1.1: Utilise Analytics Agent si disponible pour détection anomalies avancée
        """

        if not search_results:
            return None

        try:
            # 🆕 Tentative utilisation Analytics Agent (fallback gracieux si échec)
            if self.analytics_agent:
                try:
                    analytics_insight = await self._detect_anomalies_with_analytics_agent(
                        search_results, user_profile
                    )
                    if analytics_insight:
                        self.stats["analytics_insights_generated"] += 1
                        return analytics_insight
                except Exception as e:
                    logger.warning(f"Analytics Agent anomaly detection failed: {e}. Using basic method.")
                    self.stats["analytics_fallbacks"] += 1

            # Fallback: Méthode basique existante
            # Séparer par type de transaction
            debits = [tx for tx in search_results if tx.get("transaction_type") == "debit"]
            credits = [tx for tx in search_results if tx.get("transaction_type") == "credit"]

            # Analyser les dépenses inhabituelles
            if debits:
                insight = await self._analyze_unusual_expenses(debits, user_profile)
                if insight:
                    return insight

            # Analyser les revenus inhabituels
            if credits:
                insight = await self._analyze_unusual_income(credits, user_profile)
                if insight:
                    return insight

        except Exception as e:
            logger.warning(f"Erreur detection transaction inhabituelle: {str(e)}")

        return None
    
    async def _analyze_unusual_expenses(
        self, 
        debit_transactions: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Détecte les dépenses inhabituellement élevées"""
        
        abs_amounts = [abs(float(tx.get("amount", 0))) for tx in debit_transactions]
        if not abs_amounts:
            return None
        
        # Supprimer les doublons pour éviter les faux positifs
        unique_abs_amounts = list(set(abs_amounts))
        if len(unique_abs_amounts) <= 1:
            return None
            
        abs_avg_amount = sum(abs_amounts) / len(abs_amounts)
        max_abs_amount = max(abs_amounts)
        
        # Seuil plus bas pour les dépenses (2x au lieu de 3x)
        if max_abs_amount > abs_avg_amount * 2:
            max_abs_index = abs_amounts.index(max_abs_amount)
            unusual_tx = debit_transactions[max_abs_index]
            multiplier = max_abs_amount / abs_avg_amount
            
            return GeneratedInsight(
                type=InsightType.UNUSUAL_TRANSACTION,
                title="Dépense inhabituelle détectée",
                description=f"Dépense de {max_abs_amount:.2f}€ chez {unusual_tx.get('merchant_name', unusual_tx.get('merchant', 'N/A'))} - {multiplier:.1f}x votre moyenne ({abs_avg_amount:.2f}€)",
                confidence=0.85,
                data_support={
                    "max_abs_amount": max_abs_amount, 
                    "abs_avg_amount": abs_avg_amount, 
                    "multiplier": multiplier,
                    "transaction_type": "debit",
                    "merchant": unusual_tx.get('merchant_name', unusual_tx.get('merchant', 'N/A'))
                },
                actionable=True,
                priority=1
            )
        
        return None
    
    async def _analyze_unusual_income(
        self, 
        credit_transactions: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Détecte les revenus inhabituellement élevés"""
        
        abs_amounts = [abs(float(tx.get("amount", 0))) for tx in credit_transactions]
        if not abs_amounts:
            return None
        
        # Supprimer les doublons pour éviter les faux positifs
        unique_abs_amounts = list(set(abs_amounts))
        if len(unique_abs_amounts) <= 1:
            return None
            
        abs_avg_amount = sum(abs_amounts) / len(abs_amounts)
        max_abs_amount = max(abs_amounts)
        
        # Seuil plus élevé pour les revenus (4x)
        if max_abs_amount > abs_avg_amount * 4:
            max_abs_index = abs_amounts.index(max_abs_amount)
            unusual_tx = credit_transactions[max_abs_index]
            multiplier = max_abs_amount / abs_avg_amount
            
            return GeneratedInsight(
                type=InsightType.UNUSUAL_TRANSACTION,
                title="Revenu exceptionnel détecté",
                description=f"Revenu de {max_abs_amount:.2f}€ de {unusual_tx.get('merchant_name', unusual_tx.get('merchant', 'N/A'))} - {multiplier:.1f}x votre moyenne ({abs_avg_amount:.2f}€)",
                confidence=0.80,
                data_support={
                    "max_abs_amount": max_abs_amount, 
                    "abs_avg_amount": abs_avg_amount, 
                    "multiplier": multiplier,
                    "transaction_type": "credit",
                    "merchant": unusual_tx.get('merchant_name', unusual_tx.get('merchant', 'N/A'))
                },
                actionable=False,
                priority=2
            )
        
        return None
    
    async def _generate_budget_alert_insight(
        self, 
        search_results: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere alerte budget si applicable"""
        
        # Implementation simple - e enrichir selon donnees utilisateur
        user_budget = user_profile.get("monthly_budget")
        if not user_budget:
            return None
        
        try:
            total_spent = sum(float(tx.get("amount", 0)) for tx in search_results)
            budget_usage = (total_spent / user_budget) * 100
            
            if budget_usage > 80:
                return GeneratedInsight(
                    type=InsightType.BUDGET_ALERT,
                    title="Budget mensuel bientet depasse",
                    description=f"Vous avez utilise {budget_usage:.1f}% de votre budget mensuel",
                    confidence=0.90,
                    data_support={"budget_usage": budget_usage, "total_spent": total_spent},
                    actionable=True,
                    priority=1
                )
                
        except Exception as e:
            logger.warning(f"Erreur calcul budget alert: {str(e)}")
        
        return None
    
    async def _generate_trend_analysis_insight(
        self, 
        search_results: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere insight analyse de tendance"""
        
        # Placeholder - necessite donnees historiques
        return None
    
    async def _generate_recommendation_insight(
        self, 
        search_results: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere recommandation personnalisee"""
        
        # Placeholder - e implementer selon regles metier
        return None
    
    async def _generate_financial_summary_insight(
        self,
        search_results: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere un insight avec les totaux financiers basés sur les agrégations"""

        # Accéder aux agrégations via self.current_request (sera défini dans generate_automatic_insights)
        if not hasattr(self, 'current_request') or not self.current_request.search_aggregations:
            return None

        aggregations = self.current_request.search_aggregations

        # Extraire les données agrégées (nouvelles agrégations automatiques)
        transaction_count = aggregations.get('transaction_count', {}).get('value', 0)

        total_debit = 0
        total_credit = 0
        has_debit = False
        has_credit = False

        if 'total_debit' in aggregations and 'sum_amount' in aggregations['total_debit']:
            total_debit = aggregations['total_debit']['sum_amount'].get('value', 0)
            has_debit = True

        if 'total_credit' in aggregations and 'sum_amount' in aggregations['total_credit']:
            total_credit = aggregations['total_credit']['sum_amount'].get('value', 0)
            has_credit = True

        if transaction_count == 0:
            return None

        # Construire data_support dynamiquement selon les agrégations disponibles
        data_support = {
            "transaction_count": transaction_count,
            "currency": "EUR"
        }

        # N'inclure que les totaux présents dans les agrégations
        if has_debit:
            data_support["total_debit"] = total_debit
        if has_credit:
            data_support["total_credit"] = total_credit

        # Construire description dynamique
        description_parts = []
        if has_debit:
            description_parts.append(f"Débits: {total_debit:.2f}€")
        if has_credit:
            description_parts.append(f"Crédits: {total_credit:.2f}€")

        description = " | ".join(description_parts) if description_parts else f"{transaction_count} transactions"

        return GeneratedInsight(
            type=InsightType.FINANCIAL_SUMMARY,
            title=f"Résumé financier : {transaction_count} transactions",
            description=description,
            confidence=1.0,
            data_support=data_support,
            actionable=False,
            priority=0  # Priorité élevée pour affichage en premier
        )

    async def _detect_anomalies_with_analytics_agent(
        self,
        search_results: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Détecte anomalies avec Analytics Agent (Sprint 1.1)

        Convertit les transactions search_results au format Analytics Agent,
        appelle detect_anomalies(), et convertit le résultat en GeneratedInsight.

        Args:
            search_results: Liste transactions depuis search_service
            user_profile: Profil utilisateur pour contexte

        Returns:
            GeneratedInsight si anomalie détectée, None sinon
        """

        if not search_results or len(search_results) < 3:
            # Besoin d'au moins 3 transactions pour détection statistique
            return None

        try:
            # Préparer données pour Analytics Agent
            # Format requis: {'id': int, 'amount': float, 'date': str, 'merchant_name': str}
            analytics_transactions = []

            for tx in search_results:
                # Convertir au format Analytics Agent
                analytics_tx = {
                    'id': tx.get('id', tx.get('transaction_id', 0)),
                    'amount': abs(float(tx.get('amount', 0))),  # Valeur absolue pour anomalies
                    'date': tx.get('date', tx.get('transaction_date', '')),
                    'merchant_name': tx.get('merchant_name', tx.get('merchant', 'N/A'))
                }
                analytics_transactions.append(analytics_tx)

            # Appeler Analytics Agent pour détection anomalies Z-score
            # Threshold 1.5 pour équilibrer sensibilité (détecte plus d'anomalies)
            # vs précision (évite trop de faux positifs)
            anomalies = await self.analytics_agent.detect_anomalies(
                transactions=analytics_transactions,
                method='zscore',
                threshold=1.5  # Threshold abaissé pour meilleure sensibilité
            )

            if not anomalies:
                # Aucune anomalie détectée
                return None

            # Prendre la première anomalie (score le plus élevé)
            # Les anomalies sont triées par score décroissant dans Analytics Agent
            top_anomaly = anomalies[0]

            # Déterminer type de transaction (debit/credit) pour titre personnalisé
            original_tx = next(
                (tx for tx in search_results
                 if tx.get('id') == top_anomaly.transaction_id or
                    tx.get('transaction_id') == top_anomaly.transaction_id),
                None
            )

            transaction_type = "debit"
            if original_tx:
                tx_amount = float(original_tx.get('amount', 0))
                transaction_type = "credit" if tx_amount > 0 else "debit"

            # Titre personnalisé selon type
            if transaction_type == "credit":
                title = "Revenu exceptionnel détecté (Analytics)"
                description_prefix = "Revenu de"
            else:
                title = "Dépense inhabituelle détectée (Analytics)"
                description_prefix = "Dépense de"

            # Convertir AnomalyDetectionResult en GeneratedInsight
            return GeneratedInsight(
                type=InsightType.UNUSUAL_TRANSACTION,
                title=title,
                description=f"{description_prefix} {top_anomaly.amount:.2f}€ chez {top_anomaly.merchant_name} - {top_anomaly.explanation}",
                confidence=min(0.95, 0.70 + (top_anomaly.anomaly_score / 10)),  # Confidence basée sur score
                data_support={
                    "transaction_id": top_anomaly.transaction_id,
                    "amount": top_anomaly.amount,
                    "merchant_name": top_anomaly.merchant_name,
                    "anomaly_score": top_anomaly.anomaly_score,
                    "method": top_anomaly.method,
                    "threshold_exceeded": top_anomaly.threshold_exceeded,
                    "transaction_type": transaction_type,
                    "analytics_agent_used": True  # Flag pour debugging
                },
                actionable=True,
                priority=1
            )

        except Exception as e:
            # Log l'erreur et propager l'exception pour fallback
            logger.error(f"Analytics Agent anomaly detection error: {str(e)}", exc_info=True)
            raise  # Propager pour que le fallback se déclenche

    def _select_response_template(self, request: ResponseGenerationRequest) -> Dict[str, str]:
        """Selectionne le template de reponse approprie"""
        
        # Templates par defaut integres
        default_templates = {
            "financial_query": {
                "system": "Tu es un assistant financier expert. Presente les donnees financieres de maniere claire et professionnelle.",
                "structure": "1. Reponse directe, 2. Donnees detaillees, 3. Insights si pertinents"
            },
            "transaction_search": {
                "system": "Tu es un assistant personnel specialise dans l'analyse de vos transactions financieres. Reponds avec un ton chaleureux et personnel.",
                "structure": "1. Resume personnel des resultats, 2. Descriptions claires des transactions avec noms de marchands, 3. Observations utiles pour vos finances"
            },
            "account_management": {
                "system": "Tu es un assistant pour la gestion de comptes. Donne des instructions claires et securisees.",
                "structure": "1. Confirmation de l'action, 2. etapes necessaires, 3. Conseils securite"
            },
            "CONVERSATIONAL": {
                "system": "Tu es un assistant financier amical et utile. Maintiens une conversation naturelle.",
                "structure": "1. Reponse empathique, 2. Information utile, 3. Suggestion d'action"
            }
        }
        
        return default_templates.get(request.intent_group, default_templates["CONVERSATIONAL"])
    
    def _build_system_prompt(
        self, 
        request: ResponseGenerationRequest, 
        template: Dict[str, str]
    ) -> str:
        """Construit le prompt systeme avec template et contexte"""
        
        base_system = template.get("system", "Tu es un assistant financier utile.")
        structure = template.get("structure", "Reponds de maniere claire et organisee.")
        
        system_prompt = f"""{base_system}

STRUCTURE DE RePONSE: {structure}

ReGLES:
- Utilise les donnees fournies pour des reponses precises
- Mentionne les insights automatiques s'ils sont fournis
- Adapte le ton selon le contexte de conversation
- En cas de donnees manquantes, le signaler clairement
- Propose des actions concretes quand pertinent
- Utilise TOUJOURS "vos transactions", "votre compte", "vos depenses" etc. (forme personnelle)
- JAMAIS mentionner l'ID utilisateur ou les IDs techniques
- Concentre-toi sur le nom du marchand et la description pour identifier les transactions

PROFIL UTILISATEUR:
- Preferences: {request.user_profile.get('preferences', 'Aucune specifiee')}"""
        
        return system_prompt
    
    def _filter_transaction_data(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Filtre les données de transaction pour minimiser le contexte LLM

        Réduit de 14 champs à 6 champs essentiels pour améliorer la précision du LLM
        et réduire la consommation de tokens (~57% de réduction).

        Champs conservés :
        - amount: Montant de la transaction
        - date: Date de la transaction
        - primary_description: Description détaillée
        - merchant_name: Nom du marchand
        - category_name: Catégorie de dépense/revenu
        - operation_type: Type d'opération (CB, VIR, etc.)
        """

        # Champs essentiels pour le LLM (optimisation token)
        essential_fields = {
            'amount',              # Montant de la transaction
            'date',                # Date de la transaction
            'primary_description', # Description détaillée
            'merchant_name',       # Nom du marchand
            'category_name',       # Catégorie
            'operation_type'       # Type d'opération (CB, VIR, etc.)
        }

        # Filtrer en conservant uniquement les champs essentiels
        filtered_transaction = {}
        for key, value in transaction.items():
            if key in essential_fields and value is not None:
                filtered_transaction[key] = value

        return filtered_transaction

    def _build_user_prompt(
        self, 
        request: ResponseGenerationRequest, 
        insights: List[GeneratedInsight]
    ) -> str:
        """Construit le prompt utilisateur avec donnees et contexte"""
        
        prompt_parts = []
        
        # Message utilisateur
        prompt_parts.append(f"MESSAGE UTILISATEUR: \"{request.user_message}\"")
        
        # Intention classifiee
        prompt_parts.append(f"INTENTION: {request.intent_group}.{request.intent_subtype}")
        
        # Resultats de recherche - TOUTES les transactions filtrées
        if request.search_results:
            # Filtrer TOUTES les transactions pour réduire le contexte LLM
            filtered_results = [
                self._filter_transaction_data(result)
                for result in request.search_results
            ]

            # Vérification de la limite de tokens
            max_transactions = self._calculate_max_transactions_for_context(filtered_results)

            # Tronquer si nécessaire avec warning
            if len(filtered_results) > max_transactions:
                logger.warning(
                    f"Réduction du nombre de transactions pour le LLM: "
                    f"{len(filtered_results)} → {max_transactions} (limite tokens)"
                )
                filtered_results = filtered_results[:max_transactions]

            # Construire le résumé avec TOUTES les transactions filtrées
            results_summary = f"DONNEES TROUVEES ({len(filtered_results)} transactions"
            if len(filtered_results) < len(request.search_results):
                results_summary += f" sur {len(request.search_results)} total, limite appliquée"
            results_summary += "):\n"

            # Envoyer TOUTES les transactions filtrées au LLM (format compact JSON)
            results_summary += json.dumps(filtered_results, ensure_ascii=False, indent=2)

            prompt_parts.append(results_summary)
        else:
            prompt_parts.append("DONNEES: Aucune donnee trouvee")

        # Agrégations optimisées (optionnel mais utile pour les totaux)
        if request.search_aggregations:
            formatted_aggs = self._format_aggregations_for_llm(request.search_aggregations)
            if formatted_aggs:
                prompt_parts.append(f"AGREGATIONS FINANCIERES:\n{formatted_aggs}")

        # Insights automatiques
        if insights:
            insights_text = "INSIGHTS AUTOMATIQUES:"
            for insight in insights:
                insights_text += f"\n- {insight.title}: {insight.description}"
            prompt_parts.append(insights_text)
        
        # Contexte conversation
        if request.conversation_context:
            context_lines = []
            for turn in request.conversation_context[-2:]:  # 2 derniers echanges
                role = turn.get("role", "user")
                content = turn.get("content", "")[:100]
                context_lines.append(f"{role}: {content}")
            
            if context_lines:
                prompt_parts.append("CONTEXTE ReCENT:\n" + "\n".join(context_lines))
        
        prompt_parts.append("RePONSE:")
        
        return "\n\n".join(prompt_parts)
    
    def _determine_response_type(self, request: ResponseGenerationRequest, response_text: str) -> ResponseType:
        """Determine le type de reponse genere"""
        
        if request.search_results:
            return ResponseType.DATA_PRESENTATION
        elif "insight" in response_text.lower() or "analyse" in response_text.lower():
            return ResponseType.INSIGHTS_ANALYSIS
        elif request.intent_group == "CONVERSATIONAL":
            return ResponseType.CONVERSATION
        else:
            return ResponseType.DIRECT_ANSWER
    
    def _prepare_data_visualizations(self, request: ResponseGenerationRequest) -> List[Dict[str, Any]]:
        """Prepare les visualisations de donnees pour l'interface"""
        
        visualizations = []
        
        if not request.search_results:
            return visualizations
        
        try:
            # Graphique montants par date (si transactions)
            if request.intent_group == "transaction_search":
                dates_amounts = {}
                for tx in request.search_results:
                    date = tx.get("date", "N/A")
                    amount = float(tx.get("amount", 0))
                    dates_amounts[date] = dates_amounts.get(date, 0) + amount
                
                if dates_amounts:
                    visualizations.append({
                        "type": "line_chart",
                        "title": "evolution des montants",
                        "data": [{"date": d, "amount": a} for d, a in dates_amounts.items()],
                        "x_axis": "date",
                        "y_axis": "amount"
                    })
            
            # Repartition par categorie
            categories_amounts = {}
            for result in request.search_results:
                category = result.get("category", "Autres")
                amount = float(result.get("amount", 0))
                categories_amounts[category] = categories_amounts.get(category, 0) + amount
            
            if len(categories_amounts) > 1:
                visualizations.append({
                    "type": "pie_chart",
                    "title": "Repartition par categorie",
                    "data": [{"category": c, "amount": a} for c, a in categories_amounts.items()]
                })
                
        except Exception as e:
            logger.warning(f"Erreur preparation visualisations: {str(e)}")
        
        return visualizations
    
    async def _fallback_response(
        self, 
        request: ResponseGenerationRequest, 
        start_time: datetime,
        error_message: str
    ) -> ResponseGenerationResult:
        """Reponse de fallback en cas d'erreur LLM"""
        
        self.stats["fallbacks_used"] += 1
        
        # Reponse basee sur l'intention
        if request.intent_group == "financial_query":
            if request.search_results:
                response = f"J'ai trouve {len(request.search_results)} resultats pour votre recherche financiere, mais j'ai eu un probleme pour generer une analyse detaillee. Voici les donnees brutes disponibles."
            else:
                response = "Je n'ai pas pu acceder e vos donnees financieres pour le moment. Veuillez reessayer plus tard."
        
        elif request.intent_group == "transaction_search":
            if request.search_results:
                total_amount = sum(float(tx.get("amount", 0)) for tx in request.search_results)
                response = f"J'ai trouve {len(request.search_results)} transactions pour un total de {total_amount:.2f}e. Details technique indisponible temporairement."
            else:
                response = "Aucune transaction trouvee pour vos criteres de recherche."
        
        else:
            response = "Je rencontre actuellement des difficultes techniques. Pouvez-vous reformuler votre demande ?"
        
        return ResponseGenerationResult(
            success=False,
            response_text=response,
            response_type=ResponseType.ERROR_HANDLING,
            insights=[],
            data_visualizations=[],
            processing_time_ms=self._get_processing_time(start_time),
            model_used="fallback",
            tokens_used=0,
            error_message=error_message
        )
    
    async def _load_response_templates(self) -> None:
        """Charge les templates de reponse depuis la configuration"""
        
        # Templates integres par defaut
        self.response_templates = {
            "financial_query": {
                "balance": "Voici votre solde actuel et les informations associees.",
                "transactions": "Voici vos transactions recentes avec analyses.",
                "expenses": "Voici votre analyse de depenses avec recommandations.",
                "budget": "Voici votre situation budgetaire actuelle."
            },
            "transaction_search": {
                "simple": "Voici les transactions trouvees selon vos criteres.",
                "advanced": "Voici l'analyse avancee de vos transactions.",
                "filter": "Voici les transactions filtrees selon vos criteres specifiques.",
                "aggregate": "Voici l'analyse agregee de vos donnees financieres."
            },
            "account_management": {
                "create": "Voici les etapes pour creer votre nouveau compte.",
                "update": "Voici comment mettre e jour les informations de votre compte.",
                "delete": "Voici la procedure de suppression de compte.",
                "list": "Voici la liste de vos comptes et leurs informations."
            },
            "CONVERSATIONAL": {
                "greeting": "Bonjour ! Je suis le pour vous aider avec vos finances.",
                "help": "Je peux vous aider avec la consultation de soldes, recherche de transactions, et gestion de comptes.",
                "goodbye": "Au revoir ! N'hesitez pas e revenir pour vos questions financieres.",
                "clarification": "Pouvez-vous preciser votre demande ? Je suis le pour vous aider."
            }
        }
        
        self._templates_loaded = True
        logger.info(f"Charge {len(self.response_templates)} groupes de templates")
    
    def _update_stats(self, llm_response: LLMResponse, processing_time: int):
        """Met e jour les statistiques detaillees"""
        
        self.stats["responses_generated"] += 1
        self.stats["total_tokens_used"] += llm_response.usage.get("total_tokens", 0)
        
        # Moyenne mobile du temps de traitement
        current_avg = self.stats["avg_processing_time_ms"]
        total_responses = self.stats["responses_generated"]
        
        self.stats["avg_processing_time_ms"] = (
            (current_avg * (total_responses - 1) + processing_time) / total_responses
        )
    
    def _update_stats_basic(self, tokens_used: int, processing_time: int):
        """Met e jour les statistiques basiques (pour streaming)"""
        
        self.stats["responses_generated"] += 1
        self.stats["total_tokens_used"] += tokens_used
        
        current_avg = self.stats["avg_processing_time_ms"]
        total_responses = self.stats["responses_generated"]
        
        self.stats["avg_processing_time_ms"] = (
            (current_avg * (total_responses - 1) + processing_time) / total_responses
        )
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calcule le temps de traitement en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)

    def _calculate_max_transactions_for_context(
        self,
        filtered_transactions: List[Dict[str, Any]]
    ) -> int:
        """Calcule le nombre maximum de transactions à inclure sans dépasser les limites de tokens

        Limite DeepSeek : 128K tokens
        Budget alloué aux transactions : 80K tokens (laisse 48K pour le reste du contexte)
        Estimation : ~55 tokens par transaction filtrée (6 champs)

        Args:
            filtered_transactions: Liste des transactions déjà filtrées

        Returns:
            Nombre maximum de transactions à inclure dans le contexte
        """

        MAX_TOKENS_FOR_TRANSACTIONS = 80000  # Budget token pour les transactions
        AVG_TOKENS_PER_TRANSACTION = 55      # Estimation après filtrage à 6 champs

        # Calcul du nombre max basé sur le budget tokens
        max_based_on_tokens = MAX_TOKENS_FOR_TRANSACTIONS // AVG_TOKENS_PER_TRANSACTION

        # Limite de sécurité : maximum 1500 transactions
        max_transactions = min(max_based_on_tokens, 1500, len(filtered_transactions))

        logger.debug(
            f"Calcul limite transactions: {len(filtered_transactions)} disponibles, "
            f"max basé tokens: {max_based_on_tokens}, limite retenue: {max_transactions}"
        )

        return max_transactions

    def _format_aggregations_for_llm(
        self,
        aggregations: Dict[str, Any]
    ) -> str:
        """Formate les agrégations de manière compacte pour le LLM

        Optimise les agrégations en ne gardant que les métriques essentielles
        et en limitant le nombre de détails par catégorie.

        Args:
            aggregations: Dictionnaire des agrégations depuis search_service

        Returns:
            String JSON formaté avec les agrégations optimisées
        """

        if not aggregations:
            return ""

        # Extraire uniquement les métriques essentielles
        compact_aggs = {}

        # Total des transactions
        if 'transaction_count' in aggregations:
            compact_aggs['total_transactions'] = aggregations['transaction_count'].get('value', 0)

        # Total débits
        if 'total_debit' in aggregations and 'sum_amount' in aggregations['total_debit']:
            compact_aggs['total_debit'] = round(
                aggregations['total_debit']['sum_amount'].get('value', 0), 2
            )

        # Total crédits
        if 'total_credit' in aggregations and 'sum_amount' in aggregations['total_credit']:
            compact_aggs['total_credit'] = round(
                aggregations['total_credit']['sum_amount'].get('value', 0), 2
            )

        # Top marchands (seulement top 5 pour éviter la surcharge)
        top_merchants = []
        for key, value in aggregations.items():
            # Identifier les agrégations de marchands (finissent par _debit ou _credit)
            if (key.endswith('_debit') or key.endswith('_credit')) and isinstance(value, dict):
                merchant_name = key.replace('_debit', '').replace('_credit', '').replace('_', ' ').title()
                amount = value.get('sum_amount', {}).get('value', 0)
                if amount > 0:
                    top_merchants.append({
                        'merchant': merchant_name,
                        'amount': round(amount, 2)
                    })

        # Trier par montant et limiter à top 5
        if top_merchants:
            top_merchants.sort(key=lambda x: x['amount'], reverse=True)
            compact_aggs['top_merchants'] = top_merchants[:5]

        return json.dumps(compact_aggs, ensure_ascii=False, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Recupere les statistiques du generateur"""
        return {
            **self.stats,
            "templates_loaded": len(self.response_templates),
            "insight_generators": len(self.insight_generators)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check du generateur de reponses"""
        
        try:
            # Test rapide de generation
            test_request = ResponseGenerationRequest(
                intent_group="CONVERSATIONAL",
                intent_subtype="help",
                user_message="Test sante",
                search_results=[],
                conversation_context=[],
                user_profile={},
                user_id=0,
                generate_insights=False,
                search_aggregations=None
            )
            
            result = await self.generate_response(test_request)
            
            return {
                "status": "healthy" if result.success else "degraded",
                "component": "response_generator", 
                "templates_loaded": len(self.response_templates),
                "insight_generators": len(self.insight_generators),
                "test_generation_success": result.success,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "response_generator",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }