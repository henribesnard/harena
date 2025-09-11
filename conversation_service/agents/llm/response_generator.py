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
    UNUSUAL_TRANSACTION = "unusual_transaction"
    BUDGET_ALERT = "budget_alert"
    TREND_ANALYSIS = "trend_analysis"
    RECOMMENDATION = "recommendation"

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
        response_templates_path: Optional[str] = None
    ):
        self.llm_manager = llm_manager
        self.response_templates_path = response_templates_path
        
        # Templates de reponse par intention
        self.response_templates = {}
        self._templates_loaded = False
        
        # Configuration insights automatiques
        self.insight_generators = {
            InsightType.SPENDING_PATTERN: self._generate_spending_pattern_insight,
            InsightType.UNUSUAL_TRANSACTION: self._generate_unusual_transaction_insight,
            InsightType.BUDGET_ALERT: self._generate_budget_alert_insight,
            InsightType.TREND_ANALYSIS: self._generate_trend_analysis_insight,
            InsightType.RECOMMENDATION: self._generate_recommendation_insight
        }
        
        # Statistiques
        self.stats = {
            "responses_generated": 0,
            "streaming_responses": 0,
            "insights_generated": 0,
            "fallbacks_used": 0,
            "avg_processing_time_ms": 0,
            "total_tokens_used": 0
        }
        
        logger.info("ResponseGenerator initialise")
    
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
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1000,
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
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
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
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
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
        
        return insights
    
    async def _generate_spending_pattern_insight(
        self, 
        search_results: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere insight sur patterns de depenses"""
        
        if not search_results:
            return None
        
        try:
            # Analyse simple des transactions
            total_amount = sum(float(tx.get("amount", 0)) for tx in search_results)
            transaction_count = len(search_results)
            avg_transaction = total_amount / transaction_count if transaction_count > 0 else 0
            
            # Comparaison avec moyenne utilisateur (si disponible)
            user_avg = user_profile.get("avg_monthly_spending", avg_transaction * 30)
            
            if total_amount > user_avg * 1.2:  # 20% au-dessus de la moyenne
                return GeneratedInsight(
                    type=InsightType.SPENDING_PATTERN,
                    title="Depenses elevees detectees",
                    description=f"Vos depenses recentes ({total_amount:.2f}e) sont 20% au-dessus de votre moyenne habituelle",
                    confidence=0.75,
                    data_support={"total_amount": total_amount, "user_avg": user_avg},
                    actionable=True,
                    priority=1
                )
            
        except Exception as e:
            logger.warning(f"Erreur calcul spending pattern: {str(e)}")
        
        return None
    
    async def _generate_unusual_transaction_insight(
        self, 
        search_results: List[Dict[str, Any]], 
        user_profile: Dict[str, Any]
    ) -> Optional[GeneratedInsight]:
        """Genere insight sur transactions inhabituelles"""
        
        if not search_results:
            return None
        
        try:
            # Recherche montants inhabituellement eleves
            amounts = [float(tx.get("amount", 0)) for tx in search_results]
            if not amounts:
                return None
                
            avg_amount = sum(amounts) / len(amounts)
            max_amount = max(amounts)
            
            if max_amount > avg_amount * 3:  # 3x la moyenne
                unusual_tx = next(tx for tx in search_results if float(tx.get("amount", 0)) == max_amount)
                
                return GeneratedInsight(
                    type=InsightType.UNUSUAL_TRANSACTION,
                    title="Transaction inhabituelle detectee",
                    description=f"Transaction de {max_amount:.2f}e chez {unusual_tx.get('merchant', 'N/A')} - 3x votre moyenne",
                    confidence=0.80,
                    data_support={"unusual_amount": max_amount, "avg_amount": avg_amount},
                    priority=1
                )
                
        except Exception as e:
            logger.warning(f"Erreur detection transaction inhabituelle: {str(e)}")
        
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
    
    def _select_response_template(self, request: ResponseGenerationRequest) -> Dict[str, str]:
        """Selectionne le template de reponse approprie"""
        
        # Templates par defaut integres
        default_templates = {
            "financial_query": {
                "system": "Tu es un assistant financier expert. Presente les donnees financieres de maniere claire et professionnelle.",
                "structure": "1. Reponse directe, 2. Donnees detaillees, 3. Insights si pertinents"
            },
            "transaction_search": {
                "system": "Tu es un assistant specialise dans l'analyse des transactions. Organise les resultats de maniere logique.",
                "structure": "1. Resume des resultats, 2. Transactions detaillees, 3. Patterns observes"
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

PROFIL UTILISATEUR:
- ID: {request.user_id}
- Preferences: {request.user_profile.get('preferences', 'Aucune specifiee')}"""
        
        return system_prompt
    
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
        
        # Resultats de recherche
        if request.search_results:
            results_summary = f"DONNeES TROUVeES ({len(request.search_results)} resultats):"
            
            # Limiter l'affichage des donnees (eviter surcharge context)
            for i, result in enumerate(request.search_results[:5]):
                result_str = json.dumps(result, ensure_ascii=False)[:200]
                results_summary += f"\n{i+1}. {result_str}..."
            
            if len(request.search_results) > 5:
                results_summary += f"\n... et {len(request.search_results) - 5} autres resultats"
            
            prompt_parts.append(results_summary)
        else:
            prompt_parts.append("DONNeES: Aucune donnee trouvee")
        
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
                generate_insights=False
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