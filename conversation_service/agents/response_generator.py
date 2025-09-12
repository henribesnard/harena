"""
Response Generator Agent - LLM Component
Version 2.0 - Hybrid IA + Pure Logic Architecture

Responsabilité : Génération de réponses contextualisées et personnalisées basées sur les résultats de recherche.
- Génération réponse par LLM avec contexte riche
- Streaming réponse en temps réel 
- Injection insights et suggestions automatiques
- Enrichissement automatique avec analyses financières
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timezone
from dataclasses import dataclass

from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.agents.logic.context_manager import OptimizedContext
from conversation_service.clients.deepseek_client import DeepSeekClient, DeepSeekError
from conversation_service.utils.metrics_collector import metrics_collector

logger = logging.getLogger(__name__)

@dataclass
class ResponseGenerationConfig:
    """Configuration génération de réponse"""
    primary_provider: str = "deepseek"
    temperature: float = 0.3
    max_tokens: int = 1500
    streaming: bool = True
    tone: str = "professionnel_accessible"
    format: str = "conversationnel"
    include_insights: bool = True
    include_suggestions: bool = True

@dataclass
class GeneratedResponse:
    """Réponse générée avec métadonnées"""
    content: str
    insights: List[Dict[str, Any]]
    suggestions: List[str]
    metadata: Dict[str, Any]
    generation_time_ms: int
    provider_used: str
    token_count: int
    streaming_used: bool

class ResponseGeneratorAgent(BaseAgent):
    """
    Agent LLM pour génération de réponses contextualisées
    
    Architecture v2.0 - Génération enrichie avec streaming
    """
    
    def __init__(
        self,
        deepseek_client: DeepSeekClient,
        config: Optional[ResponseGenerationConfig] = None
    ):
        super().__init__()
        self.deepseek_client = deepseek_client
        self.config = config or ResponseGenerationConfig()
        
        logger.info("ResponseGeneratorAgent v2.0 initialisé")
        logger.info(f"Configuration: {self.config.primary_provider}, streaming={self.config.streaming}")
    
    async def generate_response(
        self,
        optimized_context: OptimizedContext,
        user_id: int,
        streaming: Optional[bool] = None
    ) -> GeneratedResponse:
        """
        Génération de réponse contextualisée
        
        Args:
            optimized_context: Contexte optimisé par ContextManager
            user_id: ID utilisateur
            streaming: Override streaming config
            
        Returns:
            GeneratedResponse avec contenu et métadonnées
        """
        start_time = datetime.now()
        use_streaming = streaming if streaming is not None else self.config.streaming
        
        try:
            # Construction prompt de génération
            generation_prompt = await self._build_generation_prompt(optimized_context)
            
            # Génération avec ou sans streaming
            if use_streaming:
                response_content = await self._generate_streaming_response(generation_prompt)
            else:
                response_content = await self._generate_batch_response(generation_prompt)
            
            # Extraction insights automatiques
            insights = await self._extract_financial_insights(
                optimized_context.search_results,
                optimized_context.current_query
            )
            
            # Génération suggestions contextuelles
            suggestions = await self._generate_contextual_suggestions(
                optimized_context,
                response_content
            )
            
            # Calcul métriques
            generation_time = (datetime.now() - start_time).total_seconds() * 1000
            token_count = self._estimate_token_count(response_content)
            
            # Métriques
            metrics_collector.record_histogram(
                "response_generator.generation_time_ms",
                generation_time
            )
            metrics_collector.record_histogram(
                "response_generator.response_tokens",
                token_count
            )
            metrics_collector.increment_counter("response_generator.success")
            
            return GeneratedResponse(
                content=response_content,
                insights=insights,
                suggestions=suggestions,
                metadata={
                    "user_id": user_id,
                    "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "context_tokens": optimized_context.total_tokens,
                    "compression_applied": optimized_context.compression_applied,
                    "search_results_count": len(optimized_context.search_results.data) if optimized_context.search_results else 0
                },
                generation_time_ms=int(generation_time),
                provider_used=self.config.primary_provider,
                token_count=token_count,
                streaming_used=use_streaming
            )
            
        except Exception as e:
            logger.error(f"Erreur génération réponse: {str(e)}")
            metrics_collector.increment_counter("response_generator.errors")
            
            # Réponse fallback
            return await self._generate_fallback_response(optimized_context, user_id)
    
    async def generate_response_stream(
        self,
        optimized_context: OptimizedContext,
        user_id: int
    ) -> AsyncGenerator[str, None]:
        """
        Génération streaming de réponse
        
        Args:
            optimized_context: Contexte optimisé
            user_id: ID utilisateur
            
        Yields:
            Chunks de réponse au fur et à mesure
        """
        try:
            # Construction prompt
            generation_prompt = await self._build_generation_prompt(optimized_context)
            
            # Configuration streaming
            stream_config = {
                "messages": generation_prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": True
            }
            
            # Stream depuis DeepSeek
            async for chunk in self.deepseek_client.chat_completion_stream(**stream_config):
                if chunk and "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        yield content
                        
            metrics_collector.increment_counter("response_generator.streaming_success")
            
        except Exception as e:
            logger.error(f"Erreur streaming réponse: {str(e)}")
            metrics_collector.increment_counter("response_generator.streaming_errors")
            
            # Fallback streaming
            fallback_text = f"Désolé, je rencontre un problème technique. Voici un résumé basique de vos données financières."
            for word in fallback_text.split():
                yield word + " "
                await asyncio.sleep(0.05)  # Simulation streaming
    
    async def _build_generation_prompt(self, context: OptimizedContext) -> List[Dict[str, str]]:
        """Construction prompt de génération avec contexte riche"""
        
        # System prompt adaptatif selon user profile
        system_prompt = self._build_adaptive_system_prompt(context.user_profile)
        
        # Contexte search results
        search_context = self._format_search_results_context(context.search_results)
        
        # Contexte conversationnel
        conversation_context = self._format_conversation_context(context.conversation_history)
        
        # User prompt principal
        user_prompt = f"""Contexte de la requête:
- Message utilisateur: "{context.current_query['user_message']}"
- Intention détectée: {context.current_query.get('intent_classification', {}).get('intent', 'Non définie')}
- Niveau de confiance: {context.current_query.get('intent_classification', {}).get('confidence', 0):.2f}

{search_context}

{conversation_context}

Instructions:
1. Réponds de manière {self.config.tone} et {self.config.format}
2. Utilise les données financières pour donner une réponse précise
3. {'Inclus des insights automatiques si pertinents' if self.config.include_insights else ''}
4. {'Propose des suggestions de questions de suivi' if self.config.include_suggestions else ''}
5. Adapte ton style au profil utilisateur: {context.user_profile.communication_style}

Réponds maintenant à la question de l'utilisateur:"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _build_adaptive_system_prompt(self, user_profile) -> str:
        """Construction system prompt adaptatif selon profil utilisateur"""
        
        base_prompt = """Tu es un assistant financier intelligent spécialisé dans l'analyse des données bancaires et financières.

Tes capacités principales:
- Analyse précise des transactions et soldes
- Génération d'insights financiers pertinents
- Détection de patterns et tendances dans les dépenses
- Conseils personnalisés selon le profil utilisateur
- Communication claire et professionnelle

Instructions importantes:
- UTILISE TOUJOURS les totaux précalculés fournis dans "DONNÉES FINANCIÈRES RÉSUMÉES" au début du contexte
- Ces totaux sont exacts et calculés automatiquement sur toutes les transactions
- Ne tente jamais de recalculer ou estimer des montants - utilise directement les totaux fournis
- Mentionne ces totaux de façon claire et précise dans tes réponses

Principe fondamental: Toujours baser tes réponses sur les données réelles fournies, en particulier les totaux précalculés."""
        
        # Adaptations selon profil
        language = user_profile.preferences.get("language", "fr")
        detail_level = user_profile.preferences.get("detail_level", "medium")
        style = user_profile.communication_style
        
        adaptations = []
        
        if language == "en":
            adaptations.append("- Respond in English")
        else:
            adaptations.append("- Réponds en français")
        
        if detail_level == "high":
            adaptations.append("- Fournis des analyses détaillées avec chiffres précis")
        elif detail_level == "low":
            adaptations.append("- Sois concis et va à l'essentiel")
        else:
            adaptations.append("- Equilibre entre précision et clarté")
        
        if style == "casual":
            adaptations.append("- Utilise un ton décontracté et familier")
        elif style == "formal":
            adaptations.append("- Maintiens un ton très professionnel")
        else:
            adaptations.append("- Utilise un ton professionnel mais accessible")
        
        if user_profile.preferences.get("include_insights", True):
            adaptations.append("- Inclus automatiquement des insights financiers")
        
        if adaptations:
            base_prompt += "\n\nAdaptations spécifiques:\n" + "\n".join(adaptations)
        
        return base_prompt
    
    def _format_search_results_context(self, search_results) -> str:
        """Formatage contexte résultats de recherche"""
        
        if not search_results or not search_results.data:
            return "Aucune donnée financière disponible pour cette requête."
        
        # Préparer le contexte avec agrégations automatiques en premier
        context = self._build_aggregations_summary(search_results)
        context += f"\nDétails des transactions ({search_results.total_hits} résultats au total):\n\n"
        
        # Formatage transactions
        for i, transaction in enumerate(search_results.data[:5], 1):  # Max 5 pour lisibilité, le reste via agrégations
            amount = transaction.get("amount", 0)
            currency = transaction.get("currency", "EUR")
            merchant = transaction.get("merchant_name", "N/A")
            category = transaction.get("category", "N/A")
            date = transaction.get("transaction_date", "N/A")
            
            context += f"{i}. {amount} {currency} - {merchant} ({category}) - {date}\n"
        
        if search_results.total_hits > len(search_results.data):
            remaining = search_results.total_hits - len(search_results.data)
            context += f"\n... et {remaining} autres transactions.\n"
        
        # Ajout des agrégations si disponibles
        if hasattr(search_results, 'aggregations') and search_results.aggregations:
            context += "\nStatistiques calculées :\n"
            
            aggregations = search_results.aggregations
            if 'total_amount' in aggregations:
                total_value = aggregations['total_amount'].get('value', 0)
                context += f"- Montant total : {total_value:.2f} EUR\n"
            
            if 'avg_amount' in aggregations:
                avg_value = aggregations['avg_amount'].get('value', 0)
                context += f"- Montant moyen : {avg_value:.2f} EUR\n"
            
            if 'transaction_count' in aggregations:
                count_value = aggregations['transaction_count'].get('value', 0)
                context += f"- Nombre de transactions : {count_value}\n"
        
        # Métadonnées performance
        context += f"\nRecherche effectuée en {search_results.took_ms}ms.\n"
        
        return context

    def _build_aggregations_summary(self, search_results) -> str:
        """Construit un résumé automatique des agrégations pour l'agent LLM"""
        
        if not hasattr(search_results, 'aggregations') or not search_results.aggregations:
            return "DONNÉES FINANCIÈRES RÉSUMÉES :\n(Aucune agrégation disponible)"
        
        aggregations = search_results.aggregations
        
        # Extraire les totaux
        transaction_count = aggregations.get('transaction_count', {}).get('value', 0)
        
        # Extraire total_debit et total_credit (avec structure nested)
        total_debit = 0
        total_credit = 0
        
        if 'total_debit' in aggregations and 'sum_amount' in aggregations['total_debit']:
            total_debit = aggregations['total_debit']['sum_amount'].get('value', 0)
            
        if 'total_credit' in aggregations and 'sum_amount' in aggregations['total_credit']:
            total_credit = aggregations['total_credit']['sum_amount'].get('value', 0)
        
        # Construire le résumé
        summary = "DONNÉES FINANCIÈRES RÉSUMÉES :\n"
        summary += f"• Nombre total de transactions : {transaction_count}\n"
        summary += f"• Total des débits (dépenses) : {total_debit:.2f} EUR\n" 
        summary += f"• Total des crédits (revenus) : {total_credit:.2f} EUR\n"
        
        return summary
    
    def _format_conversation_context(self, conversation_history: List) -> str:
        """Formatage contexte conversationnel"""
        
        if not conversation_history:
            return ""
        
        context = "Contexte conversationnel récent:\n"
        
        for i, turn in enumerate(conversation_history[:3], 1):  # Max 3 turns pour éviter surcharge
            user_msg = turn.user_message[:100] + "..." if len(turn.user_message) > 100 else turn.user_message
            context += f"\nÉchange {i}:\n"
            context += f"- Utilisateur: {user_msg}\n"
            
            if hasattr(turn, 'assistant_response') and turn.assistant_response:
                assistant_msg = turn.assistant_response[:100] + "..." if len(turn.assistant_response) > 100 else turn.assistant_response
                context += f"- Assistant: {assistant_msg}\n"
        
        return context + "\n"
    
    async def _generate_streaming_response(self, prompt: List[Dict[str, str]]) -> str:
        """Génération réponse avec streaming"""
        
        full_response = ""
        
        try:
            async for chunk in self.deepseek_client.chat_completion_stream(
                messages=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            ):
                if chunk and "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    full_response += content
            
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"Erreur streaming DeepSeek: {str(e)}")
            raise
    
    async def _generate_batch_response(self, prompt: List[Dict[str, str]]) -> str:
        """Génération réponse en mode batch"""
        
        try:
            response = await self.deepseek_client.chat_completion(
                messages=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            if response and "choices" in response:
                return response["choices"][0]["message"]["content"].strip()
            else:
                raise Exception("Réponse LLM invalide")
                
        except Exception as e:
            logger.error(f"Erreur génération batch: {str(e)}")
            raise
    
    async def _extract_financial_insights(
        self,
        search_results,
        current_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extraction insights financiers automatiques"""
        
        if not self.config.include_insights or not search_results:
            return []
        
        insights = []
        
        try:
            # Insight 1: Analyse montants
            if search_results.data:
                amounts = [t.get("amount", 0) for t in search_results.data if isinstance(t.get("amount"), (int, float))]
                
                if amounts:
                    total_amount = sum(amounts)
                    avg_amount = total_amount / len(amounts)
                    max_amount = max(amounts)
                    min_amount = min(amounts)
                    
                    insights.append({
                        "type": "amount_analysis",
                        "title": "Analyse des montants",
                        "data": {
                            "total": total_amount,
                            "average": round(avg_amount, 2),
                            "max": max_amount,
                            "min": min_amount,
                            "transaction_count": len(amounts)
                        },
                        "description": f"Total: {total_amount}€, Moyenne: {avg_amount:.2f}€"
                    })
            
            # Insight 2: Répartition par catégorie
            categories = {}
            for transaction in search_results.data:
                category = transaction.get("category", "Non catégorisé")
                amount = transaction.get("amount", 0)
                if isinstance(amount, (int, float)):
                    categories[category] = categories.get(category, 0) + amount
            
            if categories:
                top_category = max(categories.items(), key=lambda x: x[1])
                insights.append({
                    "type": "category_breakdown",
                    "title": "Répartition par catégorie",
                    "data": categories,
                    "description": f"Catégorie principale: {top_category[0]} ({top_category[1]}€)"
                })
            
            # Insight 3: Tendance temporelle (si applicable)
            if len(search_results.data) > 1:
                # Analyse simple des dates
                dated_transactions = [
                    t for t in search_results.data 
                    if t.get("transaction_date")
                ]
                
                if len(dated_transactions) >= 2:
                    insights.append({
                        "type": "temporal_trend",
                        "title": "Tendance temporelle",
                        "data": {
                            "period_analyzed": f"{len(dated_transactions)} transactions",
                            "date_range": f"Du {dated_transactions[-1].get('transaction_date')} au {dated_transactions[0].get('transaction_date')}"
                        },
                        "description": f"Analyse sur {len(dated_transactions)} transactions"
                    })
            
        except Exception as e:
            logger.debug(f"Erreur extraction insights: {str(e)}")
        
        return insights
    
    async def _generate_contextual_suggestions(
        self,
        context: OptimizedContext,
        response_content: str
    ) -> List[str]:
        """Génération suggestions contextuelles"""
        
        if not self.config.include_suggestions:
            return []
        
        suggestions = []
        
        try:
            # Suggestions basées sur l'intention
            intent = context.current_query.get("intent_classification", {}).get("intent", "")
            
            if intent == "BALANCE_INQUIRY":
                suggestions.extend([
                    "Voir l'évolution de mon solde sur les 3 derniers mois",
                    "Analyser mes dépenses du mois en cours",
                    "Comparer avec le mois précédent"
                ])
            
            elif intent == "TRANSACTION_SEARCH":
                suggestions.extend([
                    "Analyser les tendances de ces transactions",
                    "Voir la répartition par catégorie",
                    "Comparer avec la même période l'année dernière"
                ])
            
            elif intent == "SPENDING_ANALYSIS":
                suggestions.extend([
                    "Identifier les opportunités d'économie",
                    "Voir l'évolution mensuelle de ces dépenses",
                    "Analyser les pics de dépenses"
                ])
            
            # Suggestions génériques si pas de suggestions spécifiques
            if not suggestions:
                suggestions.extend([
                    "Analyser mes dépenses par catégorie",
                    "Voir mes plus grosses transactions du mois",
                    "Analyser l'évolution de mon budget"
                ])
            
            # Limitation à 3 suggestions max pour éviter surcharge
            return suggestions[:3]
            
        except Exception as e:
            logger.debug(f"Erreur génération suggestions: {str(e)}")
            return [
                "Analyser mes finances du mois",
                "Voir mes dépenses récentes",
                "Consulter mon solde actuel"
            ]
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimation nombre de tokens"""
        # Estimation approximative - pourrait être améliorée avec tiktoken
        return len(text.split()) * 1.3
    
    async def _generate_fallback_response(
        self,
        context: OptimizedContext,
        user_id: int
    ) -> GeneratedResponse:
        """Génération réponse fallback en cas d'erreur"""
        
        # Réponse basique basée sur les données disponibles
        if context.search_results and context.search_results.data:
            transaction_count = len(context.search_results.data)
            fallback_content = f"J'ai trouvé {transaction_count} transaction(s) correspondant à votre demande. "
            
            if transaction_count > 0:
                total_amount = sum(
                    t.get("amount", 0) for t in context.search_results.data 
                    if isinstance(t.get("amount"), (int, float))
                )
                fallback_content += f"Le montant total est de {total_amount}€."
        else:
            fallback_content = "Je n'ai pas trouvé de données correspondant à votre requête. Pouvez-vous reformuler votre demande ?"
        
        return GeneratedResponse(
            content=fallback_content,
            insights=[],
            suggestions=["Reformuler ma question", "Voir mes transactions récentes"],
            metadata={"fallback": True, "user_id": user_id},
            generation_time_ms=10,
            provider_used="fallback",
            token_count=self._estimate_token_count(fallback_content),
            streaming_used=False
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check du response generator"""
        try:
            # Test génération simple
            test_context_data = {
                "current_query": {"user_message": "Test", "intent_classification": {"intent": "TEST", "confidence": 1.0}},
                "search_results": None,
                "conversation_history": [],
                "user_profile": type('UserProfile', (), {
                    "communication_style": "professional",
                    "preferences": {"language": "fr", "detail_level": "medium"}
                })(),
                "metadata": {},
                "total_tokens": 100,
                "compression_applied": []
            }
            
            # Mock OptimizedContext pour test
            test_context = type('OptimizedContext', (), test_context_data)()
            
            # Test génération
            test_response = await self.generate_response(test_context, user_id=999)
            
            return {
                "status": "healthy",
                "component": "response_generator_agent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_generation": {
                    "success": True,
                    "provider_used": test_response.provider_used,
                    "generation_time_ms": test_response.generation_time_ms,
                    "streaming_supported": self.config.streaming
                },
                "configuration": {
                    "primary_provider": self.config.primary_provider,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "streaming": self.config.streaming,
                    "include_insights": self.config.include_insights
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "response_generator_agent",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }