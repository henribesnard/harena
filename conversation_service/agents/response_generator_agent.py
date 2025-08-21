"""
Response Generator Agent for Harena Conversation Service.

This module implements the response generation agent using AutoGen v0.4 framework.
It synthesizes search results into natural, contextual responses with financial
expertise and conversational intelligence.

Key Features:
- AutoGen AssistantAgent with specialized financial response generation
- Intent-specific response templates with few-shot learning
- Multi-modal response support: text, tables, charts, streaming
- Contextual conversation management with memory integration
- Intelligent error handling and empty result management
- Tone adaptation based on user patterns and intent type

Author: Harena Conversation Team
Created: 2025-01-31
Version: 1.0.0 - AutoGen v0.4 + Natural Language Generation
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import uuid

# AutoGen imports
from autogen import AssistantAgent
from openai import AsyncOpenAI

# Local imports
from ..models.core_models import (
    IntentType, AgentResponse, ConversationState,
    ResponseFormat, ResponseTone, HarenaValidators
)
from ..models.agent_models import AgentConfig
from ..base_agent import BaseFinancialAgent
from ..prompts.response_prompts import (
    RESPONSE_GENERATION_SYSTEM_PROMPT, RESPONSE_FEW_SHOT_EXAMPLES,
    INTENT_RESPONSE_TEMPLATES, FINANCIAL_FORMATTING_RULES
)
from ..core.cache_manager import CacheManager
from ..core.metrics_collector import MetricsCollector
from ..utils.logging import get_structured_logger

__all__ = ["ResponseGeneratorAgent", "FinancialFormatter", "ConversationContextManager"]

# Configure logging
logger = get_structured_logger(__name__)

# ================================
# FINANCIAL FORMATTING SYSTEM
# ================================

class FinancialFormatter:
    """
    Specialized formatter for financial data presentation.
    
    Handles currency formatting, number abbreviations, percentages,
    and localized financial conventions for Harena users.
    """
    
    # Currency formatting rules
    CURRENCY_FORMATS = {
        "EUR": {"symbol": "€", "position": "suffix", "decimal_places": 2},
        "USD": {"symbol": "$", "position": "prefix", "decimal_places": 2},
        "GBP": {"symbol": "£", "position": "prefix", "decimal_places": 2}
    }
    
    # Number abbreviation thresholds (French conventions)
    ABBREVIATION_THRESHOLDS = [
        (1_000_000_000, "Md€"),  # Milliards
        (1_000_000, "M€"),       # Millions  
        (1_000, "k€")            # Milliers
    ]
    
    @classmethod
    def format_amount(
        cls,
        amount: Union[float, int, str, Decimal],
        currency: str = "EUR",
        abbreviated: bool = False,
        precision: int = 2
    ) -> str:
        """Format financial amount with proper currency conventions."""
        
        try:
            # Convert to Decimal for precise calculations
            if isinstance(amount, str):
                decimal_amount = Decimal(amount)
            else:
                decimal_amount = Decimal(str(amount))
            
            # Handle negative amounts
            is_negative = decimal_amount < 0
            abs_amount = abs(decimal_amount)
            
            # Apply abbreviation if requested and amount is large
            if abbreviated and abs_amount >= 1000:
                formatted_number = cls._abbreviate_number(float(abs_amount))
            else:
                # Round to specified precision
                rounded_amount = abs_amount.quantize(
                    Decimal('0.' + '0' * precision), 
                    rounding=ROUND_HALF_UP
                )
                formatted_number = f"{rounded_amount:,.{precision}f}".replace(',', ' ')
            
            # Apply currency formatting
            currency_config = cls.CURRENCY_FORMATS.get(currency, cls.CURRENCY_FORMATS["EUR"])
            symbol = currency_config["symbol"]
            position = currency_config["position"]
            
            if position == "prefix":
                result = f"{symbol}{formatted_number}"
            else:
                result = f"{formatted_number} {symbol}"
            
            # Add negative sign
            if is_negative:
                result = f"- {result}"
            
            return result
            
        except (ValueError, TypeError, ArithmeticError):
            # Fallback to string representation
            return f"{amount} {currency}"
    
    @classmethod
    def _abbreviate_number(cls, amount: float) -> str:
        """Abbreviate large numbers using French conventions."""
        
        for threshold, suffix in cls.ABBREVIATION_THRESHOLDS:
            if amount >= threshold:
                abbreviated = amount / threshold
                if abbreviated >= 100:
                    return f"{abbreviated:.0f} {suffix}"
                elif abbreviated >= 10:
                    return f"{abbreviated:.1f} {suffix}"
                else:
                    return f"{abbreviated:.2f} {suffix}"
        
        # No abbreviation needed
        return f"{amount:,.0f}".replace(',', ' ')
    
    @classmethod
    def format_percentage(cls, value: float, precision: int = 1) -> str:
        """Format percentage with proper precision."""
        return f"{value:.{precision}f} %"
    
    @classmethod
    def format_date_range(cls, start_date: str, end_date: str) -> str:
        """Format date range in French conventions."""
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            # Same month
            if start.month == end.month and start.year == end.year:
                return f"{start.day} - {end.day} {cls._get_french_month(end.month)} {end.year}"
            
            # Different months, same year
            elif start.year == end.year:
                return f"{start.day} {cls._get_french_month(start.month)} - {end.day} {cls._get_french_month(end.month)} {end.year}"
            
            # Different years
            else:
                return f"{start.day} {cls._get_french_month(start.month)} {start.year} - {end.day} {cls._get_french_month(end.month)} {end.year}"
                
        except (ValueError, AttributeError):
            return f"{start_date} - {end_date}"
    
    @classmethod
    def _get_french_month(cls, month_num: int) -> str:
        """Get French month name."""
        months = [
            "", "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        return months[month_num] if 1 <= month_num <= 12 else str(month_num)
    
    @classmethod
    def format_transaction_summary(cls, transactions: List[Dict[str, Any]]) -> Dict[str, str]:
        """Format transaction summary with key metrics."""
        
        if not transactions:
            return {
                "total_amount": "0,00 €",
                "transaction_count": "0",
                "average_amount": "0,00 €",
                "date_range": "Aucune transaction"
            }
        
        # Calculate metrics
        total_amount = sum(float(t.get('amount', 0)) for t in transactions)
        count = len(transactions)
        average_amount = total_amount / count if count > 0 else 0
        
        # Date range
        dates = [t.get('date') for t in transactions if t.get('date')]
        if dates:
            sorted_dates = sorted(dates)
            date_range = cls.format_date_range(sorted_dates[0], sorted_dates[-1])
        else:
            date_range = "Dates non disponibles"
        
        return {
            "total_amount": cls.format_amount(total_amount),
            "transaction_count": f"{count:,}".replace(',', ' '),
            "average_amount": cls.format_amount(average_amount),
            "date_range": date_range
        }

# ================================
# CONVERSATION CONTEXT MANAGER
# ================================

@dataclass
class ConversationTurn:
    """Single conversation turn with context."""
    turn_id: str
    user_message: str
    intent: IntentType
    entities: List[Dict[str, Any]]
    response: str
    timestamp: datetime
    search_results_count: int = 0

class ConversationContextManager:
    """
    Manages conversation context and memory for response generation.
    
    Tracks conversation history, user patterns, and contextual cues
    to generate more relevant and personalized responses.
    """
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history: List[ConversationTurn] = []
        self.user_patterns: Dict[str, Any] = {}
        self.session_metadata: Dict[str, Any] = {}
    
    def add_turn(
        self,
        user_message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        response: str,
        search_results_count: int = 0
    ) -> str:
        """Add conversation turn and return turn ID."""
        
        turn_id = str(uuid.uuid4())
        turn = ConversationTurn(
            turn_id=turn_id,
            user_message=user_message,
            intent=intent,
            entities=entities,
            response=response,
            timestamp=datetime.now(),
            search_results_count=search_results_count
        )
        
        self.conversation_history.append(turn)
        
        # Maintain max turns limit
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history.pop(0)
        
        # Update user patterns
        self._update_user_patterns(turn)
        
        return turn_id
    
    def get_context_for_response(self) -> Dict[str, Any]:
        """Get relevant context for response generation."""
        
        if not self.conversation_history:
            return {}
        
        recent_turns = self.conversation_history[-3:]  # Last 3 turns
        
        return {
            "conversation_length": len(self.conversation_history),
            "recent_intents": [turn.intent.value for turn in recent_turns],
            "recent_entities": [
                entity.get('entity_type') 
                for turn in recent_turns 
                for entity in turn.entities
            ],
            "last_user_message": recent_turns[-1].user_message if recent_turns else "",
            "user_patterns": self.user_patterns,
            "session_metadata": self.session_metadata
        }
    
    def _update_user_patterns(self, turn: ConversationTurn):
        """Update user patterns based on conversation turn."""
        
        # Track frequent intents
        intent_counts = self.user_patterns.setdefault("intent_frequency", {})
        intent_counts[turn.intent.value] = intent_counts.get(turn.intent.value, 0) + 1
        
        # Track frequent entities
        for entity in turn.entities:
            entity_type = entity.get('entity_type')
            if entity_type:
                entity_counts = self.user_patterns.setdefault("entity_frequency", {})
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Track conversation preferences
        if turn.search_results_count > 0:
            self.user_patterns["prefers_detailed_results"] = True
        
        # Track temporal patterns (time of day, etc.)
        hour = turn.timestamp.hour
        time_patterns = self.user_patterns.setdefault("time_patterns", {})
        time_patterns[hour] = time_patterns.get(hour, 0) + 1

# ================================
# MAIN RESPONSE GENERATOR AGENT
# ================================

class ResponseGeneratorAgent(BaseFinancialAgent):
    """
    Response Generation Agent using AutoGen v0.4 framework.
    
    Synthesizes search results into natural, contextual responses with
    financial expertise, conversation intelligence, and user personalization.
    """
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        cache_manager: Optional[CacheManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        enable_streaming: bool = False
    ):
        """Initialize Response Generator Agent."""
        
        config = AgentConfig(
            name="response_generator",
            model_name="gpt-4-turbo",  # Use GPT-4 Turbo for better response quality
            temperature=0.3,  # Moderate creativity for natural responses
            max_tokens=1000,
            timeout_seconds=15,
            system_message=RESPONSE_GENERATION_SYSTEM_PROMPT,
            few_shot_examples=RESPONSE_FEW_SHOT_EXAMPLES
        )
        
        super().__init__(config, openai_client, cache_manager, metrics_collector)
        
        self.enable_streaming = enable_streaming
        self.formatter = FinancialFormatter()
        self.context_manager = ConversationContextManager()
        
        # Response generation statistics
        self.total_responses = 0
        self.successful_responses = 0
        self.empty_result_responses = 0
        self.error_responses = 0
        self.streaming_responses = 0
        
        logger.info(
            "Response Generator Agent initialized",
            model=config.model_name,
            streaming_enabled=enable_streaming,
            temperature=config.temperature
        )
    
    async def _process_implementation(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate contextual response from search results and conversation state."""
        
        user_message = input_data.get("user_message", "")
        intent_result = input_data.get("intent", {})
        entities = input_data.get("entities", [])
        search_results = input_data.get("search_results", {})
        context = input_data.get("context", {})
        
        if not user_message or not intent_result:
            raise ValueError("user_message and intent are required")
        
        intent_type = intent_result.get("intent")
        intent_confidence = intent_result.get("confidence", 0.0)
        
        try:
            # Update conversation context
            conversation_context = self.context_manager.get_context_for_response()
            
            # Determine response strategy based on search results
            response_strategy = self._determine_response_strategy(
                intent_type, search_results, conversation_context
            )
            
            # Generate response based on strategy
            if response_strategy == "detailed_financial":
                response_data = await self._generate_detailed_financial_response(
                    user_message, intent_type, entities, search_results, conversation_context
                )
            elif response_strategy == "empty_results":
                response_data = await self._generate_empty_results_response(
                    user_message, intent_type, entities, conversation_context
                )
            elif response_strategy == "error_handling":
                response_data = await self._generate_error_response(
                    user_message, intent_type, search_results.get("error", "Unknown error")
                )
            else:
                response_data = await self._generate_standard_response(
                    user_message, intent_type, entities, search_results, conversation_context
                )
            
            # Add conversation turn to context
            turn_id = self.context_manager.add_turn(
                user_message=user_message,
                intent=IntentType(intent_type),
                entities=entities,
                response=response_data.get("response", ""),
                search_results_count=len(search_results.get("results", []))
            )
            
            self.total_responses += 1
            self.successful_responses += 1
            
            # Enhanced response metadata
            response_data.update({
                "generation_metadata": {
                    "turn_id": turn_id,
                    "intent": intent_type,
                    "intent_confidence": intent_confidence,
                    "response_strategy": response_strategy,
                    "entities_count": len(entities),
                    "search_results_count": len(search_results.get("results", [])),
                    "conversation_turn": len(self.context_manager.conversation_history),
                    "formatting_applied": True,
                    "context_used": bool(conversation_context)
                }
            })
            
            return response_data
            
        except Exception as e:
            self.error_responses += 1
            logger.error(
                "Response generation failed",
                intent=intent_type,
                user_message=user_message[:100],
                error=str(e),
                exc_info=True
            )
            
            # Fallback response
            return await self._generate_fallback_response(user_message, intent_type, str(e))
    
    def _determine_response_strategy(
        self,
        intent_type: str,
        search_results: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> str:
        """Determine appropriate response generation strategy."""
        
        # Check for errors first
        if search_results.get("error"):
            return "error_handling"
        
        # Check for empty results
        results = search_results.get("results", [])
        if not results:
            return "empty_results"
        
        # Determine if detailed financial analysis is needed
        financial_analysis_intents = {
            "SPENDING_ANALYSIS", "CATEGORY_ANALYSIS", "MERCHANT_ANALYSIS",
            "TEMPORAL_ANALYSIS", "BUDGET_ANALYSIS", "COMPARISON_ANALYSIS"
        }
        
        if intent_type in financial_analysis_intents and len(results) > 1:
            return "detailed_financial"
        
        # Standard response for other cases
        return "standard"
    
    async def _generate_detailed_financial_response(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        search_results: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed financial analysis response."""
        
        results = search_results.get("results", [])
        
        # Format financial summary
        summary = self.formatter.format_transaction_summary(results)
        
        # Build comprehensive prompt for detailed analysis
        prompt = self._build_detailed_analysis_prompt(
            user_message, intent_type, entities, results, summary, conversation_context
        )
        
        # Generate response with LLM
        if self.enable_streaming:
            response_content = await self._generate_streaming_response(prompt)
            self.streaming_responses += 1
        else:
            llm_response = await self._call_openai(prompt)
            response_content = llm_response["content"]
        
        return {
            "response": response_content,
            "response_type": "detailed_financial_analysis",
            "financial_summary": summary,
            "transaction_count": len(results),
            "includes_charts": self._should_include_charts(intent_type, results),
            "tone": "professional_analytical"
        }
    
    async def _generate_empty_results_response(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate helpful response for empty search results."""
        
        self.empty_result_responses += 1
        
        # Build prompt for empty results handling
        prompt = self._build_empty_results_prompt(
            user_message, intent_type, entities, conversation_context
        )
        
        llm_response = await self._call_openai(prompt)
        
        return {
            "response": llm_response["content"],
            "response_type": "empty_results_guidance",
            "suggestions": self._generate_alternative_suggestions(intent_type, entities),
            "tone": "helpful_supportive"
        }
    
    async def _generate_error_response(
        self,
        user_message: str,
        intent_type: str,
        error_message: str
    ) -> Dict[str, Any]:
        """Generate user-friendly error response."""
        
        # Template-based error response (don't use LLM for errors)
        error_templates = {
            "timeout": "Je rencontre une difficulté temporaire pour accéder à vos données. Pouvez-vous réessayer dans quelques instants ?",
            "connection": "Je ne peux pas me connecter au service de données actuellement. Veuillez réessayer plus tard.",
            "default": "Je rencontre une difficulté technique pour traiter votre demande. Pouvez-vous reformuler ou réessayer ?"
        }
        
        # Determine error type
        error_type = "default"
        if "timeout" in error_message.lower():
            error_type = "timeout"
        elif "connection" in error_message.lower():
            error_type = "connection"
        
        return {
            "response": error_templates[error_type],
            "response_type": "error_handling",
            "error_type": error_type,
            "tone": "apologetic_helpful",
            "suggestions": [
                "Réessayer la même question",
                "Reformuler différemment",
                "Demander de l'aide générale"
            ]
        }
    
    async def _generate_standard_response(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        search_results: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate standard conversational response."""
        
        results = search_results.get("results", [])
        
        # Build standard response prompt
        prompt = self._build_standard_response_prompt(
            user_message, intent_type, entities, results, conversation_context
        )
        
        llm_response = await self._call_openai(prompt)
        
        return {
            "response": llm_response["content"],
            "response_type": "standard_conversational",
            "transaction_count": len(results),
            "tone": "conversational_friendly"
        }
    
    def _build_detailed_analysis_prompt(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        summary: Dict[str, str],
        conversation_context: Dict[str, Any]
    ) -> str:
        """Build prompt for detailed financial analysis."""
        
        prompt_parts = [
            f"Demande utilisateur : {user_message}",
            f"Type d'analyse : {intent_type}",
            f"Nombre de transactions : {len(results)}"
        ]
        
        # Add financial summary
        summary_text = f"""Résumé financier :
- Montant total : {summary['total_amount']}
- Nombre de transactions : {summary['transaction_count']}
- Montant moyen : {summary['average_amount']}
- Période : {summary['date_range']}"""
        prompt_parts.append(summary_text)
        
        # Add sample transactions (first 3)
        if results:
            sample_transactions = "Exemples de transactions :\n"
            for i, tx in enumerate(results[:3]):
                sample_transactions += f"{i+1}. {tx.get('merchant_name', 'N/A')} - {self.formatter.format_amount(tx.get('amount', 0))} - {tx.get('date', 'N/A')}\n"
            prompt_parts.append(sample_transactions)
        
        # Add conversation context
        if conversation_context.get("recent_intents"):
            context_text = f"Contexte conversation : {', '.join(conversation_context['recent_intents'][-2:])}"
            prompt_parts.append(context_text)
        
        prompt_parts.extend([
            "",
            "Génère une analyse financière détaillée et professionnelle.",
            "Structure ta réponse avec :",
            "1. Résumé principal en 1-2 phrases",
            "2. Analyse détaillée avec insights pertinents", 
            "3. Observations ou recommandations si pertinentes",
            "",
            "Utilise un ton professionnel mais accessible. Sois précis avec les chiffres."
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_empty_results_prompt(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        conversation_context: Dict[str, Any]
    ) -> str:
        """Build prompt for empty results response."""
        
        prompt_parts = [
            f"Demande utilisateur : {user_message}",
            f"Intention : {intent_type}",
            "Situation : Aucun résultat trouvé"
        ]
        
        # Add extracted entities
        if entities:
            entities_text = "Critères recherchés : " + ", ".join([
                f"{e.get('entity_type')}:{e.get('normalized_value', e.get('raw_value'))}"
                for e in entities
            ])
            prompt_parts.append(entities_text)
        
        prompt_parts.extend([
            "",
            "Génère une réponse utile et empathique expliquant l'absence de résultats.",
            "Inclus des suggestions alternatives constructives.",
            "Reste positif et propose des pistes pour aider l'utilisateur.",
            "Utilise un ton bienveillant et supportif."
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_standard_response_prompt(
        self,
        user_message: str,
        intent_type: str,
        entities: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        conversation_context: Dict[str, Any]
    ) -> str:
        """Build prompt for standard conversational response."""
        
        prompt_parts = [
            f"Demande utilisateur : {user_message}",
            f"Intention : {intent_type}",
            f"Résultats trouvés : {len(results)} transaction(s)"
        ]
        
        # Add key results highlights
        if results:
            if len(results) == 1:
                tx = results[0]
                highlight = f"Transaction : {tx.get('merchant_name', 'N/A')} - {self.formatter.format_amount(tx.get('amount', 0))} - {tx.get('date', 'N/A')}"
            else:
                total_amount = sum(float(tx.get('amount', 0)) for tx in results)
                highlight = f"Résumé : {len(results)} transactions pour un total de {self.formatter.format_amount(total_amount)}"
            
            prompt_parts.append(highlight)
        
        prompt_parts.extend([
            "",
            "Génère une réponse conversationnelle naturelle et claire.",
            "Réponds directement à la question posée.",
            "Utilise un ton amical et professionnel.",
            "Sois concis mais informatif."
        ])
        
        return "\n".join(prompt_parts)
    
    async def _generate_streaming_response(self, prompt: str) -> str:
        """Generate streaming response for long content."""
        
        # Note: This would implement actual streaming with AsyncOpenAI
        # For now, return standard response
        llm_response = await self._call_openai(prompt)
        return llm_response["content"]
    
    def _should_include_charts(self, intent_type: str, results: List[Dict[str, Any]]) -> bool:
        """Determine if response should include chart recommendations."""
        
        chart_worthy_intents = {
            "TEMPORAL_ANALYSIS", "CATEGORY_ANALYSIS", "SPENDING_ANALYSIS", "BUDGET_ANALYSIS"
        }
        
        return intent_type in chart_worthy_intents and len(results) >= 5
    
    def _generate_alternative_suggestions(
        self,
        intent_type: str,
        entities: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate alternative search suggestions for empty results."""
        
        suggestions = []
        
        # Intent-specific suggestions
        if intent_type == "CATEGORY_ANALYSIS":
            suggestions.extend([
                "Essayez une autre catégorie (restaurant, transport, shopping)",
                "Élargissez la période de recherche",
                "Vérifiez l'orthographe de la catégorie"
            ])
        elif intent_type == "MERCHANT_ANALYSIS":
            suggestions.extend([
                "Vérifiez le nom du marchand",
                "Essayez une recherche partielle (ex: 'Amazon' au lieu d'Amazon France')",
                "Regardez vos transactions récentes"
            ])
        elif intent_type == "TEMPORAL_ANALYSIS":
            suggestions.extend([
                "Essayez une autre période",
                "Vérifiez le format de date",
                "Consultez un mois précédent"
            ])
        else:
            suggestions.extend([
                "Reformulez votre question différemment",
                "Vérifiez la période concernée",
                "Consultez vos dernières transactions"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    async def _generate_fallback_response(
        self,
        user_message: str,
        intent_type: str,
        error: str
    ) -> Dict[str, Any]:
        """Generate fallback response for critical errors."""
        
        fallback_responses = {
            "BALANCE_INQUIRY": "Je ne peux pas accéder à votre solde actuellement. Veuillez réessayer dans quelques instants.",
            "TRANSACTION_SEARCH": "Je rencontre une difficulté pour rechercher vos transactions. Pouvez-vous reformuler votre demande ?",
            "CATEGORY_ANALYSIS": "L'analyse par catégorie n'est pas disponible en ce moment. Essayez une recherche plus simple.",
            "default": "Je rencontre une difficulté pour traiter votre demande. Comment puis-je vous aider autrement ?"
        }
        
        response_text = fallback_responses.get(intent_type, fallback_responses["default"])
        
        return {
            "response": response_text,
            "response_type": "fallback_error",
            "error_handled": True,
            "tone": "apologetic",
            "suggestions": [
                "Réessayer la même question",
                "Poser une question plus simple",
                "Demander de l'aide"
            ]
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get response generation statistics."""
        
        success_rate = self.successful_responses / max(self.total_responses, 1)
        empty_rate = self.empty_result_responses / max(self.total_responses, 1)
        error_rate = self.error_responses / max(self.total_responses, 1)
        streaming_rate = self.streaming_responses / max(self.total_responses, 1)
        
        return {
            "agent_name": self.config.name,
            "total_responses": self.total_responses,
            "successful_responses": self.successful_responses,
            "success_rate": success_rate,
            "empty_result_responses": self.empty_result_responses,
            "empty_result_rate": empty_rate,
            "error_responses": self.error_responses,
            "error_rate": error_rate,
            "streaming_responses": self.streaming_responses,
            "streaming_rate": streaming_rate,
            "conversation_turns": len(self.context_manager.conversation_history),
            "performance_metrics": {
                "response_quality": success_rate,
                "error_resilience": 1.0 - error_rate,
                "user_experience": 1.0 - (empty_rate + error_rate),
                "conversation_continuity": min(1.0, len(self.context_manager.conversation_history) / 10)
            }
        }