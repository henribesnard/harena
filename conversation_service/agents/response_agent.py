"""
Response Generation Agent for contextual financial conversations.

This agent generates natural, contextual responses based on search results
and conversation history. It formats financial data in user-friendly ways
and maintains conversation continuity.

Classes:
    - ResponseAgent: Main response generation agent
    - ResponseFormatter: Helper class for formatting financial data

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Contextual Response Generation
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_financial_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig, AgentResponse
from ..models.service_contracts import SearchServiceResponse
from ..models.conversation_models import ConversationContext
from ..core.deepseek_client import DeepSeekClient, DeepSeekResponse

logger = logging.getLogger(__name__)

# Message returned to the user when the search step fails
SEARCH_ERROR_MESSAGE = "La recherche n'a pas pu être effectuée."


class ResponseFormatter:
    """Helper class for formatting financial data in responses."""
    
    @staticmethod
    def format_amount(amount: float, currency: str = "EUR") -> str:
        """Format monetary amount for display."""
        if amount >= 0:
            return f"+{amount:.2f} {currency}"
        else:
            return f"{amount:.2f} {currency}"
    
    @staticmethod
    def format_transaction_list(transactions: List[Dict]) -> str:
        """Format a list of transactions for display."""
        if not transactions:
            return "Aucune transaction trouvée."
        
        transactions_sorted = sorted(
            transactions, key=lambda t: t.get("date", ""), reverse=True
        )

        formatted_lines = []
        for i, transaction in enumerate(transactions_sorted[:5], 1):
            date = transaction.get("date", "Date inconnue")
            amount = transaction.get("amount", 0.0)
            merchant = transaction.get("merchant_name", "Marchand inconnu")
            category = transaction.get("category_name", "")

            formatted_amount = ResponseFormatter.format_amount(amount)
            line = f"{i}. {date} | {formatted_amount} | {merchant}"
            if category:
                line += f" ({category})"
            formatted_lines.append(line)

        result = "\n".join(formatted_lines)

        total = len(transactions_sorted)
        if total > 5:
            result += f"\n... et {total - 5} autres transactions (total {total})"

        return result
    
    @staticmethod
    def format_summary_stats(search_response: SearchServiceResponse) -> str:
        """Format summary statistics from search response."""
        if not search_response.aggregations:
            return ""
        agg = search_response.aggregations

        tx_buckets = agg.get("transaction_type_terms", {}).get("buckets", [])
        if tx_buckets:
            lines = ["**Montants par type de transaction:**"]
            total = 0.0
            for bucket in tx_buckets:
                amount = bucket.get("amount_sum", {}).get("value", 0.0)
                total += amount
                tx_type = bucket.get("key", "inconnu")
                count = bucket.get("doc_count", 0)
                lines.append(
                    f"• {tx_type}: {ResponseFormatter.format_amount(amount)} ({count} transactions)"
                )
            lines.append(f"Total: {ResponseFormatter.format_amount(total)}")
            return "\n".join(lines)

        amount_total = agg.get("amount_sum", {}).get("value")
        if amount_total is not None:
            return f"**Total des montants:** {ResponseFormatter.format_amount(amount_total)}"
        return ""
    
    @staticmethod
    def format_category_breakdown(search_response: SearchServiceResponse) -> str:
        """Format category breakdown if available."""
        if (
            not search_response.aggregations
            or "category_breakdown" not in search_response.aggregations
        ):
            return ""

        breakdown = search_response.aggregations.get("category_breakdown", [])
        lines = ["**Répartition par catégorie:**"]
        
        for category_data in breakdown[:5]:  # Top 5 categories
            category = category_data.get("category", "Inconnue")
            count = category_data.get("count", 0)
            amount = category_data.get("total_amount", 0.0)
            formatted_amount = ResponseFormatter.format_amount(amount)
            
            lines.append(f"• {category}: {count} transactions, {formatted_amount}")
        
        return "\n".join(lines)


class ResponseAgent(BaseFinancialAgent):
    """
    Agent for generating contextual responses to financial queries.
    
    This agent takes search results and conversation context to generate
    natural, helpful responses that:
    1. Format financial data clearly
    2. Provide relevant insights
    3. Maintain conversation continuity
    4. Suggest follow-up actions when appropriate
    
    Attributes:
        formatter: Helper for formatting financial data
        conversation_memory: Simple conversation context storage
    """
    
    def __init__(self, deepseek_client: DeepSeekClient, config: Optional[AgentConfig] = None):
        """
        Initialize the response generation agent.
        
        Args:
            deepseek_client: Configured DeepSeek client
            config: Optional agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="response_agent",
                model_client_config={
                    "model": "deepseek-chat",
                    "api_key": deepseek_client.api_key,
                    "base_url": deepseek_client.base_url
                },
                system_message=self._get_system_message(),
                max_consecutive_auto_reply=1,
                description="Contextual response generation agent for financial conversations",
                temperature=0.3,  # Slightly higher for more natural responses
                max_tokens=300,   # Limit tokens for faster responses
                timeout_seconds=15
            )
        
        super().__init__(
            name=config.name,
            config=config,
            deepseek_client=deepseek_client
        )
        
        self.formatter = ResponseFormatter()
        self.conversation_memory = {}  # Simple in-memory storage
        
        logger.info("Initialized ResponseAgent with conversation context")
    
    async def _execute_operation(self, input_data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """
        Execute response generation operation.

        Args:
            input_data: Dict containing 'user_message', 'search_results', 'context',
                and optional 'search_error'
            user_id: ID of the requesting user

        Returns:
            Dict with generated response and metadata
        """
        user_message = input_data.get("user_message", "")
        context = input_data.get("context")
        if input_data.get("search_error"):
            return {
                "content": SEARCH_ERROR_MESSAGE,
                "metadata": {"error": "search_failed", "fallback_used": True},
                "confidence_score": 0.0,
            }

        search_results = input_data.get("search_results", {})

        if not search_results:
            raise ValueError("search_results are required for response generation")

        return await self.generate_response(user_message, search_results, user_id, context)

    async def generate_response(
        self,
        user_message: str,
        search_results: Any,
        user_id: int,
        context: Optional[ConversationContext] = None,
    ) -> Dict[str, Any]:
        """
        Generate a contextual response based on search results.

        Args:
            user_message: Original user message
            search_results: Results from SearchQueryAgent (AgentResponse or dict)
            user_id: ID of the requesting user
            context: Optional conversation context

        Returns:
            Dictionary containing the generated response
        """
        start_time = time.perf_counter()
        
        try:
            # Normalize search results to dict for easier consumption
            if isinstance(search_results, AgentResponse):
                search_results_dict = search_results.dict()
            elif isinstance(search_results, dict):
                search_results_dict = search_results
            else:
                raise ValueError("search_results must be AgentResponse or dict")

            # Extract search response from normalized results
            search_response_data = (
                search_results_dict.get("metadata", {}).get("search_response", {})
            )
            if isinstance(search_response_data, SearchServiceResponse):
                search_response = search_response_data
            else:
                search_response = SearchServiceResponse(**search_response_data)

            # Format search results into readable text
            formatted_results = await self._format_search_results(search_response)
            logger.debug("Formatted results preview: %s", formatted_results[:200])

            # Determine if a date filter was applied to the search query
            search_query = search_results_dict.get("metadata", {}).get("search_query", {})
            has_date_filter = bool(
                search_query.get("filters", {}).get("date") if isinstance(search_query, dict) else False
            )

            # Get conversation context
            conversation_context = await self._get_conversation_context(context)

            # Log context summary and search results count
            context_summary = (
                conversation_context[:200] if conversation_context else ""
            )
            logger.info(
                "Generating response with context summary: %s | returned results: %s",
                context_summary,
                search_response.response_metadata.returned_results,
            )

            # Generate contextual response using AI
            ai_response = await self._generate_ai_response(
                user_message, formatted_results, conversation_context, user_id, has_date_filter
            )

            # Extract token usage if available and log generated response
            usage = getattr(getattr(ai_response.raw, "usage", None), "__dict__", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            logger.info(
                "Generated response: %s | transactions used: %s | total tokens used: %s",
                ai_response.content,
                search_response.response_metadata.returned_results,
                total_tokens,
            )

            # Update conversation context
            if context:
                await self._update_conversation_context(
                    context.conversation_id, user_message, ai_response.content
                )

            execution_time = (time.perf_counter() - start_time) * 1000

            return {
                "content": ai_response.content,
                "metadata": {
                    "formatted_results": formatted_results,
                    "search_stats": {
                        "total_results": search_response.response_metadata.returned_results,
                        "processing_time_ms": search_response.response_metadata.processing_time_ms
                    },
                    "response_generation_time_ms": execution_time,
                    "conversation_updated": context is not None
                },
                "confidence_score": 0.85,  # Base confidence for response generation
                "token_usage": {
                    "prompt_tokens": prompt_tokens or 150,
                    "completion_tokens": completion_tokens or 80,
                    "total_tokens": total_tokens or 230,
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            
            # Fallback response
            fallback_response = self._generate_fallback_response(user_message, search_results)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "content": fallback_response,
                "metadata": {
                    "fallback_used": True,
                    "error": str(e),
                    "execution_time_ms": execution_time
                },
                "confidence_score": 0.3
            }
    
    async def _format_search_results(self, search_response: SearchServiceResponse) -> str:
        """
        Format search results into readable text.
        
        Args:
            search_response: Response from Search Service
            
        Returns:
            Formatted string representation of results
        """
        sections = []
        
        # Summary statistics
        summary = self.formatter.format_summary_stats(search_response)
        if summary:
            sections.append(summary)
        
        # Transaction list
        if search_response.results:
            transactions = [result.to_source() for result in search_response.results]
            transaction_list = self.formatter.format_transaction_list(transactions)
            sections.append(f"\n**Détail des transactions:**\n{transaction_list}")
        
        # Category breakdown
        category_breakdown = self.formatter.format_category_breakdown(search_response)
        if category_breakdown:
            sections.append(f"\n{category_breakdown}")
        
        return "\n".join(sections) if sections else "Aucun résultat trouvé."
    
    async def _get_conversation_context(self, context: Optional[ConversationContext] = None) -> str:
        """
        Get conversation context for response generation.
        
        Args:
            context: Optional conversation context object
            
        Returns:
            String representation of conversation context
        """
        if not context:
            return "Nouvelle conversation."
        
        # Build context summary
        context_parts = []
        
        if context.turns:
            recent_turns = context.turns[-3:]  # Last 3 turns
            context_parts.append(f"Historique récent ({len(recent_turns)} tours):")
            
            for turn in recent_turns:
                context_parts.append(f"- Utilisateur: {turn.user_message[:100]}...")
                if turn.intent_result and turn.intent_result.intent_type:
                    context_parts.append(f"  Intent: {turn.intent_result.intent_type}")
        
        if context.context_summary:
            context_parts.append(f"Résumé: {context.context_summary}")
        
        
        return "\n".join(context_parts) if context_parts else "Contexte minimal disponible."
    
    async def _generate_ai_response(
        self,
        user_message: str,
        formatted_results: str,
        conversation_context: str,
        user_id: int,
        has_date_filter: bool,
    ) -> DeepSeekResponse:
        """
        Generate AI response using DeepSeek.

        Args:
            user_message: Original user message
            formatted_results: Formatted search results
            conversation_context: Conversation context
            user_id: ID of the requesting user

        Returns:
            DeepSeekResponse containing response and raw completion
        """
        # Prepare prompt for response generation
        prompt = self._build_response_prompt(
            user_message, formatted_results, conversation_context, has_date_filter
        )

        try:
            logger.debug("Generating AI response for user_id=%s", user_id)
            response = await self.deepseek_client.generate_response(
                messages=[
                    {"role": "system", "content": self._get_response_generation_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                user=str(user_id),
                use_cache=True,
            )

            response.content = response.content.strip()
            return response

        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            fallback = self._generate_fallback_response(
                user_message, {"formatted_results": formatted_results}
            )
            return DeepSeekResponse(content=fallback, raw=None)
    
    def _build_response_prompt(
        self,
        user_message: str,
        formatted_results: str,
        conversation_context: str,
        has_date_filter: bool,
    ) -> str:
        """Build the complete prompt for response generation."""
        current_date = datetime.utcnow().strftime("%d/%m/%Y")
        no_results_instruction = ""
        if "Aucun résultat" in formatted_results:
            if has_date_filter:
                no_results_instruction = (
                    "Aucune transaction trouvée pour la période spécifiée."
                )
            else:
                no_results_instruction = (
                    "Aucune transaction correspondante trouvée."
                )

        return f"""Nous sommes le {current_date}.

Question utilisateur: "{user_message}"

Résultats de recherche:
{formatted_results}

Contexte conversationnel:
{conversation_context}

{no_results_instruction}

Génère une réponse naturelle et utile qui:
1. Répond directement à la question
2. Présente les données de manière claire
3. Propose des insights pertinents
4. Suggère des actions de suivi si approprié"""
    
    async def _update_conversation_context(self, conversation_id: str, 
                                         user_msg: str, response: str) -> None:
        """
        Update conversation context with new turn.
        
        Args:
            conversation_id: Conversation identifier
            user_msg: User message
            response: Generated response
        """
        try:
            if conversation_id not in self.conversation_memory:
                self.conversation_memory[conversation_id] = {
                    "turns": [],
                    "last_updated": datetime.utcnow()
                }
            
            # Add new turn
            self.conversation_memory[conversation_id]["turns"].append({
                "user_message": user_msg,
                "assistant_response": response,
                "timestamp": datetime.utcnow()
            })
            
            # Keep only last 10 turns
            turns = self.conversation_memory[conversation_id]["turns"]
            if len(turns) > 10:
                self.conversation_memory[conversation_id]["turns"] = turns[-10:]
            
            self.conversation_memory[conversation_id]["last_updated"] = datetime.utcnow()
            
            logger.debug(f"Updated conversation context for {conversation_id}")
            
        except Exception as e:
            logger.warning(f"Failed to update conversation context: {e}")
    
    def _generate_fallback_response(self, user_message: str, search_results: Any) -> str:
        """
        Generate a fallback response when AI generation fails.

        Args:
            user_message: Original user message
            search_results: Search results data or AgentResponse

        Returns:
            Fallback response string
        """
        try:
            # Normalize search results for metadata access
            if isinstance(search_results, AgentResponse):
                metadata = search_results.metadata or {}
            elif isinstance(search_results, dict):
                metadata = search_results.get("metadata", {})
            else:
                metadata = {}

            search_stats = metadata.get("search_stats", {})
            total_results = search_stats.get("total_results", 0)

            if total_results > 0:
                return (
                    f"J'ai trouvé {total_results} résultats pour votre recherche. Les données sont disponibles mais je "
                    "rencontre des difficultés pour générer une réponse détaillée. Pouvez-vous reformuler votre question ?"
                )
            else:
                return (
                    "Je n'ai pas trouvé de résultats correspondant à votre recherche. Essayez de reformuler votre question ou "
                    "d'utiliser d'autres termes."
                )

        except Exception:
            return (
                "Je rencontre des difficultés techniques pour traiter votre demande. Veuillez réessayer dans quelques instants."
            )
    
    def _get_system_message(self) -> str:
        """Get system message for the agent."""
        return """Tu es un assistant financier spécialisé dans la génération de réponses contextuelles.

Ton rôle est de:
1. Analyser les résultats de recherche financière
2. Prendre en compte le contexte conversationnel
3. Générer des réponses naturelles et utiles
4. Présenter les données de manière claire et accessible
5. Proposer des insights et actions de suivi pertinents

Principes de réponse:
- Sois précis et factuel avec les données financières
- Utilise un ton professionnel mais accessible
- Structure tes réponses de manière claire
- Propose des insights utiles basés sur les données
- Suggère des actions de suivi quand c'est pertinent
- Adapte ton niveau de détail selon le contexte

Toujours prioriser la clarté et l'utilité pour l'utilisateur."""
    
    def _get_response_generation_prompt(self) -> str:
        """Get response generation prompt for DeepSeek."""
        return """Tu es un assistant financier expert qui génère des réponses contextuelles basées sur des données de transactions.

Instructions:
1. Analyse la question utilisateur et les résultats de recherche
2. Génère une réponse naturelle qui répond directement à la question
3. Présente les données financières de manière claire et structurée
4. Ajoute des insights pertinents basés sur les données
5. Propose des actions de suivi si approprié
6. Adapte ton ton selon le contexte conversationnel

Format de réponse:
- Commence par répondre directement à la question
- Présente les données clés de manière structurée
- Ajoute des insights ou observations pertinents
- Termine par des suggestions d'actions si approprié

Ton: Professionnel mais accessible, précis avec les chiffres."""
    
    def clear_conversation_memory(self, conversation_id: Optional[str] = None) -> None:
        """
        Clear conversation memory.
        
        Args:
            conversation_id: Specific conversation to clear, or None for all
        """
        if conversation_id:
            self.conversation_memory.pop(conversation_id, None)
            logger.info(f"Cleared conversation memory for {conversation_id}")
        else:
            self.conversation_memory.clear()
            logger.info("Cleared all conversation memory")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics."""
        total_conversations = len(self.conversation_memory)
        total_turns = sum(len(conv["turns"]) for conv in self.conversation_memory.values())
        
        return {
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "average_turns_per_conversation": total_turns / total_conversations if total_conversations > 0 else 0,
            "memory_size_mb": len(str(self.conversation_memory)) / (1024 * 1024)
        }