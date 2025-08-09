"""
Search Query Agent for interfacing with Search Service.

This agent generates optimized search queries for the Search Service based on
detected intents and user messages. It includes entity extraction, query
optimization, and standardized search service communication.

Classes:
    - SearchQueryAgent: Main search query generation agent
    - QueryOptimizer: Helper class for query optimization

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Search Service Integration
"""

import time
import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .base_financial_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..models.service_contracts import SearchServiceQuery, SearchServiceResponse, QueryMetadata, SearchParameters
from ..models.financial_models import IntentResult
from ..core.deepseek_client import DeepSeekClient
from ..utils.validators import ContractValidator

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Helper class for optimizing search queries."""
    
    @staticmethod
    def optimize_search_text(user_message: str, intent_result: IntentResult) -> str:
        """
        Optimize search text based on intent and entities.
        
        Args:
            user_message: Original user message
            intent_result: Detected intent with entities
            
        Returns:
            Optimized search text
        """
        # Start with clean user message
        search_text = user_message.lower().strip()
        
        # Remove common stop words that don't help search
        stop_words = {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'mon', 'ma', 'mes'}
        words = [word for word in search_text.split() if word not in stop_words]
        
        # Add entity-based keywords
        if intent_result.entities:
            for entity_type, entity_list in intent_result.entities.items():
                if entity_type == "MERCHANT_NAME":
                    for entity in entity_list:
                        if entity.get("normalized_value"):
                            words.append(entity["normalized_value"])
                elif entity_type == "CATEGORY":
                    for entity in entity_list:
                        if entity.get("normalized_value"):
                            words.append(entity["normalized_value"])
        
        return " ".join(words)[:200]  # Limit search text length
    
    @staticmethod
    def extract_date_filters(intent_result: IntentResult) -> Dict[str, Any]:
        """Extract date range filters from intent entities."""
        date_filters = {}
        
        if not intent_result.entities:
            return date_filters
        
        # Look for date entities
        date_entities = intent_result.entities.get("DATE_RANGE", [])
        for date_entity in date_entities:
            normalized_date = date_entity.get("normalized_value")
            if normalized_date:
                if isinstance(normalized_date, dict):
                    if "start_date" in normalized_date:
                        date_filters["date_from"] = normalized_date["start_date"]
                    if "end_date" in normalized_date:
                        date_filters["date_to"] = normalized_date["end_date"]
                elif isinstance(normalized_date, str):
                    # Handle single date or month format
                    try:
                        if len(normalized_date) == 7:  # YYYY-MM format
                            year, month = normalized_date.split('-')
                            date_filters["month_year"] = normalized_date
                        elif len(normalized_date) == 10:  # YYYY-MM-DD format
                            date_filters["date_from"] = normalized_date
                            date_filters["date_to"] = normalized_date
                    except:
                        pass
        
        return date_filters
    
    @staticmethod
    def extract_amount_filters(intent_result: IntentResult) -> Dict[str, Any]:
        """Extract amount range filters from intent entities."""
        amount_filters = {}
        
        if not intent_result.entities:
            return amount_filters
        
        # Look for amount entities
        amount_entities = intent_result.entities.get("AMOUNT", [])
        for amount_entity in amount_entities:
            normalized_amount = amount_entity.get("normalized_value")
            if normalized_amount and isinstance(normalized_amount, (int, float)):
                # For single amounts, create a range around the value
                tolerance = abs(normalized_amount) * 0.1  # 10% tolerance
                amount_filters["amount_min"] = normalized_amount - tolerance
                amount_filters["amount_max"] = normalized_amount + tolerance
        
        return amount_filters


class SearchQueryAgent(BaseFinancialAgent):
    """
    Agent for generating and executing search queries.
    
    This agent takes intent detection results and user messages, then:
    1. Extracts additional entities using AI
    2. Generates optimized SearchServiceQuery contracts
    3. Executes queries against the Search Service
    4. Returns structured search results
    
    Attributes:
        search_service_url: Base URL for the Search Service
        http_client: HTTP client for service communication
        query_optimizer: Helper for query optimization
    """
    
    def __init__(self, deepseek_client: DeepSeekClient, search_service_url: str, 
                 config: Optional[AgentConfig] = None):
        """
        Initialize the search query agent.
        
        Args:
            deepseek_client: Configured DeepSeek client
            search_service_url: Base URL for Search Service
            config: Optional agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="search_query_agent",
                model_client_config={
                    "model": "deepseek-chat",
                    "api_key": deepseek_client.api_key,
                    "base_url": deepseek_client.base_url
                },
                system_message=self._get_system_message(),
                max_consecutive_auto_reply=1,
                description="Search query generation and execution agent",
                temperature=0.2,  # Low temperature for consistent entity extraction
                max_tokens=400,
                timeout_seconds=20
            )
        
        super().__init__(
            name=config.name,
            config=config,
            deepseek_client=deepseek_client
        )
        
        self.search_service_url = search_service_url.rstrip('/')
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.query_optimizer = QueryOptimizer()
        
        logger.info(f"Initialized SearchQueryAgent with service URL: {search_service_url}")
    
    async def _execute_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search query operation.
        
        Args:
            input_data: Dict containing 'intent_result' and 'user_message'
            
        Returns:
            Dict with search results and metadata
        """
        intent_result = input_data.get("intent_result")
        user_message = input_data.get("user_message", "")
        
        if not intent_result:
            raise ValueError("intent_result is required for search query generation")
        
        return await self.process_search_request(intent_result, user_message)
    
    async def process_search_request(self, intent_result: IntentResult, user_message: str) -> Dict[str, Any]:
        """
        Process a search request end-to-end.
        
        Args:
            intent_result: Detected intent with entities
            user_message: Original user message
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Extract additional entities using AI
            enhanced_entities = await self._extract_additional_entities(user_message, intent_result)
            
            # Step 2: Generate search service query contract
            search_query = await self._generate_search_contract(intent_result, user_message, enhanced_entities)
            
            # Step 3: Execute search query
            search_response = await self._execute_search_query(search_query)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "content": f"Search completed: {search_response.response_metadata.returned_hits} results",
                "metadata": {
                    "search_query": search_query.dict(),
                    "search_response": search_response.dict(),
                    "enhanced_entities": enhanced_entities,
                    "execution_time_ms": execution_time
                },
                "confidence_score": min(intent_result.confidence + 0.1, 1.0),  # Boost confidence slightly
                "token_usage": {
                    "prompt_tokens": 50,  # Estimated
                    "completion_tokens": 20,
                    "total_tokens": 70
                }
            }
            
        except Exception as e:
            logger.error(f"Search request processing failed: {e}")
            raise
    
    async def _generate_search_contract(self, intent_result: IntentResult, user_message: str,
                                      enhanced_entities: Optional[Dict] = None) -> SearchServiceQuery:
        """
        Generate a SearchServiceQuery contract from intent and message.
        
        Args:
            intent_result: Detected intent with entities
            user_message: Original user message
            enhanced_entities: Additional entities from AI extraction
            
        Returns:
            SearchServiceQuery contract for the Search Service
        """
        # Optimize search text
        search_text = self.query_optimizer.optimize_search_text(user_message, intent_result)
        
        # Extract filters from entities
        date_filters = self.query_optimizer.extract_date_filters(intent_result)
        amount_filters = self.query_optimizer.extract_amount_filters(intent_result)
        
        # Combine all entities
        all_entities = intent_result.entities.copy() if intent_result.entities else {}
        if enhanced_entities:
            for entity_type, entities in enhanced_entities.items():
                if entity_type in all_entities:
                    all_entities[entity_type].extend(entities)
                else:
                    all_entities[entity_type] = entities
        
        # Build search filters
        search_filters = {}
        search_filters.update(date_filters)
        search_filters.update(amount_filters)
        
        # Add category filters if found
        if "CATEGORY" in all_entities:
            categories = [e.get("normalized_value") for e in all_entities["CATEGORY"] if e.get("normalized_value")]
            if categories:
                search_filters["categories"] = categories
        
        # Add merchant filters if found
        if "MERCHANT_NAME" in all_entities:
            merchants = [e.get("normalized_value") for e in all_entities["MERCHANT_NAME"] if e.get("normalized_value")]
            if merchants:
                search_filters["merchants"] = merchants
        
        # Create query metadata
        query_metadata = QueryMetadata(
            conversation_id=f"conv_{int(time.time())}",  # Placeholder - should come from context
            user_id=1,  # Placeholder - should come from context
            intent_type=intent_result.intent,
            language="fr",
            priority="normal",
            source_agent=self.name
        )
        
        # Create search parameters based on intent
        search_params = SearchParameters(
            search_text=search_text,
            size=20 if intent_result.intent == "TRANSACTION_SEARCH" else 10,
            include_highlights=True,
            boost_recent=intent_result.intent in ["BALANCE_CHECK", "SPENDING_ANALYSIS"],
            fuzzy_matching=True if len(search_text.split()) > 1 else False
        )
        
        # Create complete search query
        search_query = SearchServiceQuery(
            query_metadata=query_metadata,
            search_parameters=search_params,
            filters=search_filters if search_filters else None
        )
        
        # Validate the query
        validator = ContractValidator()
        validation_errors = validator.validate_search_query(search_query.dict())
        if validation_errors:
            logger.warning(f"Search query validation warnings: {validation_errors}")
        
        return search_query
    
    async def _execute_search_query(self, query: SearchServiceQuery) -> SearchServiceResponse:
        """
        Execute search query against the Search Service.
        
        Args:
            query: SearchServiceQuery contract
            
        Returns:
            SearchServiceResponse from the service
        """
        try:
            # Prepare request
            url = f"{self.search_service_url}/search/lexical"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"ConversationService/{self.name}"
            }
            
            # Execute HTTP request
            response = await self.http_client.post(
                url=url,
                json=query.dict(),
                headers=headers
            )
            
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            search_response = SearchServiceResponse(**response_data)
            
            logger.info(f"Search query executed successfully: {search_response.response_metadata.returned_hits} results")
            
            return search_response
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Search service HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Search service error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Search service request error: {e}")
            raise Exception(f"Search service unavailable: {e}")
        except Exception as e:
            logger.error(f"Search query execution failed: {e}")
            raise
    
    async def _extract_additional_entities(self, message: str, intent_result: IntentResult) -> Dict[str, List[Dict]]:
        """
        Extract additional entities using AI that weren't caught by rules.
        
        Args:
            message: User message
            intent_result: Intent with existing entities
            
        Returns:
            Dictionary of additional entities found
        """
        try:
            # Prepare context for entity extraction
            existing_entities = intent_result.entities if intent_result.entities else {}
            context = self._prepare_entity_extraction_context(message, existing_entities)
            
            # Call DeepSeek for entity extraction
            response = await self.deepseek_client.generate_response(
                messages=[
                    {"role": "system", "content": self._get_entity_extraction_prompt()},
                    {"role": "user", "content": context}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse AI response
            additional_entities = self._parse_entity_response(response.content)
            
            return additional_entities
            
        except Exception as e:
            logger.warning(f"Additional entity extraction failed: {e}")
            return {}
    
    def _prepare_entity_extraction_context(self, message: str, existing_entities: Dict) -> str:
        """Prepare context for AI entity extraction."""
        context = f"Message: \"{message}\"\n\n"
        
        if existing_entities:
            context += "Entités déjà détectées:\n"
            for entity_type, entities in existing_entities.items():
                context += f"- {entity_type}: {entities}\n"
            context += "\n"
        
        context += "Extrais toutes les entités financières supplémentaires non détectées."
        return context
    
    def _parse_entity_response(self, ai_content: str) -> Dict[str, List[Dict]]:
        """Parse AI entity extraction response."""
        entities = {}
        
        try:
            # Simple parsing - enhance as needed
            lines = ai_content.strip().split('\n')
            
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        entity_type = parts[0].strip().upper()
                        entity_value = parts[1].strip()
                        
                        if entity_value and entity_value != "aucune":
                            if entity_type not in entities:
                                entities[entity_type] = []
                            
                            entities[entity_type].append({
                                "raw_value": entity_value,
                                "normalized_value": entity_value,
                                "confidence": 0.8,
                                "source": "ai_extraction"
                            })
        
        except Exception as e:
            logger.warning(f"Failed to parse entity response: {e}")
        
        return entities
    
    def _get_system_message(self) -> str:
        """Get system message for the agent."""
        return """Tu es un agent spécialisé dans la génération de requêtes de recherche pour les données financières.

Ton rôle est de:
1. Analyser les intentions détectées et les messages utilisateur
2. Extraire des entités financières supplémentaires 
3. Générer des requêtes optimisées pour le service de recherche
4. Exécuter les requêtes et retourner les résultats structurés

Types d'entités à extraire:
- MERCHANT_NAME: Noms de commerçants
- CATEGORY: Catégories de transactions
- AMOUNT: Montants et plages de montants
- DATE_RANGE: Dates et périodes
- TRANSACTION_TYPE: Types de transactions (débit/crédit)

Optimise les requêtes pour la pertinence et la performance."""
    
    def _get_entity_extraction_prompt(self) -> str:
        """Get entity extraction prompt for DeepSeek."""
        return """Extrais toutes les entités financières du message utilisateur.

Types d'entités à rechercher:
- MERCHANT_NAME: Noms de commerçants, magasins, services
- CATEGORY: Catégories de dépenses (alimentation, transport, etc.)
- AMOUNT: Montants, prix, valeurs monétaires
- DATE_RANGE: Dates, périodes, mois, années
- TRANSACTION_TYPE: Type de transaction (achat, virement, etc.)

Format de réponse:
TYPE_ENTITE: valeur_trouvée
TYPE_ENTITE: autre_valeur

Si aucune entité supplémentaire n'est trouvée, réponds: "aucune"."""
    
    async def close(self) -> None:
        """Close HTTP client resources."""
        await self.http_client.aclose()
        logger.info("SearchQueryAgent HTTP client closed")