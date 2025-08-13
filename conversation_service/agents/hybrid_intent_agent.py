"""
Hybrid Intent Detection Agent for financial conversations.

This agent implements a sophisticated hybrid approach to intent detection,
combining rule-based pattern matching with AI fallback using DeepSeek LLM.
It prioritizes speed and accuracy by trying rules first, then using AI for
complex or ambiguous cases.

Classes:
    - HybridIntentAgent: Main hybrid intent detection agent
    - IntentDetectionResult: Structured result from intent detection

Author: Conversation Service Team  
Created: 2025-01-31
Version: 1.0.0 MVP - Hybrid Rules + DeepSeek AI
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base_financial_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..core.deepseek_client import DeepSeekClient
from ..intent_rules.rule_engine import RuleEngine, RuleMatch
from ..models.financial_models import (
    IntentResult,
    IntentCategory,
    DetectionMethod,
    FinancialEntity,
    EntityType,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionStats:
    """Statistics for tracking detection method performance."""
    total_detections: int = 0
    rule_based_hits: int = 0
    ai_fallback_uses: int = 0
    high_confidence_detections: int = 0
    avg_rule_time_ms: float = 0.0
    avg_ai_time_ms: float = 0.0


class HybridIntentAgent(BaseFinancialAgent):
    """
    Hybrid intent detection agent using rules + AI fallback.
    
    This agent implements a two-stage intent detection process:
    1. Fast rule-based pattern matching for common financial intents
    2. DeepSeek AI fallback for complex or ambiguous cases
    
    The hybrid approach optimizes for both speed and accuracy while
    minimizing LLM costs by using rules when possible.
    
    Attributes:
        rule_engine: Rule-based intent detection engine
        detection_stats: Performance statistics tracker
        ai_confidence_threshold: Minimum confidence for AI results
    """

    
    def __init__(self, deepseek_client: DeepSeekClient, config: Optional[AgentConfig] = None):
        """
        Initialize the hybrid intent detection agent.
        
        Args:
            deepseek_client: Configured DeepSeek client
            config: Optional agent configuration (uses default if None)
        """
        # Default configuration for intent detection
        if config is None:
            config = AgentConfig(
                name="hybrid_intent_agent",
                model_client_config={
                    "model": "deepseek-chat",
                    "api_key": deepseek_client.api_key,
                    "base_url": deepseek_client.base_url
                },
                system_message=self._get_system_message(),
                max_consecutive_auto_reply=1,
                description="Hybrid intent detection agent for financial conversations",
                temperature=0.1,  # Low temperature for consistent intent detection
                max_tokens=150,   # Smaller responses for faster detection
                timeout_seconds=8
            )
        
        super().__init__(
            name=config.name,
            config=config,
            deepseek_client=deepseek_client
        )
        
        # Initialize rule engine
        self.rule_engine = RuleEngine()
        
        # Detection statistics
        self.detection_stats = DetectionStats()
        
        # Configuration parameters
        self.ai_confidence_threshold = 0.7
        self.rule_confidence_threshold = 0.8
        
        logger.info("Initialized HybridIntentAgent with rule engine and AI fallback")
    
    async def _execute_operation(self, input_data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """
        Execute hybrid intent detection operation.

        Args:
            input_data: Dict containing 'user_message' key
            user_id: ID of the requesting user

        Returns:
            Dict with intent detection results
        """
        user_message = input_data.get("user_message", "")
        if not user_message:
            raise ValueError("user_message is required for intent detection")

        return await self.detect_intent(user_message, user_id)

    async def detect_intent(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """
        Detect user intent using hybrid approach (rules + AI fallback).

        Args:
            user_message: User's input message
            user_id: ID of the requesting user

        Returns:
            Dictionary containing intent detection results
        """
        start_time = time.perf_counter()
        
        try:
            # Stage 1: Try rule-based detection first
            rule_result = await self._try_rule_based_detection(user_message)

            if rule_result and rule_result.confidence >= self.rule_confidence_threshold:
                # High confidence rule match - use it
                self.detection_stats.rule_based_hits += 1
                self._update_detection_stats(rule_time=time.perf_counter() - start_time)

                return {
                    "content": f"Intent detected: {rule_result.intent_type}",
                    "metadata": {
                        "intent_result": rule_result,
                        "detection_method": rule_result.method,
                        "confidence": rule_result.confidence,
                        "intent_type": rule_result.intent_type,
                        "entities": [e.model_dump() for e in rule_result.entities],
                        "intent_detected": rule_result.intent_type,
                        "entities_extracted": [e.model_dump() for e in rule_result.entities],
                    },
                    "confidence_score": rule_result.confidence,
                }

            # Stage 2: AI fallback for complex cases
            ai_result = await self._ai_fallback_detection(user_message, user_id, rule_result)
            self.detection_stats.ai_fallback_uses += 1
            self._update_detection_stats(ai_time=time.perf_counter() - start_time)

            return {
                "content": f"Intent detected: {ai_result.intent_type}",
                "metadata": {
                    "intent_result": ai_result,
                    "detection_method": DetectionMethod.AI_FALLBACK,
                    "confidence": ai_result.confidence,
                    "intent_type": ai_result.intent_type,
                    "entities": [e.model_dump() for e in ai_result.entities],
                    "intent_detected": ai_result.intent_type,
                    "entities_extracted": [e.model_dump() for e in ai_result.entities],
                    "rule_backup": rule_result.model_dump() if rule_result else None,
                },
                "confidence_score": ai_result.confidence,
            }
            
        except Exception as e:
            logger.error(f"Intent detection failed for message: {user_message[:100]}... Error: {e}")
            
            # Fallback to GENERAL intent
            fallback_result = IntentResult(
                intent_type="GENERAL",
                intent_category=IntentCategory.GENERAL_QUESTION,
                confidence=0.5,
                entities=[],
                method=DetectionMethod.FALLBACK,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

            return {
                "content": "Intent detected: GENERAL (fallback)",
                "metadata": {
                    "intent_result": fallback_result,
                    "detection_method": DetectionMethod.FALLBACK,
                    "confidence": 0.5,
                    "intent_type": fallback_result.intent_type,
                    "entities": [],
                    "intent_detected": fallback_result.intent_type,
                    "entities_extracted": [],
                    "error": str(e),
                },
                "confidence_score": 0.5,
            }

    def _map_rule_category(self, category: str) -> IntentCategory:
        """Map rule engine categories to IntentCategory values."""
        mapping = {
            "SEARCH": IntentCategory.TRANSACTION_SEARCH,
            "ANALYZE": IntentCategory.SPENDING_ANALYSIS,
            "CONVERSATIONAL": IntentCategory.GENERAL_QUESTION,
        }
        return mapping.get(category, IntentCategory.GENERAL_QUESTION)

    def _convert_rule_entities(self, entities: Dict[str, List]) -> List[FinancialEntity]:
        """Convert rule engine entity dict to list of FinancialEntity objects."""
        financial_entities: List[FinancialEntity] = []
        for entity_list in entities.values():
            for e in entity_list:
                try:
                    entity_type = EntityType(e.entity_type.upper())
                except ValueError:
                    logger.warning(f"Skipping unknown entity type: {e.entity_type}")
                    continue
                try:
                    financial_entities.append(
                        FinancialEntity(
                            entity_type=entity_type,
                            raw_value=e.raw_value,
                            normalized_value=e.normalized_value,
                            confidence=e.confidence,
                            start_position=e.position[0],
                            end_position=e.position[1],
                            detection_method=DetectionMethod.RULE_BASED,
                        )
                    )
                except Exception as err:
                    logger.warning(f"Failed to convert entity {e}: {err}")
        return financial_entities

    async def _try_rule_based_detection(self, message: str) -> Optional[IntentResult]:
        """
        Attempt rule-based intent detection.
        
        Args:
            message: User message to analyze
            
        Returns:
            IntentResult if rule match found, None otherwise
        """
        try:
            start_time = time.perf_counter()
            
            # Try exact matches first (fastest)
            exact_match = self.rule_engine.match_exact(message)
            if exact_match:
                execution_time = (time.perf_counter() - start_time) * 1000
                entities = self._convert_rule_entities(exact_match.entities)
                intent = exact_match.intent
                rule = self.rule_engine.all_rules.get(intent)
                no_search_needed = getattr(rule, "no_search_needed", False)
                suggestions = getattr(rule, "suggested_responses", None)
                return IntentResult(
                    intent_type=intent,
                    intent_category=self._map_rule_category(exact_match.intent_category),
                    confidence=exact_match.confidence,
                    entities=entities,
                    method=DetectionMethod.EXACT_RULE,
                    processing_time_ms=execution_time,
                    suggested_actions=suggestions,
                    search_required=not no_search_needed,
                )

            # Try pattern matching
            pattern_match = self.rule_engine.match_intent(message)
            if pattern_match:
                execution_time = (time.perf_counter() - start_time) * 1000
                entities = self._convert_rule_entities(pattern_match.entities)
                intent = pattern_match.intent
                rule = self.rule_engine.all_rules.get(intent)
                no_search_needed = getattr(rule, "no_search_needed", False)
                suggestions = getattr(rule, "suggested_responses", None)
                return IntentResult(
                    intent_type=intent,
                    intent_category=self._map_rule_category(pattern_match.intent_category),
                    confidence=pattern_match.confidence,
                    entities=entities,
                    method=DetectionMethod.PATTERN_RULE,
                    processing_time_ms=execution_time,
                    suggested_actions=suggestions,
                    search_required=not no_search_needed,
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Rule-based detection failed: {e}")
            return None
    
    async def _ai_fallback_detection(
        self,
        message: str,
        user_id: int,
        rule_backup: Optional[IntentResult] = None,
    ) -> IntentResult:
        """
        AI-powered intent detection fallback using DeepSeek.

        Args:
            message: User message to analyze
            rule_backup: Optional rule-based result providing context and
                metadata (search_required flag, suggested actions)
            user_id: ID of the requesting user

        Returns:
            IntentResult from AI analysis, enriched with rule backup metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(message, rule_backup)

            # Call DeepSeek for intent detection
            logger.debug("DeepSeek intent detection for user_id=%s", user_id)
            response = await self.deepseek_client.generate_response(
                messages=[
                    {"role": "system", "content": self._get_ai_detection_prompt()},
                    {"role": "user", "content": context}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                user=str(user_id),
                use_cache=True,
            )
            logger.debug("Raw AI response: %s", response.content)

            # Parse AI response into structured result
            result = self._parse_ai_response(response.content, message)
            logger.debug(
                "Interpreted entities: %s",
                [e.model_dump() for e in result.entities]
            )
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000

            # If rule backup indicates no search is required, propagate that metadata
            if rule_backup and rule_backup.search_required is False:
                result.search_required = False

                # Merge suggested actions from rule backup if AI result lacks them
                if not result.suggested_actions and rule_backup.suggested_actions:
                    result.suggested_actions = rule_backup.suggested_actions

            return result
            
        except Exception as e:
            logger.error(f"AI fallback detection failed: {e}")
            
            # Ultimate fallback
            return IntentResult(
                intent_type="GENERAL",
                intent_category=IntentCategory.GENERAL_QUESTION,
                confidence=0.3,
                entities=[],
                method=DetectionMethod.AI_ERROR_FALLBACK,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    def _prepare_ai_context(self, message: str, rule_backup: Optional[IntentResult]) -> str:
        """Prepare context for AI intent detection."""
        context = f"Message à analyser: \"{message}\"\n\n"
        
        if rule_backup:
            context += f"Règle trouvée (confiance faible): {rule_backup.intent_type} (confiance: {rule_backup.confidence})\n"
            if rule_backup.entities:
                serialized = [e.model_dump() for e in rule_backup.entities]
                context += f"Entités détectées: {serialized}\n"
        
        context += "\nAnalyse l'intention et extrais les entités financières pertinentes."
        
        return context
    
    def _parse_ai_response(self, ai_content: str, original_message: str) -> IntentResult:
        """Parse AI response into structured IntentResult."""
        try:
            # Simple parsing - in production, use more sophisticated JSON parsing
            lines = ai_content.strip().split('\n')
            intent_type = "GENERAL"
            confidence = 0.7
            entities: List[FinancialEntity] = []

            entity_mapping = {
                "merchant": EntityType.MERCHANT,
                "transaction_type": EntityType.TRANSACTION_TYPE,
                "amount": EntityType.AMOUNT,
                "currency": EntityType.CURRENCY,
                "date": EntityType.DATE,
                "category": EntityType.CATEGORY,
            }

            for line in lines:
                lower = line.lower()
                if lower.startswith('intention:') or lower.startswith('intent:'):
                    intent_type = line.split(':', 1)[1].strip().upper()
                elif lower.startswith('confiance:') or lower.startswith('confidence:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except Exception:
                        confidence = 0.7
                elif lower.startswith('entités:') or lower.startswith('entities:'):
                    raw_part = line.split(':', 1)[1].strip()
                    try:
                        parsed = json.loads(raw_part)
                        if isinstance(parsed, dict):
                            for key, value in parsed.items():
                                ent_type = entity_mapping.get(key.lower(), EntityType.OTHER)
                                values = value if isinstance(value, list) else [value]
                                for v in values:
                                    entities.append(
                                        FinancialEntity(
                                            entity_type=ent_type,
                                            raw_value=str(v),
                                            normalized_value=v,
                                            confidence=0.5,
                                            detection_method=DetectionMethod.LLM_BASED,
                                            start_position=None,
                                            end_position=None,
                                        )
                                    )
                        else:
                            raise ValueError("Entities JSON is not an object")
                    except Exception:
                        raw_entities = raw_part.split(',')
                        for raw in raw_entities:
                            value = raw.strip()
                            if value:
                                entities.append(
                                    FinancialEntity(
                                        entity_type=EntityType.OTHER,
                                        raw_value=value,
                                        normalized_value=value,
                                        confidence=0.5,
                                        detection_method=DetectionMethod.LLM_BASED,
                                        start_position=None,
                                        end_position=None,
                                    )
                                )

            # Determine category based on intent
            intent_upper = intent_type.upper()
            if intent_upper in ["FINANCIAL_QUERY", "TRANSACTION_SEARCH"]:
                category = IntentCategory.TRANSACTION_SEARCH
            elif intent_upper in ["BALANCE_CHECK", "BALANCE_INQUIRY"]:
                category = IntentCategory.BALANCE_INQUIRY
            elif intent_upper in ["SPENDING_ANALYSIS", "CATEGORY_ANALYSIS"]:
                category = IntentCategory.SPENDING_ANALYSIS
            else:
                category = IntentCategory.GENERAL_QUESTION

            return IntentResult(
                intent_type=intent_type,
                intent_category=category,
                confidence=min(confidence, 0.95),  # Cap AI confidence
                entities=entities,
                method=DetectionMethod.AI_DETECTION,
                processing_time_ms=0.0,
            )

        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return IntentResult(
                intent_type="GENERAL",
                intent_category=IntentCategory.GENERAL_QUESTION,
                confidence=0.5,
                entities=[],
                method=DetectionMethod.AI_PARSE_FALLBACK,
                processing_time_ms=0.0,
            )
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get intent detection performance statistics."""
        base_stats = self.get_performance_stats()
        
        # Calculate detection method ratios
        total = self.detection_stats.total_detections
        rule_ratio = self.detection_stats.rule_based_hits / total if total > 0 else 0
        ai_ratio = self.detection_stats.ai_fallback_uses / total if total > 0 else 0
        
        detection_specific_stats = {
            "detection_stats": {
                "total_detections": total,
                "rule_based_hits": self.detection_stats.rule_based_hits,
                "ai_fallback_uses": self.detection_stats.ai_fallback_uses,
                "rule_success_ratio": round(rule_ratio, 3),
                "ai_fallback_ratio": round(ai_ratio, 3),
                "high_confidence_detections": self.detection_stats.high_confidence_detections,
                "avg_rule_time_ms": round(self.detection_stats.avg_rule_time_ms, 2),
                "avg_ai_time_ms": round(self.detection_stats.avg_ai_time_ms, 2)
            },
            "thresholds": {
                "rule_confidence_threshold": self.rule_confidence_threshold,
                "ai_confidence_threshold": self.ai_confidence_threshold
            }
        }
        
        base_stats.update(detection_specific_stats)
        return base_stats
    
    def _update_detection_stats(self, rule_time: Optional[float] = None, 
                               ai_time: Optional[float] = None) -> None:
        """Update detection performance statistics."""
        self.detection_stats.total_detections += 1
        
        if rule_time is not None:
            rule_time_ms = rule_time * 1000
            if self.detection_stats.avg_rule_time_ms == 0:
                self.detection_stats.avg_rule_time_ms = rule_time_ms
            else:
                # Running average
                count = self.detection_stats.rule_based_hits
                self.detection_stats.avg_rule_time_ms = (
                    (self.detection_stats.avg_rule_time_ms * (count - 1) + rule_time_ms) / count
                )
        
        if ai_time is not None:
            ai_time_ms = ai_time * 1000
            if self.detection_stats.avg_ai_time_ms == 0:
                self.detection_stats.avg_ai_time_ms = ai_time_ms
            else:
                # Running average
                count = self.detection_stats.ai_fallback_uses
                self.detection_stats.avg_ai_time_ms = (
                    (self.detection_stats.avg_ai_time_ms * (count - 1) + ai_time_ms) / count
                )
    
    def _get_system_message(self) -> str:
        """Get system message for the agent."""
        return """Tu es un agent spécialisé dans la détection d'intentions pour les conversations financières.

Ton rôle est d'analyser les messages des utilisateurs et de déterminer leur intention principale parmi ces catégories:

INTENTIONS FINANCIÈRES:
- FINANCIAL_QUERY: Questions générales sur les finances
- TRANSACTION_SEARCH: Recherche de transactions spécifiques
- BALANCE_CHECK: Vérification du solde
- SPENDING_ANALYSIS: Analyse des dépenses
- CATEGORY_ANALYSIS: Analyse par catégorie
- BUDGET_PLANNING: Planification budgétaire

INTENTIONS CONVERSATIONNELLES:
- GREETING: Salutations
- GENERAL: Questions générales
- HELP: Demandes d'aide
- GOODBYE: Au revoir

Réponds toujours au format:
Intention: [INTENTION]
Confiance: [0.0-1.0]
Entités: {"clé": "valeur"} (JSON des entités détectées)

Privilégie la précision et la cohérence."""
    
    def _get_ai_detection_prompt(self) -> str:
        """Get AI detection prompt for DeepSeek."""
        return """Tu es un expert en détection d'intentions financières. Analyse le message utilisateur et détermine l'intention principale.

Utilise ces catégories d'intentions:

FINANCIÈRES:
- FINANCIAL_QUERY: Questions générales finances
- TRANSACTION_SEARCH: Recherche transactions
- BALANCE_CHECK: Vérification solde  
- SPENDING_ANALYSIS: Analyse dépenses
- CATEGORY_ANALYSIS: Analyse catégories
- BUDGET_PLANNING: Planification budget

CONVERSATIONNELLES:
- GREETING: Salutations
- GENERAL: Questions générales
- HELP: Demandes d'aide
- GOODBYE: Au revoir

Extrais également les entités financières (montants, dates, catégories, etc.).

Format de réponse:
Intention: [INTENTION]
Confiance: [0.0-1.0]
Entités: {"clé": "valeur"}"""