"""
Intent Classification Agent for Harena Conversation Service.

This module implements the critical intent classification agent using AutoGen v0.4
framework. It provides high-precision intent detection with special focus on
Harena's consultation scope and unsupported action redirection.

Key Features:
- AutoGen AssistantAgent with specialized prompts for Harena scope
- Multi-layer fallback: LLM → Rules → Default with confidence scoring
- Intelligent caching for frequent intent patterns
- Unsupported action detection with high precision (>90%)
- Comprehensive validation and error handling
- Structured logging for monitoring and debugging

Author: Harena Conversation Team
Created: 2025-01-31
Version: 1.0.0 - AutoGen v0.4 + Harena Scope
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# AutoGen imports
from autogen import AssistantAgent
from openai import AsyncOpenAI

# Local imports
from ..models.core_models import (
    IntentType, IntentResult, AgentResponse, ConversationState, HarenaValidators
)
from ..core.intent_taxonomy import (
    IntentTaxonomy, IntentClassificationMatrix, UnsupportedActionDetector,
    HarenaResponseTemplates, ConfidenceThresholds
)

__all__ = ["IntentClassifierAgent", "IntentClassificationCache", "IntentPromptManager"]

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# CACHING SYSTEM
# ================================

@dataclass
class CachedIntentResult:
    """Cached intent classification result with metadata."""
    intent: IntentType
    confidence: float
    reasoning: str
    timestamp: datetime
    hit_count: int = 1
    
    def is_expired(self, ttl_minutes: int = 30) -> bool:
        """Check if cached result is expired."""
        return datetime.now() - self.timestamp > timedelta(minutes=ttl_minutes)
    
    def to_intent_result(self) -> IntentResult:
        """Convert to IntentResult object."""
        return IntentResult(
            intent=self.intent,
            confidence=self.confidence,
            reasoning=f"{self.reasoning} [Cached result]",
            processing_metadata={
                "cached": True,
                "cache_hits": self.hit_count,
                "cache_timestamp": self.timestamp.isoformat()
            }
        )

class IntentClassificationCache:
    """
    Intelligent caching system for intent classification results.
    
    Implements LRU eviction with hit counting and TTL expiration.
    Optimized for frequent banking patterns and conversational intents.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl_minutes: int = 30):
        self.max_size = max_size
        self.default_ttl_minutes = default_ttl_minutes
        self.cache: Dict[str, CachedIntentResult] = {}
        self.access_times: Dict[str, datetime] = {}
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_cache_key(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from user message and context."""
        import hashlib
        
        # Normalize message for better cache hits
        normalized = user_message.lower().strip()
        
        # Include relevant context
        context_str = ""
        if context:
            # Only include stable context elements
            stable_context = {
                k: v for k, v in context.items() 
                if k in ['user_language', 'conversation_type', 'user_preferences']
            }
            if stable_context:
                context_str = json.dumps(stable_context, sort_keys=True)
        
        cache_data = f"{normalized}|{context_str}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def get(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Optional[IntentResult]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(user_message, context)
        
        if cache_key not in self.cache:
            self.misses += 1
            return None
        
        cached_result = self.cache[cache_key]
        
        # Check expiration
        if cached_result.is_expired(self.default_ttl_minutes):
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            self.misses += 1
            return None
        
        # Update access time and hit count
        cached_result.hit_count += 1
        self.access_times[cache_key] = datetime.now()
        self.hits += 1
        
        logger.debug(
            "Intent cache hit",
            cache_key=cache_key[:8],
            intent=cached_result.intent.value,
            hit_count=cached_result.hit_count
        )
        
        return cached_result.to_intent_result()
    
    def set(
        self, 
        user_message: str, 
        intent_result: IntentResult, 
        context: Optional[Dict[str, Any]] = None
    ):
        """Cache intent result with LRU eviction."""
        cache_key = self._generate_cache_key(user_message, context)
        
        # Don't cache low-confidence or unknown results
        if intent_result.confidence < 0.6 or intent_result.intent == IntentType.UNKNOWN:
            return
        
        # Create cached result
        cached_result = CachedIntentResult(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            reasoning=intent_result.reasoning,
            timestamp=datetime.now()
        )
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[cache_key] = cached_result
        self.access_times[cache_key] = datetime.now()
        
        logger.debug(
            "Intent cached",
            cache_key=cache_key[:8],
            intent=intent_result.intent.value,
            confidence=intent_result.confidence
        )
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Remove from both caches
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]
        
        self.evictions += 1
        logger.debug("Evicted LRU cache entry", cache_key=lru_key[:8])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "top_intents": self._get_top_cached_intents()
        }
    
    def _get_top_cached_intents(self) -> List[Dict[str, Any]]:
        """Get top cached intents by hit count."""
        intent_counts = {}
        for cached_result in self.cache.values():
            intent = cached_result.intent.value
            if intent not in intent_counts:
                intent_counts[intent] = {"count": 0, "total_hits": 0}
            intent_counts[intent]["count"] += 1
            intent_counts[intent]["total_hits"] += cached_result.hit_count
        
        # Sort by total hits
        sorted_intents = sorted(
            intent_counts.items(),
            key=lambda x: x[1]["total_hits"],
            reverse=True
        )
        
        return [
            {"intent": intent, **stats}
            for intent, stats in sorted_intents[:5]
        ]
    
    def clear_expired(self):
        """Clear all expired cache entries."""
        expired_keys = [
            key for key, cached_result in self.cache.items()
            if cached_result.is_expired(self.default_ttl_minutes)
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

# ================================
# PROMPT MANAGEMENT
# ================================

class IntentPromptManager:
    """
    Manages prompts for intent classification with Harena-specific optimizations.
    
    Provides few-shot examples, system prompts, and dynamic prompt generation
    based on conversation context and user patterns.
    """
    
    # Core system prompt for Harena scope
    HARENA_SYSTEM_PROMPT = """Tu es un expert en classification d'intentions pour Harena - Assistant bancaire consultatif.

⚠️ PÉRIMÈTRE HARENA - CONSULTATION UNIQUEMENT :
Harena permet UNIQUEMENT la consultation et l'analyse des données bancaires.
AUCUNE action/opération sur les comptes n'est autorisée.

✅ INTENTIONS SUPPORTÉES (Harena) :
• CONSULTATION : soldes, transactions, relevés, infos cartes
• ANALYSIS : dépenses par catégorie/marchand/période, tendances, budget
• INFORMATION_SUPPORT : aide, explications produits, frais
• CONVERSATIONAL : salutations, politesse, remerciements

❌ INTENTIONS NON SUPPORTÉES (Redirection nécessaire) :
• TRANSFER_REQUEST : virements, transferts → Classification = "TRANSFER_REQUEST"
• PAYMENT_REQUEST : paiements factures → Classification = "PAYMENT_REQUEST"  
• CARD_OPERATIONS : bloquer/débloquer carte → Classification = "CARD_OPERATIONS"
• ACCOUNT_MODIFICATION : changer coordonnées → Classification = "ACCOUNT_MODIFICATION"
• LOAN_REQUEST : demandes crédit → Classification = "LOAN_REQUEST"
• INVESTMENT_OPERATIONS : achats/ventes → Classification = "INVESTMENT_OPERATIONS"
• Pour actions non spécifiques → Classification = "UNSUPPORTED_ACTION"

🎯 RÈGLES DE CLASSIFICATION CRITIQUES :
1. Si demande = consultation/analyse → intention spécifique supportée
2. Si demande = action sur compte → intention spécifique d'action (pour redirection appropriée)
3. Si demande = hors bancaire → "OUT_OF_SCOPE"
4. Si incertain ou mal formulé → "AMBIGUOUS"
5. Si complètement incompréhensible → "UNKNOWN"

EXEMPLES CRITIQUES :
• "Faire un virement de 500€" → "TRANSFER_REQUEST" (confiance ≥ 0.8)
• "Mes dépenses restaurant" → "CATEGORY_ANALYSIS" (confiance ≥ 0.7)
• "Bloquer ma carte" → "CARD_OPERATIONS" (confiance ≥ 0.8)
• "Mon solde" → "BALANCE_INQUIRY" (confiance ≥ 0.8)
• "Payer ma facture EDF" → "PAYMENT_REQUEST" (confiance ≥ 0.8)

CONFIANCE REQUISE :
• Actions non supportées : ≥ 0.8 (redirection critique)
• Consultations/analyses : ≥ 0.5 (flexibilité utilisateur)
• Conversationnel : ≥ 0.7 (évident)
• Incertain : ≤ 0.5 (demander clarification)

FORMAT RÉPONSE OBLIGATOIRE (JSON strict) :
{"intent": "INTENT_TYPE", "confidence": 0.95, "reasoning": "explication précise du choix"}

Sois précis, confiant dans tes classifications, et respecte STRICTEMENT le périmètre Harena."""

    # Few-shot examples optimized for Harena scope
    FEW_SHOT_EXAMPLES = [
        {
            "input": "Bonjour, comment allez-vous ?",
            "output": '{"intent": "GREETING", "confidence": 0.98, "reasoning": "Salutation standard, très claire"}'
        },
        {
            "input": "Quel est mon solde actuel ?", 
            "output": '{"intent": "BALANCE_INQUIRY", "confidence": 0.97, "reasoning": "Demande consultation solde - supportée par Harena"}'
        },
        {
            "input": "Mes dépenses restaurant ce mois",
            "output": '{"intent": "CATEGORY_ANALYSIS", "confidence": 0.94, "reasoning": "Analyse par catégorie restaurant - supportée par Harena"}'
        },
        {
            "input": "Faire un virement de 500€ vers mon épargne",
            "output": '{"intent": "TRANSFER_REQUEST", "confidence": 0.96, "reasoning": "Demande de virement - action non supportée par Harena qui est consultatif uniquement"}'
        },
        {
            "input": "Bloquer ma carte bancaire immédiatement",
            "output": '{"intent": "CARD_OPERATIONS", "confidence": 0.95, "reasoning": "Opération sur carte - action non supportée par Harena, nécessite redirection"}'
        },
        {
            "input": "Combien j'ai dépensé chez Amazon cette année ?",
            "output": '{"intent": "MERCHANT_ANALYSIS", "confidence": 0.93, "reasoning": "Analyse dépenses par marchand - supportée par Harena"}'
        },
        {
            "input": "Évolution de mes dépenses transport",
            "output": '{"intent": "TEMPORAL_ANALYSIS", "confidence": 0.91, "reasoning": "Analyse temporelle catégorie transport - supportée par Harena"}'
        },
        {
            "input": "Payer ma facture EDF de 120€",
            "output": '{"intent": "PAYMENT_REQUEST", "confidence": 0.94, "reasoning": "Demande de paiement - action non supportée par Harena, consultation uniquement"}'
        },
        {
            "input": "Qu'est-ce qu'un livret A ?",
            "output": '{"intent": "PRODUCT_INFORMATION", "confidence": 0.92, "reasoning": "Demande d\'information produit bancaire - supportée par Harena"}'
        },
        {
            "input": "Ça",
            "output": '{"intent": "AMBIGUOUS", "confidence": 0.15, "reasoning": "Message trop vague, contexte insuffisant pour classification"}'
        },
        {
            "input": "Mes dernières transactions supérieures à 100€",
            "output": '{"intent": "TRANSACTION_SEARCH", "confidence": 0.89, "reasoning": "Recherche transactions avec critères - supportée par Harena"}'
        },
        {
            "input": "Changer mon adresse dans mon compte",
            "output": '{"intent": "ACCOUNT_MODIFICATION", "confidence": 0.92, "reasoning": "Modification coordonnées - action non supportée par Harena"}'
        },
        {
            "input": "Je veux...",
            "output": '{"intent": "INSUFFICIENT_CONTEXT", "confidence": 0.25, "reasoning": "Phrase incomplète, contexte insuffisant pour comprendre la demande"}'
        }
    ]
    
    @classmethod
    def build_classification_prompt(
        cls,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        include_context: bool = True
    ) -> str:
        """Build complete classification prompt with context."""
        prompt_parts = [f"Message utilisateur : {user_message}"]
        
        # Add context if available
        if include_context and context:
            context_parts = []
            
            if context.get("previous_intent"):
                context_parts.append(f"Intention précédente : {context['previous_intent']}")
            
            if context.get("conversation_history"):
                recent_history = context["conversation_history"][-2:]  # Last 2 exchanges
                if recent_history:
                    history_text = " | ".join([
                        f"User: {h.get('user', '')[:50]} → Bot: {h.get('bot', '')[:50]}"
                        for h in recent_history
                    ])
                    context_parts.append(f"Historique récent : {history_text}")
            
            if context.get("user_patterns"):
                patterns = context["user_patterns"]
                if patterns.get("frequent_intents"):
                    top_intents = patterns["frequent_intents"][:3]
                    context_parts.append(f"Intentions fréquentes : {', '.join(top_intents)}")
            
            if context_parts:
                prompt_parts.append(f"\nContexte : {' | '.join(context_parts)}")
        
        prompt_parts.extend([
            "",
            "Classe ce message selon les intentions Harena.",
            "ATTENTION : Respecte le périmètre consultatif de Harena.",
            "Réponds UNIQUEMENT en JSON valide avec la structure exacte :",
            '{"intent": "INTENT_TYPE", "confidence": 0.95, "reasoning": "explication"}'
        ])
        
        return "\n".join(prompt_parts)
    
    @classmethod
    def get_validation_prompt(cls, user_message: str, classified_intent: str, confidence: float) -> str:
        """Generate validation prompt for double-checking classification."""
        return f"""Valide cette classification d'intention :

Message : {user_message}
Classification : {classified_intent}
Confiance : {confidence}

Questions de validation :
1. Cette intention respecte-t-elle le périmètre Harena (consultation uniquement) ?
2. La confiance est-elle appropriée pour cette classification ?
3. Y a-t-il une intention plus précise ?

Réponds en JSON :
{{"is_valid": true/false, "suggested_intent": "INTENT_TYPE", "suggested_confidence": 0.XX, "reasoning": "explication"}}"""
    
    @classmethod
    def get_dynamic_examples(cls, intent_type: IntentType) -> List[Dict[str, str]]:
        """Get dynamic few-shot examples based on target intent type."""
        # Filter examples by intent category for better context
        category_examples = {
            "consultation": [ex for ex in cls.FEW_SHOT_EXAMPLES if any(intent in ex["output"] for intent in [
                "BALANCE_INQUIRY", "TRANSACTION_SEARCH", "ACCOUNT_OVERVIEW", "STATEMENT_REQUEST"
            ])],
            "analysis": [ex for ex in cls.FEW_SHOT_EXAMPLES if any(intent in ex["output"] for intent in [
                "CATEGORY_ANALYSIS", "MERCHANT_ANALYSIS", "TEMPORAL_ANALYSIS", "SPENDING_ANALYSIS"
            ])],
            "unsupported": [ex for ex in cls.FEW_SHOT_EXAMPLES if any(intent in ex["output"] for intent in [
                "TRANSFER_REQUEST", "PAYMENT_REQUEST", "CARD_OPERATIONS", "ACCOUNT_MODIFICATION"
            ])],
            "conversational": [ex for ex in cls.FEW_SHOT_EXAMPLES if any(intent in ex["output"] for intent in [
                "GREETING", "GOODBYE", "THANKS", "CLARIFICATION_REQUEST"
            ])]
        }
        
        # Select relevant examples based on intent type
        if intent_type in {IntentType.BALANCE_INQUIRY, IntentType.TRANSACTION_SEARCH, IntentType.ACCOUNT_OVERVIEW}:
            return category_examples["consultation"][:3]
        elif intent_type in {IntentType.CATEGORY_ANALYSIS, IntentType.MERCHANT_ANALYSIS, IntentType.TEMPORAL_ANALYSIS}:
            return category_examples["analysis"][:3]
        elif intent_type in {IntentType.TRANSFER_REQUEST, IntentType.PAYMENT_REQUEST, IntentType.CARD_OPERATIONS}:
            return category_examples["unsupported"][:3]
        else:
            # Return diverse examples
            return [
                category_examples["consultation"][0],
                category_examples["analysis"][0], 
                category_examples["unsupported"][0],
                category_examples["conversational"][0]
            ]

# ================================
# MAIN INTENT CLASSIFIER AGENT
# ================================

class IntentClassifierAgent:
    """
    AutoGen-based intent classification agent specialized for Harena scope.
    
    Implements sophisticated multi-layer intent classification with:
    - Primary: OpenAI LLM with Harena-optimized prompts
    - Secondary: Rule-based fallback for reliability
    - Tertiary: Conservative defaults with clarification requests
    
    Features:
    - Intelligent caching for frequent patterns
    - Unsupported action detection with high precision
    - Comprehensive validation and error handling  
    - Performance monitoring and quality metrics
    """
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str = "gpt-4",
        cache_enabled: bool = True,
        cache_size: int = 1000,
        enable_validation: bool = True
    ):
        """
        Initialize Intent Classifier Agent.
        
        Args:
            openai_client: OpenAI async client
            model_name: Model to use (gpt-4, gpt-3.5-turbo, etc.)
            cache_enabled: Enable intelligent caching
            cache_size: Maximum cache size
            enable_validation: Enable result validation
        """
        self.openai_client = openai_client
        self.model_name = model_name
        self.enable_validation = enable_validation
        
        # Initialize caching
        self.cache_enabled = cache_enabled
        if self.cache_enabled:
            self.cache = IntentClassificationCache(max_size=cache_size)
        
        # Initialize AutoGen agent
        self.agent = AssistantAgent(
            name="intent_classifier",
            system_message=IntentPromptManager.HARENA_SYSTEM_PROMPT,
            llm_config={
                "model": model_name,
                "temperature": 0.1,  # Low temperature for consistent classification
                "max_tokens": 300,   # Sufficient for JSON response
                "timeout": 10        # 10 second timeout
            }
        )
        
        # Performance tracking
        self.classification_count = 0
        self.cache_hits = 0
        self.llm_calls = 0
        self.fallback_uses = 0
        self.error_count = 0
        
        logger.info(
            "Intent Classifier Agent initialized",
            model=model_name,
            cache_enabled=cache_enabled,
            validation_enabled=enable_validation
        )
    
    async def classify_intent(
        self,
        user_message: str,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_state: Optional[ConversationState] = None
    ) -> AgentResponse:
        """
        Classify user intent with comprehensive fallback strategy.
        
        Args:
            user_message: User input text
            user_id: User identifier for personalization
            context: Additional conversation context
            conversation_state: Current conversation state
            
        Returns:
            AgentResponse with IntentResult
        """
        start_time = time.time()
        self.classification_count += 1
        
        try:
            # Step 1: Check cache
            if self.cache_enabled:
                cached_result = self.cache.get(user_message, context)
                if cached_result:
                    self.cache_hits += 1
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    logger.debug(
                        "Intent classification cache hit",
                        user_message=user_message[:50],
                        intent=cached_result.intent.value,
                        confidence=cached_result.confidence,
                        processing_time_ms=processing_time
                    )
                    
                    return AgentResponse(
                        agent_name="intent_classifier",
                        success=True,
                        result=cached_result.dict(),
                        processing_time_ms=processing_time,
                        cached=True
                    )
            
            # Step 2: Primary classification via LLM
            intent_result = None
            try:
                intent_result = await self._classify_with_llm(user_message, context)
                self.llm_calls += 1
                
                # Validate LLM result
                if self.enable_validation:
                    is_valid, warnings = await self._validate_classification(
                        intent_result, user_message, context
                    )
                    if not is_valid:
                        logger.warning(
                            "LLM classification validation failed",
                            intent=intent_result.intent.value,
                            confidence=intent_result.confidence,
                            warnings=warnings
                        )
                        # Continue to fallback
                        intent_result = None
                
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
                intent_result = None
            
            # Step 3: Fallback to rule-based classification
            if intent_result is None or intent_result.confidence < ConfidenceThresholds.REJECT_THRESHOLD:
                intent_result = await self._classify_with_rules(user_message, context)
                self.fallback_uses += 1
                
                logger.info(
                    "Using fallback classification",
                    user_message=user_message[:50],
                    intent=intent_result.intent.value,
                    confidence=intent_result.confidence
                )
            
            # Step 4: Cache successful high-confidence results
            if self.cache_enabled and intent_result.confidence >= 0.6:
                self.cache.set(user_message, intent_result, context)
            
            # Step 5: Update conversation state
            if conversation_state:
                conversation_state.update_stage("intent", intent_result.dict())
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                "Intent classification complete",
                user_message=user_message[:50],
                intent=intent_result.intent.value,
                confidence=intent_result.confidence,
                processing_time_ms=processing_time,
                method="llm" if self.llm_calls > 0 else "fallback"
            )
            
            return AgentResponse(
                agent_name="intent_classifier",
                success=True,
                result=intent_result.dict(),
                processing_time_ms=processing_time,
                cached=False
            )
            
        except Exception as e:
            self.error_count += 1
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.error(
                "Intent classification failed",
                user_message=user_message[:50],
                error=str(e),
                processing_time_ms=processing_time,
                exc_info=True
            )
            
            # Return conservative default
            fallback_result = IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.1,
                reasoning=f"Classification failed: {str(e)}"
            )
            
            return AgentResponse(
                agent_name="intent_classifier",
                success=False,
                result=fallback_result.dict(),
                error_message=str(e),
                processing_time_ms=processing_time,
                cached=False
            )
    
    async def _classify_with_llm(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentResult:
        """Classify intent using OpenAI LLM."""
        
        # Build prompt with context
        prompt = IntentPromptManager.build_classification_prompt(user_message, context)
        
        # Prepare messages with few-shot examples
        messages = [
            {"role": "system", "content": IntentPromptManager.HARENA_SYSTEM_PROMPT}
        ]
        
        # Add few-shot examples
        for example in IntentPromptManager.FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        # Call OpenAI API
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        
        # Parse response
        response_content = response.choices[0].message.content
        
        try:
            result_data = json.loads(response_content)
            
            # Validate required fields
            intent_str = result_data.get("intent")
            confidence = float(result_data.get("confidence", 0))
            reasoning = result_data.get("reasoning", "")
            
            # Validate intent exists in taxonomy
            try:
                intent = IntentType(intent_str)
            except ValueError:
                logger.warning(f"Invalid intent from LLM: {intent_str}")
                intent = IntentType.UNKNOWN
                confidence = 0.2
                reasoning = f"Invalid intent '{intent_str}' - defaulted to UNKNOWN"
            
            # Apply confidence bounds
            confidence = max(0.0, min(1.0, confidence))
            
            return IntentResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                processing_metadata={
                    "method": "llm",
                    "model": self.model_name,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Invalid LLM response format: {str(e)}")
    
    async def _classify_with_rules(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentResult:
        """Classify intent using rule-based fallback."""
        
        # First: Check for unsupported actions (high precision)
        is_action, action_confidence, action_patterns = UnsupportedActionDetector.detect_unsupported_action(user_message)
        
        if is_action and action_confidence >= ConfidenceThresholds.UNSUPPORTED_ACTION_MIN:
            action_type = UnsupportedActionDetector.categorize_action_type(user_message)
            
            return IntentResult(
                intent=action_type,
                confidence=action_confidence,
                reasoning=f"Rule-based unsupported action detection: {', '.join(action_patterns)}",
                processing_metadata={
                    "method": "rules_unsupported_action",
                    "patterns": action_patterns
                }
            )
        
        # Second: Keyword-based classification
        keyword_candidates = IntentClassificationMatrix.classify_by_keywords(user_message)
        
        if keyword_candidates:
            best_intent, best_confidence = keyword_candidates[0]
            
            # Validate confidence threshold
            min_threshold = IntentTaxonomy._get_minimum_confidence(best_intent)
            if best_confidence >= min_threshold:
                return IntentResult(
                    intent=best_intent,
                    confidence=best_confidence,
                    reasoning=f"Keyword-based classification with {len(keyword_candidates)} candidates",
                    alternative_intents=[
                        {"intent": intent.value, "confidence": conf}
                        for intent, conf in keyword_candidates[1:3]
                    ],
                    processing_metadata={
                        "method": "rules_keywords",
                        "candidates": len(keyword_candidates)
                    }
                )
        
        # Third: Conservative default
        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.3,
            reasoning="No clear classification pattern found - requesting clarification",
            processing_metadata={
                "method": "rules_default",
                "fallback_reason": "insufficient_confidence"
            }
        )
    
    async def _validate_classification(
        self,
        intent_result: IntentResult,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate classification result for consistency."""
        
        # Use taxonomy validation
        is_valid, warnings = IntentTaxonomy.validate_classification_result(
            intent_result.intent,
            intent_result.confidence,
            user_message
        )
        
        # Additional Harena-specific validations
        harena_warnings = []
        
        # Check scope compliance
        if not IntentTaxonomy.is_supported_by_harena(intent_result.intent):
            if intent_result.intent not in {
                IntentType.TRANSFER_REQUEST, IntentType.PAYMENT_REQUEST,
                IntentType.CARD_OPERATIONS, IntentType.LOAN_REQUEST,
                IntentType.ACCOUNT_MODIFICATION, IntentType.INVESTMENT_OPERATIONS,
                IntentType.UNSUPPORTED_ACTION, IntentType.OUT_OF_SCOPE,
                IntentType.UNKNOWN, IntentType.AMBIGUOUS
            }:
                harena_warnings.append("Intent not in Harena taxonomy")
                is_valid = False
        
        # Check for missed unsupported actions
        action_keywords = ['virement', 'payer', 'bloquer', 'transférer', 'modifier']
        if any(keyword in user_message.lower() for keyword in action_keywords):
            if IntentTaxonomy.is_supported_by_harena(intent_result.intent):
                harena_warnings.append("Possible unsupported action classified as supported")
        
        return is_valid, warnings + harena_warnings
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        cache_stats = self.cache.get_stats() if self.cache_enabled else {}
        
        total_classifications = self.classification_count
        cache_hit_rate = self.cache_hits / total_classifications if total_classifications > 0 else 0
        fallback_rate = self.fallback_uses / total_classifications if total_classifications > 0 else 0
        error_rate = self.error_count / total_classifications if total_classifications > 0 else 0
        
        return {
            "agent_name": "intent_classifier",
            "total_classifications": total_classifications,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "fallback_uses": self.fallback_uses,
            "error_count": self.error_count,
            "cache_hit_rate": cache_hit_rate,
            "fallback_rate": fallback_rate,
            "error_rate": error_rate,
            "cache_stats": cache_stats
        }
    
    async def cleanup(self):
        """Cleanup resources and clear expired cache entries."""
        if self.cache_enabled:
            self.cache.clear_expired()
            logger.info("Intent classifier cleanup complete", 
                       cache_size=len(self.cache.cache))
    
    # Utility methods for testing and debugging
    
    async def test_classification(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test classification with predefined cases."""
        results = {
            "total_cases": len(test_cases),
            "correct": 0,
            "incorrect": 0,
            "details": []
        }
        
        for i, case in enumerate(test_cases):
            user_message = case["input"]
            expected_intent = case["expected_intent"]
            
            response = await self.classify_intent(user_message)
            
            if response.success:
                actual_intent = response.result["intent"]
                is_correct = actual_intent == expected_intent
                
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
                
                results["details"].append({
                    "case_id": i,
                    "input": user_message,
                    "expected": expected_intent,
                    "actual": actual_intent,
                    "confidence": response.result["confidence"],
                    "correct": is_correct,
                    "processing_time_ms": response.processing_time_ms
                })
            else:
                results["incorrect"] += 1
                results["details"].append({
                    "case_id": i,
                    "input": user_message,
                    "expected": expected_intent,
                    "actual": "ERROR",
                    "error": response.error_message,
                    "correct": False
                })
        
        accuracy = results["correct"] / results["total_cases"] if results["total_cases"] > 0 else 0
        results["accuracy"] = accuracy
        
        return results
    
    def get_classification_explanation(self, user_message: str) -> Dict[str, Any]:
        """Get detailed explanation of how a message would be classified."""
        explanation = {
            "user_message": user_message,
            "analysis": {}
        }
        
        # Unsupported action analysis
        is_action, action_confidence, action_patterns = UnsupportedActionDetector.detect_unsupported_action(user_message)
        explanation["analysis"]["unsupported_action"] = {
            "detected": is_action,
            "confidence": action_confidence,
            "patterns": action_patterns
        }
        
        # Keyword analysis
        keyword_candidates = IntentClassificationMatrix.classify_by_keywords(user_message)
        explanation["analysis"]["keyword_classification"] = [
            {"intent": intent.value, "confidence": conf}
            for intent, conf in keyword_candidates[:5]
        ]
        
        # Harena scope analysis
        explanation["analysis"]["harena_scope"] = {
            "likely_supported": not is_action and len(keyword_candidates) > 0,
            "confidence_thresholds": {
                "consultation": ConfidenceThresholds.CONSULTATION_MIN,
                "analysis": ConfidenceThresholds.ANALYSIS_MIN,
                "unsupported_action": ConfidenceThresholds.UNSUPPORTED_ACTION_MIN
            }
        }
        
        return explanation