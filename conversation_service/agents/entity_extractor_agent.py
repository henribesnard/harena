"""
Entity Extraction Agent for Harena Conversation Service.

This module implements sophisticated financial entity extraction using a hybrid approach:
LLM-powered extraction combined with rule-based validation for maximum accuracy.
Designed specifically for Harena's consultation scope with dynamic normalization.

Key Features:
- Hybrid extraction: LLM + Rules for comprehensive entity detection
- Dynamic normalization without static mappings (LLM handles edge cases)
- Harena scope validation and action entity flagging
- Intelligent caching for frequent patterns
- Integration with Elasticsearch field structure
- Context-aware extraction with confidence scoring

Author: Harena Conversation Team
Created: 2025-01-31
Version: 1.0.0 - Hybrid LLM + Rules Approach
"""

import asyncio
import json
import time
import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

# AutoGen imports
from autogen import AssistantAgent
from openai import AsyncOpenAI

# Local imports
from ..models.core_models import (
    EntityType, FinancialEntity, AgentResponse, ConversationState, 
    HarenaValidators, IntentType
)

__all__ = ["EntityExtractorAgent", "EntityExtractionCache", "EntityPromptManager", "HybridEntityExtractor"]

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# ENTITY EXTRACTION CACHE
# ================================

@dataclass
class CachedEntityResult:
    """Cached entity extraction result with metadata."""
    entities: List[FinancialEntity]
    timestamp: datetime
    hit_count: int = 1
    context_hash: str = ""
    
    def is_expired(self, ttl_minutes: int = 20) -> bool:
        """Check if cached result is expired (shorter TTL for entities)."""
        return datetime.now() - self.timestamp > timedelta(minutes=ttl_minutes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "entities": [entity.dict() for entity in self.entities],
            "cached": True,
            "cache_hits": self.hit_count,
            "cache_timestamp": self.timestamp.isoformat()
        }

class EntityExtractionCache:
    """
    Intelligent caching for entity extraction results.
    
    Optimized for financial entity patterns with context-aware caching
    and intelligent invalidation strategies.
    """
    
    def __init__(self, max_size: int = 500, default_ttl_minutes: int = 20):
        self.max_size = max_size
        self.default_ttl_minutes = default_ttl_minutes
        self.cache: Dict[str, CachedEntityResult] = {}
        self.access_times: Dict[str, datetime] = {}
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_cache_key(self, user_message: str, intent: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate context-aware cache key."""
        import hashlib
        
        # Normalize message for better cache hits
        normalized = re.sub(r'\d+', 'NUM', user_message.lower().strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Include intent as it affects entity extraction
        cache_data = f"{normalized}|{intent}"
        
        # Include relevant context
        if context:
            stable_context = {k: v for k, v in context.items() if k in ['user_id', 'conversation_type']}
            if stable_context:
                cache_data += f"|{json.dumps(stable_context, sort_keys=True)}"
        
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def get(self, user_message: str, intent: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(user_message, intent, context)
        
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
            "Entity cache hit",
            cache_key=cache_key[:8],
            entities_count=len(cached_result.entities),
            hit_count=cached_result.hit_count
        )
        
        return cached_result.to_dict()
    
    def set(self, user_message: str, intent: str, entities: List[FinancialEntity], context: Optional[Dict[str, Any]] = None):
        """Cache entity extraction result."""
        cache_key = self._generate_cache_key(user_message, intent, context)
        
        # Don't cache empty results or low-confidence entities
        if not entities or all(e.confidence < 0.6 for e in entities):
            return
        
        # Create cached result
        cached_result = CachedEntityResult(
            entities=entities,
            timestamp=datetime.now(),
            context_hash=cache_key
        )
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[cache_key] = cached_result
        self.access_times[cache_key] = datetime.now()
        
        logger.debug(
            "Entities cached",
            cache_key=cache_key[:8],
            entities_count=len(entities)
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
        logger.debug("Evicted LRU entity cache entry", cache_key=lru_key[:8])
    
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
            "evictions": self.evictions
        }

# ================================
# ENTITY PROMPT MANAGEMENT
# ================================

class EntityPromptManager:
    """
    Manages prompts for entity extraction with Harena-specific optimizations.
    
    Provides dynamic prompts that adapt to intent context and entity types
    without static mappings, relying on LLM intelligence for edge cases.
    """
    
    # Core system prompt optimized for dynamic extraction
    HARENA_ENTITY_SYSTEM_PROMPT = """Tu es un expert en extraction d'entités financières pour Harena - Assistant bancaire consultatif.

🎯 TON RÔLE :
Extraire TOUTES les entités financières pertinentes du message utilisateur avec une normalisation intelligente.

📋 TYPES D'ENTITÉS À EXTRAIRE :

**ENTITÉS MONÉTAIRES :**
- AMOUNT : Montants, prix, valeurs (ex: "500€", "cinquante euros", "5 centimes")
- CURRENCY : Devises (ex: "euros", "dollars", "€", "$", "USD", "EUR") 
- PERCENTAGE : Pourcentages (ex: "15%", "quinze pour cent", "0.5 pourcent")

**ENTITÉS TEMPORELLES :**
- DATE_RANGE : Périodes, dates (ex: "ce mois", "janvier 2024", "la semaine dernière", "entre le 1er et le 15")
- FREQUENCY : Fréquences (ex: "mensuel", "tous les mois", "hebdomadaire")

**ENTITÉS TRANSACTIONNELLES :**
- MERCHANT : Noms de marchands, magasins (ex: "Amazon", "Carrefour", "SNCF", "pharmacie du coin")
- CATEGORY : Catégories de dépenses (ex: "restaurant", "transport", "alimentation", "shopping")
- TRANSACTION_TYPE : Types d'opérations (ex: "achat", "retrait", "virement", "prélèvement")

**ENTITÉS GÉOGRAPHIQUES :**
- LOCATION : Lieux, villes (ex: "Paris", "en France", "à Londres")

**ENTITÉS PRODUITS :**
- ACCOUNT_TYPE : Types de comptes (ex: "compte courant", "livret A", "épargne")
- CARD_TYPE : Types de cartes (ex: "carte visa", "mastercard", "carte bleue")

**ENTITÉS D'ACTION (À SIGNALER) :**
- BENEFICIARY : Bénéficiaires virements (ex: "Jean Dupont", "ma mère", "mon compte épargne")
- AUTHENTICATION : Codes, PIN (ex: "code secret", "PIN", "authentification")

🔧 RÈGLES DE NORMALISATION DYNAMIQUE :

1. **AMOUNT** : Format décimal (ex: "500 euros" → "500.00")
2. **CURRENCY** : Code standard ou nom (ex: "euros" → "EUR", "dollars" → "USD") 
3. **MERCHANT** : Nom propre capitalisé (ex: "amazon" → "Amazon")
4. **CATEGORY** : Minuscules, terme principal (ex: "restaurants" → "restaurant")
5. **DATE_RANGE** : Description normalisée (ex: "ce mois-ci" → "current_month")

⚠️ ATTENTION HARENA :
- Extrais TOUTES les entités même pour les actions non supportées
- Marque les entités d'action (BENEFICIARY, AUTHENTICATION) pour redirection
- Sois créatif et intelligent pour les variantes et synonymes
- Gère les fautes d'orthographe et abréviations

🎯 CONFIANCE :
- 0.9+ : Entité très claire et évidente
- 0.7-0.9 : Entité probable, bon contexte
- 0.5-0.7 : Entité possible, contexte ambigu
- <0.5 : Entité incertaine

FORMAT RÉPONSE OBLIGATOIRE (JSON strict) :
{"entities": [{"type": "ENTITY_TYPE", "raw": "texte original", "normalized": "valeur normalisée", "confidence": 0.85}]}

Sois exhaustif, intelligent et adaptatif !"""

    # Few-shot examples for complex entity scenarios
    FEW_SHOT_EXAMPLES = [
        {
            "input": "Mes achats Amazon de 150 euros cette semaine",
            "output": '{"entities": [{"type": "MERCHANT", "raw": "Amazon", "normalized": "Amazon", "confidence": 0.95}, {"type": "AMOUNT", "raw": "150 euros", "normalized": "150.00", "confidence": 0.98}, {"type": "CURRENCY", "raw": "euros", "normalized": "EUR", "confidence": 0.95}, {"type": "DATE_RANGE", "raw": "cette semaine", "normalized": "current_week", "confidence": 0.90}]}'
        },
        {
            "input": "Restaurants parisiens en janvier pour environ 200€",
            "output": '{"entities": [{"type": "CATEGORY", "raw": "Restaurants", "normalized": "restaurant", "confidence": 0.93}, {"type": "LOCATION", "raw": "parisiens", "normalized": "Paris", "confidence": 0.88}, {"type": "DATE_RANGE", "raw": "janvier", "normalized": "2024-01", "confidence": 0.85}, {"type": "AMOUNT", "raw": "200€", "normalized": "200.00", "confidence": 0.90}]}'
        },
        {
            "input": "Virement de mille dollars vers mon épargne",
            "output": '{"entities": [{"type": "AMOUNT", "raw": "mille dollars", "normalized": "1000.00", "confidence": 0.92}, {"type": "CURRENCY", "raw": "dollars", "normalized": "USD", "confidence": 0.95}, {"type": "ACCOUNT_TYPE", "raw": "épargne", "normalized": "savings", "confidence": 0.87}, {"type": "BENEFICIARY", "raw": "mon épargne", "normalized": "user_savings_account", "confidence": 0.85}]}'
        },
        {
            "input": "Mes dépenses transport ce mois-ci",
            "output": '{"entities": [{"type": "CATEGORY", "raw": "transport", "normalized": "transport", "confidence": 0.95}, {"type": "DATE_RANGE", "raw": "ce mois-ci", "normalized": "current_month", "confidence": 0.92}]}'
        },
        {
            "input": "Transactions supérieures à cinquante euros chez Carrefour",
            "output": '{"entities": [{"type": "AMOUNT", "raw": "cinquante euros", "normalized": "50.00", "confidence": 0.90}, {"type": "CURRENCY", "raw": "euros", "normalized": "EUR", "confidence": 0.95}, {"type": "MERCHANT", "raw": "Carrefour", "normalized": "Carrefour", "confidence": 0.95}]}'
        },
        {
            "input": "Frais de ma carte visa ce trimestre",
            "output": '{"entities": [{"type": "CARD_TYPE", "raw": "carte visa", "normalized": "visa", "confidence": 0.88}, {"type": "DATE_RANGE", "raw": "ce trimestre", "normalized": "current_quarter", "confidence": 0.85}]}'
        }
    ]
    
    @classmethod
    def build_extraction_prompt(
        cls,
        user_message: str,
        intent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build extraction prompt with intent-specific context."""
        prompt_parts = [f"Message utilisateur : {user_message}"]
        
        # Add intent context for better extraction
        if intent:
            intent_guidance = {
                "CATEGORY_ANALYSIS": "Focus sur l'extraction de CATEGORY et DATE_RANGE",
                "MERCHANT_ANALYSIS": "Focus sur l'extraction de MERCHANT et éventuellement DATE_RANGE",
                "SPENDING_ANALYSIS": "Focus sur AMOUNT, DATE_RANGE et CATEGORY",
                "TRANSACTION_SEARCH": "Extrais tous critères : MERCHANT, CATEGORY, AMOUNT, DATE_RANGE",
                "TRANSFER_REQUEST": "Extrais AMOUNT, CURRENCY, BENEFICIARY (action non supportée)",
                "PAYMENT_REQUEST": "Extrais AMOUNT, MERCHANT, CATEGORY (action non supportée)"
            }
            
            guidance = intent_guidance.get(intent, "Extrais toutes les entités financières pertinentes")
            prompt_parts.append(f"Intention détectée : {intent}")
            prompt_parts.append(f"Guidance : {guidance}")
        
        # Add conversation context if available
        if context:
            if context.get("previous_entities"):
                prev_entities = context["previous_entities"][:3]  # Last 3 entities
                context_text = ", ".join([f"{e.get('type')}:{e.get('normalized')}" for e in prev_entities])
                prompt_parts.append(f"Entités récentes : {context_text}")
            
            if context.get("user_patterns"):
                patterns = context["user_patterns"]
                if patterns.get("frequent_merchants"):
                    frequent = patterns["frequent_merchants"][:3]
                    prompt_parts.append(f"Marchands fréquents : {', '.join(frequent)}")
        
        prompt_parts.extend([
            "",
            "Extrais TOUTES les entités financières du message.",
            "Utilise ta connaissance pour gérer les variantes, synonymes et fautes.",
            "Réponds UNIQUEMENT en JSON valide avec la structure exacte :",
            '{"entities": [{"type": "ENTITY_TYPE", "raw": "texte", "normalized": "valeur", "confidence": 0.XX}]}'
        ])
        
        return "\n".join(prompt_parts)
    
    @classmethod
    def get_intent_specific_examples(cls, intent: str) -> List[Dict[str, str]]:
        """Get examples specific to the detected intent."""
        intent_examples = {
            "CATEGORY_ANALYSIS": [
                cls.FEW_SHOT_EXAMPLES[0],  # Amazon example
                cls.FEW_SHOT_EXAMPLES[1],  # Restaurant example
                cls.FEW_SHOT_EXAMPLES[3]   # Transport example
            ],
            "MERCHANT_ANALYSIS": [
                cls.FEW_SHOT_EXAMPLES[0],  # Amazon
                cls.FEW_SHOT_EXAMPLES[4],  # Carrefour
                cls.FEW_SHOT_EXAMPLES[1]   # Paris restaurants
            ],
            "TRANSFER_REQUEST": [
                cls.FEW_SHOT_EXAMPLES[2],  # Virement example
            ],
            "SPENDING_ANALYSIS": [
                cls.FEW_SHOT_EXAMPLES[1],  # Restaurant spending
                cls.FEW_SHOT_EXAMPLES[4],  # Amount example
                cls.FEW_SHOT_EXAMPLES[3]   # Transport spending
            ]
        }
        
        return intent_examples.get(intent, cls.FEW_SHOT_EXAMPLES[:3])

# ================================
# HYBRID ENTITY EXTRACTION SYSTEM
# ================================

class HybridEntityExtractor:
    """
    Hybrid entity extraction combining rules and LLM intelligence.
    
    Uses rule-based extraction for high-precision common patterns,
    then LLM for complex cases and normalization validation.
    """
    
    # Rule-based patterns for high-confidence extraction
    AMOUNT_PATTERNS = [
        (r'(\d+(?:[.,]\d+)?)\s*(?:€|euros?|eur)\b', 'EUR'),
        (r'(\d+(?:[.,]\d+)?)\s*(?:\$|dollars?|usd)\b', 'USD'),
        (r'(?:€|euros?)\s*(\d+(?:[.,]\d+)?)\b', 'EUR'),
        (r'\b((?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|dix-sept|dix-huit|dix-neuf|vingt|trente|quarante|cinquante|soixante|soixante-dix|quatre-vingt|quatre-vingt-dix|cent|mille)+)\s*(?:€|euros?|dollars?)\b', None)
    ]
    
    MERCHANT_PATTERNS = [
        r'\b(amazon|carrefour|fnac|sncf|uber|leclerc|auchan|monoprix|picard|ikea|h&m|zara|decathlon)\b',
        r'\bchez\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b([A-Z][A-Z0-9]{2,})\b'  # Uppercase acronyms like EDF, SNCF
    ]
    
    CATEGORY_PATTERNS = [
        r'\b(restaurant|alimentation|transport|shopping|santé|logement|loisirs|voyage|éducation|finance|assurance)\b',
        r'\b(resto|bouffe|nourriture|épicerie|courses)\b',  # Informal terms
        r'\b(métro|bus|taxi|essence|carburant|train)\b'     # Transport specifics
    ]
    
    DATE_PATTERNS = [
        (r'\b(aujourd\'hui|hier|demain)\b', lambda m: m.group(1)),
        (r'\b(cette\s+semaine|la\s+semaine\s+dernière|semaine\s+prochaine)\b', lambda m: m.group(1).replace(' ', '_')),
        (r'\b(ce\s+mois|le\s+mois\s+dernier|mois\s+prochain)\b', lambda m: m.group(1).replace(' ', '_')),
        (r'\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b', lambda m: m.group(1)),
        (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}")
    ]
    
    @classmethod
    def extract_with_rules(cls, user_message: str) -> List[FinancialEntity]:
        """Extract entities using rule-based patterns."""
        entities = []
        message_lower = user_message.lower()
        
        # Extract amounts
        for pattern, currency in cls.AMOUNT_PATTERNS:
            matches = re.finditer(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                amount_text = match.group(1) if match.groups() else match.group(0)
                
                # Handle written numbers
                if any(word in amount_text for word in ['un', 'deux', 'trois', 'cent', 'mille']):
                    normalized_amount = cls._convert_written_number(amount_text)
                else:
                    normalized_amount = amount_text.replace(',', '.')
                
                try:
                    float(normalized_amount)
                    entities.append(FinancialEntity(
                        entity_type=EntityType.AMOUNT,
                        raw_value=match.group(0),
                        normalized_value=f"{float(normalized_amount):.2f}",
                        confidence=0.9,
                        start_position=match.start(),
                        end_position=match.end()
                    ))
                    
                    # Add currency if detected
                    if currency:
                        entities.append(FinancialEntity(
                            entity_type=EntityType.CURRENCY,
                            raw_value=currency.lower(),
                            normalized_value=currency,
                            confidence=0.9
                        ))
                except ValueError:
                    continue
        
        # Extract merchants
        for pattern in cls.MERCHANT_PATTERNS:
            matches = re.finditer(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                merchant_name = match.group(1) if match.groups() else match.group(0)
                entities.append(FinancialEntity(
                    entity_type=EntityType.MERCHANT,
                    raw_value=match.group(0),
                    normalized_value=merchant_name.title(),
                    confidence=0.8,
                    start_position=match.start(),
                    end_position=match.end()
                ))
        
        # Extract categories
        for pattern in cls.CATEGORY_PATTERNS:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                category = match.group(0)
                # Dynamic normalization mapping
                normalized_category = cls._normalize_category(category)
                entities.append(FinancialEntity(
                    entity_type=EntityType.CATEGORY,
                    raw_value=category,
                    normalized_value=normalized_category,
                    confidence=0.7,
                    start_position=match.start(),
                    end_position=match.end()
                ))
        
        # Extract date ranges
        for pattern, normalizer in cls.DATE_PATTERNS:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                raw_date = match.group(0)
                normalized_date = normalizer(match) if callable(normalizer) else raw_date
                entities.append(FinancialEntity(
                    entity_type=EntityType.DATE_RANGE,
                    raw_value=raw_date,
                    normalized_value=normalized_date,
                    confidence=0.8,
                    start_position=match.start(),
                    end_position=match.end()
                ))
        
        return entities
    
    @classmethod
    def _convert_written_number(cls, text: str) -> str:
        """Convert written French numbers to digits."""
        # Basic written number conversion
        number_map = {
            'un': '1', 'une': '1', 'deux': '2', 'trois': '3', 'quatre': '4', 'cinq': '5',
            'six': '6', 'sept': '7', 'huit': '8', 'neuf': '9', 'dix': '10',
            'vingt': '20', 'trente': '30', 'quarante': '40', 'cinquante': '50',
            'soixante': '60', 'cent': '100', 'mille': '1000'
        }
        
        # Simple replacement for basic numbers
        for word, number in number_map.items():
            if word in text.lower():
                return number
        
        return text
    
    @classmethod
    def _normalize_category(cls, category: str) -> str:
        """Dynamic category normalization."""
        # Basic normalization without static mappings
        category_lower = category.lower().strip()
        
        # Handle common variations
        if category_lower in ['resto', 'bouffe', 'nourriture']:
            return 'alimentation'
        elif category_lower in ['métro', 'bus', 'taxi', 'essence', 'carburant', 'train']:
            return 'transport'
        elif category_lower in ['courses', 'épicerie']:
            return 'alimentation'
        else:
            return category_lower
    
    @classmethod
    def merge_entities(cls, rule_entities: List[FinancialEntity], llm_entities: List[FinancialEntity]) -> List[FinancialEntity]:
        """Intelligently merge rule-based and LLM entities."""
        merged = []
        
        # Start with rule-based entities (higher precision)
        for rule_entity in rule_entities:
            merged.append(rule_entity)
        
        # Add LLM entities that don't overlap
        for llm_entity in llm_entities:
            is_duplicate = False
            
            for rule_entity in rule_entities:
                # Check for overlap by type and similarity
                if (rule_entity.entity_type == llm_entity.entity_type and
                    cls._entities_overlap(rule_entity, llm_entity)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(llm_entity)
        
        # Sort by confidence and position
        return sorted(merged, key=lambda x: (x.confidence, -(x.start_position or 0)), reverse=True)
    
    @classmethod
    def _entities_overlap(cls, entity1: FinancialEntity, entity2: FinancialEntity) -> bool:
        """Check if two entities overlap in meaning or position."""
        # Position overlap
        if (entity1.start_position is not None and entity2.start_position is not None and
            entity1.end_position is not None and entity2.end_position is not None):
            
            overlap = not (entity1.end_position <= entity2.start_position or 
                          entity2.end_position <= entity1.start_position)
            if overlap:
                return True
        
        # Semantic overlap (similar normalized values)
        if (entity1.entity_type == entity2.entity_type and 
            entity1.normalized_value.lower() == entity2.normalized_value.lower()):
            return True
        
        return False

# ================================
# MAIN ENTITY EXTRACTOR AGENT
# ================================

class EntityExtractorAgent:
    """
    AutoGen-based entity extraction agent specialized for Harena scope.
    
    Implements sophisticated hybrid extraction combining rule-based patterns
    with LLM intelligence for comprehensive financial entity detection.
    
    Features:
    - Hybrid extraction: Rules + LLM for maximum coverage
    - Dynamic normalization without static mappings
    - Intent-aware extraction optimization
    - Intelligent caching for performance
    - Action entity flagging for Harena scope compliance
    """
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str = "gpt-4",
        cache_enabled: bool = True,
        cache_size: int = 500,
        enable_validation: bool = True
    ):
        """
        Initialize Entity Extractor Agent.
        
        Args:
            openai_client: OpenAI async client
            model_name: Model to use for LLM extraction
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
            self.cache = EntityExtractionCache(max_size=cache_size)
        
        # Initialize AutoGen agent
        self.agent = AssistantAgent(
            name="entity_extractor",
            system_message=EntityPromptManager.HARENA_ENTITY_SYSTEM_PROMPT,
            llm_config={
                "model": model_name,
                "temperature": 0.0,  # Deterministic for entity extraction
                "max_tokens": 500,   # Sufficient for entity JSON
                "timeout": 10
            }
        )
        
        # Hybrid extractor
        self.hybrid_extractor = HybridEntityExtractor()
        
        # Performance tracking
        self.extraction_count = 0
        self.cache_hits = 0
        self.llm_calls = 0
        self.rule_uses = 0
        self.error_count = 0
        
        logger.info(
            "Entity Extractor Agent initialized",
            model=model_name,
            cache_enabled=cache_enabled,
            validation_enabled=enable_validation
        )
    
    async def extract_entities(
        self,
        user_message: str,
        intent: str,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_state: Optional[ConversationState] = None
    ) -> AgentResponse:
        """
        Extract financial entities with hybrid approach.
        
        Args:
            user_message: User input text
            intent: Detected intent for context
            user_id: User identifier
            context: Additional context
            conversation_state: Current conversation state
            
        Returns:
            AgentResponse with extracted entities
        """
        start_time = time.time()
        self.extraction_count += 1
        
        try:
            # Step 1: Check cache
            if self.cache_enabled:
                cached_result = self.cache.get(user_message, intent, context)
                if cached_result:
                    self.cache_hits += 1
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    logger.debug(
                        "Entity extraction cache hit",
                        user_message=user_message[:50],
                        intent=intent,
                        entities_count=len(cached_result["entities"]),
                        processing_time_ms=processing_time
                    )
                    
                    return AgentResponse(
                        agent_name="entity_extractor",
                        success=True,
                        result=cached_result,
                        processing_time_ms=processing_time,
                        cached=True
                    )
            
            # Step 2: Rule-based extraction
            rule_entities = self.hybrid_extractor.extract_with_rules(user_message)
            self.rule_uses += 1
            
            logger.debug(
                "Rule-based extraction",
                entities_found=len(rule_entities),
                entity_types=[e.entity_type.value for e in rule_entities]
            )
            
            # Step 3: LLM extraction for completeness
            llm_entities = []
            try:
                llm_entities = await self._extract_with_llm(user_message, intent, context)
                self.llm_calls += 1
            except Exception as e:
                logger.warning(f"LLM extraction failed, using rules only: {e}")
            
            # Step 4: Merge and deduplicate
            final_entities = self.hybrid_extractor.merge_entities(rule_entities, llm_entities)
            
            # Step 5: Validation and filtering
            if self.enable_validation:
                final_entities = await self._validate_entities(final_entities, intent, user_message)
            
            # Step 6: Flag action entities for Harena scope
            for entity in final_entities:
                if entity.is_action_related():
                    entity.context = f"action_entity_harena_unsupported"
            
            # Step 7: Cache high-quality results
            if self.cache_enabled and final_entities and any(e.confidence >= 0.7 for e in final_entities):
                self.cache.set(user_message, intent, final_entities, context)
            
            # Step 8: Update conversation state
            if conversation_state:
                conversation_state.update_stage("entity", {
                    "entities": [entity.dict() for entity in final_entities],
                    "extraction_methods": {
                        "rules": len(rule_entities),
                        "llm": len(llm_entities),
                        "final": len(final_entities)
                    }
                })
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = {
                "entities": [entity.dict() for entity in final_entities],
                "extraction_metadata": {
                    "rule_entities_count": len(rule_entities),
                    "llm_entities_count": len(llm_entities),
                    "final_entities_count": len(final_entities),
                    "has_action_entities": any(e.is_action_related() for e in final_entities),
                    "confidence_distribution": self._get_confidence_distribution(final_entities)
                }
            }
            
            logger.info(
                "Entity extraction complete",
                user_message=user_message[:50],
                intent=intent,
                entities_count=len(final_entities),
                processing_time_ms=processing_time,
                has_action_entities=result["extraction_metadata"]["has_action_entities"]
            )
            
            return AgentResponse(
                agent_name="entity_extractor",
                success=True,
                result=result,
                processing_time_ms=processing_time,
                cached=False
            )
            
        except Exception as e:
            self.error_count += 1
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.error(
                "Entity extraction failed",
                user_message=user_message[:50],
                intent=intent,
                error=str(e),
                processing_time_ms=processing_time,
                exc_info=True
            )
            
            # Return fallback result
            fallback_result = {
                "entities": [],
                "extraction_metadata": {
                    "error": str(e),
                    "fallback_used": True
                }
            }
            
            return AgentResponse(
                agent_name="entity_extractor",
                success=False,
                result=fallback_result,
                error_message=str(e),
                processing_time_ms=processing_time,
                cached=False
            )
    
    async def _extract_with_llm(
        self,
        user_message: str,
        intent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[FinancialEntity]:
        """Extract entities using LLM with intent-specific prompts."""
        
        # Build intent-specific prompt
        prompt = EntityPromptManager.build_extraction_prompt(user_message, intent, context)
        
        # Get intent-specific examples
        examples = EntityPromptManager.get_intent_specific_examples(intent)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": EntityPromptManager.HARENA_ENTITY_SYSTEM_PROMPT}
        ]
        
        # Add few-shot examples
        for example in examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
        
        # Add current extraction request
        messages.append({"role": "user", "content": prompt})
        
        # Call OpenAI API
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        response_content = response.choices[0].message.content
        
        try:
            result_data = json.loads(response_content)
            entities = []
            
            for entity_data in result_data.get("entities", []):
                try:
                    entity_type = EntityType(entity_data.get("type"))
                    entity = FinancialEntity(
                        entity_type=entity_type,
                        raw_value=entity_data.get("raw", ""),
                        normalized_value=entity_data.get("normalized", ""),
                        confidence=float(entity_data.get("confidence", 0.5)),
                        context=f"llm_extracted_{intent}"
                    )
                    entities.append(entity)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid entity in LLM response: {entity_data}, error: {e}")
                    continue
            
            return entities
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM entity response: {e}")
            return []
    
    async def _validate_entities(
        self,
        entities: List[FinancialEntity],
        intent: str,
        user_message: str
    ) -> List[FinancialEntity]:
        """Validate and filter entities based on intent and quality."""
        validated_entities = []
        
        for entity in entities:
            # Basic confidence threshold
            if entity.confidence < 0.3:
                continue
            
            # Intent-specific validation
            if intent == "CATEGORY_ANALYSIS" and entity.entity_type != EntityType.CATEGORY:
                # Lower confidence for non-category entities in category analysis
                entity.confidence *= 0.8
            
            elif intent == "MERCHANT_ANALYSIS" and entity.entity_type != EntityType.MERCHANT:
                # Lower confidence for non-merchant entities
                entity.confidence *= 0.8
            
            elif intent in ["TRANSFER_REQUEST", "PAYMENT_REQUEST"]:
                # Boost action-related entities for unsupported actions
                if entity.entity_type in [EntityType.BENEFICIARY, EntityType.AMOUNT]:
                    entity.confidence = min(entity.confidence * 1.1, 1.0)
            
            # Content validation using Harena validators
            try:
                if entity.entity_type == EntityType.AMOUNT:
                    normalized = HarenaValidators.normalize_dynamic_amount(entity.raw_value)
                    if normalized:
                        entity.normalized_value = normalized
                    else:
                        continue  # Skip invalid amounts
                
                validated_entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Entity validation failed: {entity.dict()}, error: {e}")
                continue
        
        return validated_entities
    
    def _get_confidence_distribution(self, entities: List[FinancialEntity]) -> Dict[str, int]:
        """Get confidence distribution for monitoring."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for entity in entities:
            if entity.confidence >= 0.8:
                distribution["high"] += 1
            elif entity.confidence >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        cache_stats = self.cache.get_stats() if self.cache_enabled else {}
        
        total_extractions = self.extraction_count
        cache_hit_rate = self.cache_hits / total_extractions if total_extractions > 0 else 0
        error_rate = self.error_count / total_extractions if total_extractions > 0 else 0
        
        return {
            "agent_name": "entity_extractor",
            "total_extractions": total_extractions,
            "llm_calls": self.llm_calls,
            "rule_uses": self.rule_uses,
            "cache_hits": self.cache_hits,
            "error_count": self.error_count,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": error_rate,
            "cache_stats": cache_stats
        }
    
    async def cleanup(self):
        """Cleanup resources and cache."""
        if self.cache_enabled:
            logger.info("Entity extractor cleanup complete", 
                       cache_size=len(self.cache.cache))
    
    # Testing and debugging utilities
    
    async def test_extraction(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test entity extraction with predefined cases."""
        results = {
            "total_cases": len(test_cases),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        for i, case in enumerate(test_cases):
            user_message = case["input"]
            intent = case.get("intent", "TRANSACTION_SEARCH")
            expected_entities = case.get("expected_entities", [])
            
            response = await self.extract_entities(user_message, intent)
            
            if response.success:
                extracted_entities = response.result["entities"]
                results["successful"] += 1
                
                # Compare with expected
                precision, recall = self._calculate_extraction_metrics(extracted_entities, expected_entities)
                
                results["details"].append({
                    "case_id": i,
                    "input": user_message,
                    "intent": intent,
                    "extracted_count": len(extracted_entities),
                    "expected_count": len(expected_entities),
                    "precision": precision,
                    "recall": recall,
                    "processing_time_ms": response.processing_time_ms
                })
            else:
                results["failed"] += 1
                results["details"].append({
                    "case_id": i,
                    "input": user_message,
                    "error": response.error_message
                })
        
        return results
    
    def _calculate_extraction_metrics(self, extracted: List[Dict], expected: List[Dict]) -> Tuple[float, float]:
        """Calculate precision and recall for entity extraction."""
        if not expected:
            return 1.0 if not extracted else 0.0, 1.0
        
        if not extracted:
            return 0.0, 0.0
        
        # Simple matching by entity type (could be more sophisticated)
        extracted_types = {e["entity_type"] for e in extracted}
        expected_types = {e["entity_type"] for e in expected}
        
        true_positives = len(extracted_types & expected_types)
        precision = true_positives / len(extracted_types) if extracted_types else 0.0
        recall = true_positives / len(expected_types) if expected_types else 0.0
        
        return precision, recall