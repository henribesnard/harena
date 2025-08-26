"""
Agent de classification d'intentions financières via DeepSeek
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_DESCRIPTIONS, INTENT_CATEGORIES
from conversation_service.prompts.few_shot_examples.intent_classification import get_balanced_few_shot_examples
from conversation_service.prompts.system_prompts import INTENT_CLASSIFICATION_SYSTEM_PROMPT
from conversation_service.models.responses.conversation_responses import IntentClassificationResult, IntentAlternative
from conversation_service.utils.validation_utils import validate_intent_response, sanitize_user_input
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.intent_classifier")

class IntentClassifierAgent(BaseAgent):
    """Agent classification intentions financières Harena via DeepSeek"""
    
    def __init__(self, deepseek_client, cache_manager):
        super().__init__(
            name="intent_classifier",
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )
        
        self.supported_intents = list(HarenaIntentType)
        self.few_shot_examples = get_balanced_few_shot_examples(examples_per_intent=2)
        
        logger.info(f"IntentClassifier initialisé avec {len(self.few_shot_examples)} exemples")
    
    async def execute(
        self, 
        input_data: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> IntentClassificationResult:
        """Exécution classification intention"""
        return await self.classify_intent(input_data, context)
    
    async def classify_intent(
        self,
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> IntentClassificationResult:
        """Classification intention principale"""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Nettoyage et validation input
            clean_message = sanitize_user_input(user_message)
            
            if not clean_message:
                return self._create_unknown_intent_result("Message vide après nettoyage")
            
            if len(clean_message) > 500:
                return self._create_unclear_intent_result("Message trop long")
            
            # Vérification cache sémantique
            cache_key = self._generate_cache_key(clean_message)
            cached_result = await self.cache_manager.get_semantic_cache(
                cache_key, 
                similarity_threshold=0.85
            )
            
            if cached_result:
                self._update_metrics("cache_hit", start_time)
                result = IntentClassificationResult(**cached_result)
                result.processing_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                return result
            
            # Construction prompt avec few-shots
            prompt = self._build_classification_prompt(clean_message, user_context)
            
            # Appel DeepSeek
            response = await self.deepseek_client.chat_completion(
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parsing et validation réponse
            classification_result = self._parse_deepseek_response(response, clean_message)
            
            # Validation qualité
            validation_ok = await validate_intent_response(classification_result)
            if not validation_ok:
                logger.warning(f"Validation échouée pour message: {clean_message[:50]}...")
                return self._create_unclear_intent_result("Validation qualité échouée")
            
            # Temps de traitement
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            classification_result.processing_time_ms = processing_time
            
            # Cache résultat
            await self.cache_manager.set_semantic_cache(
                cache_key,
                classification_result.dict(),
                ttl=settings.REDIS_CONVERSATION_TTL
            )
            
            self._update_metrics("classification_success", start_time)
            return classification_result
            
        except Exception as e:
            self._update_metrics("classification_error", start_time)
            logger.error(f"Erreur classification intention: {str(e)}")
            return self._create_error_intent_result(str(e))
    
    def _build_classification_prompt(
        self, 
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construction prompt avec few-shots équilibrés"""
        
        # Few-shot examples formatés
        examples_text = "\n".join([
            f"Message: \"{ex['input']}\" → Intention: {ex['intent']} (Confiance: {ex['confidence']})"
            for ex in self.few_shot_examples[:25]  # Limiter pour tokens
        ])
        
        # Contexte utilisateur si disponible
        context_info = ""
        if user_context:
            recent_intents = user_context.get("recent_intents", [])
            if recent_intents:
                context_info = f"\nContexte: Intentions récentes: {', '.join(recent_intents[-3:])}"
        
        # Intentions supportées groupées
        intentions_text = self._format_supported_intents()
        
        prompt = f"""Classifiez ce message utilisateur selon les intentions financières Harena.

INTENTIONS HARENA:
{intentions_text}

EXEMPLES DE CLASSIFICATION:
{examples_text}

RÈGLES:
- Retournez UNIQUEMENT un JSON: {{"intent": "TYPE_INTENTION", "confidence": 0.XX, "reasoning": "explication"}}
- Confidence entre 0.0 et 1.0
- Reasoning en français, clair et concis
- Pour intentions non supportées: utilisez le type exact (TRANSFER_REQUEST, etc.)
- Messages ambigus: UNCLEAR_INTENT
- Messages incompréhensibles: UNKNOWN

{context_info}

MESSAGE: "{user_message}"

JSON:"""
        
        return prompt
    
    def _format_supported_intents(self) -> str:
        """Formate intentions par catégorie pour prompt"""
        formatted_sections = []
        
        for category, intents in INTENT_CATEGORIES.items():
            formatted_sections.append(f"\n{category}:")
            for intent in intents[:5]:  # Limiter pour éviter prompt trop long
                description = INTENT_DESCRIPTIONS.get(intent, "")
                formatted_sections.append(f"  • {intent.value}: {description}")
            
            if len(intents) > 5:
                formatted_sections.append(f"  • ... et {len(intents)-5} autres")
        
        return "\n".join(formatted_sections)
    
    def _parse_deepseek_response(
        self, 
        response: Dict[str, Any], 
        original_message: str
    ) -> IntentClassificationResult:
        """Parse et valide la réponse DeepSeek"""
        
        try:
            content = response["choices"][0]["message"]["content"].strip()
            
            # Extraction JSON (DeepSeek peut ajouter du texte autour)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
            else:
                raise ValueError(f"Pas de JSON valide trouvé: {content[:100]}")
            
            # Validation champs obligatoires
            intent_type = parsed.get("intent", "").strip()
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = parsed.get("reasoning", "").strip()
            
            # Validation intent_type
            if intent_type not in [intent.value for intent in HarenaIntentType]:
                logger.warning(f"Intention inconnue '{intent_type}', fallback vers UNCLEAR_INTENT")
                intent_type = HarenaIntentType.UNCLEAR_INTENT.value
                confidence = 0.5
                reasoning = f"Intention '{intent_type}' non reconnue"
            
            # Validation confidence
            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"Confidence invalide {confidence}, correction vers 0.5")
                confidence = 0.5
            
            if not reasoning:
                reasoning = f"Classification automatique: {intent_type}"
            
            # Détermination catégorie et support
            category = self._get_intent_category(intent_type)
            is_supported = intent_type not in [intent.value for intent in INTENT_CATEGORIES["UNSUPPORTED"]]
            
            # Construction résultat
            result = IntentClassificationResult(
                intent_type=HarenaIntentType(intent_type),
                confidence=confidence,
                reasoning=reasoning,
                original_message=original_message,
                category=category,
                is_supported=is_supported,
                alternatives=self._generate_alternatives(parsed.get("alternatives", [])),
                processing_time_ms=None  # Sera ajouté par appelant
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON invalide DeepSeek: {str(e)}")
            return self._create_error_intent_result(f"JSON parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Erreur parsing réponse DeepSeek: {str(e)}")
            return self._create_error_intent_result(f"Response parsing error: {str(e)}")
    
    def _get_intent_category(self, intent_type: str) -> str:
        """Trouve la catégorie d'une intention"""
        try:
            intent_enum = HarenaIntentType(intent_type)
            for category, intents in INTENT_CATEGORIES.items():
                if intent_enum in intents:
                    return category
            return "UNKNOWN_CATEGORY"
        except ValueError:
            return "UNKNOWN_CATEGORY"
    
    def _generate_alternatives(self, alternatives_data: List[Dict]) -> List[IntentAlternative]:
        """Génère alternatives si fournies par DeepSeek"""
        alternatives = []
        for alt in alternatives_data[:3]:  # Max 3 alternatives
            try:
                alternatives.append(IntentAlternative(
                    intent_type=HarenaIntentType(alt["intent"]),
                    confidence=float(alt["confidence"]),
                    reasoning=alt.get("reasoning", "")
                ))
            except (KeyError, ValueError, TypeError):
                continue  # Skip alternatives invalides
        
        return alternatives
    
    def _create_unknown_intent_result(self, reason: str) -> IntentClassificationResult:
        """Résultat pour intention inconnue"""
        return IntentClassificationResult(
            intent_type=HarenaIntentType.UNKNOWN,
            confidence=0.95,
            reasoning=f"Message non interprétable: {reason}",
            original_message="",
            category="UNCLEAR_INTENT",
            is_supported=False,
            alternatives=[],
            processing_time_ms=0
        )
    
    def _create_unclear_intent_result(self, reason: str) -> IntentClassificationResult:
        """Résultat pour intention ambiguë"""
        return IntentClassificationResult(
            intent_type=HarenaIntentType.UNCLEAR_INTENT,
            confidence=0.90,
            reasoning=f"Message ambigü: {reason}",
            original_message="",
            category="UNCLEAR_INTENT", 
            is_supported=False,
            alternatives=[],
            processing_time_ms=0
        )
    
    def _create_error_intent_result(self, error: str) -> IntentClassificationResult:
        """Résultat pour erreur de traitement"""
        return IntentClassificationResult(
            intent_type=HarenaIntentType.ERROR,
            confidence=0.99,
            reasoning=f"Erreur technique: {error}",
            original_message="",
            category="UNCLEAR_INTENT",
            is_supported=False,
            alternatives=[],
            processing_time_ms=0
        )