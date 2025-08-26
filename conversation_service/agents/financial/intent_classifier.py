"""
Agent de classification d'intentions financières via DeepSeek - JSON Output Forcé
"""
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_DESCRIPTIONS, INTENT_CATEGORIES
from conversation_service.prompts.few_shot_examples.intent_classification import get_balanced_few_shot_examples, format_examples_for_prompt
from conversation_service.prompts.system_prompts import INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT
from conversation_service.models.responses.conversation_responses import IntentClassificationResult, IntentAlternative
from conversation_service.utils.validation_utils import validate_intent_response, sanitize_user_input
from conversation_service.utils.metrics_collector import metrics_collector
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.intent_classifier")

class IntentClassifierAgent(BaseAgent):
    """Agent classification intentions financières Harena via DeepSeek avec JSON Output forcé"""
    
    def __init__(self, deepseek_client, cache_manager):
        super().__init__(
            name="intent_classifier",
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )

        
        self.supported_intents = list(HarenaIntentType)
        self.few_shot_examples = get_balanced_few_shot_examples(examples_per_intent=2)
        self.confidence_threshold = getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
        
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
        """Classification intention principale avec JSON Output forcé"""
        
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
            cached_result = await self._get_cached_classification(cache_key)
            
            if cached_result:
                self._update_metrics("cache_hit", start_time)
                cached_result.processing_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                return cached_result
            
            # Construction prompt avec few-shots optimisés
            prompt = self._build_json_classification_prompt(clean_message, user_context)
            
            # Appel DeepSeek avec JSON Output FORCÉ
            response = await self.deepseek_client.chat_completion(
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=getattr(settings, 'DEEPSEEK_INTENT_MAX_TOKENS', 100),
                temperature=getattr(settings, 'DEEPSEEK_INTENT_TEMPERATURE', 0.1),
                response_format={"type": "json_object"}  # JSON FORCÉ - Plus de parsing regex
            )
            
            # Parsing JSON direct - Plus de regex
            classification_result = self._parse_json_response(response, clean_message)
            
            # Validation qualité
            if not await self._validate_classification_quality(classification_result):
                logger.warning(f"Validation qualité échouée pour: {clean_message[:50]}...")
                return self._create_unclear_intent_result("Validation qualité échouée")
            
            # Calcul temps de traitement
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            classification_result.processing_time_ms = processing_time
            
            # Cache résultat pour améliorer performances
            await self._cache_classification_result(cache_key, classification_result)
            
            self._update_metrics("classification_success", start_time)
            intent_display = (
                classification_result.intent_type.value
                if hasattr(classification_result.intent_type, "value")
                else classification_result.intent_type
            )
            logger.info(
                f"Classification réussie: {intent_display} "
                f"(conf: {classification_result.confidence:.2f}, temps: {processing_time}ms)"
            )
            
            return classification_result
            
        except Exception as e:
            self._update_metrics("classification_error", start_time)
            logger.error(f"Erreur classification intention: {str(e)}")
            return self._create_error_intent_result(str(e))
    
    def _build_json_classification_prompt(
        self, 
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construction prompt optimisé pour JSON Output forcé"""
        
        # Sélection dynamique des meilleurs exemples
        relevant_examples = self._select_relevant_examples(user_message)
        examples_text = format_examples_for_prompt(relevant_examples, format_style="concise")
        
        # Contexte utilisateur si disponible
        context_section = ""
        if user_context and user_context.get("recent_intents"):
            recent = user_context["recent_intents"][-3:]  # Dernières 3 intentions
            context_section = f"\nCONTEXTE: Intentions récentes: {', '.join(recent)}"
        
        # Intentions supportées dynamiques
        intentions_summary = self._get_dynamic_intents_summary()
        
        prompt = f"""Analysez ce message utilisateur et classifiez l'intention financière Harena.

IMPORTANT: Répondez UNIQUEMENT avec un objet JSON strict, aucun autre texte.

FORMAT JSON OBLIGATOIRE:
{{"intent": "TYPE_INTENTION_EXACT", "confidence": 0.XX, "reasoning": "explication française"}}

INTENTIONS HARENA DISPONIBLES:
{intentions_summary}

EXEMPLES DE RÉFÉRENCE:
{examples_text}

RÈGLES STRICTES:
- JSON uniquement, pas de markdown ni commentaires
- Confidence entre 0.0 et 1.0 précise
- Intent doit correspondre exactement aux types Harena
- Reasoning en français, concis et informatif
- Message ambigü → UNCLEAR_INTENT
- Message incompréhensible → UNKNOWN
- Action non supportée → type exact (TRANSFER_REQUEST, etc.)

{context_section}

MESSAGE À CLASSIFIER: "{user_message}"

JSON:"""
        
        return prompt
    
    def _select_relevant_examples(self, user_message: str, max_examples: int = 15) -> List[Dict]:
        """Sélection dynamique des exemples les plus pertinents"""
        
        message_lower = user_message.lower()
        relevant_examples = []
        
        # Score de pertinence basique par mots-clés
        for example in self.few_shot_examples:
            if hasattr(example, 'keywords') and example.keywords:
                # Score basé sur les mots-clés communs
                common_keywords = sum(1 for keyword in example.keywords if keyword in message_lower)
                relevance_score = common_keywords + (example.confidence * 0.5)
            else:
                # Fallback: score basé sur la confiance uniquement
                relevance_score = example.confidence
            
            relevant_examples.append({
                'example': example,
                'relevance': relevance_score
            })
        
        # Tri par pertinence et sélection des meilleurs
        relevant_examples.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Conversion au format requis
        selected = []
        for item in relevant_examples[:max_examples]:
            ex = item['example']
            selected.append({
                'input': ex.input,
                'intent': ex.intent,
                'confidence': ex.confidence
            })
        
        return selected
    
    def _get_dynamic_intents_summary(self) -> str:
        """Génération dynamique résumé intentions par catégorie"""
        summary_lines = []
        
        for category, intents in INTENT_CATEGORIES.items():
            if category == "UNSUPPORTED":
                summary_lines.append(f"\n{category} (utilisez type exact):")
            else:
                summary_lines.append(f"\n{category}:")
            
            # Limiter à 4 intentions par catégorie pour éviter prompt trop long
            for intent in intents[:4]:
                desc = INTENT_DESCRIPTIONS.get(intent, "")[:60]  # Troncature description
                summary_lines.append(f"  • {intent.value}: {desc}")
            
            if len(intents) > 4:
                summary_lines.append(f"  • ... et {len(intents)-4} autres")
        
        return "\n".join(summary_lines)
    
    def _parse_json_response(
        self, 
        response: Dict[str, Any], 
        original_message: str
    ) -> IntentClassificationResult:
        """Parsing JSON direct - Plus de regex grâce à JSON Output forcé"""
        
        try:
            content = response["choices"][0]["message"]["content"].strip()
            
            # JSON parsing direct - Plus besoin de regex
            parsed_data = json.loads(content)

            # Extraction et validation des champs
            intent_type = self._validate_and_extract_intent(parsed_data.get("intent", "").strip())
            confidence = self._validate_and_extract_confidence(parsed_data.get("confidence", 0.0))
            reasoning = parsed_data.get("reasoning", "").strip() or f"Classification: {intent_type}"
            
            # Détermination propriétés intention
            category = self._determine_intent_category(intent_type)
            is_supported = self._is_intent_supported(intent_type)
            alternatives = self._extract_alternatives(parsed_data.get("alternatives", []))
            
            # Construction résultat
            result = IntentClassificationResult(
                intent_type=HarenaIntentType(intent_type),
                confidence=confidence,
                reasoning=reasoning,
                original_message=original_message,
                category=category,
                is_supported=is_supported,
                alternatives=alternatives,
                processing_time_ms=None  # Sera ajouté par appelant
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON invalide malgré JSON Output forcé: {content[:100]}...")
            return self._create_error_intent_result(f"JSON parsing error: {str(e)}")
        except KeyError as e:
            logger.error(f"Champ manquant dans réponse JSON: {str(e)}")
            return self._create_error_intent_result(f"Missing field: {str(e)}")
        except Exception as e:
            logger.error(f"Erreur parsing réponse DeepSeek: {str(e)}")
            return self._create_error_intent_result(f"Response parsing error: {str(e)}")
    
    def _validate_and_extract_intent(self, intent_str: str) -> str:
        """Validation et normalisation type intention"""
        if not intent_str:
            logger.warning("Intent vide, fallback vers UNCLEAR_INTENT")
            return HarenaIntentType.UNCLEAR_INTENT.value
        
        # Vérification existence dans enum
        valid_intents = [intent.value for intent in HarenaIntentType]
        if intent_str not in valid_intents:
            logger.warning(f"Intent inconnu '{intent_str}', fallback vers UNCLEAR_INTENT")
            return HarenaIntentType.UNCLEAR_INTENT.value
        
        return intent_str
    
    def _validate_and_extract_confidence(self, confidence_val: Any) -> float:
        """Validation et normalisation confidence"""
        try:
            confidence = float(confidence_val)
            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"Confidence hors limite {confidence}, correction vers 0.5")
                return 0.5
            return confidence
        except (ValueError, TypeError):
            logger.warning(f"Confidence invalide {confidence_val}, fallback vers 0.5")
            return 0.5
    
    def _determine_intent_category(self, intent_type: str) -> str:
        """Détermine dynamiquement la catégorie d'une intention"""
        try:
            intent_enum = HarenaIntentType(intent_type)
            for category, intents in INTENT_CATEGORIES.items():
                if intent_enum in intents:
                    return category
            return "UNKNOWN_CATEGORY"
        except ValueError:
            return "UNKNOWN_CATEGORY"
    
    def _is_intent_supported(self, intent_type: str) -> bool:
        """Vérification dynamique si intention est supportée"""
        try:
            intent_enum = HarenaIntentType(intent_type)
            unsupported = INTENT_CATEGORIES.get("UNSUPPORTED", []) + [HarenaIntentType.UNCLEAR_INTENT]
            return intent_enum not in unsupported
        except ValueError:
            return False
    
    def _extract_alternatives(self, alternatives_data: List[Dict]) -> List[IntentAlternative]:
        """Extraction alternatives avec validation robuste"""
        alternatives = []
        for alt_data in alternatives_data[:3]:  # Max 3 alternatives
            try:
                if not isinstance(alt_data, dict):
                    continue
                
                alt_intent = alt_data.get("intent", "").strip()
                alt_confidence = float(alt_data.get("confidence", 0.0))
                
                # Validation alternative
                if alt_intent in [intent.value for intent in HarenaIntentType]:
                    if 0.0 <= alt_confidence <= 1.0:
                        alternatives.append(IntentAlternative(
                            intent_type=HarenaIntentType(alt_intent),
                            confidence=alt_confidence,
                            reasoning=alt_data.get("reasoning", "")
                        ))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Alternative invalide ignorée: {str(e)}")
                continue
        
        return alternatives
    
    async def _get_cached_classification(self, cache_key: str) -> Optional[IntentClassificationResult]:
        """Récupération classification depuis cache"""
        try:
            if not self.cache_manager:
                return None
            
            cached_data = await self.cache_manager.get_semantic_cache(
                cache_key, 
                similarity_threshold=0.85
            )
            
            if cached_data:
                return IntentClassificationResult(**cached_data)
            
            return None
            
        except Exception as e:
            logger.debug(f"Erreur cache récupération: {str(e)}")
            return None
    
    async def _cache_classification_result(
        self, 
        cache_key: str, 
        result: IntentClassificationResult
    ) -> None:
        """Mise en cache résultat classification"""
        try:
            if not self.cache_manager:
                return
            
            cache_ttl = getattr(settings, 'CACHE_TTL_INTENT', 300)
            await self.cache_manager.set_semantic_cache(
                cache_key,
                result.dict(),
                ttl=cache_ttl
            )
            
        except Exception as e:
            logger.debug(f"Erreur cache sauvegarde: {str(e)}")
    
    async def _validate_classification_quality(
        self,
        result: IntentClassificationResult
    ) -> bool:
        """Validation qualité classification"""
        try:
            return await validate_intent_response(result)
        except Exception as e:
            logger.error(f"Erreur validation qualité: {str(e)}")
            return False

    def _update_metrics(self, event: str, start_time: datetime) -> None:
        """Collecte des métriques pour l'agent de classification"""
        try:
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            metrics_collector.increment_counter(f"intent_classifier.{event}")
            metrics_collector.record_histogram(f"intent_classifier.{event}.latency", processing_time)
        except Exception as e:
            logger.debug(f"Échec mise à jour métriques {event}: {str(e)}")

    def _create_unknown_intent_result(self, reason: str) -> IntentClassificationResult:
        """Résultat pour intention inconnue"""
        return IntentClassificationResult(
            intent_type=HarenaIntentType.UNKNOWN,
            confidence=0.95,
            reasoning=f"Message non interprétable: {reason}",
            original_message=reason,
            category="UNCLEAR_INTENT",
            is_supported=False,
            alternatives=[],
            processing_time_ms=0
        )
    
    def _create_unclear_intent_result(self, reason: str) -> IntentClassificationResult:
        """Résultat pour intention ambiguë"""
        return IntentClassificationResult(
            intent_type=HarenaIntentType.UNCLEAR_INTENT,
            confidence=0.50,
            reasoning=f"Message ambigu: {reason}",
            original_message=reason,
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
            original_message=error,
            category="UNCLEAR_INTENT",
            is_supported=False,
            alternatives=[],
            processing_time_ms=0
        )
