"""
Agent de classification d'intentions financières via DeepSeek - JSON Output Forcé avec Retry
Compatible AutoGen AssistantAgent pour collaboration multi-agents
"""
import logging
import json
import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Imports AutoGen avec fallback
try:
    from autogen import AssistantAgent
    AUTOGEN_AVAILABLE = True
    # Import optionnel Teachability
    try:
        from autogen.agentchat.contrib.capabilities import Teachability
    except ImportError:
        Teachability = None
except ImportError:
    AUTOGEN_AVAILABLE = False
    AssistantAgent = object
    Teachability = None

from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_DESCRIPTIONS, INTENT_CATEGORIES
from conversation_service.prompts.few_shot_examples.intent_classification import get_balanced_few_shot_examples, format_examples_for_prompt
from conversation_service.prompts.system_prompts import INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT
from conversation_service.prompts.autogen.collaboration_extensions import (
    get_intent_classification_with_team_collaboration,
    get_entity_focus_mapping,
    get_entity_hints_for_intent
)
from conversation_service.models.responses.conversation_responses import IntentClassificationResult, IntentAlternative
from conversation_service.utils.validation_utils import validate_intent_response, sanitize_user_input
from conversation_service.utils.metrics_collector import metrics_collector
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.intent_classifier")

# Définition des classes base selon disponibilité AutoGen
if AUTOGEN_AVAILABLE:
    class AutoGenIntentClassifierBase(AssistantAgent):
        """Base AutoGen pour IntentClassifierAgent"""
        def __init__(self, name: str = "intent_classifier", **kwargs):
            # Configuration LLM pour AutoGen
            api_key = getattr(settings, 'DEEPSEEK_API_KEY', None)
            base_url = getattr(settings, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
            
            # Vérification configuration DeepSeek
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY requis pour mode AutoGen")
            
            llm_config = {
                "config_list": [{
                    "model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
                    "api_key": api_key,
                    "base_url": base_url,
                    "response_format": {"type": "json_object"}
                }],
                "temperature": getattr(settings, 'DEEPSEEK_INTENT_TEMPERATURE', 0.1),
                "max_tokens": getattr(settings, 'DEEPSEEK_INTENT_MAX_TOKENS', 150),
                "cache_seed": 42
            }
            
            super().__init__(
                name=name,
                llm_config=llm_config,
                system_message=get_intent_classification_with_team_collaboration(),
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                **kwargs
            )
            
            # Ajout capacité Teachability pour amélioration continue
            if Teachability:
                try:
                    self.add_capability(Teachability(verbosity=0))
                    logger.debug("Capacité Teachability ajoutée à l'agent AutoGen")
                except Exception as e:
                    logger.warning(f"Impossible d'ajouter Teachability: {e}")
else:
    class AutoGenIntentClassifierBase:
        """Fallback si AutoGen non disponible"""
        def __init__(self, name: str = "intent_classifier", **kwargs):
            self.name = name


class IntentClassifierAgent(BaseAgent):
    """
    Agent classification intentions financières Harena via DeepSeek avec JSON Output forcé et retry
    Compatible AutoGen AssistantAgent pour collaboration multi-agents
    
    Modes de fonctionnement:
    - Mode Classique: Agent standalone basé sur BaseAgent
    - Mode AutoGen: Collaboration multi-agents avec AssistantAgent
    """
    
    def __init__(self, deepseek_client=None, cache_manager=None, name: str = "intent_classifier", autogen_mode: bool = False, **kwargs):
        # Mode AutoGen ou classique selon paramètres
        self._autogen_mode = autogen_mode and AUTOGEN_AVAILABLE
        self._team_collaboration_active = False
        
        if not self._autogen_mode:
            # Mode classique - BaseAgent
            super().__init__(
                name=name,
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
        else:
            # Mode AutoGen - Conservation compatibilité pour les composants existants
            BaseAgent.__init__(self, name=name, deepseek_client=deepseek_client, cache_manager=cache_manager)
            
            # Initialisation composant AutoGen
            self._autogen_agent = AutoGenIntentClassifierBase(name=name, **kwargs)
            logger.info(f"IntentClassifier initialisé en mode AutoGen: {self._autogen_agent.name}")
        
        # Configuration commune (existante)
        self.supported_intents = list(HarenaIntentType)
        self.few_shot_examples = get_balanced_few_shot_examples(examples_per_intent=2)
        self.confidence_threshold = getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
        
        # Configuration retry et tokens
        self.max_retry_attempts = getattr(settings, 'INTENT_DETECTION_RETRY', 3)
        self.max_tokens = getattr(settings, 'DEEPSEEK_INTENT_MAX_TOKENS', 150)
        self.temperature = getattr(settings, 'DEEPSEEK_INTENT_TEMPERATURE', 0.1)
        
        # Métriques équipe AutoGen
        self._team_metrics = {
            'team_classifications': 0,
            'team_collaborations_successful': 0,
            'team_collaborations_failed': 0
        }
        
        # Mapping entités pour collaboration
        self._entity_focus_mapping = get_entity_focus_mapping()
        
        mode_text = "AutoGen" if self._autogen_mode else "Classique"
        logger.info(f"IntentClassifier initialisé en mode {mode_text} avec {len(self.few_shot_examples)} exemples, retry: {self.max_retry_attempts}, max_tokens: {self.max_tokens}")
    
    def activate_team_collaboration(self) -> None:
        """Active le mode collaboration équipe AutoGen"""
        self._team_collaboration_active = True
        logger.debug("Mode collaboration équipe activé")
    
    def deactivate_team_collaboration(self) -> None:
        """Désactive le mode collaboration équipe"""
        self._team_collaboration_active = False
        logger.debug("Mode collaboration équipe désactivé")
    
    def is_autogen_mode(self) -> bool:
        """Vérifie si l'agent est en mode AutoGen"""
        return self._autogen_mode
        
    def is_team_collaboration_active(self) -> bool:
        """Vérifie si la collaboration équipe est active"""
        return self._team_collaboration_active
        
    def get_autogen_agent(self):
        """Retourne l'agent AutoGen sous-jacent si disponible"""
        return getattr(self, '_autogen_agent', None)
    
    async def execute(
        self, 
        input_data: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> IntentClassificationResult:
        """Exécution classification intention avec routage automatique"""
        if self.is_team_collaboration_active():
            return await self.classify_for_team_collaboration(input_data, context)
        else:
            return await self.classify_intent(input_data, context)
    
    async def classify_intent(
        self,
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> IntentClassificationResult:
        """Classification intention principale avec JSON Output forcé et retry"""
        
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
            
            # Classification avec retry automatique
            classification_result = await self._classify_with_retry(prompt, clean_message, start_time)
            
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
    
    async def classify_for_team_collaboration(
        self,
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> IntentClassificationResult:
        """
        Classification avec enrichissement pour collaboration équipe AutoGen
        Réutilise la logique existante et ajoute contexte équipe
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Métrique équipe
            self._team_metrics['team_classifications'] += 1
            
            # Réutilisation classification de base (existante)
            base_classification = await self.classify_intent(user_message, user_context)
            
            # Enrichissement pour équipe AutoGen
            team_enhanced_result = self._enrich_classification_for_team(
                base_classification, 
                user_message, 
                user_context
            )
            
            # Métriques collaboration
            self._update_team_metrics("team_collaboration_success", start_time)
            
            logger.info(f"Classification équipe réussie: {team_enhanced_result.intent_type.value} avec contexte équipe")
            
            return team_enhanced_result
            
        except Exception as e:
            self._update_team_metrics("team_collaboration_error", start_time)
            logger.error(f"Erreur classification équipe: {str(e)}")
            
            # Fallback: classification standard sans enrichissement
            try:
                fallback_result = await self.classify_intent(user_message, user_context)
                logger.warning("Utilisation classification standard comme fallback")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback échoué: {str(fallback_error)}")
                return self._create_error_intent_result(f"Erreur équipe et fallback: {str(e)} | {str(fallback_error)}")
    
    def _enrich_classification_for_team(
        self, 
        base_classification: IntentClassificationResult,
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> IntentClassificationResult:
        """
        Enrichissement classification existante avec contexte équipe AutoGen
        """
        try:
            intent_value = base_classification.intent_type.value
            
            # Préparation contexte équipe
            team_context = self._prepare_team_context(
                intent_value,
                base_classification.confidence,
                user_message,
                user_context
            )
            
            # Création version enrichie (conserve tout existant + ajoute team_context)
            enriched_result = IntentClassificationResult(
                intent_type=base_classification.intent_type,
                confidence=base_classification.confidence,
                reasoning=base_classification.reasoning,
                original_message=base_classification.original_message,
                category=base_classification.category,
                is_supported=base_classification.is_supported,
                alternatives=base_classification.alternatives,
                processing_time_ms=base_classification.processing_time_ms,
                
                # Extension AutoGen
                team_context=team_context
            )
            
            return enriched_result
            
        except Exception as e:
            logger.warning(f"Erreur enrichissement équipe, retour classification de base: {str(e)}")
            return base_classification
    
    def _prepare_team_context(
        self, 
        intent_type: str,
        confidence: float,
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Préparation contexte pour Entity Extractor (agent suivant)
        """
        # Récupération mapping pour cette intention
        entity_config = self._entity_focus_mapping.get(intent_type, {
            "priority": ["all_entities"],
            "strategy": "comprehensive", 
            "complexity": "medium"
        })
        
        # Suggestions entités spécifiques
        entity_hints = get_entity_hints_for_intent(intent_type)
        
        # Détermination niveau de confiance pour l'agent suivant
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        
        team_context = {
            "user_message": user_message,
            "user_id": user_context.get("user_id") if user_context else None,
            "suggested_entities_focus": {
                "priority_entities": entity_config["priority"],
                "extraction_strategy": entity_config["strategy"],
                "entity_hints": entity_hints
            },
            "processing_metadata": {
                "confidence_level": confidence_level,
                "complexity_assessment": entity_config["complexity"],
                "ready_for_entity_extraction": True,
                "intent_classification_agent": self.name
            }
        }
        
        # Ajout contexte utilisateur si disponible
        if user_context:
            team_context["user_context"] = {
                "recent_intents": user_context.get("recent_intents", []),
                "session_info": user_context.get("session_info", {}),
                "preferences": user_context.get("preferences", {})
            }
        
        return team_context
    
    def _update_team_metrics(self, event: str, start_time: datetime) -> None:
        """Mise à jour métriques spécifiques équipe AutoGen"""
        try:
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            
            # Métriques équipe
            if event == "team_collaboration_success":
                self._team_metrics['team_collaborations_successful'] += 1
            elif event == "team_collaboration_error":
                self._team_metrics['team_collaborations_failed'] += 1
            
            # Métriques globales (existantes)
            metrics_collector.increment_counter(f"intent_classifier.{event}")
            metrics_collector.record_histogram(f"intent_classifier.{event}.latency", processing_time)
            
        except Exception as e:
            logger.debug(f"Échec mise à jour métriques équipe {event}: {str(e)}")
    
    def get_team_metrics(self) -> Dict[str, Any]:
        """Retourne métriques collaboration équipe"""
        return self._team_metrics.copy()
    
    async def _classify_with_retry(
        self,
        prompt: str,
        clean_message: str,
        start_time: datetime
    ) -> IntentClassificationResult:
        """Classification avec retry automatique en cas d'erreur JSON"""
        
        last_exception = None
        
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                # Ajustement des paramètres selon la tentative
                current_max_tokens = self.max_tokens + (attempt - 1) * 50  # Augmente tokens à chaque retry
                current_temperature = self.temperature + (attempt - 1) * 0.05  # Légère augmentation température
                
                logger.debug(f"Tentative {attempt}/{self.max_retry_attempts} - tokens: {current_max_tokens}, temp: {current_temperature:.2f}")
                
                # Appel DeepSeek avec JSON Output FORCÉ
                response = await self.deepseek_client.chat_completion(
                    messages=[
                        {"role": "system", "content": INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=current_max_tokens,
                    temperature=current_temperature,
                    response_format={"type": "json_object"}  # JSON FORCÉ
                )
                
                # Parsing JSON avec validation
                classification_result = self._parse_json_response_with_validation(response, clean_message)
                
                if classification_result:
                    if attempt > 1:
                        logger.info(f"Classification réussie à la tentative {attempt}")
                        metrics_collector.increment_counter(f"intent_classifier.retry_success.attempt_{attempt}")
                    return classification_result
                    
            except json.JSONDecodeError as e:
                last_exception = e
                logger.warning(f"Tentative {attempt}: JSON invalide - {str(e)}")
                metrics_collector.increment_counter(f"intent_classifier.json_error.attempt_{attempt}")
                
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(0.1 * attempt)  # Backoff exponentiel
                    continue
                    
            except Exception as e:
                last_exception = e
                logger.error(f"Tentative {attempt}: Erreur inattendue - {str(e)}")
                metrics_collector.increment_counter(f"intent_classifier.error.attempt_{attempt}")
                
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(0.2 * attempt)
                    continue
        
        # Toutes les tentatives ont échoué
        logger.error(f"Toutes les tentatives de classification ont échoué après {self.max_retry_attempts} essais")
        metrics_collector.increment_counter("intent_classifier.retry_exhausted")
        return self._create_error_intent_result(f"JSON parsing error après {self.max_retry_attempts} tentatives: {str(last_exception)}")
    
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
        
        # Intentions supportées dynamiques (version raccourcie pour économiser tokens)
        intentions_summary = self._get_compact_intents_summary()
        
        prompt = f"""Analysez ce message utilisateur et classifiez l'intention financière Harena.

IMPORTANT: Répondez UNIQUEMENT avec un objet JSON strict, aucun autre texte.

FORMAT JSON OBLIGATOIRE:
{{"intent": "TYPE_INTENTION_EXACT", "confidence": 0.XX, "reasoning": "explication française"}}

INTENTIONS PRINCIPALES:
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
    
    def _get_compact_intents_summary(self) -> str:
        """Génération compacte résumé intentions (économie tokens)"""
        summary_lines = []
        
        # Focus sur les catégories principales
        priority_categories = ["FINANCIAL_QUERY", "SPENDING_ANALYSIS", "ACCOUNT_BALANCE", "UNSUPPORTED"]
        
        for category in priority_categories:
            intents = INTENT_CATEGORIES.get(category, [])
            if category == "UNSUPPORTED":
                summary_lines.append(f"{category}: TRANSFER_REQUEST, PAYMENT_REQUEST, CARD_BLOCK, BUDGET_INQUIRY")
            else:
                # Limiter à 3 intentions principales par catégorie
                key_intents = [intent.value for intent in intents[:3]]
                summary_lines.append(f"{category}: {', '.join(key_intents)}")
        
        # Autres intentions importantes
        summary_lines.append("CONVERSATIONAL: GREETING, CONFIRMATION")
        summary_lines.append("AMBIGUËS: UNCLEAR_INTENT, UNKNOWN, OUT_OF_SCOPE")
        
        return "\n".join(summary_lines)
    
    def _select_relevant_examples(self, user_message: str, max_examples: int = 12) -> List[Dict]:
        """Sélection dynamique des exemples les plus pertinents (réduit pour économiser tokens)"""
        
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
        for item in relevant_examples[:max_examples]:  # Réduit de 15 à 12
            ex = item['example']
            selected.append({
                'input': ex.input,
                'intent': ex.intent,
                'confidence': ex.confidence
            })
        
        return selected
    
    def _parse_json_response_with_validation(
        self, 
        response: Dict[str, Any], 
        original_message: str
    ) -> Optional[IntentClassificationResult]:
        """Parsing JSON avec validation renforcée et nettoyage"""
        
        try:
            content = response["choices"][0]["message"]["content"].strip()
            
            # Nettoyage préalable du JSON (supprime markdown potentiel)
            content = self._clean_json_content(content)
            
            # Validation que c'est bien du JSON valide
            if not self._is_valid_json_format(content):
                logger.warning(f"Format JSON invalide détecté: {content[:100]}")
                return None
            
            # JSON parsing direct
            parsed_data = json.loads(content)
            
            # Validation structure minimale requise
            if not self._validate_json_structure(parsed_data):
                logger.warning("Structure JSON incomplète")
                return None

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
            logger.warning(f"JSON invalide malgré JSON Output forcé: {content[:200] if 'content' in locals() else 'N/A'}... Erreur: {str(e)}")
            raise  # Re-raise pour retry
        except KeyError as e:
            logger.warning(f"Champ manquant dans réponse JSON: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Erreur parsing réponse DeepSeek: {str(e)}")
            return None
    
    def _clean_json_content(self, content: str) -> str:
        """Nettoyage du contenu JSON pour éliminer les artefacts"""
        # Supprime les blocs markdown potentiels
        content = content.replace("```json", "").replace("```", "")
        
        # Supprime les espaces en début/fin
        content = content.strip()
        
        # Trouve le premier { et le dernier }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            content = content[start_idx:end_idx + 1]
        
        return content
    
    def _is_valid_json_format(self, content: str) -> bool:
        """Validation basique format JSON"""
        if not content:
            return False
        
        content = content.strip()
        return content.startswith('{') and content.endswith('}')
    
    def _validate_json_structure(self, parsed_data: Dict) -> bool:
        """Validation structure JSON minimale"""
        required_fields = ["intent", "confidence"]
        return all(field in parsed_data for field in required_fields)
    
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
                result.model_dump(mode="json"),
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