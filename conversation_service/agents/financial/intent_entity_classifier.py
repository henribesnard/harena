"""
Agent unifié pour classification d'intentions et extraction d'entités
Combine intent_classifier et entity_extractor en un seul appel LLM
"""
import logging
import json
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_DESCRIPTIONS, INTENT_CATEGORIES
from conversation_service.prompts.few_shot_examples.intent_classification import get_balanced_few_shot_examples, format_examples_for_prompt
from conversation_service.models.responses.conversation_responses import IntentClassificationResult, IntentAlternative
from conversation_service.utils.validation_utils import validate_intent_response, sanitize_user_input
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.utils.entity_normalization import normalize_entities
from config_service.config import settings

logger = logging.getLogger("conversation_service.intent_entity_classifier")


class IntentEntityClassifier(BaseAgent):
    """
    Agent unifié pour classification d'intentions et extraction d'entités
    
    Remplace:
    - IntentClassifierAgent
    - EntityExtractorAgent
    
    Un seul appel LLM pour les deux tâches, réduisant la latence
    """
    
    def __init__(self, deepseek_client=None, cache_manager=None, name: str = "intent_entity_classifier"):
        super().__init__(
            name=name,
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )
        
        # Configuration
        self.supported_intents = list(HarenaIntentType)
        self.few_shot_examples = get_balanced_few_shot_examples(examples_per_intent=2)
        self.confidence_threshold = getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
        
        # Configuration retry et tokens
        self.max_retry_attempts = getattr(settings, 'INTENT_DETECTION_RETRY', 3)
        self.max_tokens = getattr(settings, 'DEEPSEEK_INTENT_MAX_TOKENS', 400)  # Augmenté pour les deux tâches
        self.temperature = getattr(settings, 'DEEPSEEK_INTENT_TEMPERATURE', 0.1)
        
        # Métriques
        self.total_classifications = 0
        self.successful_classifications = 0
        self.failed_classifications = 0
        self.cache_hits = 0
        self.average_response_time = 0.0
        
        logger.info(f"IntentEntityClassifier initialisé avec {len(self.few_shot_examples)} exemples, retry: {self.max_retry_attempts}, max_tokens: {self.max_tokens}")
    
    async def execute(
        self, 
        input_data: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classification intention + extraction entités en un seul appel
        
        Returns:
            {
                "intent_result": IntentClassificationResult,
                "entity_result": Dict[str, Any]
            }
        """
        start_time = datetime.now(timezone.utc)
        self.total_classifications += 1
        
        try:
            # Nettoyage et validation input
            clean_message = sanitize_user_input(input_data)
            
            if not clean_message:
                return self._create_error_result("Message vide après nettoyage")
            
            if len(clean_message) > 500:
                return self._create_error_result("Message trop long")
            
            # Vérification cache
            cache_key = self._generate_cache_key(clean_message)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                self.cache_hits += 1
                cached_result["intent_result"].processing_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                return cached_result
            
            # Construction prompt unifié
            prompt = self._build_unified_prompt(clean_message, context)
            
            # Classification + extraction avec retry
            result = await self._classify_and_extract_with_retry(prompt, clean_message, start_time)
            
            # Calcul temps de traitement
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            result["intent_result"].processing_time_ms = processing_time
            
            # Cache résultat
            await self._cache_result(cache_key, result)
            
            self.successful_classifications += 1
            self._update_metrics("classification_success", start_time)
            
            intent_display = (
                result["intent_result"].intent_type.value
                if hasattr(result["intent_result"].intent_type, "value")
                else result["intent_result"].intent_type
            )
            logger.info(
                f"Classification+extraction réussie: {intent_display} "
                f"(conf: {result['intent_result'].confidence:.2f}, entités: {self._count_entities(result['entity_result'])}, temps: {processing_time}ms)"
            )
            
            return result
            
        except Exception as e:
            self.failed_classifications += 1
            self._update_metrics("classification_error", start_time)
            logger.error(f"Erreur classification+extraction: {str(e)}")
            return self._create_error_result(str(e))
    
    def _build_unified_prompt(
        self, 
        user_message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construction prompt unifié pour intention + entités"""
        
        # Sélection exemples pertinents
        relevant_examples = self._select_relevant_examples(user_message)
        examples_text = format_examples_for_prompt(relevant_examples, format_style="concise")
        
        # Contexte utilisateur
        context_section = ""
        if user_context and user_context.get("recent_intents"):
            recent = user_context["recent_intents"][-3:]
            context_section = f"\nCONTEXTE: Intentions récentes: {', '.join(recent)}"
        
        # Contexte temporel intelligent
        current_date = datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        current_month_name = current_date.strftime("%B").lower()
        current_month_fr = {
            "january": "janvier", "february": "février", "march": "mars", "april": "avril",
            "may": "mai", "june": "juin", "july": "juillet", "august": "août", 
            "september": "septembre", "october": "octobre", "november": "novembre", "december": "décembre"
        }.get(current_month_name, current_month_name)
        
        # Résumé intentions
        intentions_summary = self._get_compact_intents_summary()
        
        prompt = f"""Analysez ce message utilisateur financier Harena et effectuez DEUX tâches simultanément:

1) CLASSIFICATION D'INTENTION
2) EXTRACTION D'ENTITÉS

IMPORTANT: Répondez UNIQUEMENT avec un objet JSON strict, aucun autre texte.

FORMAT JSON OBLIGATOIRE:
{{
  "intent": {{
    "type": "TYPE_INTENTION_EXACT",
    "confidence": 0.XX,
    "reasoning": "explication française"
  }},
  "entities": {{
    "amounts": [
      {{"value": 100.50, "currency": "EUR", "operator": "eq", "text": "100 euros"}}
    ],
    "dates": [
      {{"type": "specific|period|relative", "value": "YYYY-MM-DD|YYYY-MM", "text": "texte original"}}
    ],
    "merchants": [
      {{"name": "Amazon", "normalized": "Amazon", "confidence": 0.9}}
    ],
    "categories": [
      {{"name": "alimentation", "confidence": 0.8}}
    ],
    "operation_types": [
      {{"type": "card|transfer|direct_debit|withdrawal", "confidence": 0.9}}
    ],
    "transaction_types": [],
    "text_search": [
      {{"query": "terme recherche", "field": "description|merchant"}}
    ]
  }},
  "extraction_confidence": 0.XX,
  "extraction_reasoning": "explication entités extraites"
}}

INTENTIONS PRINCIPALES:
{intentions_summary}

CONTEXTE TEMPOREL IMPORTANT:
Date actuelle: {current_date_str}
Nous sommes en {current_month_fr} {current_date.year}

RÈGLES TEMPORELLES CRITIQUES:
- Pour un mois sans année mentionnée (ex: "mai"):
  * Si le mois est passé ou en cours cette année - utiliser {current_date.year}
  * Si le mois est dans le futur - utiliser {current_date.year - 1}
- "mois dernier" - mois précédent par rapport à {current_month_fr} {current_date.year}
- "ce mois" ou "ce mois-ci" - {current_date.strftime("%Y-%m")}
- "mois prochain" - mois suivant par rapport à {current_month_fr} {current_date.year}

EXEMPLES DE RÉFÉRENCE:
{examples_text}

RÈGLES STRICTES:
- JSON uniquement, pas de markdown ni commentaires
- Intent confidence entre 0.0 et 1.0 précise
- Intent doit correspondre exactement aux types Harena
- Extraction_confidence entre 0.0 et 1.0
- Message ambigü → UNCLEAR_INTENT
- Message incompréhensible → UNKNOWN
- Types d'opérations: valeurs Elasticsearch (transfer, card, direct_debit, withdrawal, etc.)
- Montants en EUR par défaut
- Dates au format ISO avec logique temporelle ci-dessus

EXEMPLE CORRECT pour "mai" en {current_month_fr} {current_date.year}:
{{"dates": [{{"type": "period", "value": "{current_date.year if 5 <= current_date.month else current_date.year - 1}-05", "text": "mai"}}]}}

{context_section}

MESSAGE À ANALYSER: "{user_message}"

JSON:"""
        
        return prompt
    
    def _get_compact_intents_summary(self) -> str:
        """Génération compacte résumé intentions"""
        summary_lines = []
        
        # Focus sur les catégories principales
        priority_categories = ["FINANCIAL_QUERY", "SPENDING_ANALYSIS", "ACCOUNT_BALANCE", "UNSUPPORTED"]
        
        for category in priority_categories:
            intents = INTENT_CATEGORIES.get(category, [])
            if category == "UNSUPPORTED":
                summary_lines.append(f"{category}: TRANSFER_REQUEST, PAYMENT_REQUEST, CARD_BLOCK, BUDGET_INQUIRY")
            else:
                key_intents = [intent.value for intent in intents[:3]]
                summary_lines.append(f"{category}: {', '.join(key_intents)}")
        
        summary_lines.append("CONVERSATIONAL: GREETING, CONFIRMATION")
        summary_lines.append("AMBIGUËS: UNCLEAR_INTENT, UNKNOWN, OUT_OF_SCOPE")
        
        return "\n".join(summary_lines)
    
    def _select_relevant_examples(self, user_message: str, max_examples: int = 8) -> List[Dict]:
        """Sélection dynamique des exemples les plus pertinents (réduit pour économiser tokens)"""
        
        message_lower = user_message.lower()
        relevant_examples = []
        
        for example in self.few_shot_examples:
            if hasattr(example, 'keywords') and example.keywords:
                common_keywords = sum(1 for keyword in example.keywords if keyword in message_lower)
                relevance_score = common_keywords + (example.confidence * 0.5)
            else:
                relevance_score = example.confidence
            
            relevant_examples.append({
                'example': example,
                'relevance': relevance_score
            })
        
        # Tri par pertinence
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
    
    async def _classify_and_extract_with_retry(
        self,
        prompt: str,
        clean_message: str,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Classification + extraction avec retry automatique"""
        
        last_exception = None
        
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                # Ajustement paramètres selon la tentative
                current_max_tokens = self.max_tokens + (attempt - 1) * 100
                current_temperature = self.temperature + (attempt - 1) * 0.05
                
                logger.debug(f"Tentative {attempt}/{self.max_retry_attempts} - tokens: {current_max_tokens}, temp: {current_temperature:.2f}")
                
                # Appel DeepSeek avec JSON Output FORCÉ
                response = await self.deepseek_client.chat_completion(
                    messages=[
                        {"role": "system", "content": "Vous êtes un expert en classification d'intentions financières et extraction d'entités. Répondez UNIQUEMENT avec du JSON valide."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=current_max_tokens,
                    temperature=current_temperature,
                    response_format={"type": "json_object"}
                )
                
                # Parsing JSON avec validation
                result = self._parse_unified_response(response, clean_message)
                
                if result:
                    if attempt > 1:
                        logger.info(f"Classification+extraction réussie à la tentative {attempt}")
                        metrics_collector.increment_counter(f"intent_entity_classifier.retry_success.attempt_{attempt}")
                    return result
                    
            except json.JSONDecodeError as e:
                last_exception = e
                logger.warning(f"Tentative {attempt}: JSON invalide - {str(e)}")
                metrics_collector.increment_counter(f"intent_entity_classifier.json_error.attempt_{attempt}")
                
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(0.1 * attempt)
                    continue
                    
            except Exception as e:
                last_exception = e
                logger.error(f"Tentative {attempt}: Erreur inattendue - {str(e)}")
                metrics_collector.increment_counter(f"intent_entity_classifier.error.attempt_{attempt}")
                
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(0.2 * attempt)
                    continue
        
        # Toutes les tentatives ont échoué
        logger.error(f"Toutes les tentatives ont échoué après {self.max_retry_attempts} essais")
        metrics_collector.increment_counter("intent_entity_classifier.retry_exhausted")
        return self._create_error_result(f"Classification échouée après {self.max_retry_attempts} tentatives: {str(last_exception)}")
    
    def _parse_unified_response(
        self, 
        response: Dict[str, Any], 
        original_message: str
    ) -> Optional[Dict[str, Any]]:
        """Parsing de la réponse unifiée intention + entités"""
        
        try:
            content = response["choices"][0]["message"]["content"].strip()
            
            # Nettoyage JSON
            content = self._clean_json_content(content)
            
            # Validation format JSON
            if not self._is_valid_json_format(content):
                logger.warning(f"Format JSON invalide détecté: {content[:100]}")
                return None
            
            # Parse JSON
            parsed_data = json.loads(content)
            
            # Validation structure
            if not self._validate_unified_structure(parsed_data):
                logger.warning("Structure JSON unifiée incomplète")
                return None
            
            # Extraction intention
            intent_data = parsed_data.get("intent", {})
            intent_type = self._validate_and_extract_intent(intent_data.get("type", "").strip())
            intent_confidence = self._validate_and_extract_confidence(intent_data.get("confidence", 0.0))
            intent_reasoning = intent_data.get("reasoning", "").strip() or f"Classification: {intent_type}"
            
            # Propriétés intention
            category = self._determine_intent_category(intent_type)
            is_supported = self._is_intent_supported(intent_type)
            
            # Construction résultat intention
            intent_result = IntentClassificationResult(
                intent_type=HarenaIntentType(intent_type),
                confidence=intent_confidence,
                reasoning=intent_reasoning,
                original_message=original_message,
                category=category,
                is_supported=is_supported,
                alternatives=[],
                processing_time_ms=None
            )
            
            # Extraction entités
            entities_data = parsed_data.get("entities", {})
            extraction_confidence = self._validate_and_extract_confidence(parsed_data.get("extraction_confidence", 0.0))
            extraction_reasoning = parsed_data.get("extraction_reasoning", "").strip() or "Entités extraites"
            
            # Normalisation des entités
            normalized_entities = normalize_entities(entities_data)
            
            # Construction résultat entités
            entity_result = {
                "entities": normalized_entities,
                "confidence": extraction_confidence,
                "reasoning": extraction_reasoning,
                "extraction_success": True,
                "original_message": original_message
            }
            
            return {
                "intent_result": intent_result,
                "entity_result": entity_result
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON invalide malgré JSON Output forcé: {content[:200] if 'content' in locals() else 'N/A'}... Erreur: {str(e)}")
            raise
        except KeyError as e:
            logger.warning(f"Champ manquant dans réponse JSON: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Erreur parsing réponse unifiée: {str(e)}")
            return None
    
    def _clean_json_content(self, content: str) -> str:
        """Nettoyage du contenu JSON"""
        content = content.replace("```json", "").replace("```", "")
        content = content.strip()
        
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
    
    def _validate_unified_structure(self, parsed_data: Dict) -> bool:
        """Validation structure JSON unifiée"""
        required_fields = ["intent", "entities"]
        return all(field in parsed_data for field in required_fields)
    
    def _validate_and_extract_intent(self, intent_str: str) -> str:
        """Validation et normalisation type intention"""
        if not intent_str:
            logger.warning("Intent vide, fallback vers UNCLEAR_INTENT")
            return HarenaIntentType.UNCLEAR_INTENT.value
        
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
        """Détermine la catégorie d'une intention"""
        try:
            intent_enum = HarenaIntentType(intent_type)
            for category, intents in INTENT_CATEGORIES.items():
                if intent_enum in intents:
                    return category
            return "UNKNOWN_CATEGORY"
        except ValueError:
            return "UNKNOWN_CATEGORY"
    
    def _is_intent_supported(self, intent_type: str) -> bool:
        """Vérification si intention est supportée"""
        try:
            intent_enum = HarenaIntentType(intent_type)
            unsupported = INTENT_CATEGORIES.get("UNSUPPORTED", []) + [HarenaIntentType.UNCLEAR_INTENT]
            return intent_enum not in unsupported
        except ValueError:
            return False
    
    def _count_entities(self, entity_result: Dict[str, Any]) -> int:
        """Compte le nombre total d'entités extraites"""
        entities = entity_result.get("entities", {})
        total = 0
        
        for entity_list in entities.values():
            if isinstance(entity_list, list):
                total += len(entity_list)
        
        return total
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupération résultat depuis cache"""
        try:
            if not self.cache_manager:
                return None
            
            cached_data = await self.cache_manager.get_semantic_cache(
                cache_key, 
                similarity_threshold=0.85
            )
            
            if cached_data:
                # Reconstruction des objets
                intent_data = cached_data.get("intent_result", {})
                if intent_data:
                    cached_data["intent_result"] = IntentClassificationResult(**intent_data)
                return cached_data
            
            return None
            
        except Exception as e:
            logger.debug(f"Erreur cache récupération: {str(e)}")
            return None
    
    async def _cache_result(
        self, 
        cache_key: str, 
        result: Dict[str, Any]
    ) -> None:
        """Mise en cache résultat unifié"""
        try:
            if not self.cache_manager:
                return
            
            cache_ttl = getattr(settings, 'CACHE_TTL_INTENT', 300)
            
            # Préparation données cache
            cache_data = {
                "intent_result": result["intent_result"].model_dump(mode="json"),
                "entity_result": result["entity_result"]
            }
            
            await self.cache_manager.set_semantic_cache(
                cache_key,
                cache_data,
                ttl=cache_ttl
            )
            
        except Exception as e:
            logger.debug(f"Erreur cache sauvegarde: {str(e)}")
    
    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Création résultat d'erreur unifié"""
        intent_result = IntentClassificationResult(
            intent_type=HarenaIntentType.ERROR,
            confidence=0.99,
            reasoning=f"Erreur technique: {error}",
            original_message=error,
            category="UNCLEAR_INTENT",
            is_supported=False,
            alternatives=[],
            processing_time_ms=0
        )
        
        entity_result = {
            "entities": {
                "amounts": [],
                "dates": [],
                "merchants": [],
                "categories": [],
                "operation_types": [],
                "transaction_types": [],
                "text_search": []
            },
            "confidence": 0.0,
            "reasoning": f"Erreur extraction: {error}",
            "extraction_success": False,
            "error": True,
            "original_message": error
        }
        
        return {
            "intent_result": intent_result,
            "entity_result": entity_result
        }
    
    def _update_metrics(self, event: str, start_time: datetime) -> None:
        """Collecte des métriques"""
        try:
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            metrics_collector.increment_counter(f"intent_entity_classifier.{event}")
            metrics_collector.record_histogram(f"intent_entity_classifier.{event}.latency", processing_time)
        except Exception as e:
            logger.debug(f"Échec mise à jour métriques {event}: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de l'agent"""
        return {
            "total_classifications": self.total_classifications,
            "successful_classifications": self.successful_classifications,
            "failed_classifications": self.failed_classifications,
            "cache_hits": self.cache_hits,
            "success_rate": (
                self.successful_classifications / self.total_classifications
                if self.total_classifications > 0 else 0.0
            ),
            "cache_hit_rate": (
                self.cache_hits / self.total_classifications
                if self.total_classifications > 0 else 0.0
            ),
            "average_response_time": self.average_response_time
        }