"""
Entity Extractor Agent AutoGen pour l'extraction d'entités financières
Réutilise l'infrastructure existante avec collaboration multi-agents
"""
import logging
import json
import asyncio
import time
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

# Réutilisation infrastructure existante
from conversation_service.prompts.system_prompts import ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT, get_prompt_config
from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.core.cache_manager import CacheManager
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.utils.entity_normalization import normalize_entities
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.entity_extractor")


# Définition de la classe base selon disponibilité AutoGen
if AUTOGEN_AVAILABLE:
    class AutoGenEntityExtractorBase(AssistantAgent):
        """Base AutoGen pour EntityExtractorAgent"""
        def __init__(self, name: str = "entity_extractor", **kwargs):
            # Configuration LLM réutilisant infrastructure existante
            entity_config = get_prompt_config("entity_extraction")
            api_key = getattr(settings, 'DEEPSEEK_API_KEY', None)
            base_url = getattr(settings, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
            
            # Vérification configuration DeepSeek
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY requis pour mode AutoGen")
            
            # Adaptation pour AutoGen format
            llm_config = {
                "config_list": [{
                    "model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
                    "api_key": api_key,
                    "base_url": base_url,
                    "response_format": entity_config.get("response_format", {"type": "json_object"})
                }],
                "temperature": entity_config.get("temperature", 0.05),
                "max_tokens": entity_config.get("max_tokens", 200),
                "cache_seed": 42
            }
            
            super().__init__(
                name=name,
                llm_config=llm_config,
                system_message=ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT,  # Réutilise prompt existant
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                **kwargs
            )
            
            # Ajout capacité Teachability pour amélioration continue
            if Teachability:
                try:
                    self.add_capability(Teachability(verbosity=0))
                    logger.debug("Capacité Teachability ajoutée à EntityExtractorAgent")
                except Exception as e:
                    logger.warning(f"Impossible d'ajouter Teachability: {e}")
else:
    class AutoGenEntityExtractorBase:
        """Fallback si AutoGen non disponible"""
        def __init__(self, name: str = "entity_extractor", **kwargs):
            self.name = name


class EntityExtractorAgent:
    """
    Agent d'extraction d'entités financières compatible AutoGen
    Réutilise 100% infrastructure existante avec collaboration équipe
    
    Modes de fonctionnement:
    - Mode AutoGen: AssistantAgent pour collaboration multi-agents
    - Mode Fallback: Agent compatible sans AutoGen
    """
    
    def __init__(self, deepseek_client=None, cache_manager=None, name: str = "entity_extractor", autogen_mode: bool = True, **kwargs):
        # Mode AutoGen ou fallback
        self._autogen_mode = autogen_mode and AUTOGEN_AVAILABLE
        self._team_collaboration_active = False
        
        if self._autogen_mode:
            # Mode AutoGen - Initialisation composant AutoGen
            self._autogen_agent = AutoGenEntityExtractorBase(name=name, **kwargs)
            logger.info(f"EntityExtractor initialisé en mode AutoGen: {self._autogen_agent.name}")
        else:
            # Mode fallback - composant simple
            self.name = name
            logger.info(f"EntityExtractor initialisé en mode fallback: {name}")
        
        # Réutilisation infrastructure existante (BaseAgent patterns)
        self.deepseek_client = deepseek_client
        self.cache_manager = cache_manager or CacheManager()
        
        # Configuration basée sur infrastructure existante
        self.entity_config = get_prompt_config("entity_extraction")
        self.max_tokens = self.entity_config.get("max_tokens", 200)
        self.temperature = self.entity_config.get("temperature", 0.05)
        
        # Métriques réutilisant infrastructure existante  
        self._extraction_metrics = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'cache_hits': 0,
            'team_collaborations': 0
        }
        
        logger.info(f"EntityExtractor configuré - Mode: {'AutoGen' if self._autogen_mode else 'Fallback'}, Max tokens: {self.max_tokens}")
    
    def activate_team_collaboration(self) -> None:
        """Active le mode collaboration équipe AutoGen"""
        self._team_collaboration_active = True
        logger.debug("Mode collaboration équipe activé pour EntityExtractor")
    
    def deactivate_team_collaboration(self) -> None:
        """Désactive le mode collaboration équipe"""
        self._team_collaboration_active = False
        logger.debug("Mode collaboration équipe désactivé pour EntityExtractor")
    
    async def extract_entities(self, user_message: str, intent_result: Optional[Dict[str, Any]] = None, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Méthode simple d'extraction d'entités pour utilisation dans l'endpoint V1
        """
        start_time = time.time()
        
        try:
            self._extraction_metrics['total_extractions'] += 1
            
            # Cache basé sur infrastructure existante
            cached_result = await self._get_cached_extraction(user_message, user_id or 0)
            if cached_result:
                self._extraction_metrics['cache_hits'] += 1
                return cached_result
            
            # Construction prompt simple
            extraction_prompt = self._build_simple_extraction_prompt(user_message, intent_result)
            
            # Extraction via AutoGen ou fallback
            extraction_result = await self._perform_extraction(extraction_prompt, user_message)
            
            # Normalisation des entités pour compatibilité Elasticsearch
            if extraction_result and "entities" in extraction_result:
                original_entities = extraction_result["entities"]
                normalized_entities = normalize_entities(original_entities)
                extraction_result["entities"] = normalized_entities
                
                logger.debug(f"Entités normalisées pour Elasticsearch: {normalized_entities}")
            
            # Cache pour réutilisation
            await self._cache_extraction_result(user_message, user_id or 0, extraction_result)
            
            # Métriques succès
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics_extraction(processing_time, True, self._count_extracted_entities(extraction_result))
            self._extraction_metrics['successful_extractions'] += 1
            
            logger.info(f"Extraction simple réussie: {len(extraction_result.get('entities', {}))} types d'entités, {processing_time:.1f}ms")
            
            return extraction_result
            
        except Exception as e:
            # Gestion erreurs cohérente existante
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics_extraction(processing_time, False, 0)
            self._extraction_metrics['failed_extractions'] += 1
            
            logger.error(f"Erreur extraction simple: {str(e)}")
            return self._create_error_extraction(f"Erreur extraction: {str(e)}")
    
    def is_autogen_mode(self) -> bool:
        """Vérifie si l'agent est en mode AutoGen"""
        return self._autogen_mode
        
    def is_team_collaboration_active(self) -> bool:
        """Vérifie si la collaboration équipe est active"""
        return self._team_collaboration_active
        
    def get_autogen_agent(self):
        """Retourne l'agent AutoGen sous-jacent si disponible"""
        return getattr(self, '_autogen_agent', None)
    
    async def extract_entities_for_team(self, team_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extraction entités avec contexte équipe AutoGen
        Réutilise infrastructure existante avec adaptation collaboration
        """
        start_time = time.time()
        
        try:
            # Métriques équipe
            self._extraction_metrics['total_extractions'] += 1
            self._extraction_metrics['team_collaborations'] += 1
            
            # Extraction données contexte équipe
            user_message = team_context.get("user_message", "")
            intent_result = team_context.get("intent_result", {})
            user_id = team_context.get("user_id")
            
            if not user_message:
                logger.error("Contexte équipe invalide: user_message manquant")
                return self._create_error_extraction("user_message manquant dans contexte équipe")
            
            # Cache basé sur infrastructure existante
            cached_result = await self._get_cached_extraction(user_message, user_id or 0)
            if cached_result:
                self._extraction_metrics['cache_hits'] += 1
                return self._enrich_cached_result_for_team(cached_result, team_context)
            
            # Construction prompt avec contexte équipe
            extraction_prompt = self._build_team_extraction_prompt(user_message, intent_result)
            
            # Extraction via AutoGen ou fallback
            extraction_result = await self._perform_extraction(extraction_prompt, user_message)
            
            # Normalisation des entités pour compatibilité Elasticsearch
            if extraction_result and "entities" in extraction_result:
                original_entities = extraction_result["entities"]
                normalized_entities = normalize_entities(original_entities)
                extraction_result["entities"] = normalized_entities
                
                logger.debug(f"Entités normalisées pour équipe: {normalized_entities}")
            
            # Enrichissement pour collaboration équipe
            team_enriched_result = self._enrich_result_for_team(extraction_result, team_context)
            
            # Cache pour réutilisation (infrastructure existante)
            await self._cache_extraction_result(user_message, user_id or 0, team_enriched_result)
            
            # Métriques succès
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics_extraction(processing_time, True, self._count_extracted_entities(extraction_result))
            self._extraction_metrics['successful_extractions'] += 1
            
            logger.info(f"Extraction équipe réussie: {len(extraction_result.get('entities', {}))} types d'entités, {processing_time:.1f}ms")
            
            return team_enriched_result
            
        except Exception as e:
            # Gestion erreurs cohérente existante
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics_extraction(processing_time, False, 0)
            self._extraction_metrics['failed_extractions'] += 1
            
            logger.error(f"Erreur extraction équipe: {str(e)}")
            return self._create_error_extraction(f"Erreur extraction: {str(e)}")
    
    async def _perform_extraction(self, extraction_prompt: str, user_message: str) -> Dict[str, Any]:
        """Perform extraction via AutoGen ou fallback DeepSeek"""
        
        if self._autogen_mode and hasattr(self._autogen_agent, 'a_generate_reply'):
            # Mode AutoGen - utilisation AssistantAgent
            try:
                response = await self._autogen_agent.a_generate_reply([{
                    "role": "user", 
                    "content": extraction_prompt
                }])
                
                if isinstance(response, str):
                    return self._parse_and_validate_response(response)
                elif isinstance(response, dict) and "content" in response:
                    return self._parse_and_validate_response(response["content"])
                else:
                    logger.warning(f"Réponse AutoGen inattendue: {type(response)}, fallback DeepSeek")
                    return await self._fallback_deepseek_extraction(extraction_prompt, user_message)
                    
            except Exception as e:
                logger.warning(f"Erreur AutoGen, fallback DeepSeek: {str(e)}")
                return await self._fallback_deepseek_extraction(extraction_prompt, user_message)
        else:
            # Mode fallback - utilisation DeepSeek direct
            return await self._fallback_deepseek_extraction(extraction_prompt, user_message)
    
    async def _fallback_deepseek_extraction(self, extraction_prompt: str, user_message: str) -> Dict[str, Any]:
        """Fallback extraction via DeepSeek direct (infrastructure existante)"""
        
        if not self.deepseek_client:
            logger.error("DeepSeek client non disponible pour fallback")
            return self._create_fallback_extraction(user_message, "DeepSeek client indisponible")
        
        try:
            # Utilisation DeepSeek avec config existante
            response = await self.deepseek_client.chat_completion(
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format=self.entity_config.get("response_format", {"type": "json_object"})
            )
            
            content = response["choices"][0]["message"]["content"]
            return self._parse_and_validate_response(content)
            
        except Exception as e:
            logger.error(f"Erreur fallback DeepSeek: {str(e)}")
            return self._create_fallback_extraction(user_message, f"Erreur DeepSeek: {str(e)}")
    
    def _build_simple_extraction_prompt(self, user_message: str, intent_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Construction prompt simple pour extraction d'entités avec contexte temporel
        """
        intent_type = "GENERAL_INQUIRY"
        confidence = 0.5
        
        if intent_result:
            if hasattr(intent_result, 'intent_type'):
                # C'est un objet IntentClassificationResult
                intent_type = getattr(intent_result.intent_type, 'value', str(intent_result.intent_type))
                confidence = getattr(intent_result, 'confidence', 0.5)
            else:
                # C'est un dictionnaire
                intent_type = intent_result.get("intent", "GENERAL_INQUIRY")
                confidence = intent_result.get("confidence", 0.5)
        
        # Contexte temporel intelligent
        from datetime import datetime
        current_date = datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        current_month_name = current_date.strftime("%B").lower()
        current_month_fr = {
            "january": "janvier", "february": "février", "march": "mars", "april": "avril",
            "may": "mai", "june": "juin", "july": "juillet", "august": "août", 
            "september": "septembre", "october": "octobre", "november": "novembre", "december": "décembre"
        }.get(current_month_name, current_month_name)
        
        # Prompt adaptatif selon intention avec contexte temporel
        simple_context = f"""
Extrayez les entités financières de ce message utilisateur.

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

CONTEXTE MÉTIER:
Intention détectée: {intent_type} (confidence: {confidence})
Message utilisateur: "{user_message}"

EXTRACTION REQUISE:
{self._get_extraction_strategy_for_intent(intent_type)}

CONSIGNES SPÉCIFIQUES:
- Réponse JSON uniquement avec format exact requis
- Pour les dates: utilisez OBLIGATOIREMENT la logique temporelle ci-dessus
- Types d'opérations: utilisez les valeurs Elasticsearch (transfer, card, direct_debit, withdrawal, etc.)
- Montants en EUR par défaut
- Confidence globale cohérente avec contexte

EXEMPLE CORRECT pour "mai" en {current_month_fr} {current_date.year}:
{{"dates": [{{"type": "period", "value": "{current_date.year if 5 <= current_date.month else current_date.year - 1}-05", "text": "mai"}}]}}

MESSAGE À ANALYSER: "{user_message}"
"""
        return simple_context

    def _build_team_extraction_prompt(self, user_message: str, intent_context: Dict[str, Any]) -> str:
        """
        Construction prompt adaptatif réutilisant infrastructure existante
        """
        intent_type = intent_context.get("intent", "GENERAL_INQUIRY")
        confidence = intent_context.get("confidence", 0.5)
        
        # Contexte temporel intelligent (cohérent avec méthode simple)
        from datetime import datetime
        current_date = datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        current_month_name = current_date.strftime("%B").lower()
        current_month_fr = {
            "january": "janvier", "february": "février", "march": "mars", "april": "avril",
            "may": "mai", "june": "juin", "july": "juillet", "august": "août", 
            "september": "septembre", "october": "octobre", "november": "novembre", "december": "décembre"
        }.get(current_month_name, current_month_name)
        
        # Prompt base existant + adaptation selon intention avec contexte temporel
        team_context_addition = f"""
CONTEXTE TEMPOREL IMPORTANT:
Date actuelle: {current_date_str}
Nous sommes en {current_month_fr} {current_date.year}

RÈGLES TEMPORELLES CRITIQUES:
- Pour un mois sans année mentionnée (ex: "mai"):
  * Si le mois est passé ou en cours cette année - utiliser {current_date.year}
  * Si le mois est dans le futur - utiliser {current_date.year - 1}
- "mois dernier" - mois précédent par rapport à {current_month_fr} {current_date.year}
- "ce mois" ou "ce mois-ci" - {current_date.strftime("%Y-%m")}

CONTEXTE ÉQUIPE AUTOGEN:
Intention détectée par agent précédent: {intent_type} (confidence: {confidence})
Message utilisateur original: "{user_message}"

ADAPTATION EXTRACTION SELON INTENTION:
{self._get_extraction_strategy_for_intent(intent_type)}

CONSIGNES SPÉCIFIQUES:
- Réponse JSON uniquement selon format ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT
- Pour les dates: utilisez OBLIGATOIREMENT la logique temporelle ci-dessus
- Types d'opérations: valeurs Elasticsearch (transfer, card, direct_debit, withdrawal, etc.)
- Focus sur entités prioritaires selon intention
- Normalisation métier Harena (montants EUR, dates ISO avec contexte temporel)
- Confidence globale cohérente avec contexte équipe

EXEMPLE CORRECT pour "mai" en {current_month_fr} {current_date.year}:
{{"dates": [{{"type": "period", "value": "{current_date.year if 5 <= current_date.month else current_date.year - 1}-05", "text": "mai"}}]}}

MESSAGE À TRAITER: "{user_message}"
"""
        
        return team_context_addition
    
    def _get_extraction_strategy_for_intent(self, intent_type: str) -> str:
        """Stratégies extraction selon intention (logique métier cohérente existante)"""
        
        strategies = {
            "SEARCH_BY_MERCHANT": "Focus PRIORITAIRE: merchants, amounts, dates. Normalise marchands avec variations courantes (Amazon→Amazon, Leclerc→E.Leclerc).",
            "SEARCH_BY_AMOUNT": "Focus PRIORITAIRE: amounts avec operators (eq, gt, lt), dates. Détecte comparaisons ('plus de', 'moins de').",
            "SPENDING_ANALYSIS": "Focus PRIORITAIRE: amounts, categories, dates, time_periods. Extraction comprehensive pour analyses.",
            "BALANCE_INQUIRY": "Focus MINIMAL: dates, accounts. Extraction temporelle principalement.",
            "SEARCH_BY_DATE": "Focus PRIORITAIRE: dates avec normalisation complète. Traite 'hier', 'mois dernier', périodes.",
            "SEARCH_BY_OPERATION_TYPE": "Focus PRIORITAIRE: operation_types, dates, amounts. Normalise types: 'virement', 'CB', 'prélèvement'.",
            "SEARCH_BY_CATEGORY": "Focus PRIORITAIRE: categories, amounts, dates. Normalise catégories Harena.",
            "TRANSACTION_HISTORY": "Focus ÉQUILIBRÉ: tous types entités avec emphasis dates et amounts.",
            "COUNT_TRANSACTIONS": "Focus: merchants, operation_types, dates, categories pour comptage.",
            "TRANSFER_REQUEST": "Focus PRIORITAIRE: amounts, recipients, dates, accounts (non supporté mais extraction utile).",
            "PAYMENT_REQUEST": "Focus PRIORITAIRE: amounts, recipients, dates, references (non supporté mais extraction utile)."
        }
        
        return strategies.get(intent_type, "Focus ÉQUILIBRÉ: extraction standard toutes entités pertinentes.")
    
    def _parse_and_validate_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse et validation réutilisant patterns existants"""
        
        try:
            # Nettoyage JSON (cohérent avec IntentClassifier existant)
            cleaned_response = self._clean_json_content(llm_response)
            
            # Parse JSON
            extraction_data = json.loads(cleaned_response)
            
            # Validation structure (réutilise logique existante)
            validated_data = self._validate_extraction_structure(extraction_data)
            
            # Validation métier (cohérent existant)
            self._validate_business_rules(validated_data)
            
            return validated_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON invalide lors extraction: {e}")
            logger.debug(f"Contenu reçu: {llm_response[:200]}...")
            return self._create_json_error_fallback()
            
        except ValueError as e:
            logger.error(f"Validation métier échouée: {e}")
            return self._create_validation_error_fallback()
    
    def _clean_json_content(self, content: str) -> str:
        """Nettoyage JSON réutilisant patterns IntentClassifier"""
        # Supprime les blocs markdown potentiels
        content = content.replace("```json", "").replace("```", "")
        content = content.strip()
        
        # Trouve le premier { et le dernier }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            content = content[start_idx:end_idx + 1]
        
        return content
    
    def _validate_extraction_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validation structure cohérente avec prompt existant"""
        
        # Champs requis selon ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT
        required_fields = ["entities", "confidence", "reasoning"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Champ requis manquant: {field}")
        
        # Validation entités selon structure existante
        entities = data.get("entities", {})
        expected_entity_types = ["amounts", "dates", "merchants", "categories", "operation_types", "transaction_types", "text_search"]
        
        for entity_type in expected_entity_types:
            if entity_type not in entities:
                entities[entity_type] = []
        
        # Validation confidence
        confidence = data.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            logger.warning(f"Confidence invalide {confidence}, correction vers 0.5")
            data["confidence"] = 0.5
        
        return data
    
    def _validate_business_rules(self, data: Dict[str, Any]) -> None:
        """Validation règles métier Harena"""
        
        entities = data.get("entities", {})
        
        # Validation amounts
        for amount in entities.get("amounts", []):
            if not isinstance(amount, dict):
                continue
            if "value" not in amount or not isinstance(amount["value"], (int, float)):
                raise ValueError(f"Amount invalide: {amount}")
            if "currency" not in amount:
                amount["currency"] = "EUR"  # Défaut Harena
            if "operator" not in amount:
                amount["operator"] = "eq"   # Défaut
        
        # Validation dates
        for date_obj in entities.get("dates", []):
            if not isinstance(date_obj, dict):
                continue
            if "value" not in date_obj and "text" not in date_obj:
                raise ValueError(f"Date invalide: {date_obj}")
        
        # Validation merchants (cohérent infrastructure existante)
        merchants = entities.get("merchants", [])
        if merchants and not isinstance(merchants, list):
            raise ValueError("Merchants doit être une liste")
    
    def _enrich_result_for_team(self, extraction_result: Dict[str, Any], team_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrichissement résultat pour collaboration équipe AutoGen
        """
        # Ajout contexte équipe pour agent suivant
        extraction_result["team_context"] = {
            "source_agent": "entity_extractor",
            "intent_context": team_context.get("intent_result", {}),
            "user_id": team_context.get("user_id"),
            "ready_for_query_generation": True,
            "entities_summary": self._generate_entities_summary(extraction_result.get("entities", {})),
            "extraction_confidence": extraction_result.get("confidence", 0.0)
        }
        
        # Méta-données pour agent suivant
        extraction_result["processing_metadata"] = {
            "extraction_mode": "team_collaboration",
            "agent_name": self.name if hasattr(self, 'name') else getattr(self._autogen_agent, 'name', 'entity_extractor'),
            "autogen_mode": self._autogen_mode,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return extraction_result
    
    def _generate_entities_summary(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Génère résumé entités pour agent suivant"""
        
        summary = {}
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                summary[entity_type] = {
                    "count": len(entity_list),
                    "present": len(entity_list) > 0
                }
                
                # Détails spécialisés
                if entity_type == "amounts" and entity_list:
                    summary[entity_type]["total_amount"] = sum(
                        item.get("value", 0) for item in entity_list 
                        if isinstance(item, dict) and isinstance(item.get("value"), (int, float))
                    )
                elif entity_type == "merchants" and entity_list:
                    summary[entity_type]["sample"] = entity_list[0] if entity_list else None
        
        return summary
    
    async def _get_cached_extraction(self, user_message: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Cache réutilisant CacheManager existant"""
        
        try:
            if not self.cache_manager:
                return None
            
            cache_key = f"entity_extraction_{user_id}_{hash(user_message)}"
            
            cached_result = await self.cache_manager.get_semantic_cache(
                cache_key, 
                similarity_threshold=0.85
            )
            
            if cached_result:
                logger.info(f"Cache hit extraction entités: {cache_key}")
                return cached_result
            
            return None
            
        except Exception as e:
            logger.debug(f"Erreur cache récupération: {str(e)}")
            return None
    
    async def _cache_extraction_result(self, user_message: str, user_id: int, result: Dict[str, Any]) -> None:
        """Mise en cache résultat extraction (infrastructure existante)"""
        
        try:
            if not self.cache_manager:
                return
            
            cache_key = f"entity_extraction_{user_id}_{hash(user_message)}"
            cache_ttl = getattr(settings, 'CACHE_TTL_EXTRACTION', 3600)  # 1h par défaut
            
            # Préparation données cache (sans team_context temporaire)
            cache_data = result.copy()
            if "team_context" in cache_data:
                del cache_data["team_context"]  # Éviter pollution cache
            
            await self.cache_manager.set_semantic_cache(
                cache_key,
                cache_data,
                ttl=cache_ttl
            )
            
        except Exception as e:
            logger.debug(f"Erreur cache sauvegarde: {str(e)}")
    
    def _enrich_cached_result_for_team(self, cached_result: Dict[str, Any], team_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichissement résultat caché pour équipe"""
        
        enriched_result = cached_result.copy()
        
        # Ajout contexte équipe même pour résultat caché
        enriched_result = self._enrich_result_for_team(enriched_result, team_context)
        
        # Marquage cache
        enriched_result["cache_used"] = True
        
        return enriched_result
    
    def _count_extracted_entities(self, extraction_result: Dict[str, Any]) -> int:
        """Compte entités extraites pour métriques"""
        
        entities = extraction_result.get("entities", {})
        total = 0
        
        for entity_list in entities.values():
            if isinstance(entity_list, list):
                total += len(entity_list)
        
        return total
    
    def _update_metrics_extraction(self, processing_time: float, success: bool, entities_count: int) -> None:
        """Métriques intégrées avec infrastructure existante"""
        
        try:
            # Métriques globales (existantes)
            event = "extraction_success" if success else "extraction_error"
            metrics_collector.increment_counter(f"entity_extractor.{event}")
            metrics_collector.record_histogram(f"entity_extractor.{event}.latency", processing_time)
            
            if success:
                metrics_collector.record_histogram("entity_extractor.entities_extracted", entities_count)
            
        except Exception as e:
            logger.debug(f"Échec mise à jour métriques extraction {success}: {str(e)}")
    
    def _create_fallback_extraction(self, user_message: str, error_reason: str) -> Dict[str, Any]:
        """Fallback cohérent avec patterns existants"""
        
        return {
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
            "reasoning": f"Extraction échouée: {error_reason}",
            "extraction_success": False,
            "fallback_used": True,
            "original_message": user_message
        }
    
    def _create_error_extraction(self, error_reason: str) -> Dict[str, Any]:
        """Création extraction d'erreur"""
        
        return {
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
            "reasoning": f"Erreur extraction: {error_reason}",
            "extraction_success": False,
            "error": True,
            "team_context": {
                "ready_for_query_generation": False,
                "error_handled": True
            }
        }
    
    def _create_json_error_fallback(self) -> Dict[str, Any]:
        """Fallback pour erreurs JSON"""
        return self._create_fallback_extraction("", "JSON invalide reçu du LLM")
    
    def _create_validation_error_fallback(self) -> Dict[str, Any]:
        """Fallback pour erreurs validation"""
        return self._create_fallback_extraction("", "Validation structure/métier échouée")
    
    def get_extraction_metrics(self) -> Dict[str, Any]:
        """Retourne métriques extraction"""
        return self._extraction_metrics.copy()