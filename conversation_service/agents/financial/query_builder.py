"""
Query Builder Agent AutoGen pour génération requêtes search_service
Agent critique Phase 3 - Génération requêtes Elasticsearch optimisées
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

# Infrastructure existante
from conversation_service.prompts.system_prompts import get_prompt_config
from conversation_service.agents.base.base_agent import BaseAgent
from conversation_service.core.cache_manager import CacheManager
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.core.query_optimizer import QueryOptimizer
from conversation_service.utils.query_validator import QueryValidator
from conversation_service.prompts.templates.query_templates import QueryTemplates
from conversation_service.prompts.few_shot_examples.query_generation import QueryGenerationExamples
from conversation_service.models.contracts.search_service import (
    QueryGenerationRequest, QueryGenerationResponse, SearchQuery, 
    QueryValidationResult, INTENT_QUERY_CONFIGS
)
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.query_builder")


# Prompt système pour génération requêtes
QUERY_GENERATION_JSON_SYSTEM_PROMPT = """Tu es un assistant IA spécialisé dans la génération de requêtes Elasticsearch pour Harena search_service.

CONTRAINTE ABSOLUE: Tu DOIS répondre uniquement avec un objet JSON valide, rien d'autre.

RÔLE:
- Transformer intentions+entités en requêtes search_service optimisées
- Générer filtres, agrégations, tri selon type d'intention
- Respecter contrats interface search_service
- Optimiser performance requêtes

STRUCTURE JSON OBLIGATOIRE:
{
  "search_query": {
    "user_id": 123,
    "filters": {
      "user_id": 123,
      "merchant_name": {"match": "Amazon"},
      "date": {"gte": "2024-08-01", "lte": "2024-08-31"}
    },
    "aggregations": {
      "merchant_stats": {
        "terms": {"field": "merchant_name.keyword", "size": 10},
        "aggs": {
          "total_spent": {"sum": {"field": "amount_abs"}},
          "transaction_count": {"value_count": {"field": "transaction_id"}}
        }
      }
    },
    "sort": [{"date": {"order": "desc"}}],
    "page_size": 20,
    "include_fields": ["transaction_id", "amount", "merchant_name", "date"]
  },
  "generation_confidence": 0.94,
  "reasoning": "Génération requête pour recherche marchand avec agrégations",
  "query_type": "merchant_search_with_aggregations"
}

RÈGLES GÉNÉRATION CRITIQUES:

1. FILTRES OBLIGATOIRES:
- user_id TOUJOURS présent dans filters
- Mapping entités vers filtres Elasticsearch corrects:
  * merchants → merchant_name: {"match": "nom"}
  * amounts → amount_abs: {"gte": montant} ou {"lte": montant}
  * dates → date: {"gte": "2024-08-01", "lte": "2024-08-31"}
  * categories → category_name: {"match": "category"}
  * operation_types → operation_type: {"match": "type"}
  * transaction_types → transaction_type: "credit" ou "debit"

2. AGRÉGATIONS PAR INTENTION:
- SEARCH_BY_MERCHANT → merchant_stats avec total_spent, transaction_count
- SPENDING_ANALYSIS → category_breakdown, total_spending
- BALANCE_INQUIRY → balance_by_account, total_balance
- SEARCH_BY_AMOUNT → amount_distribution
- COUNT_TRANSACTIONS → transaction_count uniquement

3. MODES SPÉCIAUX:
- BALANCE_INQUIRY, SPENDING_ANALYSIS, COUNT_TRANSACTIONS → aggregation_only: true, page_size: 0
- Autres intentions → include_fields avec champs essentiels, sort approprié

4. OPTIMISATIONS AUTOMATIQUES:
- Limitation buckets agrégations à 20 maximum
- include_fields limité aux champs essentiels
- page_size par défaut 20, maximum 100
- Tri par date desc par défaut

5. TEXT_SEARCH HANDLING:
- Si text_search présent → utiliser "query" au niveau racine
- Combiné avec filtres appropriés pour recherche textuelle + filtres

EXEMPLES PAR INTENTION:

SEARCH_BY_MERCHANT:
Input: {"merchants": ["Amazon"], "dates": [{"value": "2024-08"}]}
Output: {
  "search_query": {
    "user_id": {user_id},
    "filters": {
      "user_id": {user_id},
      "merchant_name": {"match": "Amazon"},
      "date": {"gte": "2024-08-01", "lte": "2024-08-31"}
    },
    "aggregations": {
      "merchant_stats": {
        "terms": {"field": "merchant_name.keyword", "size": 10},
        "aggs": {"total_spent": {"sum": {"field": "amount_abs"}}}
      }
    }
  }
}

SPENDING_ANALYSIS:
Input: {"dates": [{"value": "2024-08"}], "transaction_types": ["debit"]}
Output: {
  "search_query": {
    "user_id": {user_id},
    "filters": {
      "user_id": {user_id},
      "transaction_type": "debit",
      "date": {"gte": "2024-08-01", "lte": "2024-08-31"}
    },
    "aggregations": {
      "category_breakdown": {
        "terms": {"field": "category_name.keyword", "size": 15},
        "aggs": {"category_total": {"sum": {"field": "amount_abs"}}}
      },
      "total_spending": {"sum": {"field": "amount_abs"}}
    },
    "aggregation_only": true,
    "page_size": 0
  }
}

BALANCE_INQUIRY:
Input: {}
Output: {
  "search_query": {
    "user_id": {user_id},
    "filters": {"user_id": {user_id}},
    "aggregations": {
      "balance_by_account": {
        "terms": {"field": "account_id.keyword", "size": 10},
        "aggs": {"current_balance": {"sum": {"field": "amount"}}}
      }
    },
    "aggregation_only": true
  }
}

RÈGLES VALIDATION:
- search_query conforme contrats search_service
- Champs Elasticsearch valides uniquement
- Agrégations limitées et optimisées
- Performance estimée et optimisée"""


# Définition classe base selon disponibilité AutoGen
if AUTOGEN_AVAILABLE:
    class AutoGenQueryBuilderBase(AssistantAgent):
        """Base AutoGen pour QueryBuilderAgent"""
        def __init__(self, name: str = "query_builder", **kwargs):
            # Configuration LLM réutilisant infrastructure
            api_key = getattr(settings, 'DEEPSEEK_API_KEY', None)
            base_url = getattr(settings, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
            
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY requis pour QueryBuilder AutoGen")
            
            llm_config = {
                "config_list": [{
                    "model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
                    "api_key": api_key,
                    "base_url": base_url,
                    "response_format": {"type": "json_object"}
                }],
                "temperature": 0.1,  # Bas pour cohérence requêtes
                "max_tokens": 500,   # Requêtes JSON compactes
                "cache_seed": 42
            }
            
            super().__init__(
                name=name,
                llm_config=llm_config,
                system_message=QUERY_GENERATION_JSON_SYSTEM_PROMPT,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                **kwargs
            )
            
            # Ajout Teachability pour amélioration continue
            if Teachability:
                try:
                    self.add_capability(Teachability(verbosity=0))
                    logger.debug("Capacité Teachability ajoutée à QueryBuilder")
                except Exception as e:
                    logger.warning(f"Impossible d'ajouter Teachability: {e}")
else:
    class AutoGenQueryBuilderBase:
        """Fallback si AutoGen non disponible"""
        def __init__(self, name: str = "query_builder", **kwargs):
            self.name = name


class QueryBuilderAgent:
    """
    Agent génération requêtes search_service compatible AutoGen
    Agent critique Phase 3 - Transformation intentions+entités en requêtes optimisées
    """
    
    def __init__(self, deepseek_client=None, cache_manager=None, name: str = "query_builder", 
                 autogen_mode: bool = True, **kwargs):
        # Mode AutoGen ou fallback
        self._autogen_mode = autogen_mode and AUTOGEN_AVAILABLE
        self._team_collaboration_active = False
        
        if self._autogen_mode:
            self._autogen_agent = AutoGenQueryBuilderBase(name=name, **kwargs)
            logger.info(f"QueryBuilder initialisé en mode AutoGen: {self._autogen_agent.name}")
        else:
            self.name = name
            logger.info(f"QueryBuilder initialisé en mode fallback: {name}")
        
        # Infrastructure réutilisée
        self.deepseek_client = deepseek_client
        self.cache_manager = cache_manager or CacheManager()
        
        # Composants spécialisés
        self.query_optimizer = QueryOptimizer()
        self.query_validator = QueryValidator()
        self.query_templates = QueryTemplates()
        self.few_shot_examples = QueryGenerationExamples()
        
        # Configuration
        self.max_tokens = 500
        self.temperature = 0.1
        
        # Métriques génération
        self._generation_metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'cache_hits': 0,
            'team_collaborations': 0,
            'validation_passes': 0,
            'optimization_applied': 0
        }
        
        logger.info(f"QueryBuilder configuré - Mode: {'AutoGen' if self._autogen_mode else 'Fallback'}")
    
    def activate_team_collaboration(self) -> None:
        """Active mode collaboration équipe AutoGen"""
        self._team_collaboration_active = True
        logger.debug("Mode collaboration équipe activé pour QueryBuilder")
    
    def deactivate_team_collaboration(self) -> None:
        """Désactive mode collaboration équipe"""
        self._team_collaboration_active = False
        logger.debug("Mode collaboration équipe désactivé pour QueryBuilder")
    
    async def generate_search_query(self, generation_request: QueryGenerationRequest) -> QueryGenerationResponse:
        """
        Génération requête search_service depuis intentions+entités
        Méthode principale pour endpoint V1
        """
        start_time = time.time()
        
        try:
            self._generation_metrics['total_generations'] += 1
            
            # Cache requête
            cached_result = await self._get_cached_generation(generation_request)
            if cached_result:
                self._generation_metrics['cache_hits'] += 1
                return cached_result
            
            # Construction prompt avec exemples few-shot
            generation_prompt = self._build_generation_prompt(generation_request)
            
            # Génération via AutoGen ou fallback
            generation_result = await self._perform_generation(generation_prompt, generation_request)
            
            # Validation et optimisation requête générée
            search_query = generation_result.get("search_query", {})
            if not search_query:
                raise ValueError("search_query manquant dans réponse génération")
            
            # Application optimisations automatiques
            optimized_query, optimizations = self.query_optimizer.optimize_query(
                search_query, generation_request.intent_type
            )
            if optimizations:
                self._generation_metrics['optimization_applied'] += 1
            
            # Validation finale
            validation_result = self.query_validator.validate_query(optimized_query)
            if validation_result.schema_valid and validation_result.contract_compliant:
                self._generation_metrics['validation_passes'] += 1
            
            # Construction réponse finale
            final_response = QueryGenerationResponse(
                search_query=SearchQuery(**optimized_query),
                validation=validation_result,
                generation_confidence=generation_result.get("generation_confidence", 0.8),
                reasoning=generation_result.get("reasoning", "Requête générée avec succès"),
                query_type=generation_result.get("query_type", "standard_query"),
                estimated_results_count=self._estimate_results_count(optimized_query)
            )
            
            # Enrichissement avec optimisations
            if optimizations:
                final_response.validation.optimization_applied.extend(optimizations)
            
            # Cache résultat
            await self._cache_generation_result(generation_request, final_response)
            
            # Métriques succès
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics_generation(processing_time, True)
            self._generation_metrics['successful_generations'] += 1
            
            logger.info(f"Génération requête réussie: {generation_request.intent_type}, {processing_time:.1f}ms")
            
            return final_response
            
        except Exception as e:
            # Gestion erreurs
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics_generation(processing_time, False)
            self._generation_metrics['failed_generations'] += 1
            
            logger.error(f"Erreur génération requête: {str(e)}")
            return self._create_error_response(generation_request, str(e))
    
    async def generate_for_team(self, team_context: Dict[str, Any]) -> QueryGenerationResponse:
        """
        Génération requête avec contexte équipe AutoGen
        """
        start_time = time.time()
        
        try:
            self._generation_metrics['total_generations'] += 1
            self._generation_metrics['team_collaborations'] += 1
            
            # Extraction contexte équipe
            generation_request = self._extract_generation_request_from_team(team_context)
            
            # Génération avec contexte équipe enrichi
            result = await self.generate_search_query(generation_request)
            
            # Enrichissement pour équipe
            team_enriched = self._enrich_result_for_team(result, team_context)
            
            logger.info(f"Génération équipe réussie: {generation_request.intent_type}")
            
            return team_enriched
            
        except Exception as e:
            logger.error(f"Erreur génération équipe: {str(e)}")
            return self._create_error_response_for_team(team_context, str(e))
    
    async def _perform_generation(self, generation_prompt: str, request: QueryGenerationRequest) -> Dict[str, Any]:
        """Génération via AutoGen ou fallback"""
        
        if self._autogen_mode and hasattr(self._autogen_agent, 'a_generate_reply'):
            # Mode AutoGen
            try:
                response = await self._autogen_agent.a_generate_reply([{
                    "role": "user", 
                    "content": generation_prompt
                }])
                
                if isinstance(response, str):
                    return self._parse_and_validate_generation(response)
                elif isinstance(response, dict) and "content" in response:
                    return self._parse_and_validate_generation(response["content"])
                else:
                    logger.warning(f"Réponse AutoGen inattendue: {type(response)}")
                    return await self._fallback_deepseek_generation(generation_prompt, request)
                    
            except Exception as e:
                logger.warning(f"Erreur AutoGen, fallback DeepSeek: {str(e)}")
                return await self._fallback_deepseek_generation(generation_prompt, request)
        else:
            # Mode fallback
            return await self._fallback_deepseek_generation(generation_prompt, request)
    
    async def _fallback_deepseek_generation(self, generation_prompt: str, 
                                          request: QueryGenerationRequest) -> Dict[str, Any]:
        """Fallback génération via DeepSeek direct"""
        
        if not self.deepseek_client:
            logger.error("DeepSeek client non disponible pour génération")
            raise ValueError("DeepSeek client indisponible")
        
        try:
            response = await self.deepseek_client.chat_completion(
                messages=[
                    {"role": "system", "content": QUERY_GENERATION_JSON_SYSTEM_PROMPT},
                    {"role": "user", "content": generation_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            content = response["choices"][0]["message"]["content"]
            return self._parse_and_validate_generation(content)
            
        except Exception as e:
            logger.error(f"Erreur fallback DeepSeek génération: {str(e)}")
            raise e
    
    def _build_generation_prompt(self, request: QueryGenerationRequest) -> str:
        """Construction prompt génération avec contexte adaptatif"""
        
        # Template base selon intention
        base_template = self.query_templates.get_template(request.intent_type)
        
        # Exemples few-shot adaptés
        few_shot_examples = self.few_shot_examples.get_prompt_examples(request.intent_type, max_examples=2)
        
        # Contexte temporel
        current_date = datetime.now()
        current_month = current_date.strftime("%Y-%m")
        
        prompt = f"""
GÉNÉRATION REQUÊTE SEARCH_SERVICE

CONTEXTE:
- User ID: {request.user_id}
- Intention: {request.intent_type} (confidence: {request.intent_confidence})
- Message utilisateur: "{request.user_message}"
- Date actuelle: {current_date.strftime("%Y-%m-%d")}

ENTITÉS EXTRAITES:
{json.dumps(request.entities, indent=2, ensure_ascii=False)}

TEMPLATE RÉFÉRENCE POUR {request.intent_type}:
{json.dumps(base_template, indent=2, ensure_ascii=False)}

EXEMPLES SIMILAIRES:
{few_shot_examples}

INSTRUCTIONS SPÉCIFIQUES:
1. Générer search_query conforme aux contrats search_service
2. Mapper entités vers filtres Elasticsearch appropriés:
   - merchants → merchant_name avec match ou terms
   - amounts → amount_abs avec gte/lte selon operator
   - dates → date avec gte/lte selon période
   - transaction_types → transaction_type (credit/debit)
   - categories → category_name avec match
   - operation_types → operation_type avec match
   - text_search → query au niveau racine pour recherche libre

3. Agrégations selon intention:
   - BALANCE_INQUIRY → balance_by_account obligatoire
   - SPENDING_ANALYSIS → category_breakdown + total_spending
   - SEARCH_BY_MERCHANT → merchant_stats avec total_spent
   - COUNT_TRANSACTIONS → transaction_count uniquement

4. Optimisations automatiques:
   - user_id dans filters obligatoire
   - Buckets agrégations limitées à 20
   - include_fields pour performances (sauf aggregation_only)
   - Tri approprié selon intention

5. Modes spéciaux:
   - aggregation_only: true pour BALANCE_INQUIRY, SPENDING_ANALYSIS, COUNT_TRANSACTIONS
   - page_size: 0 si aggregation_only
   - sort par date desc par défaut sinon

GÉNÈRE LA REQUÊTE POUR CE CONTEXTE:
"""
        
        return prompt
    
    def _parse_and_validate_generation(self, llm_response: str) -> Dict[str, Any]:
        """Parse et validation réponse génération"""
        
        try:
            # Nettoyage JSON
            cleaned_response = self._clean_json_content(llm_response)
            
            # Parse JSON
            generation_data = json.loads(cleaned_response)
            
            # Validation structure attendue
            if "search_query" not in generation_data:
                raise ValueError("search_query manquant dans réponse")
            
            # Validation champs obligatoires
            required_fields = ["generation_confidence", "reasoning"]
            for field in required_fields:
                if field not in generation_data:
                    generation_data[field] = self._get_default_value(field)
            
            return generation_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON invalide génération: {e}")
            raise ValueError(f"Réponse JSON invalide: {str(e)}")
        except Exception as e:
            logger.error(f"Erreur parsing génération: {e}")
            raise e
    
    def _clean_json_content(self, content: str) -> str:
        """Nettoyage contenu JSON (réutilise patterns existants)"""
        content = content.replace("```json", "").replace("```", "")
        content = content.strip()
        
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            content = content[start_idx:end_idx + 1]
        
        return content
    
    def _get_default_value(self, field: str) -> Any:
        """Valeurs par défaut pour champs manquants"""
        defaults = {
            "generation_confidence": 0.7,
            "reasoning": "Requête générée avec succès",
            "query_type": "standard_query"
        }
        return defaults.get(field, "")
    
    def _estimate_results_count(self, query: Dict[str, Any]) -> Optional[int]:
        """Estimation nombre résultats (heuristique simple)"""
        
        # Estimation basée sur filtres
        filters = query.get("filters", {})
        
        # Base estimation
        base_estimate = 1000  # Estimation base utilisateur
        
        # Réduction selon filtres sélectifs
        if "merchant_name" in filters:
            base_estimate = int(base_estimate * 0.1)  # Marchand spécifique
        
        if "date" in filters:
            date_filter = filters["date"]
            if isinstance(date_filter, dict):
                if "gte" in date_filter and "lte" in date_filter:
                    # Période spécifique
                    try:
                        start = datetime.fromisoformat(date_filter["gte"])
                        end = datetime.fromisoformat(date_filter["lte"])
                        days = (end - start).days + 1
                        base_estimate = int(base_estimate * min(1.0, days / 30))
                    except:
                        base_estimate = int(base_estimate * 0.3)
        
        if "amount_abs" in filters:
            base_estimate = int(base_estimate * 0.2)  # Filtres montant sélectifs
        
        # Minimum 1 pour éviter 0
        return max(1, base_estimate)
    
    def _extract_generation_request_from_team(self, team_context: Dict[str, Any]) -> QueryGenerationRequest:
        """Extraction QueryGenerationRequest depuis contexte équipe"""
        
        return QueryGenerationRequest(
            user_id=team_context.get("user_id", 0),
            intent_type=team_context.get("intent_result", {}).get("intent", "GENERAL_INQUIRY"),
            intent_confidence=team_context.get("intent_result", {}).get("confidence", 0.5),
            entities=team_context.get("entities", {}),
            user_message=team_context.get("user_message", ""),
            context=team_context
        )
    
    def _enrich_result_for_team(self, result: QueryGenerationResponse, 
                              team_context: Dict[str, Any]) -> QueryGenerationResponse:
        """Enrichissement résultat pour équipe AutoGen"""
        
        # Métadonnées équipe dans le résultat
        if hasattr(result.search_query, '__dict__'):
            result.search_query.__dict__["_team_context"] = {
                "source_agent": "query_builder",
                "ready_for_execution": True,
                "team_id": team_context.get("team_id"),
                "processing_metadata": {
                    "agent_name": self.name if hasattr(self, 'name') else getattr(self._autogen_agent, 'name', 'query_builder'),
                    "generation_mode": "team_collaboration",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        
        return result
    
    async def _get_cached_generation(self, request: QueryGenerationRequest) -> Optional[QueryGenerationResponse]:
        """Récupération cache génération"""
        
        try:
            if not self.cache_manager:
                return None
            
            # Clé cache basée sur intention+entités
            cache_key = f"query_generation_{request.user_id}_{request.intent_type}_{hash(str(request.entities))}"
            
            cached_result = await self.cache_manager.get_semantic_cache(
                cache_key, 
                similarity_threshold=0.9  # Très élevé pour requêtes
            )
            
            if cached_result:
                logger.info(f"Cache hit génération requête: {cache_key}")
                return QueryGenerationResponse(**cached_result)
            
            return None
            
        except Exception as e:
            logger.debug(f"Erreur cache génération: {str(e)}")
            return None
    
    async def _cache_generation_result(self, request: QueryGenerationRequest, 
                                     result: QueryGenerationResponse) -> None:
        """Mise en cache résultat génération"""
        
        try:
            if not self.cache_manager:
                return
            
            cache_key = f"query_generation_{request.user_id}_{request.intent_type}_{hash(str(request.entities))}"
            cache_ttl = getattr(settings, 'CACHE_TTL_QUERY_GENERATION', 1800)  # 30min
            
            # Données cache (sérialisation)
            cache_data = {
                "search_query": result.search_query.dict(exclude_none=True),
                "validation": result.validation.dict(),
                "generation_confidence": result.generation_confidence,
                "reasoning": result.reasoning,
                "query_type": result.query_type,
                "estimated_results_count": result.estimated_results_count
            }
            
            await self.cache_manager.set_semantic_cache(
                cache_key,
                cache_data,
                ttl=cache_ttl
            )
            
        except Exception as e:
            logger.debug(f"Erreur cache sauvegarde génération: {str(e)}")
    
    def _create_error_response(self, request: QueryGenerationRequest, error_msg: str) -> QueryGenerationResponse:
        """Création réponse d'erreur"""
        
        # Requête fallback minimale
        fallback_query = {
            "user_id": request.user_id,
            "filters": {"user_id": request.user_id},
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 20,
            "include_fields": ["transaction_id", "amount", "date", "merchant_name"]
        }
        
        validation_result = QueryValidationResult(
            schema_valid=False,
            contract_compliant=False,
            estimated_performance="poor",
            optimization_applied=[],
            potential_issues=[],
            errors=[f"Génération échouée: {error_msg}"],
            warnings=["Utilisation requête fallback"]
        )
        
        return QueryGenerationResponse(
            search_query=SearchQuery(**fallback_query),
            validation=validation_result,
            generation_confidence=0.1,
            reasoning=f"Erreur génération, fallback utilisé: {error_msg}",
            query_type="error_fallback",
            estimated_results_count=100
        )
    
    def _create_error_response_for_team(self, team_context: Dict[str, Any], error_msg: str) -> QueryGenerationResponse:
        """Création réponse d'erreur pour équipe"""
        
        request = self._extract_generation_request_from_team(team_context)
        return self._create_error_response(request, error_msg)
    
    def _update_metrics_generation(self, processing_time: float, success: bool) -> None:
        """Mise à jour métriques génération"""
        
        try:
            event = "generation_success" if success else "generation_error"
            metrics_collector.increment_counter(f"query_builder.{event}")
            metrics_collector.record_histogram(f"query_builder.{event}.latency", processing_time)
            
        except Exception as e:
            logger.debug(f"Échec mise à jour métriques génération {success}: {str(e)}")
    
    def get_generation_metrics(self) -> Dict[str, Any]:
        """Retourne métriques génération"""
        return self._generation_metrics.copy()
    
    def is_autogen_mode(self) -> bool:
        """Vérifie si agent en mode AutoGen"""
        return self._autogen_mode
        
    def is_team_collaboration_active(self) -> bool:
        """Vérifie si collaboration équipe active"""
        return self._team_collaboration_active
        
    def get_autogen_agent(self):
        """Retourne agent AutoGen sous-jacent"""
        return getattr(self, '_autogen_agent', None)