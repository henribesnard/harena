"""
Query Builder - Agent Logique Phase 3
Architecture v2.0 - Composant déterministe

Responsabilité : Sélection de templates et injection de paramètres
- Mapping intention  template approprié
- Extraction et validation des entités
- Construction de requêtes search_service
- Validation des paramètres selon configuration
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from conversation_service.core.template_engine import TemplateEngine, CompiledTemplate
from conversation_service.config.settings import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class QueryBuildRequest:
    """Requête pour construire une query"""
    intent_group: str
    intent_subtype: Optional[str]
    entities: Dict[str, Any]
    user_context: Dict[str, Any]
    user_id: int
    conversation_id: Optional[str] = None

@dataclass  
class QueryBuildResult:
    """Résultat de construction de query"""
    success: bool
    query: Optional[Dict[str, Any]]
    template_used: Optional[str]
    parameters_injected: Dict[str, Any]
    validation_errors: List[str]
    processing_time_ms: int
    fallback_used: bool = False

class QueryBuilder:
    """
    Agent logique pour construction de requêtes search_service
    
    Sélectionne le bon template selon l'intention et injecte les paramètres
    """
    
    def __init__(
        self,
        template_engine: TemplateEngine,
        config_manager: ConfigManager
    ):
        self.template_engine = template_engine
        self.config_manager = config_manager
        
        # Cache des mappings intention  template
        self._intent_template_cache: Dict[str, str] = {}
        
        # Statistiques 
        self.stats = {
            "queries_built": 0,
            "template_cache_hits": 0,
            "validation_failures": 0,
            "fallbacks_used": 0
        }
        
        logger.info("QueryBuilder initialisé")
    
    async def build_query(self, request: QueryBuildRequest) -> QueryBuildResult:
        """
        Construit une requête search_service à partir d'une intention et entités
        
        Args:
            request: Requête de construction avec intention et entités
            
        Returns:
            QueryBuildResult avec query générée ou erreurs
        """
        start_time = datetime.now()
        
        try:
            # Cas spécial : les intents CONVERSATIONAL n'ont pas besoin de templates
            if request.intent_group == "CONVERSATIONAL":
                return QueryBuildResult(
                    success=True,
                    query=None,  # Pas de query pour les intents conversationnels
                    template_used="conversational_bypass",
                    parameters_injected={},
                    validation_errors=[],
                    processing_time_ms=self._get_processing_time(start_time)
                )
            
            # 1. Sélection du template approprié
            template_result = await self._select_template(
                request.intent_group, 
                request.intent_subtype
            )
            
            if not template_result[0]:
                return QueryBuildResult(
                    success=False,
                    query=None,
                    template_used=None,
                    parameters_injected={},
                    validation_errors=[f"No template found for {request.intent_group}.{request.intent_subtype}"],
                    processing_time_ms=self._get_processing_time(start_time)
                )
                
            template = template_result[1]
            
            # 2. Préparation des paramètres
            template_parameters = self._prepare_template_parameters(request)
            
            # 3. Validation des paramètres requis
            validation_errors = self._validate_parameters(template, template_parameters)
            
            if validation_errors:
                self.stats["validation_failures"] += 1
                return QueryBuildResult(
                    success=False,
                    query=None,
                    template_used=template.name,
                    parameters_injected=template_parameters,
                    validation_errors=validation_errors,
                    processing_time_ms=self._get_processing_time(start_time)
                )
            
            # 4. Rendu du template avec paramètres
            logger.info(f"🎯 Using template: {template.name} for intent {request.intent_group}.{request.intent_subtype}")
            try:
                query = await self.template_engine.render_template(template, template_parameters)
            except Exception as e:
                logger.error(f"Erreur rendu template {template.name}: {str(e)}")
                return QueryBuildResult(
                    success=False,
                    query=None,
                    template_used=template.name,
                    parameters_injected=template_parameters,
                    validation_errors=[f"Template rendering failed: {str(e)}"],
                    processing_time_ms=self._get_processing_time(start_time)
                )
            
            # 5. Post-validation de la query générée
            query_validation_errors = self._validate_generated_query(query)
            
            if query_validation_errors:
                return QueryBuildResult(
                    success=False,
                    query=query,
                    template_used=template.name,
                    parameters_injected=template_parameters,
                    validation_errors=query_validation_errors,
                    processing_time_ms=self._get_processing_time(start_time)
                )
            
            # Success !
            self.stats["queries_built"] += 1
            
            return QueryBuildResult(
                success=True,
                query=query,
                template_used=template.name,
                parameters_injected=template_parameters,
                validation_errors=[],
                processing_time_ms=self._get_processing_time(start_time)
            )
            
        except Exception as e:
            logger.error(f"Erreur inattendue construction query: {str(e)}")
            return QueryBuildResult(
                success=False,
                query=None,
                template_used=None,
                parameters_injected={},
                validation_errors=[f"Unexpected error: {str(e)}"],
                processing_time_ms=self._get_processing_time(start_time)
            )
    
    async def _select_template(
        self,
        intent_group: str,
        intent_subtype: Optional[str]
    ) -> Tuple[bool, Optional[CompiledTemplate]]:
        """Sélectionne le template approprié pour une intention"""

        # Mapping des intents vers les templates appropriés
        # financial_query -> transaction_search avec aggregations
        intent_mapping = {
            "FINANCIAL_QUERY": "TRANSACTION_SEARCH"
        }

        # Appliquer le mapping si nécessaire
        original_intent = intent_group
        intent_group = intent_mapping.get(intent_group, intent_group)

        if original_intent != intent_group:
            logger.info(f"Mapping intent {original_intent} -> {intent_group}")

        # Clé de cache pour mapping intention  template
        cache_key = f"{intent_group}.{intent_subtype or 'default'}"
        
        # Vérifier cache d'abord
        if cache_key in self._intent_template_cache:
            template_name = self._intent_template_cache[cache_key]
            template = await self.template_engine.load_template(intent_group, template_name)
            if template:
                self.stats["template_cache_hits"] += 1
                return True, template
        
        # Stratégie de sélection basée sur la configuration
        intentions_config = self.config_manager.get_intentions_config()
        
        if not intentions_config or intent_group not in intentions_config.get("intent_groups", {}):
            logger.warning(f"Intent group {intent_group} not found in configuration")
            return False, None
        
        intent_config = intentions_config["intent_groups"][intent_group]
        
        # Recherche du template par intent_subtype
        template_name = None
        
        if intent_subtype:
            # Chercher un template spécifique pour le sous-type
            template_name = intent_subtype
            template = await self.template_engine.load_template(intent_group, template_name)
            
            if template:
                # Mise en cache du mapping
                self._intent_template_cache[cache_key] = template_name
                return True, template
        
        # Fallback : chercher un template par défaut
        # composite est le template universel qui gère toutes les combinaisons d'entités
        default_templates = ["composite", "default", "simple", "base"]

        for template_candidate in default_templates:
            template = await self.template_engine.load_template(intent_group, template_candidate)
            if template:
                self._intent_template_cache[cache_key] = template_candidate
                self.stats["fallbacks_used"] += 1
                logger.info(f"Using fallback template {template_candidate} for {cache_key}")
                return True, template
        
        logger.error(f"No template found for {intent_group}.{intent_subtype}")
        return False, None
    
    def _prepare_template_parameters(self, request: QueryBuildRequest) -> Dict[str, Any]:
        """Prépare les paramètres pour le template"""
        
        return {
            "context": {
                "user_id": request.user_id,
                "conversation_id": request.conversation_id,
                **request.user_context
            },
            "entities": request.entities,
            "intent": {
                "group": request.intent_group,
                "subtype": request.intent_subtype
            }
        }
    
    def _validate_parameters(
        self, 
        template: CompiledTemplate, 
        parameters: Dict[str, Any]
    ) -> List[str]:
        """Valide que tous les paramètres requis sont présents"""
        
        errors = []
        
        for param_name, param_config in template.parameter_mappings.items():
            if param_config.get("required", False):
                source = param_config.get("source", "")
                
                # Vérifier que le paramètre source existe
                value = self._get_nested_parameter(parameters, source)
                
                if value is None:
                    errors.append(f"Required parameter '{param_name}' missing from source '{source}'")
        
        return errors
    
    def _get_nested_parameter(self, data: Dict[str, Any], path: str) -> Any:
        """Récupère un paramètre nested (ex: 'context.user_id')"""
        if not path:
            return None
            
        parts = path.split('.')
        current = data
        
        try:
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None
    
    def _validate_generated_query(self, query: Dict[str, Any]) -> List[str]:
        """Valide que la query générée est cohérente"""
        
        errors = []
        
        # Vérifications de base
        if not isinstance(query, dict):
            errors.append("Generated query is not a dictionary")
            return errors
        
        # Vérifier présence user_id
        if "user_id" not in query:
            errors.append("Generated query missing required 'user_id' field")
        
        # Vérifier que user_id est un entier
        user_id = query.get("user_id")
        if user_id is not None and not isinstance(user_id, int):
            errors.append(f"user_id must be integer, got {type(user_id)}")
        
        # Vérifier structure des filtres si présents
        if "filters" in query:
            filters = query["filters"]
            if not isinstance(filters, dict):
                errors.append("Query 'filters' must be a dictionary")
        
        # Vérifier structure des agrégations si présentes
        if "aggregations" in query:
            aggregations = query["aggregations"]
            if not isinstance(aggregations, dict):
                errors.append("Query 'aggregations' must be a dictionary")
        
        return errors
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calcule le temps de traitement en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    async def get_available_templates(self, intent_group: str) -> List[str]:
        """Récupère la liste des templates disponibles pour un groupe d'intention"""
        
        # Utiliser le template engine pour lister les templates
        templates = []
        
        for template_name, template in self.template_engine.compiled_templates.items():
            if template.template_data.get("target_intention", "").startswith(intent_group):
                templates.append(template_name)
        
        return templates
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du QueryBuilder"""
        
        return {
            **self.stats,
            "cache_size": len(self._intent_template_cache),
            "template_engine_stats": self.template_engine.get_cache_stats()
        }
    
    async def clear_cache(self) -> None:
        """Vide le cache des mappings intention  template"""
        
        self._intent_template_cache.clear()
        logger.info("QueryBuilder cache cleared")