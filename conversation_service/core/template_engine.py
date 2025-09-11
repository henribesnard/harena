"""
Template Engine pour Conversation Service v2.0
Compilation et parameterization des templates de requêtes JSON
"""

import json
import re
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from jinja2 import Template, Environment, FileSystemLoader, meta
from jinja2.exceptions import TemplateError

logger = logging.getLogger(__name__)

@dataclass
class CompiledTemplate:
    """Template compilé avec métadonnées"""
    name: str
    template_data: Dict[str, Any]
    jinja_template: Template
    parameter_mappings: Dict[str, Any]
    compiled_at: datetime
    cache_duration: timedelta
    file_path: Path

@dataclass
class TemplateCompilationResult:
    """Résultat de compilation d'un template"""
    success: bool
    template: Optional[CompiledTemplate] = None
    compilation_time_ms: float = 0.0
    error_message: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)

class TemplateEngine:
    """Moteur de templates pour génération de requêtes search_service"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent.parent / "templates" / "query"
        self.compiled_templates: Dict[str, CompiledTemplate] = {}
        self.jinja_env = Environment()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "compilations": 0
        }
        
        logger.info(f"TemplateEngine initialized with directory: {self.templates_dir}")

    async def initialize(self) -> bool:
        """Initialise le moteur de templates"""
        try:
            if not self.templates_dir.exists():
                logger.error(f"Templates directory not found: {self.templates_dir}")
                return False
                
            # Précompiler tous les templates
            await self._precompile_all_templates()
            logger.info(f"Template engine initialized with {len(self.compiled_templates)} templates")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize template engine: {e}")
            return False

    async def load_template(self, intent_group: str, intent_subtype: str) -> Optional[CompiledTemplate]:
        """Charge un template par intention"""
        template_key = f"{intent_group}.{intent_subtype}"
        
        # Vérifier le cache d'abord
        if template_key in self.compiled_templates:
            template = self.compiled_templates[template_key]
            if self._is_template_cache_valid(template):
                self.cache_stats["hits"] += 1
                return template
            else:
                # Template expiré, le supprimer du cache
                del self.compiled_templates[template_key]
        
        # Cache miss - compiler le template
        self.cache_stats["misses"] += 1
        return await self._compile_template_by_intent(intent_group, intent_subtype)

    async def compile_template(self, template_path: Path) -> TemplateCompilationResult:
        """Compile un template depuis un fichier"""
        start_time = time.perf_counter()
        
        try:
            if not template_path.exists():
                return TemplateCompilationResult(
                    success=False,
                    error_message=f"Template file not found: {template_path}"
                )
            
            # Charger le fichier JSON
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Valider la structure du template
            validation_errors = self._validate_template_structure(template_data)
            if validation_errors:
                return TemplateCompilationResult(
                    success=False,
                    validation_errors=validation_errors,
                    error_message=f"Template validation failed: {'; '.join(validation_errors)}"
                )
            
            # Compiler le template Jinja2
            parameters_json = json.dumps(template_data["parameters"])
            jinja_template = Template(parameters_json)
            
            # Calculer la durée de cache
            optimizations = template_data.get("optimizations", {})
            cache_duration = self._parse_cache_duration(optimizations.get("cache_duration", "1h"))
            
            # Créer le template compilé
            compiled_template = CompiledTemplate(
                name=template_data["template_name"],
                template_data=template_data,
                jinja_template=jinja_template,
                parameter_mappings=template_data.get("parameter_mappings", {}),
                compiled_at=datetime.now(),
                cache_duration=cache_duration,
                file_path=template_path
            )
            
            compilation_time = (time.perf_counter() - start_time) * 1000
            self.cache_stats["compilations"] += 1
            
            logger.debug(f"Template compiled successfully: {template_data['template_name']} ({compilation_time:.2f}ms)")
            
            return TemplateCompilationResult(
                success=True,
                template=compiled_template,
                compilation_time_ms=compilation_time
            )
            
        except json.JSONDecodeError as e:
            return TemplateCompilationResult(
                success=False,
                error_message=f"Invalid JSON in template: {e}",
                compilation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        except TemplateError as e:
            return TemplateCompilationResult(
                success=False,
                error_message=f"Jinja2 template error: {e}",
                compilation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        except Exception as e:
            return TemplateCompilationResult(
                success=False,
                error_message=f"Unexpected error compiling template: {e}",
                compilation_time_ms=(time.perf_counter() - start_time) * 1000
            )

    async def render_template(self, template: CompiledTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Rend un template avec les paramètres fournis"""
        try:
            # Préparer les données pour le template
            template_context = self._prepare_template_context(template, parameters)
            
            # Rendre le template
            rendered_json = template.jinja_template.render(**template_context)
            
            # Parser le JSON résultant
            rendered_query = json.loads(rendered_json)
            
            # Post-traitement
            cleaned_query = self._post_process_query(template, rendered_query)
            
            # Restaurer les types corrects des paramètres
            typed_query = self._restore_parameter_types(template, cleaned_query)
            
            return typed_query
            
        except Exception as e:
            logger.error(f"Error rendering template {template.name}: {e}")
            raise

    def _prepare_template_context(self, template: CompiledTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare le contexte pour le template Jinja2"""
        context = {}
        
        for param_name, param_config in template.parameter_mappings.items():
            value = self._extract_parameter_value(param_config, parameters)
            context[param_name] = value
            
        return context

    def _extract_parameter_value(self, param_config: Dict[str, Any], parameters: Dict[str, Any]) -> Any:
        """Extrait une valeur de paramètre selon sa configuration"""
        source = param_config.get("source", "")
        required = param_config.get("required", False)
        default_value = param_config.get("default")
        param_type = param_config.get("type", "string")
        
        # Parser le chemin source (ex: "entities.periode_temporelle.date")
        value = self._get_nested_value(parameters, source)
        
        # Appliquer la valeur par défaut si nécessaire
        if value is None and default_value is not None:
            value = default_value
        
        # Validation requis
        if required and value is None:
            raise ValueError(f"Required parameter missing: {source}")
        
        # Validation du type
        if value is not None:
            value = self._cast_parameter_type(value, param_type)
            
        # Validation des valeurs autorisées
        allowed_values = param_config.get("allowed_values")
        if allowed_values and value not in allowed_values:
            if required:
                raise ValueError(f"Parameter {source} value '{value}' not in allowed values: {allowed_values}")
            else:
                value = default_value
                
        return value

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Récupère une valeur nested par chemin (ex: 'entities.date.gte')"""
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

    def _cast_parameter_type(self, value: Any, param_type: str) -> Any:
        """Cast une valeur vers le type spécifié"""
        if value is None or value == "None":
            return None
            
        try:
            if param_type == "integer":
                return int(value)
            elif param_type == "float":
                return float(value)
            elif param_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif param_type == "string":
                return str(value)
            elif param_type == "merchant_name":
                # Normalisation pour les noms de marchands (première lettre majuscule)
                return str(value).strip().title() if value else value
            elif param_type == "object":
                return value  # Garder tel quel pour les objets
            else:
                return value
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast value {value} to type {param_type}: {e}")
            return value

    def _post_process_query(self, template: CompiledTemplate, query: Dict[str, Any]) -> Dict[str, Any]:
        """Post-traite la requête générée"""
        # Nettoyage selon les règles du template
        filter_cleanup = template.template_data.get("filter_cleanup", {})
        
        if filter_cleanup.get("remove_null_values", False):
            query = self._remove_null_values(query)
            
        if filter_cleanup.get("remove_empty_objects", False):
            query = self._remove_empty_objects(query)
            
        return query

    def _remove_null_values(self, obj: Any) -> Any:
        """Supprime les valeurs null récursivement"""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Exclure les valeurs null ET les chaînes "None" (de Jinja2)
                if v is not None and v != "None" and str(v).strip() != "":
                    cleaned_value = self._remove_null_values(v)
                    if cleaned_value is not None and cleaned_value != "None":
                        result[k] = cleaned_value
            return result
        elif isinstance(obj, list):
            result = []
            for item in obj:
                if item is not None and item != "None":
                    cleaned_item = self._remove_null_values(item)
                    if cleaned_item is not None and cleaned_item != "None":
                        result.append(cleaned_item)
            return result
        else:
            # Convertir "None" en None réel
            if obj == "None":
                return None
            return obj

    def _remove_empty_objects(self, obj: Any) -> Any:
        """Supprime les objets vides récursivement"""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_v = self._remove_empty_objects(v)
                if cleaned_v or cleaned_v == 0 or cleaned_v is False:  # Garder 0 et False
                    cleaned[k] = cleaned_v
            return cleaned if cleaned else None
        elif isinstance(obj, list):
            return [self._remove_empty_objects(item) for item in obj if item]
        else:
            return obj

    def _restore_parameter_types(self, template: CompiledTemplate, query: Dict[str, Any]) -> Dict[str, Any]:
        """Restaure les types corrects des paramètres après rendu Jinja2"""
        # Créer un mapping des paramètres et leurs types
        type_mappings = {}
        for param_name, param_config in template.parameter_mappings.items():
            param_type = param_config.get("type", "string")
            type_mappings[param_name] = param_type
        
        return self._apply_type_mappings(query, type_mappings)
    
    def _apply_type_mappings(self, obj: Any, type_mappings: Dict[str, str], current_path: str = "") -> Any:
        """Applique les mappings de type récursivement"""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                # Si on a un type mapping direct pour cette clé
                if key in type_mappings:
                    result[key] = self._cast_parameter_type(value, type_mappings[key])
                else:
                    result[key] = self._apply_type_mappings(value, type_mappings, new_path)
            return result
        elif isinstance(obj, list):
            return [self._apply_type_mappings(item, type_mappings, current_path) for item in obj]
        else:
            return obj

    def _validate_template_structure(self, template_data: Dict[str, Any]) -> List[str]:
        """Valide la structure d'un template"""
        errors = []
        
        required_fields = ["template_name", "description", "target_intention", "parameters"]
        for field in required_fields:
            if field not in template_data:
                errors.append(f"Missing required field: {field}")
        
        # Valider les parameter_mappings si présents
        if "parameter_mappings" in template_data:
            for param_name, param_config in template_data["parameter_mappings"].items():
                if not isinstance(param_config, dict):
                    errors.append(f"Parameter mapping for {param_name} must be a dictionary")
                elif "source" not in param_config:
                    errors.append(f"Parameter mapping for {param_name} missing 'source' field")
        
        return errors

    def _parse_cache_duration(self, duration_str: str) -> timedelta:
        """Parse une durée de cache (ex: '1h', '30m', '5s')"""
        match = re.match(r"(\d+)([smhd])", duration_str.lower())
        if not match:
            return timedelta(hours=1)  # Défaut
            
        amount = int(match.group(1))
        unit = match.group(2)
        
        if unit == 's':
            return timedelta(seconds=amount)
        elif unit == 'm':
            return timedelta(minutes=amount)
        elif unit == 'h':
            return timedelta(hours=amount)
        elif unit == 'd':
            return timedelta(days=amount)
        else:
            return timedelta(hours=1)

    def _is_template_cache_valid(self, template: CompiledTemplate) -> bool:
        """Vérifie si un template en cache est encore valide"""
        return datetime.now() - template.compiled_at < template.cache_duration

    async def _precompile_all_templates(self):
        """Précompile tous les templates au démarrage"""
        for template_file in self.templates_dir.rglob("*.json"):
            result = await self.compile_template(template_file)
            if result.success and result.template:
                # Déterminer la clé depuis le target_intention
                target_intention = result.template.template_data.get("target_intention", "")
                if "." in target_intention:
                    self.compiled_templates[target_intention] = result.template
                else:
                    logger.warning(f"Template {result.template.name} has invalid target_intention: {target_intention}")
            else:
                logger.error(f"Failed to precompile template {template_file}: {result.error_message}")

    async def _compile_template_by_intent(self, intent_group: str, intent_subtype: str) -> Optional[CompiledTemplate]:
        """Compile un template par intention"""
        # Chercher le fichier template correspondant
        group_dir = self.templates_dir / intent_group.lower()
        
        possible_files = [
            group_dir / f"{intent_subtype}.json",
            group_dir / f"by_{intent_subtype}.json",
            group_dir / f"{intent_subtype}_search.json"
        ]
        
        # Log pour debugging
        logger.info(f"Recherche template pour {intent_group}.{intent_subtype}")
        logger.info(f"Répertoire de base: {self.templates_dir}")
        logger.info(f"Répertoire groupe: {group_dir}")
        logger.info(f"Fichiers possibles: {[str(f) for f in possible_files]}")
        
        for template_file in possible_files:
            logger.info(f"Test existence: {template_file} -> {template_file.exists()}")
            if template_file.exists():
                result = await self.compile_template(template_file)
                if result.success and result.template:
                    template_key = f"{intent_group}.{intent_subtype}"
                    self.compiled_templates[template_key] = result.template
                    return result.template
                else:
                    logger.error(f"Failed to compile template {template_file}: {result.error_message}")
        
        logger.warning(f"No template found for intent {intent_group}.{intent_subtype}")
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de cache"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "total_compilations": self.cache_stats["compilations"],
            "templates_in_cache": len(self.compiled_templates)
        }

    async def reload_templates(self):
        """Recharge tous les templates (hot-reload)"""
        logger.info("Reloading all templates...")
        self.compiled_templates.clear()
        await self._precompile_all_templates()
        logger.info(f"Reloaded {len(self.compiled_templates)} templates")

# Instance globale
_template_engine: Optional[TemplateEngine] = None

async def get_template_engine() -> TemplateEngine:
    """Retourne l'instance globale du moteur de templates"""
    global _template_engine
    if _template_engine is None:
        _template_engine = TemplateEngine()
        await _template_engine.initialize()
    return _template_engine