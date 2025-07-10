"""
Exceptions pour les Templates - Search Service

Définit toutes les exceptions spécialisées pour la gestion d'erreurs
dans les templates de requêtes Elasticsearch.
"""

from typing import Optional, Dict, Any


class QueryTemplateError(Exception):
    """Erreur de base pour tous les templates de requête."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TemplateNotFoundError(QueryTemplateError):
    """Template de requête non trouvé."""
    
    def __init__(self, template_name: str, available_templates: Optional[list] = None):
        message = f"Template '{template_name}' non trouvé"
        if available_templates:
            message += f". Templates disponibles: {', '.join(available_templates)}"
        
        super().__init__(message, {
            "template_name": template_name,
            "available_templates": available_templates
        })
        self.template_name = template_name


class TemplateValidationError(QueryTemplateError):
    """Erreur de validation de template."""
    
    def __init__(self, message: str, template: Optional[Dict[str, Any]] = None, path: Optional[str] = None):
        super().__init__(message, {
            "template": template,
            "validation_path": path
        })
        self.template = template
        self.path = path


class TemplateRenderError(QueryTemplateError):
    """Erreur de rendu de template."""
    
    def __init__(self, message: str, template: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(message, {
            "template": template,
            "params": params
        })
        self.template = template
        self.params = params


class InvalidParametersError(QueryTemplateError):
    """Paramètres de template invalides."""
    
    def __init__(self, missing_params: list = None, invalid_params: list = None, provided_params: Optional[Dict[str, Any]] = None):
        messages = []
        if missing_params:
            messages.append(f"Paramètres manquants: {', '.join(missing_params)}")
        if invalid_params:
            messages.append(f"Paramètres invalides: {', '.join(invalid_params)}")
        
        message = "; ".join(messages) if messages else "Paramètres invalides"
        
        super().__init__(message, {
            "missing_params": missing_params or [],
            "invalid_params": invalid_params or [],
            "provided_params": provided_params
        })
        self.missing_params = missing_params or []
        self.invalid_params = invalid_params or []


class QueryBuilderError(QueryTemplateError):
    """Erreur dans la construction de requête."""
    
    def __init__(self, message: str, builder_state: Optional[Dict[str, Any]] = None):
        super().__init__(message, {
            "builder_state": builder_state
        })
        self.builder_state = builder_state


class TemplateConfigurationError(QueryTemplateError):
    """Erreur de configuration des templates."""
    
    def __init__(self, message: str, config_section: Optional[str] = None):
        super().__init__(message, {
            "config_section": config_section
        })
        self.config_section = config_section


class CacheError(QueryTemplateError):
    """Erreur liée au cache des templates."""
    
    def __init__(self, message: str, cache_key: Optional[str] = None):
        super().__init__(message, {
            "cache_key": cache_key
        })
        self.cache_key = cache_key


class PerformanceError(QueryTemplateError):
    """Erreur liée aux performances (trop de clauses, etc.)."""
    
    def __init__(self, message: str, performance_metrics: Optional[Dict[str, Any]] = None):
        super().__init__(message, {
            "performance_metrics": performance_metrics
        })
        self.performance_metrics = performance_metrics


# ==================== FONCTIONS UTILITAIRES ====================

def handle_template_error(error: Exception, context: str = "") -> QueryTemplateError:
    """
    Convertit une exception générique en exception de template spécialisée.
    """
    if isinstance(error, QueryTemplateError):
        return error
    
    error_message = str(error)
    if context:
        error_message = f"{context}: {error_message}"
    
    # Déterminer le type d'erreur spécialisée selon le message
    if "not found" in error_message.lower() or "trouvé" in error_message.lower():
        return TemplateNotFoundError(error_message)
    elif "validation" in error_message.lower():
        return TemplateValidationError(error_message)
    elif "parameter" in error_message.lower() or "paramètre" in error_message.lower():
        return InvalidParametersError()
    elif "render" in error_message.lower():
        return TemplateRenderError(error_message)
    else:
        return QueryTemplateError(error_message)


def create_validation_error(message: str, template: Dict[str, Any], path: str = "") -> TemplateValidationError:
    """Helper pour créer une erreur de validation avec contexte."""
    return TemplateValidationError(message, template=template, path=path)


def create_parameter_error(required: list = None, missing: list = None, invalid: list = None) -> InvalidParametersError:
    """Helper pour créer une erreur de paramètres avec détails."""
    return InvalidParametersError(missing_params=missing, invalid_params=invalid)