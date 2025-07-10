"""
Validateur de sécurité pour le Search Service.

Ce module fournit la validation de sécurité complète incluant
la détection d'injections, XSS, patterns malicieux et validations d'accès.
"""

import re
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime, timedelta

from .base import (
    BaseValidator, ValidationResult, ValidationLevel,
    SecurityValidationError, DANGEROUS_PATTERNS
)
from .config import (
    VALIDATION_LIMITS, VALIDATION_CONFIG, ERROR_MESSAGES
)

logger = logging.getLogger(__name__)

class SecurityValidator(BaseValidator):
    """
    Validateur spécialisé pour la sécurité.
    
    Détecte et prévient les attaques par injection, XSS, CSRF,
    et autres vulnérabilités de sécurité dans les requêtes de recherche.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        super().__init__(validation_level)
        self.config = VALIDATION_CONFIG[validation_level.value]
        
        # Patterns de sécurité étendus
        self.security_patterns = DANGEROUS_PATTERNS + [
            r'(?i)(script|javascript|vbscript|onload|onerror)',
            r'(?i)(select.*from|insert.*into|update.*set|delete.*from)',
            r'(?i)(union.*select|or.*1.*=.*1|and.*1.*=.*1)',
            r'(?i)(exec|execute|sp_|xp_)',
            r'(?i)(<.*>|&lt;.*&gt;)',
            r'(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()',
            r'(?i)(document\.|window\.|location\.)',
            r'(?i)(\.\.\/|\.\.\\)',
            r'(?i)(cmd|command|shell|bash|sh\s)',
            r'(?i)(wget|curl|fetch)',
            r'(?i)(\${.*}|<%.*%>)',
            r'(?i)(eval\s*\(|setTimeout\s*\(|setInterval\s*\()',
        ]
        
        # Whitelist de caractères autorisés par contexte
        self.allowed_chars = {
            'query_text': r'^[a-zA-Z0-9\s\-_.,!?àâäéèêëïîôöùûüÿç]*$',
            'merchant_name': r'^[a-zA-Z0-9\s\-_.,&\']*$',
            'field_name': r'^[a-zA-Z_][a-zA-Z0-9_]*$',
            'safe_string': r'^[a-zA-Z0-9\s\-_.]*$'
        }
    
    def validate(self, data: Any, context: str = "general", **kwargs) -> ValidationResult:
        """
        Valide la sécurité de données.
        
        Args:
            data: Données à valider
            context: Contexte de validation (query, filter, parameter)
            **kwargs: Options supplémentaires
            
        Returns:
            ValidationResult avec détails de sécurité
        """
        start_time = time.time()
        result = self._create_result()
        
        try:
            # Validation selon le type de données
            if isinstance(data, str):
                self._validate_string_security(data, context, result)
            elif isinstance(data, dict):
                self._validate_dict_security(data, context, result)
            elif isinstance(data, list):
                self._validate_list_security(data, context, result)
            else:
                # Types primitifs considérés comme sûrs
                pass
            
            # Validation globale de sécurité
            self._validate_global_security(data, result)
            
            # Génération de données sanitisées
            if result.is_valid or self.validation_level != ValidationLevel.PARANOID:
                result.sanitized_data = self._sanitize_data(data, context)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation de sécurité: {e}")
            result.add_error(f"Erreur interne de validation de sécurité: {str(e)}")
        
        result.validation_time_ms = self._measure_validation_time(start_time)
        return result
    
    def validate_query_injection(self, query: str) -> ValidationResult:
        """
        Valide spécifiquement contre les injections dans les requêtes.
        
        Args:
            query: Requête à valider
            
        Returns:
            ValidationResult
        """
        result = self._create_result()
        
        if not isinstance(query, str):
            result.add_error("La requête doit être une chaîne")
            return result
        
        # Détection d'injection SQL
        sql_patterns = [
            r'(?i)(union.*select)',
            r'(?i)(or.*1.*=.*1)',
            r'(?i)(and.*1.*=.*1)',
            r'(?i)(select.*from)',
            r'(?i)(insert.*into)',
            r'(?i)(update.*set)',
            r'(?i)(delete.*from)',
            r'(?i)(drop.*table)',
            r'(?i)(alter.*table)',
            r'(?i)(exec|execute)',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, query):
                result.add_error("Injection SQL potentielle détectée")
                result.add_security_flag(f"SQL injection pattern: {pattern}")
                break
        
        # Détection d'injection NoSQL/Elasticsearch
        nosql_patterns = [
            r'(?i)(\$where|\$regex|\$ne)',
            r'(?i)(script.*source)',
            r'(?i)(painless.*script)',
        ]
        
        for pattern in nosql_patterns:
            if re.search(pattern, query):
                result.add_error("Injection NoSQL potentielle détectée")
                result.add_security_flag(f"NoSQL injection pattern: {pattern}")
                break
        
        return result
    
    def validate_xss_prevention(self, text: str) -> ValidationResult:
        """
        Valide contre les attaques XSS.
        
        Args:
            text: Texte à valider
            
        Returns:
            ValidationResult
        """
        result = self._create_result()
        
        if not isinstance(text, str):
            return result
        
        # Patterns XSS courants
        xss_patterns = [
            r'(?i)<script[^>]*>.*?</script>',
            r'(?i)<.*?javascript:.*?>',
            r'(?i)<.*?on\w+\s*=.*?>',
            r'(?i)<.*?src\s*=.*?>',
            r'(?i)(alert|confirm|prompt)\s*\(',
            r'(?i)document\.(cookie|domain|location)',
            r'(?i)window\.(location|open)',
            r'(?i)<iframe.*?>',
            r'(?i)<object.*?>',
            r'(?i)<embed.*?>',
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, text):
                result.add_error("Contenu XSS potentiel détecté")
                result.add_security_flag(f"XSS pattern: {pattern}")
                break
        
        # Vérification des entités HTML suspectes
        suspicious_entities = [
            '&lt;script', '&gt;', '&#x', '&#[0-9]',
            '%3Cscript', '%3E', '%22', '%27'
        ]
        
        for entity in suspicious_entities:
            if entity in text.lower():
                result.add_warning(f"Entité HTML suspecte: {entity}")
                result.add_security_flag(f"Suspicious HTML entity: {entity}")
        
        return result
    
    def validate_access_control(self, user_id: int, requested_resources: List[str]) -> ValidationResult:
        """
        Valide le contrôle d'accès aux ressources.
        
        Args:
            user_id: ID de l'utilisateur
            requested_resources: Ressources demandées
            
        Returns:
            ValidationResult
        """
        result = self._create_result()
        
        # Validation de base de l'utilisateur
        if not isinstance(user_id, int) or user_id <= 0:
            result.add_error("ID utilisateur invalide")
            return result
        
        # Validation des ressources demandées
        allowed_resources = {
            'transactions', 'accounts', 'categories', 'merchants',
            'search_history', 'preferences'
        }
        
        for resource in requested_resources:
            if resource not in allowed_resources:
                result.add_warning(f"Ressource non standard demandée: {resource}")
                result.add_security_flag(f"Non-standard resource: {resource}")
        
        # Vérification des tentatives d'accès à des données sensibles
        sensitive_resources = {'admin', 'system', 'config', 'users'}
        for resource in requested_resources:
            if any(sensitive in resource.lower() for sensitive in sensitive_resources):
                result.add_error(f"Tentative d'accès à une ressource sensible: {resource}")
                result.add_security_flag(f"Sensitive resource access: {resource}")
        
        return result
    
    def _validate_string_security(self, text: str, context: str, result: ValidationResult):
        """Valide la sécurité d'une chaîne."""
        # Validation contre les patterns dangereux
        for pattern in self.security_patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    if self.validation_level == ValidationLevel.PARANOID:
                        result.add_error(f"Pattern dangereux détecté: {pattern}")
                    else:
                        result.add_warning(f"Pattern suspect détecté")
                    result.add_security_flag(f"Dangerous pattern: {pattern}")
            except re.error:
                continue
        
        # Validation contre injection
        injection_result = self.validate_query_injection(text)
        result.errors.extend(injection_result.errors)
        result.security_flags.extend(injection_result.security_flags)
        
        # Validation contre XSS
        xss_result = self.validate_xss_prevention(text)
        result.errors.extend(xss_result.errors)
        result.warnings.extend(xss_result.warnings)
        result.security_flags.extend(xss_result.security_flags)
        
        # Validation de la longueur (prévention DoS)
        max_length = VALIDATION_LIMITS.get("max_query_length", 1000)
        if len(text) > max_length:
            result.add_error(f"Texte trop long (max: {max_length})")
            result.add_security_flag("Potential DoS: excessive length")
        
        # Validation whitelist selon le contexte
        if context in self.allowed_chars:
            pattern = self.allowed_chars[context]
            if not re.match(pattern, text):
                if self.validation_level == ValidationLevel.PARANOID:
                    result.add_error(f"Caractères non autorisés pour le contexte {context}")
                else:
                    result.add_warning(f"Caractères suspects pour le contexte {context}")
    
    def _validate_dict_security(self, data: Dict[str, Any], context: str, result: ValidationResult):
        """Valide la sécurité d'un dictionnaire."""
        # Limitation du nombre de clés (prévention DoS)
        if len(data) > 100:
            result.add_error("Trop de clés dans le dictionnaire (max: 100)")
            result.add_security_flag("Potential DoS: excessive keys")
        
        # Validation récursive
        for key, value in data.items():
            # Validation de la clé
            if isinstance(key, str):
                key_result = self.validate(key, f"{context}_key")
                result.errors.extend(key_result.errors)
                result.warnings.extend(key_result.warnings)
                result.security_flags.extend(key_result.security_flags)
            
            # Validation de la valeur
            value_result = self.validate(value, f"{context}_value")
            result.errors.extend(value_result.errors)
            result.warnings.extend(value_result.warnings)
            result.security_flags.extend(value_result.security_flags)
    
    def _validate_list_security(self, data: List[Any], context: str, result: ValidationResult):
        """Valide la sécurité d'une liste."""
        # Limitation de la taille (prévention DoS)
        if len(data) > VALIDATION_LIMITS.get("max_filter_values", 100):
            result.add_error(f"Liste trop longue (max: {VALIDATION_LIMITS.get('max_filter_values', 100)})")
            result.add_security_flag("Potential DoS: excessive list size")
        
        # Validation récursive des éléments
        for i, item in enumerate(data):
            item_result = self.validate(item, f"{context}_item")
            if item_result.errors:
                result.errors.extend([f"Index {i}: {error}" for error in item_result.errors])
            result.warnings.extend(item_result.warnings)
            result.security_flags.extend(item_result.security_flags)
    
    def _validate_global_security(self, data: Any, result: ValidationResult):
        """Validation de sécurité globale."""
        # Conversion en string pour analyse globale
        data_str = str(data)
        
        # Détection de tentatives d'évasion
        evasion_patterns = [
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'\\u[0-9a-fA-F]{4}',  # Unicode encoding
            r'%[0-9a-fA-F]{2}',    # URL encoding
            r'&[#\w]+;',           # HTML entities
        ]
        
        for pattern in evasion_patterns:
            if re.search(pattern, data_str):
                result.add_warning("Tentative d'encodage/évasion détectée")
                result.add_security_flag(f"Encoding detected: {pattern}")
        
        # Détection de payloads de test de sécurité
        security_test_patterns = [
            r'(?i)(test|poc|exploit)',
            r'(?i)(nmap|sqlmap|burp)',
            r'(?i)(payload|injection)',
            r'(?i)(xxe|ssrf|lfi|rfi)',
        ]
        
        for pattern in security_test_patterns:
            if re.search(pattern, data_str):
                result.add_security_flag(f"Security test pattern: {pattern}")
    
    def _sanitize_data(self, data: Any, context: str) -> Any:
        """Sanitise les données selon le contexte."""
        if isinstance(data, str):
            return self._sanitize_string_security(data, context)
        elif isinstance(data, dict):
            return {k: self._sanitize_data(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item, context) for item in data]
        else:
            return data
    
    def _sanitize_string_security(self, text: str, context: str) -> str:
        """Sanitise une chaîne pour la sécurité."""
        sanitized = text
        
        # Suppression des caractères de contrôle
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # Suppression/échappement selon le contexte
        if context == "query_text":
            # Suppression des caractères dangereux pour les requêtes
            sanitized = re.sub(r'[<>"\';\\]', '', sanitized)
            # Normalisation des espaces
            sanitized = re.sub(r'\s+', ' ', sanitized)
        elif context == "merchant_name":
            # Autoriser plus de caractères pour les noms de marchands
            sanitized = re.sub(r'[<>"\'\\]', '', sanitized)
        elif context == "field_name":
            # Très restrictif pour les noms de champs
            sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
        else:
            # Sanitisation générale
            sanitized = re.sub(r'[<>"\';\\]', '', sanitized)
        
        # Limitation de longueur
        max_length = VALIDATION_LIMITS.get("max_query_length", 1000)
        sanitized = sanitized[:max_length]
        
        return sanitized.strip()

# ==================== VALIDATIONS SPÉCIALISÉES ====================

class RateLimitValidator:
    """Validateur pour les limites de taux."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.request_history: Dict[str, List[datetime]] = {}
    
    def validate_rate_limit(self, user_id: int, endpoint: str = "search") -> ValidationResult:
        """
        Valide les limites de taux pour un utilisateur.
        
        Args:
            user_id: ID utilisateur
            endpoint: Endpoint appelé
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        now = datetime.utcnow()
        key = f"{user_id}:{endpoint}"
        
        # Initialisation de l'historique
        if key not in self.request_history:
            self.request_history[key] = []
        
        # Nettoyage des anciennes requêtes (plus d'1 minute)
        cutoff = now - timedelta(minutes=1)
        self.request_history[key] = [
            req_time for req_time in self.request_history[key] 
            if req_time > cutoff
        ]
        
        # Vérification de la limite
        if len(self.request_history[key]) >= self.max_requests:
            result.is_valid = False
            result.add_error("Limite de taux dépassée")
            result.add_security_flag(f"Rate limit exceeded for user {user_id}")
        else:
            # Ajout de la requête actuelle
            self.request_history[key].append(now)
        
        return result

class InputSizeValidator:
    """Validateur pour les tailles d'input."""
    
    def __init__(self):
        self.max_sizes = {
            'query': 1000,
            'filters': 5000,
            'parameters': 2000,
            'total_request': 10000
        }
    
    def validate_input_size(self, data: Any, data_type: str = "query") -> ValidationResult:
        """
        Valide la taille des données d'entrée.
        
        Args:
            data: Données à valider
            data_type: Type de données
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        # Calcul de la taille
        if isinstance(data, str):
            size = len(data)
        else:
            size = len(str(data))
        
        max_size = self.max_sizes.get(data_type, 1000)
        
        if size > max_size:
            result.is_valid = False
            result.add_error(f"Taille d'input trop importante pour {data_type} (max: {max_size})")
            result.add_security_flag(f"Input size violation: {size} > {max_size}")
        elif size > max_size * 0.8:
            result.add_warning(f"Taille d'input importante pour {data_type}")
        
        return result

# ==================== FONCTIONS UTILITAIRES ====================

def validate_security(data: Any, context: str = "general", 
                     validation_level: ValidationLevel = ValidationLevel.STRICT) -> ValidationResult:
    """
    Fonction utilitaire pour validation de sécurité.
    
    Args:
        data: Données à valider
        context: Contexte de validation
        validation_level: Niveau de validation
        
    Returns:
        ValidationResult
    """
    validator = SecurityValidator(validation_level)
    return validator.validate(data, context)

def sanitize_for_security(data: Any, context: str = "general") -> Any:
    """
    Sanitise les données pour la sécurité.
    
    Args:
        data: Données à sanitiser
        context: Contexte de sanitisation
        
    Returns:
        Données sanitisées
    """
    validator = SecurityValidator()
    result = validator.validate(data, context)
    return result.sanitized_data if result.sanitized_data is not None else data

def check_injection_patterns(text: str) -> Tuple[bool, List[str]]:
    """
    Vérifie les patterns d'injection dans un texte.
    
    Args:
        text: Texte à vérifier
        
    Returns:
        Tuple (has_injection, patterns_found)
    """
    validator = SecurityValidator()
    result = validator.validate_query_injection(text)
    
    patterns_found = [
        flag for flag in result.security_flags 
        if "injection pattern" in flag
    ]
    
    return not result.is_valid, patterns_found

def generate_security_hash(data: Any) -> str:
    """
    Génère un hash de sécurité pour les données.
    
    Args:
        data: Données à hasher
        
    Returns:
        Hash de sécurité
    """
    data_str = str(data) if not isinstance(data, str) else data
    return hashlib.sha256(data_str.encode()).hexdigest()

def validate_user_permissions(user_id: int, action: str, resource: str) -> ValidationResult:
    """
    Valide les permissions utilisateur.
    
    Args:
        user_id: ID utilisateur
        action: Action demandée (read, write, delete)
        resource: Ressource ciblée
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    # Validation de base
    if not isinstance(user_id, int) or user_id <= 0:
        result.is_valid = False
        result.add_error("ID utilisateur invalide")
        return result
    
    # Validation de l'action
    allowed_actions = {"read", "write", "delete", "search"}
    if action not in allowed_actions:
        result.add_warning(f"Action non standard: {action}")
    
    # Validation de la ressource
    allowed_resources = {
        "transactions", "accounts", "categories", "merchants", 
        "search_history", "preferences"
    }
    if resource not in allowed_resources:
        result.add_warning(f"Ressource non standard: {resource}")
    
    # Vérification des actions sensibles
    if action in ["write", "delete"] and resource in ["accounts", "transactions"]:
        result.add_security_flag(f"Sensitive action: {action} on {resource}")
    
    return result

def create_security_context(user_id: int, ip_address: str = None, 
                          user_agent: str = None) -> Dict[str, Any]:
    """
    Crée un contexte de sécurité pour une requête.
    
    Args:
        user_id: ID utilisateur
        ip_address: Adresse IP (optionnelle)
        user_agent: User agent (optionnel)
        
    Returns:
        Contexte de sécurité
    """
    context = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "security_level": "standard",
        "flags": []
    }
    
    if ip_address:
        context["ip_address"] = ip_address
        # Vérification d'IP suspecte (exemple basique)
        if ip_address.startswith(("127.", "192.168.", "10.")):
            context["flags"].append("internal_ip")
    
    if user_agent:
        context["user_agent"] = user_agent
        # Détection de bots/crawlers
        bot_indicators = ["bot", "crawler", "spider", "scraper"]
        if any(indicator in user_agent.lower() for indicator in bot_indicators):
            context["flags"].append("bot_detected")
    
    return context

# ==================== EXPORTS ====================

__all__ = [
    'SecurityValidator',
    'RateLimitValidator', 
    'InputSizeValidator',
    'validate_security',
    'sanitize_for_security',
    'check_injection_patterns',
    'generate_security_hash',
    'validate_user_permissions',
    'create_security_context'
]