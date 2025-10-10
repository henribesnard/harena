"""
Modèles Pydantic V2 optimisés pour les requêtes conversation service
"""
from pydantic import BaseModel, field_validator, ConfigDict, computed_field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """Types de messages supportés"""
    TEXT = "text"
    VOICE_TO_TEXT = "voice_to_text"  # Pour phases futures
    STRUCTURED = "structured"        # Pour phases futures


class RequestPriority(str, Enum):
    """Priorités de traitement"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ConversationRequest(BaseModel):
    """
    Requête POST /conversation/{user_id} optimisée - Phase 1
    
    Features:
    - Validation stricte avec messages d'erreur clairs
    - Nettoyage automatique des inputs
    - Support métadonnées pour phases futures
    - Validation sécurité contre injections
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "message": "Combien j'ai dépensé chez Amazon ce mois ?",
                "message_type": "text",
                "priority": "normal",
                "client_info": {
                    "version": "1.0.0",
                    "platform": "web"
                }
            }
        }
    )
    
    # Champ principal
    message: str

    # Conversation tracking
    conversation_id: Optional[int] = None

    # Métadonnées optionnelles
    message_type: MessageType = MessageType.TEXT
    priority: RequestPriority = RequestPriority.NORMAL
    client_info: Optional[Dict[str, Any]] = None
    context_hints: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

    # Debugging et tracing (non-production uniquement)
    debug_mode: bool = False
    trace_id: Optional[str] = None
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validation complète du message avec sécurité"""
        if not v or not v.strip():
            raise ValueError("Le message ne peut pas être vide")
        
        # Nettoyage
        cleaned = v.strip()
        
        # Validation longueur
        if len(cleaned) > 1000:
            raise ValueError("Le message ne peut pas dépasser 1000 caractères")
        
        if len(cleaned) < 1:
            raise ValueError("Le message doit contenir au moins 1 caractère")
        
        # Validation sécurité - détection patterns malveillants
        suspicious_patterns = [
            '<script', '</script>', 
            'javascript:', 'data:',
            '<?php', '<?xml',
            'SELECT ', 'DROP ', 'INSERT ', 'UPDATE ', 'DELETE ',
            'UNION SELECT', '--', ';--'
        ]
        
        message_upper = cleaned.upper()
        for pattern in suspicious_patterns:
            if pattern.upper() in message_upper:
                raise ValueError(f"Pattern non autorisé détecté: contenu potentiellement malveillant")
        
        # Validation caractères de contrôle
        control_chars = [c for c in cleaned if ord(c) < 32 and c not in '\n\t\r']
        if control_chars:
            raise ValueError("Caractères de contrôle non autorisés dans le message")
        
        # Validation répétition excessive (protection spam)
        if len(set(cleaned.replace(' ', ''))) < max(3, len(cleaned) / 20):
            raise ValueError("Message avec répétition excessive de caractères")
        
        # Validation URL et email (basique pour éviter spam)
        url_count = cleaned.lower().count('http://') + cleaned.lower().count('https://')
        email_count = cleaned.count('@')
        if url_count > 2 or email_count > 2:
            raise ValueError("Trop d'URLs ou d'emails dans le message")
        
        return cleaned
    
    @field_validator('client_info')
    @classmethod
    def validate_client_info(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validation informations client"""
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError("client_info doit être un objet")
        
        # Limitation taille
        if len(str(v)) > 1000:
            raise ValueError("client_info trop volumineux")
        
        # Validation des clés attendues
        allowed_keys = {
            'version', 'platform', 'user_agent', 'language', 
            'timezone', 'screen_resolution', 'device_type'
        }
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Clé non autorisée dans client_info: {key}")
        
        # Validation version si présente
        if 'version' in v:
            version = str(v['version'])
            if len(version) > 20:
                raise ValueError("Version trop longue")
        
        # Validation platform
        if 'platform' in v:
            valid_platforms = {'web', 'mobile', 'desktop', 'api', 'other'}
            if v['platform'] not in valid_platforms:
                raise ValueError(f"Platform invalide. Doit être: {valid_platforms}")
        
        return v
    
    @field_validator('context_hints')
    @classmethod
    def validate_context_hints(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validation indices contextuels"""
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError("context_hints doit être un objet")
        
        # Limitation taille
        if len(str(v)) > 500:
            raise ValueError("context_hints trop volumineux")
        
        # Clés autorisées pour hints contextuels
        allowed_keys = {
            'previous_intent', 'user_location', 'time_context',
            'conversation_context', 'user_preferences', 'session_info'
        }
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Clé non autorisée dans context_hints: {key}")
        
        return v
    
    @field_validator('preferences')
    @classmethod
    def validate_preferences(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validation préférences utilisateur"""
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError("preferences doit être un objet")
        
        # Limitation taille
        if len(str(v)) > 500:
            raise ValueError("preferences trop volumineux")
        
        # Clés autorisées pour préférences
        allowed_keys = {
            'language', 'response_format', 'verbosity_level',
            'include_alternatives', 'max_processing_time', 'cache_preference'
        }
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Clé non autorisée dans preferences: {key}")
        
        # Validation langue
        if 'language' in v:
            valid_languages = {'fr', 'en', 'es', 'de', 'it'}
            if v['language'] not in valid_languages:
                raise ValueError(f"Langue non supportée. Supportées: {valid_languages}")
        
        # Validation verbosity_level
        if 'verbosity_level' in v:
            try:
                level = int(v['verbosity_level'])
                if not (1 <= level <= 5):
                    raise ValueError("verbosity_level doit être entre 1 et 5")
            except (ValueError, TypeError):
                raise ValueError("verbosity_level doit être un entier")
        
        return v
    
    @field_validator('trace_id')
    @classmethod
    def validate_trace_id(cls, v: Optional[str]) -> Optional[str]:
        """Validation trace ID pour debugging"""
        if v is None:
            return v
        
        cleaned = v.strip()
        if not cleaned:
            return None
        
        # Format trace ID (UUID-like ou custom)
        if len(cleaned) > 64:
            raise ValueError("trace_id trop long (max 64 caractères)")
        
        # Seulement alphanumériques, tirets et underscores
        if not all(c.isalnum() or c in '-_' for c in cleaned):
            raise ValueError("trace_id contient des caractères non autorisés")
        
        return cleaned
    
    @computed_field
    @property
    def message_length(self) -> int:
        """Longueur du message pour métrics"""
        return len(self.message)
    
    @computed_field
    @property
    def message_word_count(self) -> int:
        """Nombre de mots dans le message"""
        return len(self.message.split())
    
    @computed_field
    @property
    def has_client_info(self) -> bool:
        """Indique si des infos client sont fournies"""
        return self.client_info is not None and bool(self.client_info)
    
    @computed_field
    @property
    def estimated_complexity(self) -> str:
        """Estimation de la complexité du message"""
        word_count = self.message_word_count
        
        if word_count <= 3:
            return "simple"
        elif word_count <= 10:
            return "medium"
        else:
            return "complex"
    
    def get_processing_hints(self) -> Dict[str, Any]:
        """Récupère les indices pour optimiser le traitement"""
        hints = {
            "message_length": self.message_length,
            "word_count": self.message_word_count,
            "complexity": self.estimated_complexity,
            "priority": self.priority.value,
            "has_context": bool(self.context_hints),
            "debug_mode": self.debug_mode
        }
        
        # Ajout hints depuis client_info
        if self.client_info:
            hints["platform"] = self.client_info.get("platform", "unknown")
            hints["client_version"] = self.client_info.get("version", "unknown")
        
        # Ajout préférences de traitement
        if self.preferences:
            hints["max_processing_time"] = self.preferences.get("max_processing_time")
            hints["cache_preference"] = self.preferences.get("cache_preference", "enabled")
            hints["include_alternatives"] = self.preferences.get("include_alternatives", True)
        
        return hints
    
    def sanitize_for_logging(self) -> Dict[str, Any]:
        """Version sécurisée pour logging (sans données sensibles)"""
        return {
            "message_length": self.message_length,
            "word_count": self.message_word_count,
            "message_preview": self.message[:20] + "..." if len(self.message) > 20 else self.message,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "complexity": self.estimated_complexity,
            "has_client_info": self.has_client_info,
            "debug_mode": self.debug_mode,
            "trace_id": self.trace_id
        }


class BatchConversationRequest(BaseModel):
    """
    Requête pour traitement en lot (Phase 2+)
    Préparé pour les futures phases mais non utilisé en Phase 1
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    messages: List[ConversationRequest]
    batch_id: Optional[str] = None
    processing_mode: str = "sequential"  # sequential, parallel, adaptive
    max_concurrent: int = 3
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: List[ConversationRequest]) -> List[ConversationRequest]:
        if not v:
            raise ValueError("La liste de messages ne peut pas être vide")
        
        if len(v) > 10:  # Limitation pour éviter surcharge
            raise ValueError("Maximum 10 messages par batch")
        
        return v
    
    @field_validator('max_concurrent')
    @classmethod
    def validate_max_concurrent(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("max_concurrent doit être entre 1 et 5")
        return v
    
    @computed_field
    @property
    def total_messages(self) -> int:
        return len(self.messages)


class HealthCheckRequest(BaseModel):
    """Requête health check avec options"""
    model_config = ConfigDict(validate_assignment=True)
    
    include_details: bool = False
    include_metrics: bool = False
    check_dependencies: bool = True
    timeout_seconds: int = 30
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1 or v > 300:  # 5 minutes max
            raise ValueError("timeout_seconds doit être entre 1 et 300")
        return v


class MetricsRequest(BaseModel):
    """Requête métriques avec filtres"""
    model_config = ConfigDict(validate_assignment=True)
    
    include_histograms: bool = True
    include_rates: bool = True
    include_metadata: bool = False
    time_window_minutes: int = 60
    format_type: str = "json"  # json, prometheus
    
    @field_validator('time_window_minutes')
    @classmethod
    def validate_time_window(cls, v: int) -> int:
        if v < 1 or v > 1440:  # 24 heures max
            raise ValueError("time_window_minutes doit être entre 1 et 1440")
        return v
    
    @field_validator('format_type')
    @classmethod
    def validate_format_type(cls, v: str) -> str:
        valid_formats = {'json', 'prometheus', 'csv'}
        if v not in valid_formats:
            raise ValueError(f"format_type doit être: {valid_formats}")
        return v


# Factory functions pour créer des requêtes standardisées
def create_simple_request(message: str) -> ConversationRequest:
    """Crée une requête simple avec validation"""
    return ConversationRequest(message=message)


def create_detailed_request(
    message: str,
    priority: RequestPriority = RequestPriority.NORMAL,
    client_info: Optional[Dict[str, Any]] = None,
    context_hints: Optional[Dict[str, Any]] = None
) -> ConversationRequest:
    """Crée une requête détaillée"""
    return ConversationRequest(
        message=message,
        priority=priority,
        client_info=client_info,
        context_hints=context_hints
    )


def create_debug_request(
    message: str,
    trace_id: str,
    debug_mode: bool = True
) -> ConversationRequest:
    """Crée une requête avec debug activé"""
    return ConversationRequest(
        message=message,
        debug_mode=debug_mode,
        trace_id=trace_id,
        priority=RequestPriority.HIGH
    )


# Validation utilities
class MessageValidator:
    """Validateur de messages avec règles avancées"""
    
    @staticmethod
    def is_financial_related(message: str) -> bool:
        """Détecte si un message semble financier"""
        financial_keywords = {
            # Français
            'euro', 'euros', '€', 'argent', 'dépense', 'dépenses', 'achat', 'achats',
            'virement', 'transaction', 'compte', 'solde', 'carte', 'banque',
            'amazon', 'carrefour', 'restaurant', 'essence', 'courses',
            # Anglais (support basique)
            'dollar', 'money', 'spend', 'purchase', 'bank', 'account'
        }
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in financial_keywords)
    
    @staticmethod
    def estimate_processing_complexity(message: str) -> str:
        """Estime la complexité de traitement"""
        factors = {
            'length': len(message),
            'words': len(message.split()),
            'questions': message.count('?'),
            'numbers': sum(c.isdigit() for c in message),
            'special_chars': sum(not c.isalnum() and c != ' ' for c in message)
        }
        
        complexity_score = 0
        
        # Facteurs de complexité
        if factors['length'] > 100:
            complexity_score += 2
        if factors['words'] > 15:
            complexity_score += 2
        if factors['questions'] > 1:
            complexity_score += 1
        if factors['numbers'] > 5:
            complexity_score += 1
        if factors['special_chars'] > 10:
            complexity_score += 1
        
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def detect_language(message: str) -> str:
        """Détection basique de langue"""
        french_indicators = [
            'le', 'la', 'les', 'de', 'du', 'des', 'mon', 'ma', 'mes',
            'combien', 'quand', 'comment', 'pourquoi', 'où',
            'j\'ai', 'je', 'tu', 'il', 'elle'
        ]
        
        english_indicators = [
            'the', 'and', 'or', 'but', 'my', 'your', 'his', 'her',
            'how', 'what', 'when', 'where', 'why',
            'i', 'you', 'he', 'she'
        ]
        
        message_words = message.lower().split()
        
        french_score = sum(1 for word in message_words if word in french_indicators)
        english_score = sum(1 for word in message_words if word in english_indicators)
        
        if french_score > english_score:
            return 'fr'
        elif english_score > french_score:
            return 'en'
        else:
            return 'unknown'


# Constantes pour validation
MAX_MESSAGE_LENGTH = 1000
MAX_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 30
SUPPORTED_LANGUAGES = {'fr', 'en', 'es', 'de', 'it'}
SUPPORTED_PLATFORMS = {'web', 'mobile', 'desktop', 'api', 'other'}