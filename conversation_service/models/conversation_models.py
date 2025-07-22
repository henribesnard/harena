"""
💬 Modèles Pydantic pour conversation service

Modèles de données avec évitement des imports circulaires,
validation robuste et sérialization optimisée.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import time

# ✅ Pas d'imports de modules conversation_service pour éviter circuits
# Tous les modèles sont définis dans ce module uniquement

# ==========================================
# ENUMS ET TYPES DE BASE
# ==========================================

class FinancialIntent(str, Enum):
    """Types d'intentions financières supportées"""
    BALANCE_CHECK = "BALANCE_CHECK"
    EXPENSE_ANALYSIS = "EXPENSE_ANALYSIS"
    TRANSFER = "TRANSFER"
    BILL_PAYMENT = "BILL_PAYMENT"
    INVESTMENT_QUERY = "INVESTMENT_QUERY"
    LOAN_INQUIRY = "LOAN_INQUIRY"
    CARD_MANAGEMENT = "CARD_MANAGEMENT"
    TRANSACTION_HISTORY = "TRANSACTION_HISTORY"
    BUDGET_PLANNING = "BUDGET_PLANNING"
    SAVINGS_GOAL = "SAVINGS_GOAL"
    ACCOUNT_MANAGEMENT = "ACCOUNT_MANAGEMENT"
    FINANCIAL_ADVICE = "FINANCIAL_ADVICE"
    GREETING = "GREETING"
    HELP = "HELP"
    UNKNOWN = "UNKNOWN"

class ProcessingLevel(str, Enum):
    """Niveaux de traitement du pipeline"""
    L0_PATTERN = "L0_PATTERN"
    L1_LIGHTWEIGHT = "L1_LIGHTWEIGHT" 
    L2_LLM = "L2_LLM"
    ERROR_FALLBACK = "ERROR_FALLBACK"
    ERROR_TIMEOUT = "ERROR_TIMEOUT"
    ERROR_VALIDATION = "ERROR_VALIDATION"

class ConfidenceLevel(str, Enum):
    """Niveaux de confiance"""
    VERY_HIGH = "VERY_HIGH"    # > 0.9
    HIGH = "HIGH"              # 0.8 - 0.9
    MEDIUM = "MEDIUM"          # 0.6 - 0.8
    LOW = "LOW"                # 0.4 - 0.6
    VERY_LOW = "VERY_LOW"      # < 0.4

# ==========================================
# MODÈLES DE MÉTADONNÉES
# ==========================================

class ProcessingMetadata(BaseModel):
    """Métadonnées de traitement de la requête"""
    request_id: str
    level_used: str
    processing_time_ms: float
    cache_hit: bool = False
    engine_latency_ms: Optional[float] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "level_used": "L1_LIGHTWEIGHT",
                "processing_time_ms": 45.2,
                "cache_hit": False,
                "engine_latency_ms": 42.1,
                "timestamp": 1640995200
            }
        }

class ConfidenceScore(BaseModel):
    """Score de confiance avec niveau"""
    score: float = Field(..., ge=0.0, le=1.0)
    level: ConfidenceLevel
    
    @validator('level', always=True)
    def determine_confidence_level(cls, v, values):
        """Détermine automatiquement le niveau selon le score"""
        score = values.get('score', 0.0)
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.85,
                "level": "HIGH"
            }
        }

# ==========================================
# MODÈLES DE REQUÊTE ET RÉPONSE
# ==========================================

class ChatRequest(BaseModel):
    """Requête de chat utilisateur"""
    message: str = Field(..., min_length=1, max_length=2000, description="Message utilisateur")
    user_id: int = Field(..., gt=0, description="ID utilisateur")
    conversation_id: Optional[str] = Field(None, max_length=100, description="ID conversation optionnel")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contexte additionnel")
    
    @validator('message')
    def validate_message(cls, v):
        """Validation et nettoyage du message"""
        if not v or not v.strip():
            raise ValueError("Message ne peut pas être vide")
        return v.strip()
    
    @validator('context')
    def validate_context(cls, v):
        """Validation du contexte"""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("Context doit être un dictionnaire")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Quel est mon solde ?",
                "user_id": 123,
                "conversation_id": "conv_456",
                "context": {"channel": "web"}
            }
        }

class ChatResponse(BaseModel):
    """Réponse du service conversation"""
    request_id: str = Field(..., description="ID unique de la requête")
    intent: str = Field(..., description="Intention détectée")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score de confiance")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Entités extraites")
    message: Optional[str] = Field(None, description="Message de réponse")
    suggested_actions: Optional[List[str]] = Field(None, description="Actions suggérées")
    processing_metadata: ProcessingMetadata = Field(..., description="Métadonnées de traitement")
    success: bool = Field(True, description="Succès du traitement")
    error: Optional[str] = Field(None, description="Message d'erreur si échec")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "intent": "BALANCE_CHECK",
                "confidence": 0.95,
                "entities": {"account_type": "checking"},
                "message": "Je vais vérifier votre solde",
                "suggested_actions": ["show_balance", "show_transactions"],
                "success": True,
                "processing_metadata": {
                    "request_id": "req_1234567890",
                    "level_used": "L1_LIGHTWEIGHT", 
                    "processing_time_ms": 45.2,
                    "cache_hit": False,
                    "timestamp": 1640995200
                }
            }
        }

# ==========================================
# MODÈLES D'ENTITÉS ET CONTEXTE
# ==========================================

class FinancialEntity(BaseModel):
    """Entité financière extraite"""
    type: str = Field(..., description="Type d'entité (amount, date, account, etc.)")
    value: Union[str, int, float] = Field(..., description="Valeur de l'entité")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance extraction")
    position: Optional[Dict[str, int]] = Field(None, description="Position dans le texte")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "amount",
                "value": 150.50,
                "confidence": 0.9,
                "position": {"start": 10, "end": 16}
            }
        }

class ConversationContext(BaseModel):
    """Contexte de conversation multi-tours"""
    conversation_id: str
    user_id: int
    turn_number: int = Field(default=1, ge=1)
    previous_intents: List[str] = Field(default_factory=list)
    session_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "user_id": 456,
                "turn_number": 3,
                "previous_intents": ["GREETING", "BALANCE_CHECK"],
                "session_data": {"preferred_language": "fr"},
                "created_at": 1640995200,
                "updated_at": 1640995260
            }
        }

# ==========================================
# MODÈLES DE MÉTRIQUES ET MONITORING
# ==========================================

class PerformanceMetrics(BaseModel):
    """Métriques de performance du service"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    level_distribution: Dict[str, int] = Field(default_factory=dict)
    confidence_distribution: Dict[str, int] = Field(default_factory=dict)
    error_distribution: Dict[str, int] = Field(default_factory=dict)
    cache_hit_rate: float = 0.0
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    
    @property
    def success_rate(self) -> float:
        """Calcul du taux de succès"""
        total = self.total_requests
        return (self.successful_requests / total) if total > 0 else 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_requests": 1000,
                "successful_requests": 950,
                "failed_requests": 50,
                "avg_latency_ms": 45.2,
                "level_distribution": {"L0": 300, "L1": 600, "L2": 100},
                "cache_hit_rate": 0.15,
                "timestamp": 1640995200
            }
        }

class ServiceHealth(BaseModel):
    """État de santé du service"""
    status: str = Field(..., description="healthy, degraded, unhealthy")
    service_name: str = "conversation_service"
    version: str = "1.0.0"
    uptime_seconds: Optional[int] = None
    dependencies: Dict[str, str] = Field(default_factory=dict)
    performance: Optional[PerformanceMetrics] = None
    last_check: int = Field(default_factory=lambda: int(time.time()))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service_name": "conversation_service",
                "version": "1.0.0", 
                "dependencies": {
                    "deepseek_api": "healthy",
                    "redis_cache": "healthy"
                },
                "last_check": 1640995200,
                "warnings": [],
                "errors": []
            }
        }

# ==========================================
# MODÈLES DE BATCH ET REQUÊTES MULTIPLES
# ==========================================

class BatchChatRequest(BaseModel):
    """Requête de chat en lot"""
    requests: List[ChatRequest] = Field(..., min_items=1, max_items=10)
    batch_id: Optional[str] = Field(None, description="ID du lot")
    
    @validator('requests')
    def validate_batch_size(cls, v):
        """Validation de la taille du lot"""
        if len(v) > 10:
            raise ValueError("Maximum 10 requêtes par lot")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "requests": [
                    {"message": "Mon solde ?", "user_id": 123},
                    {"message": "Mes dépenses", "user_id": 123}
                ],
                "batch_id": "batch_456"
            }
        }

class BatchChatResponse(BaseModel):
    """Réponse de chat en lot"""
    batch_id: str
    responses: List[ChatResponse]
    batch_metadata: Dict[str, Any] = Field(default_factory=dict)
    total_processing_time_ms: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch_456",
                "responses": [],  # Liste des ChatResponse
                "total_processing_time_ms": 150.5,
                "batch_metadata": {"processed_at": 1640995200}
            }
        }

# ==========================================
# MODÈLES D'ERREUR STRUCTURÉS
# ==========================================

class ConversationError(BaseModel):
    """Erreur structurée du service"""
    error_type: str = Field(..., description="Type d'erreur")
    error_code: str = Field(..., description="Code d'erreur")
    message: str = Field(..., description="Message d'erreur")
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_type": "validation_error",
                "error_code": "INVALID_MESSAGE",
                "message": "Message trop court",
                "details": {"min_length": 1},
                "request_id": "req_123",
                "timestamp": 1640995200
            }
        }

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_error_response(
    request_id: str,
    error_type: str,
    message: str,
    processing_time_ms: float = 0.0
) -> ChatResponse:
    """Helper pour créer une réponse d'erreur standardisée"""
    return ChatResponse(
        request_id=request_id,
        intent="UNKNOWN",
        confidence=0.0,
        entities={},
        success=False,
        error=message,
        processing_metadata=ProcessingMetadata(
            request_id=request_id,
            level_used="ERROR_FALLBACK",
            processing_time_ms=processing_time_ms,
            cache_hit=False,
            error=error_type
        )
    )

def create_success_response(
    request_id: str,
    intent: str,
    confidence: float,
    entities: Dict[str, Any],
    processing_time_ms: float,
    level_used: str,
    cache_hit: bool = False,
    message: str = None,
    suggested_actions: List[str] = None
) -> ChatResponse:
    """Helper pour créer une réponse de succès standardisée"""
    return ChatResponse(
        request_id=request_id,
        intent=intent,
        confidence=confidence,
        entities=entities,
        message=message,
        suggested_actions=suggested_actions,
        success=True,
        processing_metadata=ProcessingMetadata(
            request_id=request_id,
            level_used=level_used,
            processing_time_ms=processing_time_ms,
            cache_hit=cache_hit,
            engine_latency_ms=processing_time_ms
        )
    )