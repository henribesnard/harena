"""Modèles Pydantic ultra-simples"""
from pydantic import BaseModel, Field
from typing import Optional
import time

class IntentRequest(BaseModel):
    """Requête détection intention"""
    query: str = Field(..., description="Texte utilisateur", min_length=1, max_length=500)
    user_id: Optional[str] = Field(None, description="ID utilisateur optionnel")

class IntentResponse(BaseModel):
    """Réponse détection intention"""
    intent: str = Field(..., description="Intention détectée")
    confidence: float = Field(..., description="Score de confiance [0-1]")
    processing_time_ms: float = Field(..., description="Temps traitement en ms")
    query: str = Field(..., description="Requête originale")
    model: str = Field(default="TinyBERT", description="Modèle utilisé")
    timestamp: float = Field(default_factory=time.time, description="Timestamp")

class HealthResponse(BaseModel):
    """Réponse santé service"""
    status: str = Field(default="healthy")
    model_loaded: bool = Field(..., description="Modèle TinyBERT chargé")
    total_requests: int = Field(default=0, description="Nombre total requêtes")
    average_latency_ms: float = Field(default=0.0, description="Latence moyenne")