"""Service conversation ultra-minimaliste avec TinyBERT"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import IntentRequest, IntentResponse, HealthResponse
from .intent_detector import TinyBERTDetector
from .config import API_HOST, API_PORT, DEBUG

# Configuration logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Instance globale détecteur
detector = TinyBERTDetector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion cycle de vie application"""
    logger.info("🚀 Démarrage Conversation Service")
    
    # Chargement modèle au démarrage
    await detector.load_model()
    logger.info("✅ Service prêt")
    
    yield
    
    logger.info("🛑 Arrêt service")

# Application FastAPI
app = FastAPI(
    title="Conversation Service - TinyBERT",
    description="Détection intentions financières ultra-rapide",
    version="1.0.0",
    lifespan=lifespan
)

# CORS simple
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ================================
# ENDPOINTS
# ================================

@app.post("/detect-intent", response_model=IntentResponse)
async def detect_intent(request: IntentRequest):
    """
    🎯 ENDPOINT UNIQUE - Détection intention financière
    
    Analyse le texte utilisateur avec TinyBERT et retourne
    l'intention détectée avec métriques de performance.
    """
    try:
        intent, confidence, processing_time_ms = await detector.detect_intent(request.query)
        
        response = IntentResponse(
            intent=intent,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            query=request.query
        )
        
        logger.info(f"✅ Intent détecté: {intent} ({confidence:.3f}) en {processing_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Erreur détection: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur détection: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Santé du service"""
    stats = detector.get_stats()
    
    return HealthResponse(
        status="healthy" if stats["model_loaded"] else "unhealthy",
        model_loaded=stats["model_loaded"],
        total_requests=stats["total_requests"],
        average_latency_ms=stats["average_latency_ms"]
    )

@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "service": "conversation_service",
        "version": "1.0.0",
        "model": "TinyBERT",
        "endpoint": "/detect-intent",
        "health": "/health"
    }

# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info"
    )