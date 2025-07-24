"""Détecteur intentions TinyBERT ultra-optimisé"""
import time
import logging
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from .config import TINYBERT_MODEL, DEVICE, MAX_LENGTH, FINANCIAL_INTENTS, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

class TinyBERTDetector:
    """Détecteur intentions financières avec TinyBERT"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = DEVICE
        self.is_loaded = False
        
        # Métriques simples
        self.total_requests = 0
        self.total_processing_time = 0.0
        
        # Labels mapping (à adapter selon votre dataset)
        self.id2label = {
            0: "BALANCE_CHECK",
            1: "TRANSFER", 
            2: "EXPENSE_ANALYSIS",
            3: "CARD_MANAGEMENT",
            4: "GREETING",
            5: "HELP",
            6: "GOODBYE",
            7: "UNKNOWN"
        }
    
    async def load_model(self):
        """Chargement modèle TinyBERT"""
        try:
            logger.info(f"🤖 Chargement TinyBERT: {TINYBERT_MODEL}")
            start = time.time()
            
            self.tokenizer = AutoTokenizer.from_pretrained(TINYBERT_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                TINYBERT_MODEL,
                num_labels=len(self.id2label)
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            load_time = (time.time() - start) * 1000
            logger.info(f"✅ TinyBERT chargé en {load_time:.2f}ms")
            
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement TinyBERT: {e}")
            raise
    
    async def detect_intent(self, query: str) -> Tuple[str, float, float]:
        """
        Détecte intention avec mesure précise latence
        
        Returns:
            (intent, confidence, processing_time_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Modèle TinyBERT non chargé")
        
        start_time = time.time()
        
        try:
            # Tokenisation
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            ).to(self.device)
            
            # Inférence
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
                
                # Meilleure prédiction
                predicted_id = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_id].item()
                
                # Mapping vers intention
                intent = self.id2label.get(predicted_id, "UNKNOWN")
                
                # Seuil de confiance
                if confidence < CONFIDENCE_THRESHOLD:
                    intent = "UNKNOWN"
                    confidence = 1.0 - confidence
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Métriques
            self.total_requests += 1
            self.total_processing_time += processing_time_ms
            
            logger.debug(f"Intent: {intent}, Confidence: {confidence:.3f}, Time: {processing_time_ms:.2f}ms")
            
            return intent, confidence, processing_time_ms
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Erreur détection: {e}")
            return "UNKNOWN", 0.0, processing_time_ms
    
    def get_average_latency(self) -> float:
        """Latence moyenne"""
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time / self.total_requests
    
    def get_stats(self) -> Dict:
        """Statistiques détecteur"""
        return {
            "model_loaded": self.is_loaded,
            "total_requests": self.total_requests,
            "average_latency_ms": self.get_average_latency(),
            "device": self.device,
            "model_name": TINYBERT_MODEL
        }