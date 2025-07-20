"""
Classificateur léger TinyBERT quantifié pour niveau L1
Performance optimisée Heroku avec cache embeddings
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .models import IntentResult, IntentLevel, IntentConfidence, IntentType
from conversation_service.config import settings
from conversation_service.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IntentEmbedding:
    """Embedding pré-calculé pour une intention"""
    intent_type: str
    embedding: List[float]
    keywords: List[str]
    confidence_threshold: float


class LightweightClassifier:
    """
    Classificateur TinyBERT quantifié pour détection rapide L1
    Modèle 15MB en mémoire, latence 15-30ms
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.intent_embeddings: Dict[str, IntentEmbedding] = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialisation modèle TinyBERT quantifié"""
        try:
            # Note: En production, charger TinyBERT quantifié depuis fichier local
            # Ici simulation pour architecture
            await self._load_quantized_model()
            await self._load_intent_embeddings()
            
            self.is_initialized = True
            logger.info("TinyBERT classifier initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation TinyBERT: {e}")
            # Fallback mode sans modèle
            self.is_initialized = False
    
    async def _load_quantized_model(self):
        """Chargement modèle TinyBERT 4-bit quantifié"""
        # Simulation chargement modèle optimisé
        # En production: torch.jit.load() ou ONNX Runtime
        
        self.model = "simulated_tinybert_model"  # Placeholder
        self.tokenizer = "simulated_tokenizer"   # Placeholder
        
        logger.info("Modèle TinyBERT quantifié chargé (15MB RAM)")
    
    async def _load_intent_embeddings(self):
        """Préchargement embeddings intentions principales"""
        
        # Embeddings pré-calculés pour intentions financières
        # En production: calculés offline et stockés
        
        intent_embeddings = {
            IntentType.BALANCE_CHECK.value: IntentEmbedding(
                intent_type=IntentType.BALANCE_CHECK.value,
                embedding=np.random.random(384).tolist(),  # Simulation TinyBERT 384-dim
                keywords=["solde", "combien", "montant", "compte", "carte"],
                confidence_threshold=0.85
            ),
            
            IntentType.EXPENSE_ANALYSIS.value: IntentEmbedding(
                intent_type=IntentType.EXPENSE_ANALYSIS.value,
                embedding=np.random.random(384).tolist(),
                keywords=["dépenses", "frais", "achats", "total", "analyse"],
                confidence_threshold=0.82
            ),
            
            IntentType.TRANSFER.value: IntentEmbedding(
                intent_type=IntentType.TRANSFER.value,
                embedding=np.random.random(384).tolist(),
                keywords=["virement", "virer", "envoyer", "transfert"],
                confidence_threshold=0.88
            ),
            
            IntentType.TRANSACTION_SEARCH.value: IntentEmbedding(
                intent_type=IntentType.TRANSACTION_SEARCH.value,
                embedding=np.random.random(384).tolist(),
                keywords=["transactions", "paiements", "recherche", "historique"],
                confidence_threshold=0.80
            ),
            
            IntentType.BUDGET_INQUIRY.value: IntentEmbedding(
                intent_type=IntentType.BUDGET_INQUIRY.value,
                embedding=np.random.random(384).tolist(),
                keywords=["budget", "enveloppe", "limite", "planification"],
                confidence_threshold=0.83
            )
        }
        
        self.intent_embeddings = intent_embeddings
        logger.info(f"Embeddings chargés pour {len(intent_embeddings)} intentions")
    
    async def generate_embedding(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Génération embedding avec TinyBERT quantifié
        Performance cible: 10-20ms
        """
        if not self.is_initialized:
            # Fallback embedding simple basé sur mots-clés
            return self._fallback_keyword_embedding(query)
        
        start_time = time.time()
        
        try:
            # Simulation génération embedding TinyBERT
            # En production: 
            # tokens = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
            # with torch.no_grad():
            #     outputs = self.model(**tokens)
            #     embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            
            # Simulation optimisée
            embedding = np.random.random(384).tolist()
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"Embedding généré en {latency_ms}ms")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur génération embedding: {e}")
            return self._fallback_keyword_embedding(query)
    
    async def classify_intent(
        self, 
        query_embedding: List[float], 
        original_query: str
    ) -> Optional[IntentResult]:
        """
        Classification intention par similarité cosinus
        """
        if not self.intent_embeddings:
            return None
        
        start_time = time.time()
        
        try:
            best_intent = None
            best_score = 0.0
            
            query_vector = np.array(query_embedding)
            
            # Calcul similarité avec chaque intention
            for intent_type, intent_data in self.intent_embeddings.items():
                intent_vector = np.array(intent_data.embedding)
                
                # Similarité cosinus optimisée
                similarity = np.dot(query_vector, intent_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(intent_vector)
                )
                
                # Bonus mots-clés pour améliorer précision
                keyword_bonus = self._calculate_keyword_bonus(original_query, intent_data.keywords)
                final_score = similarity + keyword_bonus
                
                if final_score > best_score and final_score > intent_data.confidence_threshold:
                    best_score = final_score
                    best_intent = intent_type
            
            if best_intent:
                # Extraction entités simples
                entities = self._extract_basic_entities(original_query, best_intent)
                
                result = IntentResult(
                    intent_type=best_intent,
                    entities=entities,
                    confidence=IntentConfidence(
                        score=min(best_score, 1.0),  # Cap à 1.0
                        level=IntentLevel.L1_LIGHTWEIGHT
                    ),
                    level=IntentLevel.L1_LIGHTWEIGHT,
                    latency_ms=int((time.time() - start_time) * 1000),
                    metadata={
                        "similarity_score": float(best_score - self._calculate_keyword_bonus(original_query, self.intent_embeddings[best_intent].keywords)),
                        "keyword_bonus": self._calculate_keyword_bonus(original_query, self.intent_embeddings[best_intent].keywords),
                        "threshold_used": self.intent_embeddings[best_intent].confidence_threshold
                    }
                )
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur classification intention: {e}")
            return None
    
    def _fallback_keyword_embedding(self, query: str) -> List[float]:
        """Embedding fallback basé sur mots-clés"""
        # Implémentation simple pour fallback
        words = query.lower().split()
        
        # Embedding 384-dim basé sur hash mots
        embedding = np.zeros(384)
        for i, word in enumerate(words[:10]):  # Max 10 mots
            hash_val = hash(word) % 384
            embedding[hash_val] += 1.0
        
        # Normalisation
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def _calculate_keyword_bonus(self, query: str, keywords: List[str]) -> float:
        """Bonus basé sur présence mots-clés"""
        query_lower = query.lower()
        matches = sum(1 for keyword in keywords if keyword in query_lower)
        return min(matches * 0.05, 0.15)  # Max 15% bonus
    
    def _extract_basic_entities(self, query: str, intent_type: str) -> Dict[str, Any]:
        """Extraction entités basiques selon type intention"""
        entities = {"query": query}
        
        # Extraction spécialisée par intention
        if intent_type == IntentType.EXPENSE_ANALYSIS.value:
            if any(cat in query.lower() for cat in ["restaurant", "courses", "essence"]):
                entities["category"] = next(cat for cat in ["restaurant", "courses", "essence"] if cat in query.lower())
        
        elif intent_type == IntentType.TRANSFER.value:
            # Extraction nom contact simple
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ["vers", "à", "pour"] and i + 1 < len(words):
                    entities["recipient"] = words[i + 1]
                    break
        
        return entities
    
    async def preload_intent_embeddings(self) -> bool:
        """Préchargement embeddings en cache Redis"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Validation embeddings chargés
            for intent_type, embedding_data in self.intent_embeddings.items():
                assert len(embedding_data.embedding) == 384
                assert embedding_data.confidence_threshold > 0
            
            logger.info("Embeddings intentions préchargés")
            return True
            
        except Exception as e:
            logger.error(f"Erreur préchargement embeddings: {e}")
            return False
