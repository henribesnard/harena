"""
Service de génération d'embeddings.

Ce module gère la génération d'embeddings vectoriels pour les transactions
en utilisant l'API OpenAI.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from config_service.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service pour générer des embeddings via OpenAI."""
    
    def __init__(self):
        self.client = None
        self.model = settings.EMBEDDING_MODEL
        self.batch_size = min(settings.BATCH_SIZE, 100)  # OpenAI limite à 100
        
    async def initialize(self):
        """Initialise le client OpenAI."""
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY non définie")
            raise ValueError("OpenAI API key is required")
            
        self.client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.DEEPSEEK_TIMEOUT  # Réutiliser le timeout config
        )
        
        logger.info(f"EmbeddingService initialisé avec le modèle {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Génère un embedding pour un texte donné.
        
        Args:
            text: Texte à vectoriser
            
        Returns:
            List[float]: Vecteur d'embedding
        """
        if not self.client:
            raise ValueError("EmbeddingService not initialized")
            
        if not text or not text.strip():
            logger.warning("Texte vide fourni pour l'embedding")
            # Retourner un vecteur zéro de la bonne dimension (1536 pour text-embedding-3-small)
            return [0.0] * 1536
            
        try:
            logger.debug(f"Génération embedding pour: {text[:100]}...")
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Embedding généré: dimension {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embedding: {e}")
            raise
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_id: Optional[str] = None
    ) -> List[List[float]]:
        """
        Génère des embeddings pour un lot de textes.
        
        Args:
            texts: Liste des textes à vectoriser
            batch_id: Identifiant optionnel du lot pour les logs
            
        Returns:
            List[List[float]]: Liste des vecteurs d'embedding
        """
        if not self.client:
            raise ValueError("EmbeddingService not initialized")
            
        if not texts:
            return []
            
        # Nettoyer les textes
        clean_texts = [text.strip() if text else "" for text in texts]
        
        # Traiter par lots pour respecter les limites d'API
        all_embeddings = []
        total_batches = (len(clean_texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Génération d'embeddings pour {len(clean_texts)} textes en {total_batches} lots")
        
        for i in range(0, len(clean_texts), self.batch_size):
            batch_texts = clean_texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            try:
                logger.debug(f"Traitement lot {batch_num}/{total_batches} ({len(batch_texts)} textes)")
                
                # Générer les embeddings pour ce lot
                batch_embeddings = await self._generate_batch_openai(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Lot {batch_num} traité avec succès")
                
                # Petite pause entre les lots pour éviter le rate limiting
                if batch_num < total_batches:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement du lot {batch_num}: {e}")
                # Générer des embeddings individuellement en cas d'échec du lot
                batch_embeddings = await self._fallback_individual_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Génération terminée: {len(all_embeddings)} embeddings générés")
        return all_embeddings
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Génère un lot d'embeddings via l'API OpenAI."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        
        # Extraire les embeddings en respectant l'ordre
        embeddings = []
        for data_point in response.data:
            embeddings.append(data_point.embedding)
            
        return embeddings
    
    async def _fallback_individual_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings individuellement en cas d'échec du lot."""
        logger.warning(f"Passage en mode fallback pour {len(texts)} textes")
        
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Erreur embedding individuel {i}: {e}")
                # Ajouter un vecteur zéro en cas d'échec
                embeddings.append([0.0] * 1536)
                
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings selon le modèle utilisé."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        return model_dimensions.get(self.model, 1536)
    
    async def close(self):
        """Nettoie les ressources du service."""
        if self.client:
            await self.client.close()
            logger.info("EmbeddingService fermé")

# Instance globale
embedding_service = EmbeddingService()