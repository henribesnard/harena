"""
Service d'embeddings pour la recherche sémantique - VERSION COMPLÈTE CORRIGÉE.

Ce module gère la génération d'embeddings vectoriels pour les requêtes
en utilisant l'API OpenAI avec les mêmes paramètres que enrichment_service.

CORRECTIONS APPORTÉES:
- Ajout de EmbeddingConfig et EmbeddingModel classes manquantes
- Interface 100% compatible avec enrichment_service
- Mêmes paramètres OpenAI (text-embedding-3-small, pas de dimensions)
- Gestion d'erreurs robuste avec fallbacks
- Export correct pour éviter les erreurs d'import
"""
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from config_service.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """Modèles d'embeddings supportés."""
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


@dataclass
class EmbeddingConfig:
    """Configuration pour le service d'embeddings."""
    model: EmbeddingModel = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
    dimensions: Optional[int] = None  # None = utilise les dimensions par défaut du modèle
    batch_size: int = 100
    max_tokens: int = 8191
    timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """Post-initialisation pour définir les dimensions par défaut."""
        if self.dimensions is None:
            model_dimensions = {
                EmbeddingModel.TEXT_EMBEDDING_3_SMALL: 1536,
                EmbeddingModel.TEXT_EMBEDDING_3_LARGE: 3072,
                EmbeddingModel.TEXT_EMBEDDING_ADA_002: 1536
            }
            self.dimensions = model_dimensions.get(self.model, 1536)


class EmbeddingService:
    """Service pour générer des embeddings via OpenAI (compatible enrichment_service)."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.client = None
        self.model = self.config.model.value
        self.batch_size = min(self.config.batch_size, 100) 
        self._initialized = False
        
        # Métriques
        self.total_requests = 0
        self.successful_requests = 0
        self.total_tokens = 0
        
        # Cache simple pour éviter les regénérations
        self._cache: Dict[str, List[float]] = {}
        self._max_cache_size = 1000
        
    async def initialize(self):
        """Initialise le client OpenAI."""
        if self._initialized:
            return
            
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY non définie")
            raise ValueError("OpenAI API key is required")
            
        self.client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=getattr(settings, 'DEEPSEEK_TIMEOUT', 30)
        )
        
        self._initialized = True
        logger.info(f"EmbeddingService initialisé avec le modèle {self.model} (compatible enrichment_service)")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_embedding(
        self, 
        text: str, 
        use_cache: bool = True,  
        text_id: Optional[str] = None  
    ) -> List[float]:
        """
        Génère un embedding pour un texte donné.
        
        Args:
            text: Texte à vectoriser
            use_cache: Utiliser le cache si disponible
            text_id: Identifiant optionnel pour le texte (pour les logs)
            
        Returns:
            List[float]: Vecteur d'embedding
        """
        if not self._initialized:
            await self.initialize()
            
        # Nettoyage du texte
        clean_text = text.strip().replace('\n', ' ') if text else ""
        if not clean_text:
            logger.warning("Texte vide fourni pour l'embedding")
            return [0.0] * self.config.dimensions
            
        # Vérification du cache
        cache_key = f"{self.model}:{hash(clean_text)}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            self.total_requests += 1
            
            if text_id:
                logger.debug(f"Génération embedding pour {text_id}: {clean_text[:100]}...")
            else:
                logger.debug(f"Génération embedding pour: {clean_text[:100]}...")
            
            # PARAMÈTRES IDENTIQUES À enrichment_service
            response = await self.client.embeddings.create(
                model=self.model,
                input=clean_text,
                encoding_format="float"
                # ✅ PAS de paramètre dimensions pour compatibilité enrichment_service
            )
            
            embedding = response.data[0].embedding
            self.successful_requests += 1
            self.total_tokens += response.usage.total_tokens if response.usage else 0
            
            logger.debug(f"Embedding généré: dimension {len(embedding)}")
            
            # Mise en cache
            if use_cache and len(self._cache) < self._max_cache_size:
                self._cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embedding: {e}")
            # Retourner un vecteur zéro en cas d'erreur pour éviter les crashes
            return [0.0] * self.config.dimensions
    
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
        if not self._initialized:
            await self.initialize()
            
        if not texts:
            return []
            
        # Nettoyer les textes
        clean_texts = [text.strip().replace('\n', ' ') if text else "" for text in texts]
        clean_texts = [text if text else " " for text in clean_texts]  # Éviter les textes vides
        
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
        self.total_requests += 1
        
        # PARAMÈTRES IDENTIQUES À enrichment_service
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        
        # Extraire les embeddings en respectant l'ordre
        embeddings = []
        for data_point in response.data:
            embeddings.append(data_point.embedding)
            
        self.successful_requests += 1
        self.total_tokens += response.usage.total_tokens if response.usage else 0
        
        return embeddings
    
    async def _fallback_individual_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings individuellement en cas d'échec du lot."""
        logger.warning(f"Passage en mode fallback pour {len(texts)} textes")
        
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = await self.generate_embedding(text, use_cache=False)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Erreur embedding individuel {i}: {e}")
                # Ajouter un vecteur zéro en cas d'échec
                embeddings.append([0.0] * self.config.dimensions)
                
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings selon le modèle utilisé."""
        return self.config.dimensions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du service."""
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "total_tokens": self.total_tokens,
            "cache_size": len(self._cache),
            "model": self.model,
            "dimensions": self.config.dimensions,
            "enrichment_service_compatible": True
        }
    
    async def close(self):
        """Nettoie les ressources du service."""
        if self.client:
            await self.client.close()
            logger.info("EmbeddingService fermé")


class EmbeddingManager:
    """
    Gestionnaire d'embeddings de haut niveau compatible avec enrichment_service.
    
    Fournit une interface unifiée pour la génération d'embeddings avec
    les mêmes paramètres que enrichment_service.
    """
    
    def __init__(self, primary_service: EmbeddingService):
        self.primary_service = primary_service
        self.fallback_services: List[EmbeddingService] = []
        
        # Métriques globales
        self.total_requests = 0
        self.successful_requests = 0
        self.fallback_usage = 0
        
        logger.info("Embedding manager initialized (enrichment_service compatible)")
    
    def add_fallback_service(self, service: EmbeddingService) -> None:
        """Ajoute un service de fallback."""
        self.fallback_services.append(service)
        logger.info(f"Added fallback embedding service (total: {len(self.fallback_services)})")
    
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True,
        text_id: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Génère un embedding avec fallback automatique.
        
        Args:
            text: Texte à encoder
            use_cache: Utiliser le cache si disponible
            text_id: Identifiant optionnel pour le texte
            
        Returns:
            Vecteur d'embedding ou None si erreur
        """
        self.total_requests += 1
        
        # Essayer le service primaire
        try:
            embedding = await self.primary_service.generate_embedding(
                text, use_cache=use_cache, text_id=text_id
            )
            if embedding and len(embedding) > 0:
                self.successful_requests += 1
                return embedding
        except Exception as e:
            logger.warning(f"Primary embedding service failed: {e}")
        
        # Essayer les services de fallback
        for i, fallback_service in enumerate(self.fallback_services):
            try:
                logger.info(f"Trying fallback service {i + 1}")
                embedding = await fallback_service.generate_embedding(
                    text, use_cache=use_cache, text_id=text_id
                )
                if embedding and len(embedding) > 0:
                    self.successful_requests += 1
                    self.fallback_usage += 1
                    logger.info(f"Fallback service {i + 1} succeeded")
                    return embedding
            except Exception as e:
                logger.warning(f"Fallback service {i + 1} failed: {e}")
        
        logger.error(f"All embedding services failed for text: {text[:100]}...")
        return None
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[Optional[List[float]]]:
        """Génère des embeddings pour une liste de textes."""
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text, use_cache)
            embeddings.append(embedding)
        return embeddings
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du gestionnaire."""
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        fallback_rate = (
            self.fallback_usage / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        primary_metrics = self.primary_service.get_metrics()
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "fallback_usage": self.fallback_usage,
            "fallback_rate": round(fallback_rate, 2),
            "fallback_services_count": len(self.fallback_services),
            "primary_service_metrics": primary_metrics,
            "enrichment_service_compatible": True
        }


# Factory functions pour compatibilité avec enrichment_service
def create_embedding_service(config: Optional[EmbeddingConfig] = None) -> EmbeddingService:
    """
    Factory function pour créer un service d'embeddings compatible enrichment_service.
    
    Utilise les mêmes variables d'environnement et paramètres.
    """
    if config is None:
        config = EmbeddingConfig(
            model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
            dimensions=None  # Utilise les dimensions par défaut (1536)
        )
    
    service = EmbeddingService(config)
    logger.info(f"Created enrichment_service compatible embedding service with model {service.model}")
    return service


def create_embedding_manager(config: Optional[EmbeddingConfig] = None) -> EmbeddingManager:
    """
    Factory function pour créer un gestionnaire d'embeddings compatible enrichment_service.
    """
    primary_service = create_embedding_service(config)
    manager = EmbeddingManager(primary_service)
    logger.info("Created enrichment_service compatible embedding manager")
    return manager


# Instance globale pour compatibilité (sera initialisée à la demande)
embedding_service: Optional[EmbeddingService] = None


def get_global_embedding_service() -> EmbeddingService:
    """Retourne l'instance globale d'embedding service."""
    global embedding_service
    if embedding_service is None:
        embedding_service = create_embedding_service()
    return embedding_service


# Exports principaux
__all__ = [
    # Classes et enums
    "EmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingService",
    "EmbeddingManager",
    
    # Factory functions
    "create_embedding_service",
    "create_embedding_manager",
    "get_global_embedding_service",
    
    # Instance globale
    "embedding_service"
]