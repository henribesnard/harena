"""
Configuration centralisée pour le service d'enrichissement.

Ce module utilise la configuration globale de config_service et ajoute
des paramètres spécifiques au service d'enrichissement.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from config_service.config import settings as global_settings

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration pour les embeddings."""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    max_tokens: int = 8192
    encoding_format: str = "float"
    
@dataclass
class QdrantConfig:
    """Configuration pour Qdrant."""
    url: str = ""
    api_key: str = ""
    timeout: int = 30
    max_retries: int = 3
    
    # Configuration des collections
    collections: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "enriched_transactions": {
            "vectors_config": {
                "size": 1536,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        },
        "financial_patterns": {
            "vectors_config": {
                "size": 1536,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        },
        "financial_insights": {
            "vectors_config": {
                "size": 1536,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        },
        "financial_summaries": {
            "vectors_config": {
                "size": 1536,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        },
        "enriched_accounts": {
            "vectors_config": {
                "size": 1536,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        }
    })

@dataclass  
class EnrichmentSettings:
    """Configuration complète pour le service d'enrichissement."""
    
    # Configuration générale héritée
    project_name: str = global_settings.PROJECT_NAME
    api_v1_str: str = global_settings.API_V1_STR
    environment: str = global_settings.ENVIRONMENT
    
    # Configuration base de données
    database_url: str = global_settings.SQLALCHEMY_DATABASE_URI
    
    # Configuration logging
    log_level: str = global_settings.LOG_LEVEL
    log_file: str = global_settings.LOG_FILE
    log_to_file: bool = global_settings.LOG_TO_FILE
    
    # Configuration sécurité
    secret_key: str = global_settings.SECRET_KEY
    access_token_expire_minutes: int = global_settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    # Configuration CORS
    cors_origins: str = global_settings.CORS_ORIGINS
    
    # Configuration spécifique à l'enrichissement
    embedding_config: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(
        model=global_settings.EMBEDDING_MODEL,
        batch_size=global_settings.BATCH_SIZE
    ))
    
    qdrant_config: QdrantConfig = field(default_factory=lambda: QdrantConfig(
        url=global_settings.QDRANT_URL,
        api_key=global_settings.QDRANT_API_KEY
    ))
    
    # Configuration OpenAI pour embeddings
    openai_api_key: str = global_settings.OPENAI_API_KEY
    
    # Configuration traitement par lots
    batch_size: int = global_settings.BATCH_SIZE
    cache_ttl: int = global_settings.CACHE_TTL
    memory_cache_ttl: int = global_settings.MEMORY_CACHE_TTL
    memory_cache_max_size: int = global_settings.MEMORY_CACHE_MAX_SIZE
    
    # Configuration de taux de limite
    rate_limit_enabled: bool = global_settings.RATE_LIMIT_ENABLED
    rate_limit_period: int = global_settings.RATE_LIMIT_PERIOD
    rate_limit_requests: int = global_settings.RATE_LIMIT_REQUESTS
    
    # Configuration d'enrichissement spécifique
    enrichment_batch_size: int = 50
    pattern_detection_window_days: int = 90
    pattern_minimum_occurrences: int = 3
    pattern_confidence_threshold: float = 0.7
    
    # Configuration des mises à jour en temps réel
    enable_real_time_updates: bool = True
    trigger_listen_timeout: int = 5
    max_reconnection_attempts: int = 10
    reconnection_delay_seconds: int = 30
    
    # Configuration des résumés périodiques
    generate_monthly_summaries: bool = True
    generate_weekly_summaries: bool = True
    summary_generation_hour: int = 2  # Heure de génération des résumés (2h du matin)
    
    # Configuration des insights
    insight_generation_enabled: bool = True
    insight_confidence_threshold: float = 0.6
    max_insights_per_user: int = 20
    
    # Configuration des catégories
    default_category_confidence: float = 0.5
    enable_category_learning: bool = True
    
    # Configuration des marchands
    merchant_normalization_enabled: bool = True
    merchant_similarity_threshold: float = 0.8
    
    # Timeouts et limites
    embedding_timeout: int = 30
    qdrant_timeout: int = 30
    max_concurrent_enrichments: int = 10
    
    # Configuration pour les tâches d'arrière-plan
    enable_background_tasks: bool = True
    task_queue_max_size: int = 1000
    task_worker_count: int = 4
    
    def __post_init__(self):
        """Validation et ajustements post-initialisation."""
        # Validation des seuils
        if not 0 <= self.pattern_confidence_threshold <= 1:
            logger.warning(f"pattern_confidence_threshold doit être entre 0 et 1, valeur reçue: {self.pattern_confidence_threshold}")
            self.pattern_confidence_threshold = 0.7
            
        if not 0 <= self.insight_confidence_threshold <= 1:
            logger.warning(f"insight_confidence_threshold doit être entre 0 et 1, valeur reçue: {self.insight_confidence_threshold}")
            self.insight_confidence_threshold = 0.6
        
        # Validation des URLs et clés API
        if not self.qdrant_config.url:
            logger.warning("URL Qdrant non configurée, les fonctionnalités vectorielles seront désactivées")
            
        if not self.openai_api_key:
            logger.warning("Clé API OpenAI non configurée, les embeddings ne fonctionneront pas")
        
        # Log de la configuration en mode debug
        if self.log_level.upper() == "DEBUG":
            self._log_configuration()
    
    def _log_configuration(self):
        """Journalise la configuration actuelle pour le débogage."""
        logger.debug("Configuration du service d'enrichissement:")
        logger.debug(f"  - Environnement: {self.environment}")
        logger.debug(f"  - Niveau de log: {self.log_level}")
        logger.debug(f"  - Modèle embedding: {self.embedding_config.model}")
        logger.debug(f"  - Dimensions embedding: {self.embedding_config.dimensions}")
        logger.debug(f"  - Taille de lot: {self.batch_size}")
        logger.debug(f"  - URL Qdrant configurée: {bool(self.qdrant_config.url)}")
        logger.debug(f"  - Mises à jour temps réel: {self.enable_real_time_updates}")
        logger.debug(f"  - Génération d'insights: {self.insight_generation_enabled}")
        logger.debug(f"  - Détection de patterns: fenêtre={self.pattern_detection_window_days}j, min={self.pattern_minimum_occurrences}")
        
    @property
    def is_production(self) -> bool:
        """Vérifie si l'environnement est en production."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Vérifie si l'environnement est en développement."""
        return self.environment.lower() in ["development", "dev", "local"]
    
    def get_collection_config(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Récupère la configuration d'une collection Qdrant."""
        return self.qdrant_config.collections.get(collection_name)
    
    def get_embedding_batch_size(self) -> int:
        """Retourne la taille de lot optimale pour les embeddings."""
        return min(self.embedding_config.batch_size, self.enrichment_batch_size)
    
    def should_enable_feature(self, feature: str) -> bool:
        """Vérifie si une fonctionnalité doit être activée selon l'environnement."""
        feature_mapping = {
            "real_time_updates": self.enable_real_time_updates,
            "background_tasks": self.enable_background_tasks,
            "insight_generation": self.insight_generation_enabled,
            "pattern_detection": True,  # Toujours activé
            "merchant_normalization": self.merchant_normalization_enabled,
            "category_learning": self.enable_category_learning
        }
        
        return feature_mapping.get(feature, False)

# Instance globale de configuration
enrichment_settings = EnrichmentSettings()

# Vérification de configuration au chargement du module
if __name__ == "__main__":
    logger.info("Vérification de la configuration du service d'enrichissement...")
    enrichment_settings._log_configuration()
    logger.info("Configuration du service d'enrichissement chargée avec succès")