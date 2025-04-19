"""
Interface avec Qdrant pour le service de recherche.

Ce module gère l'accès au stockage vectoriel déjà configuré dans sync_service,
en réutilisant la même instance Qdrant.
"""
import logging
from typing import Optional, Dict, Any, List
import asyncio

# Import conditionnel de Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from config_service.config import settings

# Tenter d'importer le service vectoriel de sync_service
try:
    from sync_service.services.vector_storage import VectorStorageService as SyncVectorService
    SYNC_VECTOR_SERVICE_AVAILABLE = True
except ImportError:
    SYNC_VECTOR_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Client Qdrant global partagé avec sync_service
_qdrant_client = None
_sync_vector_service = None

async def get_qdrant_client() -> Optional[Any]:
    """
    Obtient une instance du client Qdrant, si possible en réutilisant
    celle de sync_service.
    
    Returns:
        Client Qdrant ou exception si indisponible
    """
    global _qdrant_client, _sync_vector_service
    
    if not QDRANT_AVAILABLE:
        logger.error("Module qdrant_client non disponible. Veuillez l'installer: 'pip install qdrant_client'")
        raise ImportError("Module qdrant_client requis pour la recherche vectorielle")
    
    # Si nous avons déjà un client, le retourner
    if _qdrant_client is not None:
        return _qdrant_client
    
    # Essayer d'abord de récupérer le client depuis sync_service
    if SYNC_VECTOR_SERVICE_AVAILABLE:
        try:
            _sync_vector_service = SyncVectorService()
            if _sync_vector_service.client:
                logger.info("Réutilisation du client Qdrant de sync_service")
                _qdrant_client = _sync_vector_service.client
                return _qdrant_client
        except Exception as e:
            logger.warning(f"Impossible de réutiliser le client Qdrant de sync_service: {str(e)}")
    
    # Sinon, créer un nouveau client
    try:
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60.0
        )
        # Tester la connexion
        collections = _qdrant_client.get_collections()
        logger.info(f"Client Qdrant connecté à {settings.QDRANT_URL}. Collections disponibles: {[c.name for c in collections.collections]}")
        return _qdrant_client
    except Exception as e:
        logger.error(f"Impossible de se connecter à Qdrant: {str(e)}")
        raise

async def init_qdrant() -> Optional[Any]:
    """
    Initialise la connexion Qdrant et vérifie les collections.
    
    Returns:
        Client Qdrant ou None en cas d'erreur
    """
    try:
        client = await get_qdrant_client()
        
        # Vérifier les collections nécessaires
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        required_collections = ["transactions", "accounts", "merchants"]
        for collection in required_collections:
            if collection not in collection_names:
                logger.warning(f"Collection Qdrant requise non trouvée: {collection}")
        
        logger.info(f"Qdrant initialisé avec succès. Collections disponibles: {collection_names}")
        return client
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de Qdrant: {str(e)}")
        return None

def get_sync_vector_service() -> Optional[Any]:
    """
    Obtient l'instance du service vectoriel de sync_service.
    
    Returns:
        Instance de VectorStorageService de sync_service ou None
    """
    global _sync_vector_service
    
    if _sync_vector_service is not None:
        return _sync_vector_service
    
    if SYNC_VECTOR_SERVICE_AVAILABLE:
        try:
            _sync_vector_service = SyncVectorService()
            return _sync_vector_service
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation du service vectoriel de sync_service: {str(e)}")
    
    return None