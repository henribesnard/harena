"""
Interface avec Elasticsearch pour le service de recherche.

Ce module gère la connexion et les interactions avec le moteur de recherche
Elasticsearch pour la recherche lexicale.
"""
import logging
from typing import Optional, Dict, Any, List
import asyncio

# Import conditionnel d'Elasticsearch
try:
    from elasticsearch import AsyncElasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

from search_service.core.config import settings

logger = logging.getLogger(__name__)

# Client Elasticsearch global
_es_client = None

async def get_es_client() -> Optional[Any]:
    """
    Obtient une instance du client Elasticsearch (singleton).
    
    Returns:
        Client Elasticsearch ou exception si indisponible
    """
    global _es_client
    
    if not ELASTICSEARCH_AVAILABLE:
        logger.error("Module elasticsearch non disponible. Veuillez l'installer: 'pip install elasticsearch'")
        raise ImportError("Module elasticsearch requis pour la recherche lexicale")
    
    if _es_client is None:
        try:
            auth = None
            if settings.ELASTICSEARCH_USERNAME and settings.ELASTICSEARCH_PASSWORD:
                auth = (settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD)
            
            _es_client = AsyncElasticsearch(
                [settings.ELASTICSEARCH_URL],
                basic_auth=auth,
                request_timeout=30
            )
            logger.info(f"Client Elasticsearch connecté à {settings.ELASTICSEARCH_URL}")
        except Exception as e:
            logger.error(f"Impossible de se connecter à Elasticsearch: {str(e)}")
            raise
    
    return _es_client

async def init_elasticsearch() -> Optional[Any]:
    """
    Initialise la connexion Elasticsearch et vérifie les index.
    
    Returns:
        Client Elasticsearch ou None en cas d'erreur
    """
    try:
        client = await get_es_client()
        
        # Vérifier la connectivité
        info = await client.info()
        logger.info(f"Elasticsearch connecté: version {info['version']['number']}")
        
        return client
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation d'Elasticsearch: {str(e)}")
        return None

async def ensure_index_exists(index_name: str, settings_body: Dict[str, Any], mappings_body: Dict[str, Any]) -> bool:
    """
    Vérifie si un index existe et le crée si nécessaire.
    
    Args:
        index_name: Nom de l'index
        settings_body: Configuration de l'index
        mappings_body: Mappings des champs
        
    Returns:
        True si l'index existe ou a été créé, False sinon
    """
    client = await get_es_client()
    
    try:
        # Vérifier si l'index existe
        exists = await client.indices.exists(index=index_name)
        if exists:
            logger.debug(f"Index Elasticsearch existant: {index_name}")
            return True
        
        # Créer l'index avec les configurations et mappings
        body = {
            "settings": settings_body,
            "mappings": mappings_body
        }
        
        await client.indices.create(index=index_name, body=body)
        logger.info(f"Index Elasticsearch créé: {index_name}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'index {index_name}: {str(e)}")
        return False

async def close_es_client():
    """Ferme la connexion au client Elasticsearch."""
    global _es_client
    
    if _es_client is not None:
        await _es_client.close()
        _es_client = None
        logger.info("Connexion Elasticsearch fermée")