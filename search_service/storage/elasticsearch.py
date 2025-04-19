"""
Interface avec Elasticsearch via SearchBox pour le service de recherche.

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

from config_service.config import settings

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
            # Utiliser l'URL de SearchBox
            url = settings.SEARCHBOX_URL
            api_key = settings.SEARCHBOX_API_KEY
            
            if not url:
                raise ValueError("SEARCHBOX_URL est manquante dans les variables d'environnement")
            
            # Options de compatibilité pour les services hébergés comme SearchBox
            es_options = {
                "request_timeout": 30,
                "verify_certs": True,
                "retry_on_timeout": True,
                "max_retries": 3,
                #"ignore_status": [400, 401, 403, 404],  # Ignorer certains codes d'erreur
                "headers": {
                    "X-Elastic-Product": "Elasticsearch"  # Aide à l'identification
                }
            }
            
            # Si l'URL contient déjà des identifiants, ne pas ajouter d'authentification séparée
            if "@" in url:
                _es_client = AsyncElasticsearch(
                    [url],
                    **es_options
                )
                logger.info(f"Client Elasticsearch connecté à SearchBox avec authentification intégrée dans l'URL")
            # Sinon utiliser l'API key si présente
            elif api_key:
                _es_client = AsyncElasticsearch(
                    [url],
                    api_key=api_key,
                    **es_options
                )
                logger.info(f"Client Elasticsearch connecté à SearchBox avec API key")
            else:
                # Connexion sans authentification (selon la configuration de SearchBox)
                _es_client = AsyncElasticsearch(
                    [url],
                    **es_options
                )
                logger.info(f"Client Elasticsearch connecté à SearchBox sans authentification")
        except Exception as e:
            logger.error(f"Impossible de se connecter à Elasticsearch (SearchBox): {str(e)}")
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
        
        # Vérifier la connectivité - ignorer l'erreur "unknown product"
        try:
            info = await client.info()
            logger.info(f"Elasticsearch connecté: version {info['version']['number']}")
        except Exception as e:
            if "not Elasticsearch" in str(e):
                logger.warning(f"Serveur non reconnu comme Elasticsearch standard, mais la connexion est établie.")
                # On continue malgré cette erreur spécifique
            else:
                raise
        
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