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
                "ignore_status": [400, 401, 403, 404],  # Ignorer certains codes d'erreur
                "headers": {
                    "X-Elastic-Product": "Elasticsearch",  # Aide à l'identification
                    "User-Agent": "HarenaSearchService/1.0"  # Ajouter un User-Agent personnalisé
                }
            }
            
            # Si l'URL contient déjà des identifiants, ne pas ajouter d'authentification séparée
            if "@" in url:
                _es_client = AsyncElasticsearch(
                    [url],
                    **es_options
                )
                logger.info("Client Elasticsearch connecté à SearchBox avec authentification intégrée dans l'URL")
            # Sinon utiliser l'API key si présente
            elif api_key:
                _es_client = AsyncElasticsearch(
                    [url],
                    api_key=api_key,
                    **es_options
                )
                logger.info("Client Elasticsearch connecté à SearchBox avec API key")
            else:
                # Connexion sans authentification (selon la configuration de SearchBox)
                _es_client = AsyncElasticsearch(
                    [url],
                    **es_options
                )
                logger.info("Client Elasticsearch connecté à SearchBox sans authentification")
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
        
        # Vérifier la connectivité avec gestion adaptée des erreurs
        try:
            info = await client.info()
            logger.info(f"Elasticsearch connecté: version {info['version']['number']}")
        except Exception as e:
            if "not Elasticsearch" in str(e) or "unknown product" in str(e).lower():
                logger.warning("Serveur non reconnu comme Elasticsearch standard, mais la connexion est établie.")
                # On continue malgré cette erreur spécifique à SearchBox
                return client
            else:
                logger.error(f"Erreur lors de la vérification de la connexion Elasticsearch: {str(e)}")
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

async def search_transactions(user_id: int, query_text: str, filters: Dict[str, Any] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Effectue une recherche dans les transactions d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        query_text: Texte de la requête
        filters: Filtres additionnels à appliquer
        limit: Nombre de résultats maximum
        
    Returns:
        Liste des résultats de recherche
    """
    client = await get_es_client()
    index_name = f"transactions_{user_id}"
    
    try:
        # Vérifier si l'index existe
        exists = await client.indices.exists(index=index_name)
        if not exists:
            logger.warning(f"Index {index_name} n'existe pas encore. Aucun résultat.")
            return []
        
        # Préparer la requête Elasticsearch
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {
                            "query": query_text,
                            "fields": ["description^3", "merchant_name^4", "category^2", "clean_description^3.5"],
                            "type": "best_fields",
                            "operator": "or",
                            "fuzziness": "AUTO"
                        }}
                    ]
                }
            },
            "size": limit,
            "highlight": {
                "fields": {
                    "description": {},
                    "merchant_name": {},
                    "clean_description": {}
                },
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"]
            }
        }
        
        # Ajouter les filtres si présents
        if filters:
            es_query["query"]["bool"]["filter"] = filters
        
        # Exécuter la recherche
        response = await client.search(index=index_name, body=es_query)
        
        # Transformer les résultats
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "id": hit["_id"],
                "content": hit["_source"],
                "score": hit["_score"],
                "highlight": hit.get("highlight", {})
            }
            results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la recherche Elasticsearch: {str(e)}")
        # En cas d'erreur spécifique à SearchBox, retourner une liste vide plutôt que lever une exception
        if "not Elasticsearch" in str(e) or "unknown product" in str(e).lower():
            logger.warning("Erreur SearchBox spécifique détectée, retour d'une liste vide")
            return []
        raise

async def close_es_client():
    """Ferme la connexion au client Elasticsearch."""
    global _es_client
    
    if _es_client is not None:
        try:
            await _es_client.close()
            _es_client = None
            logger.info("Connexion Elasticsearch fermée")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la connexion Elasticsearch: {str(e)}")