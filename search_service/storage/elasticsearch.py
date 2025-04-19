"""
Interface avec Elasticsearch via Bonsai pour le service de recherche.

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
            # Utiliser l'URL de Bonsai avec priorité, puis fallback sur SearchBox si présent
            url = settings.BONSAI_URL or settings.SEARCHBOX_URL
            
            if not url:
                raise ValueError("URL Elasticsearch (BONSAI_URL) manquante dans les variables d'environnement")
            
            # Configuration simplifiée pour Bonsai
            # L'URL contient déjà les identifiants de connexion
            logger.info(f"Tentative de connexion à Elasticsearch avec l'URL: {url[:20]}...")
            _es_client = AsyncElasticsearch([url])
            logger.info("Client Elasticsearch connecté avec succès")
            
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
        try:
            info = await client.info()
            logger.info(f"Elasticsearch connecté: version {info['version']['number']}")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de la connexion Elasticsearch: {str(e)}")
            return None
        
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
        return []

async def create_transaction_index(user_id: int) -> bool:
    """
    Crée l'index de transactions pour un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        True si l'index a été créé avec succès, False sinon
    """
    index_name = f"transactions_{user_id}"
    
    # Configuration de base pour l'index
    settings_body = {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "transaction_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"]
                }
            }
        }
    }
    
    # Mapping des champs pour les transactions
    mappings_body = {
        "properties": {
            "user_id": {"type": "integer"},
            "account_id": {"type": "integer"},
            "bridge_transaction_id": {"type": "keyword"},
            "amount": {"type": "float"},
            "currency_code": {"type": "keyword"},
            "description": {"type": "text", "analyzer": "transaction_analyzer"},
            "clean_description": {"type": "text", "analyzer": "transaction_analyzer"},
            "transaction_date": {"type": "date"},
            "booking_date": {"type": "date"},
            "value_date": {"type": "date"},
            "category_id": {"type": "integer"},
            "operation_type": {"type": "keyword"},
            "is_recurring": {"type": "boolean"},
            "merchant_id": {"type": "keyword"},
            "merchant_name": {"type": "text", "analyzer": "transaction_analyzer"}
        }
    }
    
    return await ensure_index_exists(index_name, settings_body, mappings_body)

async def get_indices_stats() -> Dict[str, Any]:
    """
    Récupère des statistiques sur les indices Elasticsearch.
    
    Returns:
        Dictionnaire contenant les statistiques des indices
    """
    client = await get_es_client()
    
    try:
        stats = await client.indices.stats()
        return stats
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques des indices: {str(e)}")
        return {"error": str(e)}

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