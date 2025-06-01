"""
Client Elasticsearch pour la recherche lexicale.

Ce module gère les opérations de recherche full-text dans Elasticsearch,
incluant l'indexation et la recherche avec highlighting.
"""
import logging
from typing import List, Dict, Any, Optional
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from config_service.config import settings

logger = logging.getLogger(__name__)


class ElasticClient:
    """Client pour interagir avec Elasticsearch."""
    
    def __init__(self):
        self.client = None
        self.index_name = "harena_transactions"
        self._initialized = False
        
    async def initialize(self):
        """Initialise la connexion Elasticsearch."""
        # Déterminer quelle URL utiliser
        if settings.SEARCHBOX_URL:
            es_url = settings.SEARCHBOX_URL
            logger.info("Utilisation de SearchBox Elasticsearch")
        elif settings.BONSAI_URL:
            es_url = settings.BONSAI_URL
            logger.info("Utilisation de Bonsai Elasticsearch")
        else:
            logger.warning("Aucune URL Elasticsearch configurée")
            return
        
        try:
            # Créer le client
            # Bonsai inclut l'auth dans l'URL, donc pas besoin de config supplémentaire
            self.client = AsyncElasticsearch(
                [es_url],
                verify_certs=True,
                ssl_show_warn=False,
                max_retries=3,
                retry_on_timeout=True,
                timeout=30,
                # Pour Bonsai, on peut ajouter des headers supplémentaires si nécessaire
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Vérifier la connexion
            info = await self.client.info()
            logger.info(f"Connecté à Elasticsearch {info['version']['number']}")
            
            # Créer l'index si nécessaire
            await self._setup_index()
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation Elasticsearch: {e}")
            self.client = None
            self._initialized = False
    
    async def _setup_index(self):
        """Configure l'index Elasticsearch."""
        try:
            # Vérifier si l'index existe
            exists = await self.client.indices.exists(index=self.index_name)
            
            if not exists:
                logger.info(f"Création de l'index {self.index_name}")
                
                # Mapping optimisé pour les transactions
                mapping = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "french_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase",
                                        "french_stop",
                                        "french_stemmer",
                                        "asciifolding"
                                    ]
                                }
                            },
                            "filter": {
                                "french_stop": {
                                    "type": "stop",
                                    "stopwords": "_french_"
                                },
                                "french_stemmer": {
                                    "type": "stemmer",
                                    "language": "french"
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            # Identifiants
                            "transaction_id": {"type": "long"},
                            "user_id": {"type": "long"},
                            "account_id": {"type": "long"},
                            
                            # Texte recherchable
                            "searchable_text": {
                                "type": "text",
                                "analyzer": "french_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "primary_description": {
                                "type": "text",
                                "analyzer": "french_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "merchant_name": {
                                "type": "text",
                                "analyzer": "french_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "category_name": {
                                "type": "text",
                                "analyzer": "french_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            
                            # Données numériques
                            "amount": {"type": "float"},
                            "amount_abs": {"type": "float"},
                            "category_id": {"type": "integer"},
                            
                            # Dates
                            "date": {"type": "date"},
                            "timestamp": {"type": "long"},
                            
                            # Métadonnées
                            "transaction_type": {"type": "keyword"},
                            "currency_code": {"type": "keyword"},
                            "is_deleted": {"type": "boolean"},
                            
                            # Données additionnelles
                            "metadata": {"type": "object", "enabled": False}
                        }
                    }
                }
                
                await self.client.indices.create(
                    index=self.index_name,
                    body=mapping
                )
                
                logger.info(f"Index {self.index_name} créé avec succès")
            else:
                logger.info(f"Index {self.index_name} existe déjà")
                
        except Exception as e:
            logger.error(f"Erreur lors de la configuration de l'index: {e}")
    
    async def index_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Indexe une transaction dans Elasticsearch.
        
        Args:
            transaction: Données de la transaction
            
        Returns:
            bool: True si l'indexation a réussi
        """
        if not self.client:
            return False
        
        try:
            # Créer l'ID unique
            doc_id = f"user_{transaction['user_id']}_tx_{transaction['transaction_id']}"
            
            # Indexer le document
            await self.client.index(
                index=self.index_name,
                id=doc_id,
                body=transaction,
                refresh=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation: {e}")
            return False
    
    async def search(
        self,
        user_id: int,
        query: Dict[str, Any],
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        include_highlights: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche dans Elasticsearch.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête Elasticsearch
            limit: Nombre de résultats
            filters: Filtres additionnels
            include_highlights: Inclure le highlighting
            
        Returns:
            List[Dict]: Résultats de recherche
        """
        if not self.client:
            return []
        
        try:
            # Construire la requête complète
            must_clauses = [
                {"term": {"user_id": user_id}},
                {"term": {"is_deleted": False}},
                query
            ]
            
            # Ajouter les filtres
            if filters:
                if "date_from" in filters:
                    must_clauses.append({
                        "range": {
                            "date": {"gte": filters["date_from"]}
                        }
                    })
                
                if "date_to" in filters:
                    must_clauses.append({
                        "range": {
                            "date": {"lte": filters["date_to"]}
                        }
                    })
                
                if "amount_min" in filters:
                    must_clauses.append({
                        "range": {
                            "amount_abs": {"gte": filters["amount_min"]}
                        }
                    })
                
                if "amount_max" in filters:
                    must_clauses.append({
                        "range": {
                            "amount_abs": {"lte": filters["amount_max"]}
                        }
                    })
                
                if "categories" in filters:
                    must_clauses.append({
                        "terms": {"category_id": filters["categories"]}
                    })
                
                if "transaction_types" in filters:
                    must_clauses.append({
                        "terms": {"transaction_type": filters["transaction_types"]}
                    })
            
            # Construire le corps de la requête
            search_body = {
                "query": {
                    "bool": {
                        "must": must_clauses
                    }
                },
                "size": limit,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"date": {"order": "desc"}}
                ]
            }
            
            # Ajouter le highlighting si demandé
            if include_highlights:
                search_body["highlight"] = {
                    "fields": {
                        "searchable_text": {},
                        "primary_description": {},
                        "merchant_name": {},
                        "category_name": {}
                    },
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fragment_size": 150
                }
            
            # Exécuter la recherche
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Extraire les résultats
            hits = response.get("hits", {}).get("hits", [])
            return hits
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche Elasticsearch: {e}")
            return []
    
    async def delete_user_transactions(self, user_id: int) -> bool:
        """
        Supprime toutes les transactions d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si la suppression a réussi
        """
        if not self.client:
            return False
        
        try:
            # Supprimer par requête
            await self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"user_id": user_id}
                    }
                },
                conflicts="proceed"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            return False
    
    async def is_healthy(self) -> bool:
        """Vérifie l'état de santé du client."""
        if not self.client:
            return False
        
        try:
            # Ping le cluster
            return await self.client.ping()
        except:
            return False
    
    async def close(self):
        """Ferme la connexion Elasticsearch."""
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.info("Connexion Elasticsearch fermée")