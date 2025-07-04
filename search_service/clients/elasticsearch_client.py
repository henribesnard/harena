"""
Client Elasticsearch/Bonsai pour le service de recherche.

Ce module fournit une interface optimisée pour interagir avec
Elasticsearch/Bonsai pour les recherches lexicales de transactions financières.
"""
import logging
import ssl
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import aiohttp

from search_service.clients.base_client import BaseClient, RetryConfig, CircuitBreakerConfig, HealthCheckConfig
from search_service.models.search_types import FINANCIAL_SYNONYMS

logger = logging.getLogger(__name__)


class ElasticsearchClient(BaseClient):
    """
    Client pour Elasticsearch/Bonsai optimisé pour les transactions financières.
    
    Responsabilités:
    - Recherches lexicales optimisées
    - Construction de requêtes complexes
    - Gestion des filtres et agrégations
    - Highlighting des résultats
    - Monitoring des performances
    """
    
    def __init__(
        self,
        bonsai_url: str,
        index_name: str = "harena_transactions",
        timeout: float = 5.0,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        # Configuration SSL pour Bonsai
        health_check_config = HealthCheckConfig(
            enabled=True,
            interval_seconds=30.0,
            timeout_seconds=3.0,
            endpoint="/"
        )
        
        super().__init__(
            base_url=bonsai_url,
            service_name="elasticsearch",
            timeout=timeout,
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            health_check_config=health_check_config
        )
        
        self.index_name = index_name
        self.ssl_context = ssl.create_default_context()
        
        # Cache des requêtes fréquentes
        self._query_cache: Dict[str, Dict] = {}
        
        logger.info(f"Elasticsearch client initialized for index: {index_name}")
    
    async def start(self):
        """Démarre le client avec configuration SSL pour Bonsai."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self.headers
            )
            logger.info(f"{self.service_name} client started with SSL")
    
    async def test_connection(self) -> bool:
        """Teste la connectivité de base à Elasticsearch."""
        try:
            async def _test():
                async with self.session.get(self.base_url) as response:
                    return response.status == 200
            
            return await self.execute_with_retry(_test, "connection_test")
        except Exception as e:
            logger.error(f"Elasticsearch connection test failed: {e}")
            return False
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Effectue une vérification de santé spécifique à Elasticsearch."""
        try:
            # Vérifier le cluster
            async with self.session.get(self.base_url) as response:
                if response.status == 200:
                    cluster_info = await response.json()
                    
                    # Vérifier l'existence de l'index
                    index_exists = await self._check_index_exists()
                    
                    return {
                        "cluster_name": cluster_info.get("cluster_name", "unknown"),
                        "version": cluster_info.get("version", {}).get("number", "unknown"),
                        "index_exists": index_exists,
                        "status": "healthy" if index_exists else "degraded"
                    }
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_index_exists(self) -> bool:
        """Vérifie si l'index existe."""
        try:
            async with self.session.head(f"{self.base_url}/{self.index_name}") as response:
                return response.status == 200
        except Exception:
            return False
    
    # ============================================================================
    # MÉTHODES MANQUANTES AJOUTÉES POUR COMPATIBILITÉ AVEC LES MOTEURS
    # ============================================================================
    
    async def search(
        self,
        index: str,
        body: Dict[str, Any],
        size: int = 20,
        from_: int = 0
    ) -> Dict[str, Any]:
        """
        Effectue une recherche Elasticsearch (méthode générique).
        
        Cette méthode est utilisée par le lexical_engine.py et doit être
        compatible avec l'interface attendue.
        
        Args:
            index: Nom de l'index
            body: Corps de la requête Elasticsearch
            size: Nombre de résultats
            from_: Offset pour pagination
            
        Returns:
            Résultats de la recherche
        """
        # Ajouter size et from_ au body si pas déjà présents
        if "size" not in body:
            body["size"] = size
        if "from" not in body:
            body["from"] = from_
        
        async def _search():
            async with self.session.post(
                f"{self.base_url}/{index}/_search",
                json=body
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Search failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_search, "search")
    
    async def health(self) -> Dict[str, Any]:
        """
        Vérifie la santé d'Elasticsearch (méthode générique).
        
        Cette méthode est utilisée par les health checks et doit retourner
        un format standard.
        
        Returns:
            Statut de santé du cluster
        """
        async def _health():
            async with self.session.get(
                f"{self.base_url}/_cluster/health"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Health check failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_health, "health")
    
    async def count(
        self,
        index: str,
        body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compte les documents correspondant à une requête (méthode générique).
        
        Cette méthode est utilisée par les moteurs pour compter les documents.
        
        Args:
            index: Nom de l'index
            body: Corps de la requête de comptage
            
        Returns:
            Résultat du comptage
        """
        async def _count():
            async with self.session.post(
                f"{self.base_url}/{index}/_count",
                json=body
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Count failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_count, "count")
    
    # ============================================================================
    # MÉTHODES SPÉCIALISÉES EXISTANTES
    # ============================================================================
    
    async def search_transactions(
        self,
        query: str,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        include_highlights: bool = True,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Recherche de transactions avec requête optimisée.
        
        Args:
            query: Terme de recherche
            user_id: ID de l'utilisateur
            limit: Nombre de résultats
            offset: Décalage pour pagination
            filters: Filtres additionnels
            include_highlights: Inclure le highlighting
            explain: Inclure l'explication du scoring
            
        Returns:
            Résultats de recherche Elasticsearch
        """
        search_body = self._build_optimized_search_query(
            query, user_id, limit, offset, filters, include_highlights, explain
        )
        
        async def _search():
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_search",
                json=search_body
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Search failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_search, "transaction_search")
    
    def _build_optimized_search_query(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        filters: Optional[Dict[str, Any]] = None,
        include_highlights: bool = True,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch optimisée basée sur les résultats du validateur.
        
        Le validateur montre que:
        - Les requêtes fonctionnent mais la pertinence peut être améliorée
        - Les scores max sont corrects (139.33 pour virement, 96.30 pour carte)
        - Certaines requêtes ne retournent aucun résultat (essence, pharmacie)
        """
        expanded_query = self._expand_financial_query(query)
        query_words = query.lower().split()
        
        # Requête multi-stratégie optimisée
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}}
                    ],
                    "should": [
                        # 1. Correspondance exacte de phrase (boost très élevé)
                        {
                            "match_phrase": {
                                "searchable_text": {
                                    "query": query,
                                    "boost": 10.0  # Augmenté de 8.0 à 10.0
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "primary_description": {
                                    "query": query,
                                    "boost": 8.0  # Augmenté de 6.0 à 8.0
                                }
                            }
                        },
                        
                        # 2. Correspondance dans merchant_name (critique pour les marchands)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["merchant_name^6.0", "merchant_name.keyword^8.0"],  # Augmenté
                                "type": "best_fields",
                                "boost": 5.0  # Augmenté de 4.0 à 5.0
                            }
                        },
                        
                        # 3. Correspondance multi-champs avec fuzziness améliorée
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "searchable_text^4.0",      # Augmenté de 3.0 à 4.0
                                    "primary_description^3.0",  # Augmenté de 2.5 à 3.0
                                    "clean_description^2.5",    # Augmenté de 2.0 à 2.5
                                    "provider_description^2.0", # Augmenté de 1.5 à 2.0
                                    "merchant_name^3.5"         # Ajouté
                                ],
                                "type": "best_fields",
                                "operator": "or",
                                "fuzziness": "AUTO",
                                "boost": 4.0  # Augmenté de 3.0 à 4.0
                            }
                        },
                        
                        # 4. Correspondance avec requête étendue (synonymes)
                        {
                            "multi_match": {
                                "query": expanded_query,
                                "fields": [
                                    "searchable_text^2.5",      # Augmenté de 2.0 à 2.5
                                    "primary_description^2.0",  # Augmenté de 1.5 à 2.0
                                    "merchant_name^2.5"         # Ajouté
                                ],
                                "type": "cross_fields",
                                "operator": "or",
                                "boost": 2.5  # Augmenté de 2.0 à 2.5
                            }
                        },
                        
                        # 5. Correspondance partielle avec wildcards (améliorée)
                        {
                            "bool": {
                                "should": [
                                    {
                                        "wildcard": {
                                            "searchable_text": {
                                                "value": f"*{word}*",
                                                "boost": 2.0  # Augmenté de 1.5 à 2.0
                                            }
                                        }
                                    } for word in query_words if len(word) > 2  # Changé de 3 à 2
                                ] + [
                                    {
                                        "wildcard": {
                                            "merchant_name": {
                                                "value": f"*{word}*",
                                                "boost": 2.5  # Nouveau
                                            }
                                        }
                                    } for word in query_words if len(word) > 2
                                ]
                            }
                        },
                        
                        # 6. Correspondance simple pour fallback
                        {
                            "simple_query_string": {
                                "query": query,
                                "fields": [
                                    "searchable_text^1.5",      # Augmenté de 1.0 à 1.5
                                    "primary_description^1.2", 
                                    "merchant_name^1.5"
                                ],
                                "default_operator": "or",
                                "boost": 1.5  # Augmenté de 1.0 à 1.5
                            }
                        },
                        
                        # 7. NOUVEAU: Correspondance fuzzy pour capturer les variantes
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "searchable_text^1.5",
                                    "primary_description^1.2",
                                    "merchant_name^1.8"
                                ],
                                "type": "most_fields",
                                "fuzziness": "AUTO",
                                "prefix_length": 1,
                                "boost": 1.8
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "from": offset,
            "sort": [
                {"_score": {"order": "desc"}},
                {"transaction_date": {"order": "desc", "unmapped_type": "date"}}
            ],
            "_source": [
                "transaction_id", "primary_description", "merchant_name",
                "amount", "transaction_date", "searchable_text", 
                "category_id", "user_id", "account_id", "transaction_type",
                "currency_code", "operation_type"
            ]
        }
        
        # Ajouter les filtres si fournis
        if filters:
            self._apply_filters(search_query, filters)
        
        # Ajouter le highlighting si demandé
        if include_highlights:
            search_query["highlight"] = {
                "fields": {
                    "searchable_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    },
                    "primary_description": {
                        "fragment_size": 100,
                        "number_of_fragments": 2,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    },
                    "merchant_name": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            }
        
        # Ajouter l'explication si demandée
        if explain:
            search_query["explain"] = True
        
        return search_query
    
    def _expand_financial_query(self, query: str) -> str:
        """Expand les requêtes avec des synonymes financiers."""
        query_lower = query.lower()
        expanded_terms = [query]
        
        for term, synonyms in FINANCIAL_SYNONYMS.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        return " ".join(set(expanded_terms))
    
    def _apply_filters(self, search_query: Dict[str, Any], filters: Dict[str, Any]):
        """Applique les filtres à la requête de recherche."""
        bool_query = search_query["query"]["bool"]
        
        if "must" not in bool_query:
            bool_query["must"] = []
        
        # Filtre de montant
        if "amount_min" in filters or "amount_max" in filters:
            amount_filter = {"range": {"amount": {}}}
            if "amount_min" in filters:
                amount_filter["range"]["amount"]["gte"] = filters["amount_min"]
            if "amount_max" in filters:
                amount_filter["range"]["amount"]["lte"] = filters["amount_max"]
            bool_query["must"].append(amount_filter)
        
        # Filtre de date
        if "date_from" in filters or "date_to" in filters:
            date_filter = {"range": {"transaction_date": {}}}
            if "date_from" in filters:
                date_filter["range"]["transaction_date"]["gte"] = filters["date_from"]
            if "date_to" in filters:
                date_filter["range"]["transaction_date"]["lte"] = filters["date_to"]
            bool_query["must"].append(date_filter)
        
        # Filtre de catégories
        if "category_ids" in filters and filters["category_ids"]:
            bool_query["must"].append({"terms": {"category_id": filters["category_ids"]}})
        
        # Filtre de comptes
        if "account_ids" in filters and filters["account_ids"]:
            bool_query["must"].append({"terms": {"account_id": filters["account_ids"]}})
        
        # Filtre de type de transaction
        if "transaction_type" in filters and filters["transaction_type"] != "all":
            bool_query["must"].append({"term": {"transaction_type": filters["transaction_type"]}})
    
    async def get_suggestions(
        self,
        partial_query: str,
        user_id: int,
        max_suggestions: int = 10
    ) -> Dict[str, Any]:
        """
        Obtient des suggestions d'auto-complétion.
        
        Args:
            partial_query: Début de requête
            user_id: ID de l'utilisateur
            max_suggestions: Nombre max de suggestions
            
        Returns:
            Suggestions groupées par type
        """
        suggestions_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}}
                    ],
                    "should": [
                        {
                            "prefix": {
                                "merchant_name.keyword": {
                                    "value": partial_query,
                                    "boost": 3.0
                                }
                            }
                        },
                        {
                            "prefix": {
                                "primary_description": {
                                    "value": partial_query,
                                    "boost": 2.0
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "merchants": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": max_suggestions,
                        "include": f"{partial_query}.*"
                    }
                },
                "descriptions": {
                    "terms": {
                        "field": "primary_description.keyword",
                        "size": max_suggestions,
                        "include": f".*{partial_query}.*"
                    }
                }
            },
            "size": 0
        }
        
        async def _get_suggestions():
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_search",
                json=suggestions_query
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Suggestions failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_get_suggestions, "suggestions")
    
    async def count_transactions(
        self,
        user_id: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Compte le nombre de transactions correspondant aux critères.
        
        Args:
            user_id: ID de l'utilisateur
            filters: Filtres additionnels
            
        Returns:
            Nombre de transactions
        """
        count_query = {
            "query": {
                "bool": {
                    "must": [{"term": {"user_id": user_id}}]
                }
            }
        }
        
        if filters:
            self._apply_filters(count_query, filters)
        
        async def _count():
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_count",
                json=count_query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("count", 0)
                else:
                    error_text = await response.text()
                    raise Exception(f"Count failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_count, "count_transactions")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Obtient les statistiques de l'index.
        
        Returns:
            Statistiques de l'index
        """
        async def _get_stats():
            async with self.session.get(
                f"{self.base_url}/{self.index_name}/_stats"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Stats failed: HTTP {response.status} - {error_text}")
        
        return await self.execute_with_retry(_get_stats, "index_stats")