import logging
import time
import json
from datetime import datetime
from typing import Dict, Any
from search_service.models.request import SearchRequest
from search_service.models.response import SearchResult
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)

class SearchEngine:
    """Moteur de recherche unifié utilisant le client Elasticsearch existant"""
    
    def __init__(self, elasticsearch_client=None):
        self.elasticsearch_client = elasticsearch_client
        self.query_builder = QueryBuilder()
        self.index_name = "harena_transactions"  # Basé sur votre config existante
    
    def set_elasticsearch_client(self, client):
        """Définit le client Elasticsearch à utiliser"""
        self.elasticsearch_client = client
        if hasattr(client, 'index_name'):
            self.index_name = client.index_name
    
    async def search(self, request: SearchRequest) -> Dict[str, Any]:
        """Execute a search and always return a structured response.

        This method now guarantees that even in error scenarios a predictable
        response structure is returned.  This structure contains a
        ``response_metadata`` block with default timings and counts so that
        higher level services (e.g. ``SearchServiceResponse``) can safely
        parse the output.
        """

        start_time = time.time()

        try:
            if not self.elasticsearch_client:
                raise RuntimeError("Client Elasticsearch non initialisé")

            # Construction requête Elasticsearch
            es_query = self.query_builder.build_query(request)

            logger.debug(
                f"Executing search for user {request.user_id} with query: '{request.query}'"
            )

            # Exécution via le client existant
            es_response = await self._execute_search(es_query, request)

            # Traitement des résultats
            results = self._process_results(es_response)

            # Calcul temps d'exécution
            execution_time = int((time.time() - start_time) * 1000)

            total_hits = self._get_total_hits(es_response)
            returned_hits = len(results)

            response = {
                "results": [r.model_dump() for r in results],
                "aggregations": es_response.get("aggregations"),
                "success": True,
                "error_message": None,
                "response_metadata": {
                    "query_id": (request.metadata or {}).get("query_id", "unknown"),
                    "response_timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": execution_time,
                    "total_results": total_hits,
                    "returned_results": returned_hits,
                    "has_more_results": total_hits > (returned_hits + request.offset),
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": es_response.get("took", 0),
                    "cache_hit": False,
                },
            }

            if request.metadata.get("debug"):
                response["response_metadata"]["debug_info"] = self._build_debug_info(
                    request, es_query
                )

            logger.info(
                f"Search completed: {returned_hits}/{total_hits} results in {execution_time}ms"
            )
            return response

        except Exception as e:
            logger.error(f"Search failed for user {request.user_id}: {str(e)}")
            execution_time = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "aggregations": None,
                "success": False,
                "error_message": str(e),
                "response_metadata": {
                    "query_id": (request.metadata or {}).get("query_id", "unknown"),
                    "response_timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": execution_time,
                    "total_results": 0,
                    "returned_results": 0,
                    "has_more_results": False,
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": 0,
                    "cache_hit": False,
                },
            }
    
    async def _execute_search(self, es_query: Dict[str, Any], request: SearchRequest) -> Dict[str, Any]:
        """Exécute la recherche via le client Elasticsearch"""
        
        # Utiliser la méthode search du client existant
        if hasattr(self.elasticsearch_client, 'search'):
            # Client avec méthode search directe
            response = await self.elasticsearch_client.search(
                index=self.index_name,
                body=es_query,
                size=request.limit,
                from_=request.offset
            )
        else:
            # Fallback: utiliser une requête HTTP directe
            search_url = f"/{self.index_name}/_search"
            
            # Ajouter les paramètres de pagination à la requête
            es_query["size"] = request.limit
            es_query["from"] = request.offset
            
            # Utiliser la session HTTP du client
            async with self.elasticsearch_client.session.post(
                f"{self.elasticsearch_client.base_url}{search_url}",
                json=es_query
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Elasticsearch error {resp.status}: {error_text}")
                response = await resp.json()
        
        return response
    
    def _process_results(self, es_response: Dict[str, Any]) -> list[SearchResult]:
        """
        Traite les résultats Elasticsearch en objets SearchResult
        VERSION CORRIGÉE - Robuste aux données manquantes/nulles
        """
        results = []
        
        hits = es_response.get('hits', {}).get('hits', [])
        logger.debug(f"Processing {len(hits)} hits from Elasticsearch")
        
        for i, hit in enumerate(hits):
            source = hit.get('_source', {})
            score = hit.get('_score')
            
            try:
                # ✅ CORRECTION : Gestion robuste de tous les champs avec valeurs par défaut sécurisées
                result = SearchResult(
                    # Champs obligatoires avec fallbacks robustes
                    transaction_id=str(source.get('transaction_id', f'tx_{i}_{int(time.time())}')),
                    user_id=int(source.get('user_id', 0)),
                    amount=float(source.get('amount', 0.0)),
                    amount_abs=float(source.get('amount_abs', abs(float(source.get('amount', 0.0))))),
                    currency_code=str(source.get('currency_code', 'EUR')),
                    
                    # ✅ CORRECTION CRITIQUE : Gérer les champs obligatoires qui peuvent être vides
                    transaction_type=str(source.get('transaction_type', 'unknown')),
                    date=str(source.get('date', '')),
                    primary_description=str(source.get('primary_description', 'Description non disponible')),
                    
                    # Champs optionnels - gestion explicite des None
                    account_id=source.get('account_id'),  # Peut être None
                    month_year=source.get('month_year'),  # Peut être None
                    weekday=source.get('weekday'),        # Peut être None
                    merchant_name=source.get('merchant_name'),      # Peut être None ou ""
                    category_name=source.get('category_name'),      # Peut être None ou ""
                    operation_type=source.get('operation_type'),    # Peut être None ou ""
                    
                    # Métadonnées de recherche
                    score=float(score) if score is not None else 0.0,
                    highlights=hit.get('highlight')
                )
                results.append(result)
                
                # Log de succès pour debug
                logger.debug(f"✅ Successfully processed result {i+1}: {result.transaction_id} - {result.primary_description[:50]}")
                
            except ValueError as ve:
                # Erreur de conversion de type (int, float)
                logger.error(f"❌ ValueError processing search result {i+1}: {str(ve)}")
                logger.error(f"   Problematic source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}")
                
                # Essayer de créer un résultat minimal avec des types corrects
                try:
                    minimal_result = SearchResult(
                        transaction_id=str(source.get('transaction_id', f'error_tx_{i}')),
                        user_id=int(source.get('user_id', 0)) if source.get('user_id') is not None else 0,
                        amount=0.0,
                        amount_abs=0.0,
                        currency_code='EUR',
                        transaction_type='error',
                        date='',
                        primary_description=f'Erreur conversion: {str(ve)[:100]}',
                        score=0.0
                    )
                    results.append(minimal_result)
                    logger.warning(f"⚠️ Created minimal result for failed conversion {i+1}")
                except Exception as e2:
                    logger.error(f"❌ Failed to create minimal result: {str(e2)}")
                    continue
                    
            except Exception as e:
                # Autres erreurs (validation Pydantic, etc.)
                logger.error(f"❌ General error processing search result {i+1}: {str(e)}")
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}")
                
                # Log des champs spécifiques pour debug
                logger.error(f"   transaction_id: {repr(source.get('transaction_id'))}")
                logger.error(f"   user_id: {repr(source.get('user_id'))}")
                logger.error(f"   transaction_type: {repr(source.get('transaction_type'))}")
                logger.error(f"   primary_description: {repr(source.get('primary_description'))}")
                logger.error(f"   amount: {repr(source.get('amount'))}")
                
                continue
        
        success_count = len(results)
        total_count = len(hits)
        
        if success_count < total_count:
            logger.warning(f"⚠️ Processed only {success_count}/{total_count} results due to errors")
        else:
            logger.info(f"✅ Successfully processed all {success_count}/{total_count} Elasticsearch hits")
        
        return results
    
    def _get_total_hits(self, es_response: Dict[str, Any]) -> int:
        """Extrait le nombre total de résultats"""
        hits = es_response.get('hits', {})
        total = hits.get('total', 0)
        
        # Gestion des différents formats de total d'Elasticsearch
        if isinstance(total, dict):
            return total.get('value', 0)
        else:
            return int(total)
    
    def _build_debug_info(self, request: SearchRequest, es_query: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les informations de debug"""
        return {
            "original_request": {
                "user_id": request.user_id,
                "query": request.query,
                "filters": request.filters,
                "limit": request.limit,
                "offset": request.offset
            },
            "elasticsearch_query": es_query,
            "index_used": self.index_name
        }
    
    async def count(self, request: SearchRequest) -> int:
        """Compte le nombre total de résultats sans les récupérer"""
        if not self.elasticsearch_client:
            raise RuntimeError("Client Elasticsearch non initialisé")
        
        try:
            # Construction requête de comptage
            count_query = {
                "query": self.query_builder.build_query(request)["query"]
            }
            
            # Exécution via le client
            if hasattr(self.elasticsearch_client, 'count'):
                response = await self.elasticsearch_client.count(
                    index=self.index_name,
                    body=count_query
                )
                return response.get('count', 0)
            else:
                # Fallback: utiliser _count endpoint
                count_url = f"/{self.index_name}/_count"
                async with self.elasticsearch_client.session.post(
                    f"{self.elasticsearch_client.base_url}{count_url}",
                    json=count_query
                ) as resp:
                    if resp.status != 200:
                        return 0
                    response = await resp.json()
                    return response.get('count', 0)
                    
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any
from search_service.models.request import SearchRequest
from search_service.models.response import SearchResult
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)

class SearchEngine:
    """Moteur de recherche unifié utilisant le client Elasticsearch existant"""
    
    def __init__(self, elasticsearch_client=None):
        self.elasticsearch_client = elasticsearch_client
        self.query_builder = QueryBuilder()
        self.index_name = "harena_transactions"  # Basé sur votre config existante
    
    def set_elasticsearch_client(self, client):
        """Définit le client Elasticsearch à utiliser"""
        self.elasticsearch_client = client
        if hasattr(client, 'index_name'):
            self.index_name = client.index_name
    
    async def search(self, request: SearchRequest) -> Dict[str, Any]:
        """Execute a search and always return a structured response.

        This method now guarantees that even in error scenarios a predictable
        response structure is returned.  This structure contains a
        ``response_metadata`` block with default timings and counts so that
        higher level services (e.g. ``SearchServiceResponse``) can safely
        parse the output.
        """

        start_time = time.time()

        try:
            if not self.elasticsearch_client:
                raise RuntimeError("Client Elasticsearch non initialisé")

            # Construction requête Elasticsearch
            es_query = self.query_builder.build_query(request)

            logger.debug(
                f"Executing search for user {request.user_id} with query: '{request.query}'"
            )

            # Exécution via le client existant
            es_response = await self._execute_search(es_query, request)

            # Traitement des résultats
            results = self._process_results(es_response)

            # Calcul temps d'exécution
            execution_time = int((time.time() - start_time) * 1000)

            total_hits = self._get_total_hits(es_response)
            returned_hits = len(results)

            response = {
                "results": [r.model_dump() for r in results],
                "aggregations": es_response.get("aggregations"),
                "success": True,
                "error_message": None,
                "response_metadata": {
                    "query_id": (request.metadata or {}).get("query_id", "unknown"),
                    "response_timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": execution_time,
                    "total_results": total_hits,
                    "returned_results": returned_hits,
                    "has_more_results": total_hits > (returned_hits + request.offset),
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": es_response.get("took", 0),
                    "cache_hit": False,
                },
            }

            if request.metadata.get("debug"):
                response["response_metadata"]["debug_info"] = self._build_debug_info(
                    request, es_query
                )

            logger.info(
                f"Search completed: {returned_hits}/{total_hits} results in {execution_time}ms"
            )
            return response

        except Exception as e:
            logger.error(f"Search failed for user {request.user_id}: {str(e)}")
            execution_time = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "aggregations": None,
                "success": False,
                "error_message": str(e),
                "response_metadata": {
                    "query_id": (request.metadata or {}).get("query_id", "unknown"),
                    "response_timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": execution_time,
                    "total_results": 0,
                    "returned_results": 0,
                    "has_more_results": False,
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": 0,
                    "cache_hit": False,
                },
            }
    
    async def _execute_search(self, es_query: Dict[str, Any], request: SearchRequest) -> Dict[str, Any]:
        """Exécute la recherche via le client Elasticsearch"""
        
        # Utiliser la méthode search du client existant
        if hasattr(self.elasticsearch_client, 'search'):
            # Client avec méthode search directe
            response = await self.elasticsearch_client.search(
                index=self.index_name,
                body=es_query,
                size=request.limit,
                from_=request.offset
            )
        else:
            # Fallback: utiliser une requête HTTP directe
            search_url = f"/{self.index_name}/_search"
            
            # Ajouter les paramètres de pagination à la requête
            es_query["size"] = request.limit
            es_query["from"] = request.offset
            
            # Utiliser la session HTTP du client
            async with self.elasticsearch_client.session.post(
                f"{self.elasticsearch_client.base_url}{search_url}",
                json=es_query
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Elasticsearch error {resp.status}: {error_text}")
                response = await resp.json()
        
        return response
    
    def _process_results(self, es_response: Dict[str, Any]) -> list[SearchResult]:
        """
        Traite les résultats Elasticsearch en objets SearchResult
        VERSION CORRIGÉE - Robuste aux données manquantes/nulles
        """
        results = []
        
        hits = es_response.get('hits', {}).get('hits', [])
        logger.debug(f"Processing {len(hits)} hits from Elasticsearch")
        
        for i, hit in enumerate(hits):
            source = hit.get('_source', {})
            score = hit.get('_score')
            
            try:
                # ✅ CORRECTION : Gestion robuste de tous les champs avec valeurs par défaut sécurisées
                result = SearchResult(
                    # Champs obligatoires avec fallbacks robustes
                    transaction_id=str(source.get('transaction_id', f'tx_{i}_{int(time.time())}')),
                    user_id=int(source.get('user_id', 0)),
                    amount=float(source.get('amount', 0.0)),
                    amount_abs=float(source.get('amount_abs', abs(float(source.get('amount', 0.0))))),
                    currency_code=str(source.get('currency_code', 'EUR')),
                    
                    # ✅ CORRECTION CRITIQUE : Gérer les champs obligatoires qui peuvent être vides
                    transaction_type=str(source.get('transaction_type', 'unknown')),
                    date=str(source.get('date', '')),
                    primary_description=str(source.get('primary_description', 'Description non disponible')),
                    
                    # Champs optionnels - gestion explicite des None
                    account_id=source.get('account_id'),  # Peut être None
                    month_year=source.get('month_year'),  # Peut être None
                    weekday=source.get('weekday'),        # Peut être None
                    merchant_name=source.get('merchant_name'),      # Peut être None ou ""
                    category_name=source.get('category_name'),      # Peut être None ou ""
                    operation_type=source.get('operation_type'),    # Peut être None ou ""
                    
                    # Métadonnées de recherche
                    score=float(score) if score is not None else 0.0,
                    highlights=hit.get('highlight')
                )
                results.append(result)
                
                # Log de succès pour debug
                logger.debug(f"✅ Successfully processed result {i+1}: {result.transaction_id} - {result.primary_description[:50]}")
                
            except ValueError as ve:
                # Erreur de conversion de type (int, float)
                logger.error(f"❌ ValueError processing search result {i+1}: {str(ve)}")
                logger.error(f"   Problematic source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}")
                
                # Essayer de créer un résultat minimal avec des types corrects
                try:
                    minimal_result = SearchResult(
                        transaction_id=str(source.get('transaction_id', f'error_tx_{i}')),
                        user_id=int(source.get('user_id', 0)) if source.get('user_id') is not None else 0,
                        amount=0.0,
                        amount_abs=0.0,
                        currency_code='EUR',
                        transaction_type='error',
                        date='',
                        primary_description=f'Erreur conversion: {str(ve)[:100]}',
                        score=0.0
                    )
                    results.append(minimal_result)
                    logger.warning(f"⚠️ Created minimal result for failed conversion {i+1}")
                except Exception as e2:
                    logger.error(f"❌ Failed to create minimal result: {str(e2)}")
                    continue
                    
            except Exception as e:
                # Autres erreurs (validation Pydantic, etc.)
                logger.error(f"❌ General error processing search result {i+1}: {str(e)}")
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}")
                
                # Log des champs spécifiques pour debug
                logger.error(f"   transaction_id: {repr(source.get('transaction_id'))}")
                logger.error(f"   user_id: {repr(source.get('user_id'))}")
                logger.error(f"   transaction_type: {repr(source.get('transaction_type'))}")
                logger.error(f"   primary_description: {repr(source.get('primary_description'))}")
                logger.error(f"   amount: {repr(source.get('amount'))}")
                
                continue
        
        success_count = len(results)
        total_count = len(hits)
        
        if success_count < total_count:
            logger.warning(f"⚠️ Processed only {success_count}/{total_count} results due to errors")
        else:
            logger.info(f"✅ Successfully processed all {success_count}/{total_count} Elasticsearch hits")
        
        return results
    
    def _get_total_hits(self, es_response: Dict[str, Any]) -> int:
        """Extrait le nombre total de résultats"""
        hits = es_response.get('hits', {})
        total = hits.get('total', 0)
        
        # Gestion des différents formats de total d'Elasticsearch
        if isinstance(total, dict):
            return total.get('value', 0)
        else:
            return int(total)
    
    def _build_debug_info(self, request: SearchRequest, es_query: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les informations de debug"""
        return {
            "original_request": {
                "user_id": request.user_id,
                "query": request.query,
                "filters": request.filters,
                "limit": request.limit,
                "offset": request.offset
            },
            "elasticsearch_query": es_query,
            "index_used": self.index_name
        }
    
    async def count(self, request: SearchRequest) -> int:
        """Compte le nombre total de résultats sans les récupérer"""
        if not self.elasticsearch_client:
            raise RuntimeError("Client Elasticsearch non initialisé")
        
        try:
            # Construction requête de comptage
            count_query = {
                "query": self.query_builder.build_query(request)["query"]
            }
            
            # Exécution via le client
            if hasattr(self.elasticsearch_client, 'count'):
                response = await self.elasticsearch_client.count(
                    index=self.index_name,
                    body=count_query
                )
                return response.get('count', 0)
            else:
                # Fallback: utiliser _count endpoint
                count_url = f"/{self.index_name}/_count"
                async with self.elasticsearch_client.session.post(
                    f"{self.elasticsearch_client.base_url}{count_url}",
                    json=count_query
                ) as resp:
                    if resp.status != 200:
                        return 0
                    response = await resp.json()
                    return response.get('count', 0)
                    
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0
