import logging
import time
from typing import Dict, Any
from search_service.models.request import SearchRequest
from search_service.models.response import SearchResponse, SearchResult
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
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Recherche unifiée gérant tous les cas d'usage
        Utilise le client Elasticsearch existant de votre architecture
        """
        if not self.elasticsearch_client:
            raise RuntimeError("Client Elasticsearch non initialisé")
        
        start_time = time.time()
        
        try:
            # Construction requête Elasticsearch
            es_query = self.query_builder.build_query(request)
            
            logger.debug(f"Executing search for user {request.user_id} with query: '{request.query}'")
            
            # Exécution via le client existant
            es_response = await self._execute_search(es_query, request)
            
            # Traitement des résultats
            results = self._process_results(es_response)
            
            # Calcul temps d'exécution
            execution_time = int((time.time() - start_time) * 1000)
            
            response = SearchResponse(
                results=results,
                total_hits=self._get_total_hits(es_response),
                returned_hits=len(results),
                execution_time_ms=execution_time,
                elasticsearch_took=es_response.get('took', 0),
                cache_hit=False,  # TODO: implémenter cache si nécessaire
                query_info=self._build_debug_info(request, es_query) if request.metadata.get('debug') else None
            )
            
            logger.info(f"Search completed: {response.returned_hits}/{response.total_hits} results in {execution_time}ms")
            return response
            
        except Exception as e:
            logger.error(f"Search failed for user {request.user_id}: {str(e)}")
            # Retourner une réponse vide en cas d'erreur
            execution_time = int((time.time() - start_time) * 1000)
            return SearchResponse(
                results=[],
                total_hits=0,
                returned_hits=0,
                execution_time_ms=execution_time,
                query_info={"error": str(e)} if request.metadata.get('debug') else None
            )
    
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
        """Traite les résultats Elasticsearch en objets SearchResult"""
        results = []
        
        hits = es_response.get('hits', {}).get('hits', [])
        
        for hit in hits:
            source = hit.get('_source', {})
            score = hit.get('_score')
            
            try:
                result = SearchResult(
                    transaction_id=source.get('transaction_id', ''),
                    user_id=source.get('user_id', 0),
                    account_id=source.get('account_id'),
                    amount=source.get('amount', 0.0),
                    amount_abs=source.get('amount_abs', 0.0),
                    currency_code=source.get('currency_code', 'EUR'),
                    transaction_type=source.get('transaction_type', ''),
                    date=source.get('date', ''),
                    month_year=source.get('month_year'),
                    weekday=source.get('weekday'),
                    primary_description=source.get('primary_description', ''),
                    merchant_name=source.get('merchant_name'),
                    category_name=source.get('category_name'),
                    operation_type=source.get('operation_type'),
                    score=score,
                    highlights=hit.get('highlight')
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
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