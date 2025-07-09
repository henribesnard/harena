"""
Result Processor - Traitement et formatage des résultats Elasticsearch

Responsabilité : Transforme les réponses brutes Elasticsearch en réponses standardisées
SearchServiceResponse avec enrichissement, déduplication et optimisation.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

from ..models.service_contracts import (
    SearchServiceResponse, 
    SearchServiceQuery,
    ResponseMetadata,
    SearchResult,
    AggregationResult
)

logger = logging.getLogger(__name__)


class ResultProcessor:
    """
    Processeur de résultats Elasticsearch haute performance.
    
    Responsabilités:
    - Transformation réponses Elasticsearch → contrats standardisés
    - Enrichissement résultats avec métadonnées
    - Déduplication intelligente
    - Highlighting et scoring
    - Formatage agrégations financières
    - Optimisation performance traitement
    """
    
    def __init__(self):
        # Configuration traitement
        self.max_highlights_per_field = 3
        self.duplicate_threshold = 0.95
        self.score_boost_factors = {
            "exact_match": 2.0,
            "merchant_match": 1.5,
            "category_match": 1.3,
            "recent_transaction": 1.2
        }
        
        # Statistiques de traitement
        self.processing_stats = {
            "processed_results": 0,
            "deduplication_count": 0,
            "enrichment_count": 0,
            "total_processing_time": 0.0
        }
        
        logger.info("ResultProcessor initialisé")
    
    async def process_search_response(
        self,
        es_response: Dict[str, Any],
        query_contract: SearchServiceQuery,
        execution_time_ms: int
    ) -> SearchServiceResponse:
        """
        Traite une réponse Elasticsearch complète.
        
        Args:
            es_response: Réponse brute Elasticsearch
            query_contract: Contrat de requête original
            execution_time_ms: Temps d'exécution en ms
            
        Returns:
            SearchServiceResponse: Réponse standardisée et enrichie
        """
        start_time = datetime.now()
        
        try:
            # 1. Extraction des métadonnées de réponse
            response_metadata = self._extract_response_metadata(
                es_response, query_contract, execution_time_ms
            )
            
            # 2. Traitement des résultats principaux
            results = await self._process_search_hits(
                es_response.get("hits", {}), query_contract
            )
            
            # 3. Traitement des agrégations
            aggregations = None
            if es_response.get("aggregations"):
                aggregations = self._process_aggregations(
                    es_response["aggregations"], query_contract
                )
            
            # 4. Enrichissement contextuel
            if query_contract.options.include_explanation:
                results = self._enrich_with_explanations(results, es_response)
            
            # 5. Construction de la réponse finale
            response = SearchServiceResponse(
                response_metadata=response_metadata,
                results=results,
                aggregations=aggregations,
                performance=self._generate_performance_metadata(
                    es_response, execution_time_ms
                ),
                context_enrichment=self._generate_context_enrichment(
                    results, query_contract
                ),
                debug=self._generate_debug_info(
                    es_response, query_contract
                ) if query_contract.options.return_raw_elasticsearch else None
            )
            
            # 6. Mise à jour des statistiques
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_processing_stats(len(results), processing_time)
            
            logger.info(
                f"Réponse traitée: {len(results)} résultats, "
                f"{processing_time:.2f}ms traitement"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la réponse: {str(e)}")
            raise
    
    async def process_multi_search_response(
        self,
        es_responses: List[Dict[str, Any]],
        query_contracts: List[SearchServiceQuery],
        execution_times_ms: List[int]
    ) -> List[SearchServiceResponse]:
        """
        Traite une réponse multi-search Elasticsearch.
        
        Args:
            es_responses: Liste des réponses Elasticsearch
            query_contracts: Liste des contrats de requête
            execution_times_ms: Liste des temps d'exécution
            
        Returns:
            List[SearchServiceResponse]: Liste des réponses traitées
        """
        if len(es_responses) != len(query_contracts):
            raise ValueError("Nombre de réponses != nombre de contrats")
        
        processed_responses = []
        
        for es_response, query_contract, exec_time in zip(
            es_responses, query_contracts, execution_times_ms
        ):
            if "error" in es_response:
                # Gestion des erreurs individuelles
                error_response = self._create_error_response(
                    es_response["error"], query_contract, exec_time
                )
                processed_responses.append(error_response)
            else:
                processed_response = await self.process_search_response(
                    es_response, query_contract, exec_time
                )
                processed_responses.append(processed_response)
        
        return processed_responses
    
    def _extract_response_metadata(
        self,
        es_response: Dict[str, Any],
        query_contract: SearchServiceQuery,
        execution_time_ms: int
    ) -> ResponseMetadata:
        """
        Extrait les métadonnées de réponse depuis Elasticsearch.
        """
        hits_info = es_response.get("hits", {})
        total_info = hits_info.get("total", {})
        
        # Gestion des différentes versions d'Elasticsearch
        if isinstance(total_info, dict):
            total_hits = total_info.get("value", 0)
        else:
            total_hits = total_info or 0
        
        returned_hits = len(hits_info.get("hits", []))
        has_more = (
            query_contract.search_parameters.offset + returned_hits < total_hits
        )
        
        return ResponseMetadata(
            query_id=query_contract.query_metadata.query_id,
            execution_time_ms=execution_time_ms,
            total_hits=total_hits,
            returned_hits=returned_hits,
            has_more=has_more,
            cache_hit=False,  # Sera mis à jour par le cache manager
            elasticsearch_took=es_response.get("took", 0),
            agent_context={
                "requesting_agent": query_contract.query_metadata.agent_name,
                "requesting_team": query_contract.query_metadata.team_name,
                "next_suggested_agent": self._suggest_next_agent(query_contract, total_hits)
            } if hasattr(query_contract.query_metadata, 'agent_name') else None
        )
    
    async def _process_search_hits(
        self,
        hits_data: Dict[str, Any],
        query_contract: SearchServiceQuery
    ) -> List[SearchResult]:
        """
        Traite les résultats de recherche Elasticsearch.
        """
        raw_hits = hits_data.get("hits", [])
        if not raw_hits:
            return []
        
        results = []
        seen_transactions: Set[str] = set()
        
        for hit in raw_hits:
            # Extraction des données source
            source = hit.get("_source", {})
            
            # Déduplication basée sur transaction_id
            transaction_id = source.get("transaction_id")
            if transaction_id in seen_transactions:
                self.processing_stats["deduplication_count"] += 1
                continue
            seen_transactions.add(transaction_id)
            
            # Traitement du highlighting
            highlights = self._process_highlighting(
                hit.get("highlight", {}), query_contract
            )
            
            # Calcul du score enrichi
            enriched_score = self._calculate_enriched_score(
                hit.get("_score", 0.0), source, query_contract
            )
            
            # Construction du résultat standardisé
            result = SearchResult(
                transaction_id=transaction_id,
                user_id=source.get("user_id"),
                account_id=source.get("account_id"),
                amount=source.get("amount"),
                amount_abs=source.get("amount_abs"),
                transaction_type=source.get("transaction_type"),
                currency_code=source.get("currency_code"),
                date=source.get("date"),
                primary_description=source.get("primary_description"),
                merchant_name=source.get("merchant_name"),
                category_name=source.get("category_name"),
                operation_type=source.get("operation_type"),
                month_year=source.get("month_year"),
                weekday=source.get("weekday"),
                score=enriched_score,
                highlights=highlights if query_contract.options.include_highlights else None,
                # Champs additionnels selon la requête
                **self._extract_additional_fields(source, query_contract)
            )
            
            results.append(result)
        
        # Tri final par score enrichi
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _process_highlighting(
        self,
        highlight_data: Dict[str, Any],
        query_contract: SearchServiceQuery
    ) -> Optional[Dict[str, List[str]]]:
        """
        Traite les highlights Elasticsearch.
        """
        if not highlight_data or not query_contract.options.include_highlights:
            return None
        
        processed_highlights = {}
        
        for field, highlights in highlight_data.items():
            if isinstance(highlights, list):
                # Limiter le nombre de highlights par champ
                processed_highlights[field] = highlights[:self.max_highlights_per_field]
            else:
                processed_highlights[field] = [str(highlights)]
        
        return processed_highlights if processed_highlights else None
    
    def _calculate_enriched_score(
        self,
        base_score: float,
        source: Dict[str, Any],
        query_contract: SearchServiceQuery
    ) -> float:
        """
        Calcule un score enrichi basé sur des facteurs business.
        """
        score = base_score
        
        # Boost pour correspondance exacte merchant
        if (query_contract.filters.text_search and 
            query_contract.filters.text_search.get("query", "").lower() in 
            source.get("merchant_name", "").lower()):
            score *= self.score_boost_factors["merchant_match"]
        
        # Boost pour correspondance catégorie
        if any(f.field == "category_name" for f in query_contract.filters.required):
            score *= self.score_boost_factors["category_match"]
        
        # Boost pour transactions récentes (derniers 30 jours)
        transaction_date = source.get("date")
        if transaction_date:
            try:
                from datetime import datetime, timedelta
                date_obj = datetime.strptime(transaction_date, "%Y-%m-%d")
                days_ago = (datetime.now() - date_obj).days
                if days_ago <= 30:
                    score *= self.score_boost_factors["recent_transaction"]
            except:
                pass  # Ignore si format de date invalide
        
        return round(score, 4)
    
    def _extract_additional_fields(
        self,
        source: Dict[str, Any],
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Extrait les champs additionnels demandés dans la requête.
        """
        additional_fields = {}
        requested_fields = query_contract.search_parameters.fields or []
        
        # Champs standards déjà traités
        standard_fields = {
            "transaction_id", "user_id", "account_id", "amount", "amount_abs",
            "transaction_type", "currency_code", "date", "primary_description",
            "merchant_name", "category_name", "operation_type", "month_year", "weekday"
        }
        
        for field in requested_fields:
            if field not in standard_fields and field in source:
                additional_fields[field] = source[field]
        
        return additional_fields
    
    def _process_aggregations(
        self,
        aggs_data: Dict[str, Any],
        query_contract: SearchServiceQuery
    ) -> AggregationResult:
        """
        Traite les agrégations Elasticsearch en format financier.
        """
        aggregation_result = AggregationResult()
        
        # Extraction des métriques globales
        for metric_field in query_contract.aggregations.metrics or []:
            # Somme totale
            if f"total_{metric_field}" in aggs_data:
                total_value = aggs_data[f"total_{metric_field}"]["value"]
                if metric_field == "amount":
                    aggregation_result.total_amount = total_value
                elif metric_field == "amount_abs":
                    aggregation_result.total_amount_abs = total_value
            
            # Moyenne
            if f"avg_{metric_field}" in aggs_data:
                avg_value = aggs_data[f"avg_{metric_field}"]["value"]
                if metric_field == "amount":
                    aggregation_result.average_amount = avg_value
            
            # Min/Max
            if f"min_{metric_field}" in aggs_data:
                min_value = aggs_data[f"min_{metric_field}"]["value"]
                if not hasattr(aggregation_result, 'statistics'):
                    aggregation_result.statistics = {}
                aggregation_result.statistics[f"min_{metric_field}"] = min_value
            
            if f"max_{metric_field}" in aggs_data:
                max_value = aggs_data[f"max_{metric_field}"]["value"]
                if not hasattr(aggregation_result, 'statistics'):
                    aggregation_result.statistics = {}
                aggregation_result.statistics[f"max_{metric_field}"] = max_value
        
        # Comptage de transactions
        if "transaction_count" in aggs_data:
            aggregation_result.transaction_count = aggs_data["transaction_count"]["value"]
        elif any("count_" in key for key in aggs_data.keys()):
            # Prendre le premier comptage trouvé
            count_key = next(key for key in aggs_data.keys() if "count_" in key)
            aggregation_result.transaction_count = aggs_data[count_key]["value"]
        
        # Agrégations par groupement
        grouping_results = []
        
        for group_field in query_contract.aggregations.group_by or []:
            agg_key = f"by_{group_field}"
            if agg_key in aggs_data:
                buckets = aggs_data[agg_key].get("buckets", [])
                
                for bucket in buckets:
                    group_result = {
                        "key": bucket["key"],
                        "doc_count": bucket["doc_count"]
                    }
                    
                    # Ajout des sous-agrégations
                    for metric_field in query_contract.aggregations.metrics or []:
                        if f"total_{metric_field}" in bucket:
                            group_result[f"total_{metric_field}"] = bucket[f"total_{metric_field}"]["value"]
                        if f"avg_{metric_field}" in bucket:
                            group_result[f"avg_{metric_field}"] = bucket[f"avg_{metric_field}"]["value"]
                    
                    grouping_results.append(group_result)
        
        # Assignation selon le type de groupement
        if query_contract.aggregations.group_by:
            first_group = query_contract.aggregations.group_by[0]
            if first_group == "month_year":
                aggregation_result.by_month = grouping_results
            elif first_group == "category_name":
                aggregation_result.by_category = grouping_results
            elif first_group == "merchant_name":
                aggregation_result.by_merchant = grouping_results
            else:
                # Groupement générique
                if not hasattr(aggregation_result, 'custom_groupings'):
                    aggregation_result.custom_groupings = {}
                aggregation_result.custom_groupings[first_group] = grouping_results
        
        return aggregation_result
    
    def _enrich_with_explanations(
        self,
        results: List[SearchResult],
        es_response: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        Enrichit les résultats avec les explications de score Elasticsearch.
        """
        explanations = es_response.get("hits", {}).get("hits", [])
        
        for i, result in enumerate(results):
            if i < len(explanations) and "_explanation" in explanations[i]:
                explanation = explanations[i]["_explanation"]
                # Ajouter l'explication comme métadonnée
                if not hasattr(result, 'metadata'):
                    result.metadata = {}
                result.metadata['score_explanation'] = {
                    "value": explanation.get("value"),
                    "description": explanation.get("description")
                }
        
        return results
    
    def _generate_performance_metadata(
        self,
        es_response: Dict[str, Any],
        execution_time_ms: int
    ) -> Dict[str, Any]:
        """
        Génère les métadonnées de performance.
        """
        return {
            "query_complexity": self._assess_query_complexity(es_response),
            "optimization_applied": self._get_applied_optimizations(es_response),
            "index_used": "harena_transactions",  # Index principal
            "shards_queried": es_response.get("_shards", {}).get("total", 1),
            "elasticsearch_took": es_response.get("took", 0),
            "total_time_ms": execution_time_ms
        }
    
    def _generate_context_enrichment(
        self,
        results: List[SearchResult],
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Génère l'enrichissement contextuel pour les agents conversationnels.
        """
        if not results:
            return {
                "search_intent_matched": False,
                "result_quality_score": 0.0,
                "suggested_followup_questions": [
                    "Essayez une recherche plus large",
                    "Vérifiez les filtres appliqués"
                ]
            }
        
        # Calcul de la qualité des résultats
        avg_score = sum(r.score for r in results) / len(results)
        quality_score = min(avg_score / 10.0, 1.0)  # Normalisation sur 1.0
        
        # Génération de questions de suivi basées sur les résultats
        followup_questions = self._generate_followup_questions(results, query_contract)
        
        return {
            "search_intent_matched": quality_score > 0.5,
            "result_quality_score": round(quality_score, 2),
            "suggested_followup_questions": followup_questions,
            "results_summary": {
                "total_results": len(results),
                "avg_score": round(avg_score, 2),
                "top_categories": self._get_top_categories(results),
                "date_range": self._get_date_range(results)
            }
        }
    
    def _generate_debug_info(
        self,
        es_response: Dict[str, Any],
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Génère les informations de debug si demandées.
        """
        if not query_contract.options.return_raw_elasticsearch:
            return None
        
        return {
            "elasticsearch_query": "Query would be included here",  # Serait la requête originale
            "elasticsearch_response": {
                "took": es_response.get("took"),
                "timed_out": es_response.get("timed_out"),
                "shards": es_response.get("_shards"),
                "hits_total": es_response.get("hits", {}).get("total")
            },
            "query_explanation": self._explain_query_execution(es_response)
        }
    
    def _assess_query_complexity(self, es_response: Dict[str, Any]) -> str:
        """
        Évalue la complexité de la requête basée sur la réponse.
        """
        took_ms = es_response.get("took", 0)
        total_hits = es_response.get("hits", {}).get("total", {})
        
        if isinstance(total_hits, dict):
            hits_count = total_hits.get("value", 0)
        else:
            hits_count = total_hits or 0
        
        if took_ms < 10 and hits_count < 1000:
            return "simple"
        elif took_ms < 50 and hits_count < 10000:
            return "medium"
        else:
            return "complex"
    
    def _get_applied_optimizations(self, es_response: Dict[str, Any]) -> List[str]:
        """
        Identifie les optimisations appliquées basées sur la réponse.
        """
        optimizations = []
        
        # Détection des optimisations basées sur la structure de réponse
        if es_response.get("took", 0) < 20:
            optimizations.append("fast_execution")
        
        if "aggregations" in es_response:
            optimizations.append("aggregation_optimization")
        
        # Ça serait mieux avec accès à la requête originale
        optimizations.extend(["user_filter", "index_optimization"])
        
        return optimizations
    
    def _suggest_next_agent(
        self,
        query_contract: SearchServiceQuery,
        total_hits: int
    ) -> Optional[str]:
        """
        Suggère le prochain agent dans le pipeline conversationnel.
        """
        if not hasattr(query_contract.query_metadata, 'agent_name'):
            return None
        
        current_agent = query_contract.query_metadata.agent_name
        
        # Logique de suggestion basée sur l'agent actuel et les résultats
        if current_agent == "query_generator_agent":
            if total_hits > 0:
                return "response_generator_agent"
            else:
                return "intent_classifier_agent"  # Re-classification
        
        elif current_agent == "intent_classifier_agent":
            return "entity_extractor_agent"
        
        elif current_agent == "entity_extractor_agent":
            return "query_generator_agent"
        
        return None
    
    def _generate_followup_questions(
        self,
        results: List[SearchResult],
        query_contract: SearchServiceQuery
    ) -> List[str]:
        """
        Génère des questions de suivi basées sur les résultats.
        """
        if not results:
            return ["Essayez une recherche plus large", "Modifiez vos critères"]
        
        questions = []
        
        # Basé sur les catégories trouvées
        categories = set(r.category_name for r in results if r.category_name)
        if len(categories) > 1:
            questions.append("Filtrer par catégorie spécifique")
        
        # Basé sur les montants
        amounts = [r.amount_abs for r in results if r.amount_abs]
        if amounts:
            avg_amount = sum(amounts) / len(amounts)
            questions.append(f"Voir transactions > {avg_amount:.0f}€")
        
        # Basé sur les dates
        dates = set(r.month_year for r in results if r.month_year)
        if len(dates) > 1:
            questions.append("Analyser par période")
        
        # Questions génériques
        questions.extend([
            "Voir détails de ces transactions",
            "Comparer avec période précédente"
        ])
        
        return questions[:4]  # Limiter à 4 suggestions
    
    def _get_top_categories(self, results: List[SearchResult]) -> List[str]:
        """
        Extrait les top catégories des résultats.
        """
        from collections import Counter
        
        categories = [r.category_name for r in results if r.category_name]
        counter = Counter(categories)
        return [cat for cat, _ in counter.most_common(3)]
    
    def _get_date_range(self, results: List[SearchResult]) -> Optional[Dict[str, str]]:
        """
        Extrait la plage de dates des résultats.
        """
        dates = [r.date for r in results if r.date]
        if not dates:
            return None
        
        return {
            "start_date": min(dates),
            "end_date": max(dates)
        }
    
    def _explain_query_execution(self, es_response: Dict[str, Any]) -> str:
        """
        Explique l'exécution de la requête.
        """
        took = es_response.get("took", 0)
        hits = es_response.get("hits", {}).get("total", {})
        
        if isinstance(hits, dict):
            total_hits = hits.get("value", 0)
        else:
            total_hits = hits or 0
        
        return f"Requête exécutée en {took}ms, {total_hits} résultats trouvés"
    
    def _create_error_response(
        self,
        error_data: Dict[str, Any],
        query_contract: SearchServiceQuery,
        execution_time_ms: int
    ) -> SearchServiceResponse:
        """
        Crée une réponse d'erreur standardisée.
        """
        return SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id=query_contract.query_metadata.query_id,
                execution_time_ms=execution_time_ms,
                total_hits=0,
                returned_hits=0,
                has_more=False,
                cache_hit=False,
                elasticsearch_took=0,
                error=error_data
            ),
            results=[],
            aggregations=None,
            performance={"error": True, "error_details": error_data},
            context_enrichment={
                "search_intent_matched": False,
                "result_quality_score": 0.0,
                "suggested_followup_questions": ["Vérifiez votre requête", "Contactez le support"]
            }
        )
    
    def _update_processing_stats(self, result_count: int, processing_time_ms: float) -> None:
        """
        Met à jour les statistiques de traitement.
        """
        self.processing_stats["processed_results"] += result_count
        self.processing_stats["total_processing_time"] += processing_time_ms
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de traitement.
        """
        stats = self.processing_stats.copy()
        
        if stats["processed_results"] > 0:
            stats["avg_processing_time_per_result"] = (
                stats["total_processing_time"] / stats["processed_results"]
            )
        
        return stats
    
    def reset_stats(self) -> None:
        """
        Remet à zéro les statistiques de traitement.
        """
        self.processing_stats = {
            "processed_results": 0,
            "deduplication_count": 0,
            "enrichment_count": 0,
            "total_processing_time": 0.0
        }