"""
Moteur de recherche lexicale pour Elasticsearch/Bonsai.

Ce module implémente la recherche lexicale optimisée pour les transactions
financières, basé sur les résultats du validateur harena_search_validator.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from search_service.clients.elasticsearch_client import ElasticsearchClient
from search_service.core.query_processor import QueryProcessor, QueryAnalysis
from search_service.models.search_types import SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem

logger = logging.getLogger(__name__)


@dataclass
class LexicalSearchConfig:
    """Configuration pour la recherche lexicale."""
    max_results: int = 50
    min_score: float = 1.0
    highlight_enabled: bool = True
    fuzzy_enabled: bool = True
    synonym_expansion: bool = True
    boost_merchant_field: float = 5.0
    boost_description_field: float = 3.0
    boost_exact_phrase: float = 10.0


@dataclass
class LexicalSearchResult:
    """Résultat d'une recherche lexicale."""
    results: List[SearchResultItem]
    total_found: int
    max_score: float
    avg_score: float
    processing_time_ms: float
    quality: SearchQuality
    query_used: str
    debug_info: Optional[Dict[str, Any]] = None


class LexicalSearchEngine:
    """
    Moteur de recherche lexicale optimisé pour Elasticsearch.
    
    Implémente les optimisations identifiées par le validateur:
    - Requêtes multi-stratégies avec boost améliorés
    - Correspondances exactes privilégiées
    - Champs merchant_name avec boost élevé
    - Gestion des wildcards et fuzzy search
    - Highlighting optimisé
    """
    
    def __init__(
        self,
        elasticsearch_client: ElasticsearchClient,
        query_processor: QueryProcessor,
        config: Optional[LexicalSearchConfig] = None
    ):
        self.es_client = elasticsearch_client
        self.query_processor = query_processor
        self.config = config or LexicalSearchConfig()
        
        # Métriques
        self.search_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        logger.info("Lexical search engine initialized")
    
    async def search(
        self,
        query: str,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        sort_order: SortOrder = SortOrder.RELEVANCE,
        filters: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> LexicalSearchResult:
        """
        Effectue une recherche lexicale optimisée.
        
        Args:
            query: Terme de recherche
            user_id: ID de l'utilisateur
            limit: Nombre de résultats
            offset: Décalage pour pagination
            sort_order: Ordre de tri
            filters: Filtres additionnels
            debug: Inclure informations de debug
            
        Returns:
            Résultats de recherche lexicale
        """
        start_time = time.time()
        self.search_count += 1
        
        try:
            # 1. Analyser et traiter la requête
            query_analysis = self.query_processor.process_query(query)
            optimized_query = self.query_processor.optimize_for_lexical_search(query_analysis)
            
            # 2. Construire et exécuter la requête Elasticsearch
            es_results = await self.es_client.search_transactions(
                query=optimized_query,
                user_id=user_id,
                limit=min(limit, self.config.max_results),
                offset=offset,
                filters=filters,
                include_highlights=self.config.highlight_enabled,
                explain=debug
            )
            
            # 3. Traiter les résultats
            processed_results = self._process_elasticsearch_results(
                es_results, query_analysis, debug
            )
            
            # 4. Appliquer le tri si nécessaire
            if sort_order != SortOrder.RELEVANCE:
                processed_results = self._apply_custom_sorting(processed_results, sort_order)
            
            # 5. Calculer les métriques de qualité
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            quality = self._assess_search_quality(processed_results, query_analysis)
            
            # 6. Construire le résultat final
            lexical_result = LexicalSearchResult(
                results=processed_results.results[:limit],
                total_found=processed_results.total_found,
                max_score=processed_results.max_score,
                avg_score=processed_results.avg_score,
                processing_time_ms=processing_time,
                quality=quality,
                query_used=optimized_query,
                debug_info=processed_results.debug_info if debug else None
            )
            
            logger.debug(
                f"Lexical search completed: {len(processed_results.results)} results, "
                f"quality: {quality}, time: {processing_time:.2f}ms"
            )
            
            return lexical_result
            
        except Exception as e:
            self.error_count += 1
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Lexical search failed: {e}")
            
            return LexicalSearchResult(
                results=[],
                total_found=0,
                max_score=0.0,
                avg_score=0.0,
                processing_time_ms=processing_time,
                quality=SearchQuality.FAILED,
                query_used=query,
                debug_info={"error": str(e)} if debug else None
            )
    
    def _process_elasticsearch_results(
        self,
        es_results: Dict[str, Any],
        query_analysis: QueryAnalysis,
        debug: bool = False
    ) -> 'ProcessedSearchResults':
        """Traite les résultats bruts d'Elasticsearch."""
        hits = es_results.get("hits", {}).get("hits", [])
        total = es_results.get("hits", {}).get("total", {})
        total_found = total.get("value", 0) if isinstance(total, dict) else total
        
        processed_results = []
        scores = []
        
        for hit in hits:
            result_item = self._convert_es_hit_to_result_item(hit, query_analysis)
            if result_item and result_item.score >= self.config.min_score:
                processed_results.append(result_item)
                scores.append(result_item.score)
        
        # Calculer les statistiques
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Informations de debug
        debug_info = None
        if debug:
            debug_info = {
                "elasticsearch_query": es_results.get("_query", {}),
                "total_hits": total_found,
                "max_score_es": es_results.get("hits", {}).get("max_score", 0),
                "filtered_results": len(processed_results),
                "score_distribution": self._calculate_score_distribution(scores),
                "query_analysis": {
                    "original_query": query_analysis.original_query,
                    "cleaned_query": query_analysis.cleaned_query,
                    "expanded_query": query_analysis.expanded_query,
                    "detected_entities": query_analysis.detected_entities
                }
            }
        
        return ProcessedSearchResults(
            results=processed_results,
            total_found=total_found,
            max_score=max_score,
            avg_score=avg_score,
            debug_info=debug_info
        )
    
    def _convert_es_hit_to_result_item(
        self,
        hit: Dict[str, Any],
        query_analysis: QueryAnalysis
    ) -> Optional[SearchResultItem]:
        """Convertit un hit Elasticsearch en SearchResultItem."""
        try:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            highlights = hit.get("highlight", {})
            
            # Calculer des scores détaillés si disponibles
            lexical_score = score
            semantic_score = None  # N/A pour recherche lexicale
            combined_score = score
            
            # Créer l'item de résultat
            result_item = SearchResultItem(
                transaction_id=source.get("transaction_id"),
                user_id=source.get("user_id"),
                account_id=source.get("account_id"),
                score=score,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                combined_score=combined_score,
                primary_description=source.get("primary_description", ""),
                searchable_text=source.get("searchable_text", ""),
                merchant_name=source.get("merchant_name"),
                amount=source.get("amount", 0.0),
                currency_code=source.get("currency_code", "EUR"),
                transaction_type=source.get("transaction_type", ""),
                transaction_date=source.get("transaction_date", ""),
                created_at=source.get("created_at"),
                category_id=source.get("category_id"),
                operation_type=source.get("operation_type"),
                highlights=highlights if highlights else None,
                metadata={
                    "elasticsearch_score": score,
                    "search_type": "lexical",
                    "query_type": query_analysis.query_type,
                    "boost_applied": self._identify_boost_factors(hit, query_analysis)
                }
            )
            
            return result_item
            
        except Exception as e:
            logger.warning(f"Failed to convert ES hit to result item: {e}")
            return None
    
    def _identify_boost_factors(
        self,
        hit: Dict[str, Any],
        query_analysis: QueryAnalysis
    ) -> List[str]:
        """Identifie les facteurs de boost appliqués à un résultat."""
        boost_factors = []
        
        source = hit.get("_source", {})
        highlights = hit.get("highlight", {})
        query_terms = query_analysis.cleaned_query.lower().split()
        
        # Boost pour correspondance exacte
        searchable_text = source.get("searchable_text", "").lower()
        primary_desc = source.get("primary_description", "").lower()
        
        if query_analysis.cleaned_query.lower() in searchable_text:
            boost_factors.append("exact_phrase_match")
        
        if query_analysis.cleaned_query.lower() in primary_desc:
            boost_factors.append("description_exact_match")
        
        # Boost pour merchant_name
        merchant_name = source.get("merchant_name", "").lower()
        if any(term in merchant_name for term in query_terms):
            boost_factors.append("merchant_name_match")
        
        # Boost pour highlighting (indique correspondance forte)
        if highlights:
            if "merchant_name" in highlights:
                boost_factors.append("merchant_highlighted")
            if "searchable_text" in highlights:
                boost_factors.append("searchable_text_highlighted")
            if "primary_description" in highlights:
                boost_factors.append("description_highlighted")
        
        # Boost pour catégorie détectée
        if query_analysis.detected_entities.get("categories"):
            categories = query_analysis.detected_entities["categories"]
            for category in categories:
                if category.lower() in searchable_text or category.lower() in primary_desc:
                    boost_factors.append(f"category_match_{category}")
        
        return boost_factors
    
    def _apply_custom_sorting(
        self,
        results: 'ProcessedSearchResults',
        sort_order: SortOrder
    ) -> 'ProcessedSearchResults':
        """Applique un tri personnalisé aux résultats."""
        if sort_order == SortOrder.RELEVANCE:
            # Déjà trié par pertinence
            return results
        
        sorted_results = results.results.copy()
        
        if sort_order == SortOrder.DATE_DESC:
            sorted_results.sort(
                key=lambda x: (x.transaction_date, x.score), 
                reverse=True
            )
        elif sort_order == SortOrder.DATE_ASC:
            sorted_results.sort(
                key=lambda x: (x.transaction_date, x.score)
            )
        elif sort_order == SortOrder.AMOUNT_DESC:
            sorted_results.sort(
                key=lambda x: (abs(x.amount), x.score), 
                reverse=True
            )
        elif sort_order == SortOrder.AMOUNT_ASC:
            sorted_results.sort(
                key=lambda x: (abs(x.amount), x.score)
            )
        
        return ProcessedSearchResults(
            results=sorted_results,
            total_found=results.total_found,
            max_score=results.max_score,
            avg_score=results.avg_score,
            debug_info=results.debug_info
        )
    
    def _assess_search_quality(
        self,
        results: 'ProcessedSearchResults',
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité des résultats de recherche."""
        if not results.results:
            return SearchQuality.FAILED
        
        # Facteurs de qualité basés sur le validateur
        quality_score = 0.0
        
        # 1. Score moyen des résultats
        if results.avg_score >= 50:
            quality_score += 0.4
        elif results.avg_score >= 20:
            quality_score += 0.2
        
        # 2. Nombre de résultats pertinents
        if len(results.results) >= 5:
            quality_score += 0.2
        elif len(results.results) >= 1:
            quality_score += 0.1
        
        # 3. Correspondance avec entités détectées
        if self._check_entity_relevance(results, query_analysis):
            quality_score += 0.2
        
        # 4. Diversity des résultats (pas tous identiques)
        if self._check_result_diversity(results):
            quality_score += 0.1
        
        # 5. Présence de highlights (indique correspondance forte)
        highlighted_results = sum(1 for r in results.results if r.highlights)
        if highlighted_results > 0:
            quality_score += 0.1 * (highlighted_results / len(results.results))
        
        # Convertir en enum
        if quality_score >= 0.9:
            return SearchQuality.EXCELLENT
        elif quality_score >= 0.7:
            return SearchQuality.GOOD
        elif quality_score >= 0.5:
            return SearchQuality.MEDIUM
        elif quality_score >= 0.3:
            return SearchQuality.POOR
        else:
            return SearchQuality.FAILED
    
    def _check_entity_relevance(
        self,
        results: 'ProcessedSearchResults',
        query_analysis: QueryAnalysis
    ) -> bool:
        """Vérifie si les résultats correspondent aux entités détectées."""
        entities = query_analysis.detected_entities
        
        # Vérifier correspondance des montants
        if entities.get("amounts"):
            expected_amounts = [a["value"] for a in entities["amounts"]]
            for result in results.results[:5]:  # Top 5 résultats
                result_amount = abs(result.amount)
                for expected in expected_amounts:
                    if abs(result_amount - expected) / max(result_amount, expected) < 0.2:  # 20% de tolérance
                        return True
        
        # Vérifier correspondance des catégories
        if entities.get("categories"):
            expected_categories = set(cat.lower() for cat in entities["categories"])
            for result in results.results[:5]:
                result_text = f"{result.searchable_text} {result.primary_description}".lower()
                if any(cat in result_text for cat in expected_categories):
                    return True
        
        # Si pas d'entités spécifiques, considérer comme pertinent
        if not entities.get("amounts") and not entities.get("categories"):
            return True
        
        return False
    
    def _check_result_diversity(self, results: 'ProcessedSearchResults') -> bool:
        """Vérifie la diversité des résultats."""
        if len(results.results) < 2:
            return True
        
        # Vérifier diversité des descriptions
        descriptions = set()
        merchants = set()
        
        for result in results.results[:10]:
            descriptions.add(result.primary_description.lower().strip())
            if result.merchant_name:
                merchants.add(result.merchant_name.lower().strip())
        
        # Au moins 60% de descriptions uniques
        description_diversity = len(descriptions) / len(results.results[:10])
        return description_diversity >= 0.6
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Calcule la distribution des scores."""
        if not scores:
            return {}
        
        return {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
            "count": len(scores),
            "std_dev": self._calculate_std_dev(scores)
        }
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calcule l'écart-type."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    async def suggest_query_improvements(
        self,
        query: str,
        user_id: int,
        search_results: LexicalSearchResult
    ) -> List[str]:
        """Suggère des améliorations de requête basées sur les résultats."""
        suggestions = []
        
        # Si peu de résultats, suggérer des variantes
        if len(search_results.results) < 3:
            query_analysis = self.query_processor.process_query(query)
            alternatives = self.query_processor.generate_alternative_queries(query_analysis)
            
            for alt_query in alternatives[:3]:
                suggestions.append(f"Essayez: '{alt_query}'")
        
        # Si résultats de faible qualité, suggérer simplification
        if search_results.quality in [SearchQuality.POOR, SearchQuality.FAILED]:
            keywords = query.split()
            if len(keywords) > 2:
                simplified = ' '.join(keywords[:2])
                suggestions.append(f"Simplifiez: '{simplified}'")
        
        # Suggérer ajout de filtres si entités détectées
        query_analysis = self.query_processor.process_query(query)
        if query_analysis.suggested_filters:
            suggestions.append("Utilisez des filtres de date ou montant pour affiner")
        
        return suggestions[:5]  # Maximum 5 suggestions
    
    async def explain_search_results(
        self,
        query: str,
        transaction_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Explique pourquoi une transaction spécifique apparaît dans les résultats."""
        try:
            # Rechercher avec explication activée
            es_results = await self.es_client.search_transactions(
                query=query,
                user_id=user_id,
                limit=50,
                filters={"transaction_ids": [transaction_id]},
                explain=True
            )
            
            # Trouver la transaction dans les résultats
            target_hit = None
            for hit in es_results.get("hits", {}).get("hits", []):
                if hit["_source"]["transaction_id"] == transaction_id:
                    target_hit = hit
                    break
            
            if not target_hit:
                return {"error": "Transaction not found in search results"}
            
            # Analyser l'explication
            explanation = target_hit.get("_explanation", {})
            score = target_hit.get("_score", 0)
            source = target_hit["_source"]
            
            return {
                "transaction_id": transaction_id,
                "score": score,
                "explanation": self._parse_es_explanation(explanation),
                "matching_fields": self._identify_matching_fields(target_hit, query),
                "transaction_data": {
                    "description": source.get("primary_description"),
                    "merchant": source.get("merchant_name"),
                    "amount": source.get("amount"),
                    "date": source.get("transaction_date")
                },
                "boost_factors": self._identify_boost_factors(target_hit, self.query_processor.process_query(query))
            }
            
        except Exception as e:
            return {"error": f"Failed to explain search result: {e}"}
    
    def _parse_es_explanation(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Parse l'explication Elasticsearch en format lisible."""
        if not explanation:
            return {}
        
        return {
            "total_score": explanation.get("value", 0),
            "description": explanation.get("description", ""),
            "details": self._extract_explanation_details(explanation.get("details", []))
        }
    
    def _extract_explanation_details(self, details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extrait les détails de l'explication."""
        extracted = []
        
        for detail in details:
            extracted.append({
                "component": detail.get("description", ""),
                "score": detail.get("value", 0),
                "sub_details": self._extract_explanation_details(detail.get("details", []))
            })
        
        return extracted
    
    def _identify_matching_fields(self, hit: Dict[str, Any], query: str) -> List[str]:
        """Identifie les champs qui correspondent à la requête."""
        matching_fields = []
        source = hit["_source"]
        highlights = hit.get("highlight", {})
        
        query_terms = query.lower().split()
        
        # Vérifier les highlights d'abord
        for field in highlights:
            matching_fields.append(f"{field} (highlighted)")
        
        # Vérifier les correspondances dans les champs principaux
        fields_to_check = {
            "searchable_text": source.get("searchable_text", ""),
            "primary_description": source.get("primary_description", ""),
            "merchant_name": source.get("merchant_name", "")
        }
        
        for field_name, field_value in fields_to_check.items():
            if field_value and any(term in field_value.lower() for term in query_terms):
                if f"{field_name} (highlighted)" not in matching_fields:
                    matching_fields.append(field_name)
        
        return matching_fields
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du moteur lexical."""
        avg_processing_time = (
            self.total_processing_time / self.search_count
            if self.search_count > 0 else 0
        )
        
        error_rate = self.error_count / self.search_count if self.search_count > 0 else 0
        
        return {
            "engine_type": "lexical",
            "search_count": self.search_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_processing_time_ms": avg_processing_time,
            "config": {
                "max_results": self.config.max_results,
                "min_score": self.config.min_score,
                "highlight_enabled": self.config.highlight_enabled,
                "fuzzy_enabled": self.config.fuzzy_enabled,
                "synonym_expansion": self.config.synonym_expansion
            },
            "elasticsearch_client_stats": self.es_client.get_metrics()
        }
    
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.search_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        logger.info("Lexical search engine stats reset")
    
    async def warmup(self, user_id: int) -> bool:
        """Réchauffe le moteur avec des requêtes de test."""
        warmup_queries = [
            "restaurant", "virement", "carte bancaire", 
            "supermarché", "essence", "pharmacie"
        ]
        
        success_count = 0
        
        for query in warmup_queries:
            try:
                result = await self.search(query, user_id, limit=5)
                if result.quality != SearchQuality.FAILED:
                    success_count += 1
                    
                # Petit délai entre les requêtes
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Warmup query '{query}' failed: {e}")
        
        warmup_success = success_count >= len(warmup_queries) // 2
        logger.info(f"Lexical engine warmup: {success_count}/{len(warmup_queries)} queries successful")
        
        return warmup_success


@dataclass
class ProcessedSearchResults:
    """Résultats traités de recherche."""
    results: List[SearchResultItem]
    total_found: int
    max_score: float
    avg_score: float
    debug_info: Optional[Dict[str, Any]] = None


class LexicalSearchOptimizer:
    """Optimiseur pour les requêtes lexicales."""
    
    def __init__(self):
        # Patterns de requêtes optimisées
        self.optimization_patterns = {
            "single_word": self._optimize_single_word,
            "amount_query": self._optimize_amount_query,
            "merchant_query": self._optimize_merchant_query,
            "category_query": self._optimize_category_query,
            "date_query": self._optimize_date_query,
            "complex_query": self._optimize_complex_query
        }
        
        logger.info("Lexical search optimizer initialized")
    
    def optimize_query(
        self,
        query_analysis: QueryAnalysis,
        search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Optimise une requête pour la recherche lexicale.
        
        Args:
            query_analysis: Analyse de la requête
            search_context: Contexte de recherche (historique, préférences)
            
        Returns:
            Requête optimisée
        """
        query_type = query_analysis.query_type
        
        # Sélectionner l'optimiseur approprié
        if len(query_analysis.cleaned_query.split()) == 1:
            optimizer = self.optimization_patterns["single_word"]
        elif query_analysis.detected_entities.get("amounts"):
            optimizer = self.optimization_patterns["amount_query"]
        elif query_analysis.detected_entities.get("categories"):
            optimizer = self.optimization_patterns["category_query"]
        elif "merchant" in query_type or "restaurant" in query_analysis.cleaned_query:
            optimizer = self.optimization_patterns["merchant_query"]
        elif query_analysis.detected_entities.get("dates"):
            optimizer = self.optimization_patterns["date_query"]
        else:
            optimizer = self.optimization_patterns["complex_query"]
        
        return optimizer(query_analysis, search_context)
    
    def _optimize_single_word(
        self,
        query_analysis: QueryAnalysis,
        search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimise les requêtes à un seul mot."""
        word = query_analysis.cleaned_query
        
        # Ajouter des variantes et synonymes
        variations = [word]
        
        # Ajouter des synonymes financiers
        if word in FINANCIAL_SYNONYMS:
            variations.extend(FINANCIAL_SYNONYMS[word][:3])
        
        # Ajouter des wildcard pour capturer les variantes
        if len(word) > 3:
            variations.append(f"{word}*")
        
        return " ".join(set(variations))
    
    def _optimize_amount_query(
        self,
        query_analysis: QueryAnalysis,
        search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimise les requêtes avec montants."""
        base_query = query_analysis.cleaned_query
        
        # Supprimer les montants pour se concentrer sur le contexte
        text_only = base_query
        for amount in query_analysis.detected_entities.get("amounts", []):
            text_only = text_only.replace(amount["raw"], "")
        
        text_only = " ".join(text_only.split())  # Nettoyer les espaces
        
        if text_only:
            return text_only
        else:
            # Si seul montant, rechercher des transactions similaires
            return "transaction paiement achat"
    
    def _optimize_merchant_query(
        self,
        query_analysis: QueryAnalysis,
        search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimise les requêtes de marchands."""
        query = query_analysis.cleaned_query
        
        # Prioriser le champ merchant_name
        merchant_terms = []
        
        # Extraire les termes qui pourraient être des noms de marchands
        words = query.split()
        for word in words:
            if len(word) > 2 and word not in ["chez", "au", "à", "le", "la", "les"]:
                merchant_terms.append(word)
        
        # Ajouter des variantes courantes
        if "restaurant" in query or "resto" in query:
            merchant_terms.extend(["restaurant", "resto", "brasserie"])
        
        return " ".join(set(merchant_terms))
    
    def _optimize_category_query(
        self,
        query_analysis: QueryAnalysis,
        search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimise les requêtes de catégories."""
        categories = query_analysis.detected_entities.get("categories", [])
        base_query = query_analysis.cleaned_query
        
        # Combiner la requête originale avec les termes de catégorie
        category_terms = []
        for category in categories:
            if category in FINANCIAL_SYNONYMS:
                category_terms.extend(FINANCIAL_SYNONYMS[category][:3])
            else:
                category_terms.append(category)
        
        all_terms = [base_query] + category_terms
        return " ".join(set(all_terms))
    
    def _optimize_date_query(
        self,
        query_analysis: QueryAnalysis,
        search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimise les requêtes avec dates."""
        base_query = query_analysis.cleaned_query
        
        # Supprimer les dates pour se concentrer sur le contenu
        text_only = base_query
        for date_info in query_analysis.detected_entities.get("dates", []):
            text_only = text_only.replace(date_info["raw"], "")
        
        text_only = " ".join(text_only.split())
        
        # Note: Les filtres de date seront appliqués séparément
        return text_only if text_only else "transaction"
    
    def _optimize_complex_query(
        self,
        query_analysis: QueryAnalysis,
        search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimise les requêtes complexes."""
        # Pour les requêtes complexes, utiliser la requête étendue
        expanded = query_analysis.expanded_query
        
        # Limiter le nombre de termes pour éviter la dilution
        terms = expanded.split()
        if len(terms) > 8:
            # Garder les termes les plus importants
            important_terms = []
            
            # Prioriser les termes de la requête originale
            original_terms = set(query_analysis.cleaned_query.split())
            for term in terms:
                if term in original_terms:
                    important_terms.append(term)
            
            # Ajouter d'autres termes jusqu'à 8
            for term in terms:
                if term not in important_terms and len(important_terms) < 8:
                    important_terms.append(term)
            
            return " ".join(important_terms)
        
        return expanded


# Import pour éviter la référence circulaire
import asyncio
from search_service.models.search_types import FINANCIAL_SYNONYMS