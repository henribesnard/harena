"""
Processeur de résultats Elasticsearch - Composant Core #2
Traite et formate les résultats Elasticsearch
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import re
from collections import defaultdict

from search_service.models.service_contracts import SearchServiceResponse, SearchResult, ResultMetadata
from search_service.models.responses import SearchHit, AggregationResult, HighlightResult
from search_service.utils.elasticsearch_helpers import ElasticsearchHelper

logger = logging.getLogger(__name__)


class ResultProcessor:
    """
    Processeur avancé de résultats Elasticsearch
    
    Responsabilités:
    - Formatage des résultats Elasticsearch standards
    - Highlighting intelligent des termes recherchés
    - Calcul des scores de pertinence
    - Déduplication et nettoyage des résultats
    """
    
    def __init__(self):
        self.helper = ElasticsearchHelper()
        
        # Configuration du processing
        self.max_snippet_length = 200
        self.max_highlights_per_field = 3
        self.min_score_threshold = 0.1
        
        # Cache pour la déduplication
        self.deduplication_cache: Set[str] = set()
        
        logger.info("✅ ResultProcessor initialisé")
    
    async def process_search_results(
        self, 
        es_results: Dict[str, Any], 
        query_text: Optional[str] = None,
        include_aggregations: bool = True
    ) -> SearchServiceResponse:
        """
        Traite les résultats de recherche Elasticsearch
        
        Args:
            es_results: Résultats bruts Elasticsearch
            query_text: Texte de la requête original
            include_aggregations: Inclure les agrégations
            
        Returns:
            Réponse de recherche formatée
        """
        try:
            # Extraction des métadonnées
            metadata = self._extract_metadata(es_results)
            
            # Traitement des hits
            hits = self._process_hits(es_results.get("hits", {}), query_text)
            
            # Traitement des agrégations
            aggregations = []
            if include_aggregations and "aggregations" in es_results:
                aggregations = self._process_aggregations(es_results["aggregations"])
            
            # Construction de la réponse
            response = SearchServiceResponse(
                hits=hits,
                aggregations=aggregations,
                metadata=metadata,
                query_text=query_text,
                processing_time=metadata.processing_time
            )
            
            logger.info(f"✅ Résultats traités: {len(hits)} hits, {len(aggregations)} agrégations")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des résultats: {e}")
            raise
    
    async def process_aggregation_results(self, es_results: Dict[str, Any]) -> List[AggregationResult]:
        """
        Traite les résultats d'agrégation uniquement
        
        Args:
            es_results: Résultats Elasticsearch avec agrégations
            
        Returns:
            Liste des résultats d'agrégation
        """
        try:
            if "aggregations" not in es_results:
                return []
            
            aggregations = self._process_aggregations(es_results["aggregations"])
            
            logger.info(f"✅ Agrégations traitées: {len(aggregations)}")
            
            return aggregations
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des agrégations: {e}")
            raise
    
    def _extract_metadata(self, es_results: Dict[str, Any]) -> ResultMetadata:
        """
        Extrait les métadonnées des résultats Elasticsearch
        
        Args:
            es_results: Résultats Elasticsearch
            
        Returns:
            Métadonnées formatées
        """
        try:
            hits_info = es_results.get("hits", {})
            total_info = hits_info.get("total", {})
            
            # Extraction des informations de base
            total_hits = total_info.get("value", 0) if isinstance(total_info, dict) else total_info
            max_score = hits_info.get("max_score", 0.0)
            took = es_results.get("took", 0)
            
            # Informations sur la recherche
            timed_out = es_results.get("timed_out", False)
            shards_info = es_results.get("_shards", {})
            
            metadata = ResultMetadata(
                total_hits=total_hits,
                max_score=max_score,
                processing_time=took,
                timed_out=timed_out,
                shards_total=shards_info.get("total", 0),
                shards_successful=shards_info.get("successful", 0),
                shards_failed=shards_info.get("failed", 0)
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction des métadonnées: {e}")
            # Retour des métadonnées par défaut
            return ResultMetadata(
                total_hits=0,
                max_score=0.0,
                processing_time=0,
                timed_out=False,
                shards_total=0,
                shards_successful=0,
                shards_failed=0
            )
    
    def _process_hits(self, hits_data: Dict[str, Any], query_text: Optional[str] = None) -> List[SearchHit]:
        """
        Traite les hits Elasticsearch
        
        Args:
            hits_data: Données des hits
            query_text: Texte de la requête
            
        Returns:
            Liste des hits formatés
        """
        try:
            processed_hits = []
            raw_hits = hits_data.get("hits", [])
            
            for hit in raw_hits:
                # Vérification du score minimal
                score = hit.get("_score", 0.0)
                if score < self.min_score_threshold:
                    continue
                
                # Extraction des données source
                source = hit.get("_source", {})
                
                # Traitement des highlights
                highlights = self._process_highlights(hit.get("highlight", {}))
                
                # Création du hit formaté
                processed_hit = SearchHit(
                    id=hit.get("_id"),
                    index=hit.get("_index"),
                    score=score,
                    source=source,
                    highlights=highlights,
                    # Extraction des champs principaux
                    title=source.get("title", ""),
                    content=source.get("content", ""),
                    summary=source.get("summary", ""),
                    document_type=source.get("document_type", ""),
                    created_at=source.get("created_at"),
                    updated_at=source.get("updated_at"),
                    # Snippets enrichis
                    snippet=self._create_snippet(source, highlights, query_text)
                )
                
                processed_hits.append(processed_hit)
            
            # Déduplication
            deduplicated_hits = self._deduplicate_hits(processed_hits)
            
            # Tri par score
            deduplicated_hits.sort(key=lambda x: x.score, reverse=True)
            
            return deduplicated_hits
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des hits: {e}")
            return []
    
    def _process_highlights(self, highlight_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Traite les highlights Elasticsearch
        
        Args:
            highlight_data: Données de highlight
            
        Returns:
            Highlights formatés
        """
        try:
            processed_highlights = {}
            
            for field, highlights in highlight_data.items():
                if isinstance(highlights, list):
                    # Limitation du nombre de highlights
                    limited_highlights = highlights[:self.max_highlights_per_field]
                    
                    # Nettoyage des highlights
                    cleaned_highlights = []
                    for highlight in limited_highlights:
                        # Nettoyage des balises HTML en trop
                        cleaned = self._clean_highlight(highlight)
                        if cleaned and len(cleaned) > 10:  # Minimum de caractères
                            cleaned_highlights.append(cleaned)
                    
                    if cleaned_highlights:
                        processed_highlights[field] = cleaned_highlights
            
            return processed_highlights
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des highlights: {e}")
            return {}
    
    def _clean_highlight(self, highlight: str) -> str:
        """
        Nettoie un highlight individuel
        
        Args:
            highlight: Highlight à nettoyer
            
        Returns:
            Highlight nettoyé
        """
        try:
            # Suppression des espaces multiples
            cleaned = re.sub(r'\s+', ' ', highlight)
            
            # Suppression des caractères de contrôle
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
            
            # Limitation de la longueur
            if len(cleaned) > self.max_snippet_length:
                cleaned = cleaned[:self.max_snippet_length] + "..."
            
            return cleaned.strip()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du nettoyage du highlight: {e}")
            return highlight
    
    def _create_snippet(
        self, 
        source: Dict[str, Any], 
        highlights: Dict[str, List[str]], 
        query_text: Optional[str] = None
    ) -> str:
        """
        Crée un snippet enrichi pour un document
        
        Args:
            source: Données source du document
            highlights: Highlights du document
            query_text: Texte de la requête
            
        Returns:
            Snippet enrichi
        """
        try:
            # Priorité aux highlights
            if highlights:
                # Recherche du meilleur highlight
                best_highlight = self._find_best_highlight(highlights, query_text)
                if best_highlight:
                    return best_highlight
            
            # Fallback sur le résumé
            if source.get("summary"):
                summary = source["summary"]
                if len(summary) > self.max_snippet_length:
                    summary = summary[:self.max_snippet_length] + "..."
                return summary
            
            # Fallback sur le début du contenu
            if source.get("content"):
                content = source["content"]
                if len(content) > self.max_snippet_length:
                    content = content[:self.max_snippet_length] + "..."
                return content
            
            # Fallback sur le titre
            return source.get("title", "Document sans titre")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du snippet: {e}")
            return "Erreur lors de la création du snippet"
    
    def _find_best_highlight(self, highlights: Dict[str, List[str]], query_text: Optional[str] = None) -> Optional[str]:
        """
        Trouve le meilleur highlight parmi ceux disponibles
        
        Args:
            highlights: Highlights disponibles
            query_text: Texte de la requête
            
        Returns:
            Meilleur highlight ou None
        """
        try:
            # Priorité aux champs par ordre d'importance
            field_priority = ["title", "summary", "content"]
            
            for field in field_priority:
                if field in highlights and highlights[field]:
                    # Retour du premier highlight du champ prioritaire
                    return highlights[field][0]
            
            # Si aucun champ prioritaire, retour du premier disponible
            for field_highlights in highlights.values():
                if field_highlights:
                    return field_highlights[0]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche du meilleur highlight: {e}")
            return None
    
    def _deduplicate_hits(self, hits: List[SearchHit]) -> List[SearchHit]:
        """
        Déduplique les hits par contenu similaire
        
        Args:
            hits: Liste des hits
            
        Returns:
            Liste des hits dédupliqués
        """
        try:
            deduplicated = []
            seen_signatures = set()
            
            for hit in hits:
                # Création d'une signature pour le document
                signature = self._create_document_signature(hit)
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    deduplicated.append(hit)
                else:
                    logger.debug(f"🔄 Document dédupliqué: {hit.id}")
            
            logger.info(f"✅ Déduplication: {len(hits)} → {len(deduplicated)} hits")
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la déduplication: {e}")
            return hits
    
    def _create_document_signature(self, hit: SearchHit) -> str:
        """
        Crée une signature unique pour un document
        
        Args:
            hit: Hit à signer
            
        Returns:
            Signature du document
        """
        try:
            # Combinaison de champs pour créer une signature unique
            title = hit.title or ""
            content_preview = (hit.content or "")[:100]  # Premier 100 caractères
            doc_type = hit.document_type or ""
            
            # Hash simple basé sur le contenu
            signature_text = f"{title}|{content_preview}|{doc_type}"
            signature = str(hash(signature_text))
            
            return signature
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de la signature: {e}")
            return hit.id or "unknown"
    
    def _process_aggregations(self, agg_data: Dict[str, Any]) -> List[AggregationResult]:
        """
        Traite les agrégations Elasticsearch
        
        Args:
            agg_data: Données d'agrégation
            
        Returns:
            Liste des résultats d'agrégation
        """
        try:
            aggregations = []
            
            for agg_name, agg_result in agg_data.items():
                processed_agg = self._process_single_aggregation(agg_name, agg_result)
                if processed_agg:
                    aggregations.append(processed_agg)
            
            return aggregations
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des agrégations: {e}")
            return []
    
    def _process_single_aggregation(self, name: str, agg_data: Dict[str, Any]) -> Optional[AggregationResult]:
        """
        Traite une agrégation individuelle
        
        Args:
            name: Nom de l'agrégation
            agg_data: Données de l'agrégation
            
        Returns:
            Résultat d'agrégation formaté
        """
        try:
            agg_result = AggregationResult(name=name)
            
            # Terms aggregation
            if "buckets" in agg_data:
                agg_result.type = "terms"
                agg_result.buckets = []
                
                for bucket in agg_data["buckets"]:
                    bucket_data = {
                        "key": bucket.get("key"),
                        "doc_count": bucket.get("doc_count", 0)
                    }
                    
                    # Sous-agrégations
                    sub_aggs = {}
                    for key, value in bucket.items():
                        if key not in ["key", "doc_count"] and isinstance(value, dict):
                            sub_agg = self._process_single_aggregation(key, value)
                            if sub_agg:
                                sub_aggs[key] = sub_agg
                    
                    if sub_aggs:
                        bucket_data["sub_aggregations"] = sub_aggs
                    
                    agg_result.buckets.append(bucket_data)
            
            # Stats aggregation
            elif "count" in agg_data:
                agg_result.type = "stats"
                agg_result.stats = {
                    "count": agg_data.get("count", 0),
                    "min": agg_data.get("min"),
                    "max": agg_data.get("max"),
                    "avg": agg_data.get("avg"),
                    "sum": agg_data.get("sum")
                }
            
            # Date histogram
            elif "buckets" in agg_data and isinstance(agg_data["buckets"], list):
                agg_result.type = "date_histogram"
                agg_result.buckets = []
                
                for bucket in agg_data["buckets"]:
                    bucket_data = {
                        "key": bucket.get("key"),
                        "key_as_string": bucket.get("key_as_string"),
                        "doc_count": bucket.get("doc_count", 0)
                    }
                    agg_result.buckets.append(bucket_data)
            
            # Value aggregation (single value)
            elif "value" in agg_data:
                agg_result.type = "value"
                agg_result.value = agg_data["value"]
            
            else:
                logger.warning(f"⚠️ Type d'agrégation non reconnu pour {name}: {agg_data}")
                return None
            
            return agg_result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de l'agrégation {name}: {e}")
            return None
    
    def enhance_results_with_context(
        self, 
        results: SearchServiceResponse, 
        context_data: Optional[Dict[str, Any]] = None
    ) -> SearchServiceResponse:
        """
        Enrichit les résultats avec du contexte additionnel
        
        Args:
            results: Résultats de base
            context_data: Données de contexte
            
        Returns:
            Résultats enrichis
        """
        try:
            if not context_data:
                return results
            
            # Enrichissement des hits
            for hit in results.hits:
                # Ajout de métadonnées contextuelles
                if "document_metadata" in context_data:
                    hit.source["context_metadata"] = context_data["document_metadata"]
                
                # Enrichissement du snippet avec contexte
                if "related_terms" in context_data:
                    hit.snippet = self._enhance_snippet_with_terms(
                        hit.snippet, 
                        context_data["related_terms"]
                    )
            
            logger.info(f"✅ Résultats enrichis avec contexte")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enrichissement: {e}")
            return results
    
    def _enhance_snippet_with_terms(self, snippet: str, related_terms: List[str]) -> str:
        """
        Enrichit un snippet avec des termes liés
        
        Args:
            snippet: Snippet original
            related_terms: Termes liés à mettre en évidence
            
        Returns:
            Snippet enrichi
        """
        try:
            enhanced_snippet = snippet
            
            for term in related_terms:
                if term.lower() in snippet.lower():
                    # Mise en évidence des termes liés
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    enhanced_snippet = pattern.sub(
                        f"<strong>{term}</strong>", 
                        enhanced_snippet
                    )
            
            return enhanced_snippet
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enrichissement du snippet: {e}")
            return snippet
    
    def calculate_relevance_scores(self, results: SearchServiceResponse) -> SearchServiceResponse:
        """
        Recalcule les scores de pertinence avec des facteurs additionnels
        
        Args:
            results: Résultats originaux
            
        Returns:
            Résultats avec scores recalculés
        """
        try:
            max_original_score = max((hit.score for hit in results.hits), default=1.0)
            
            for hit in results.hits:
                # Score original normalisé
                normalized_score = hit.score / max_original_score
                
                # Facteurs de boost
                title_boost = 1.2 if hit.title and len(hit.title) > 0 else 1.0
                recent_boost = self._calculate_recency_boost(hit.created_at)
                content_quality_boost = self._calculate_content_quality_boost(hit.content)
                
                # Score final
                final_score = normalized_score * title_boost * recent_boost * content_quality_boost
                hit.relevance_score = final_score
            
            # Re-tri par score de pertinence
            results.hits.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"✅ Scores de pertinence recalculés")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du recalcul des scores: {e}")
            return results
    
    def _calculate_recency_boost(self, created_at: Optional[str]) -> float:
        """
        Calcule un boost basé sur la récence du document
        
        Args:
            created_at: Date de création
            
        Returns:
            Facteur de boost (1.0 à 1.3)
        """
        try:
            if not created_at:
                return 1.0
            
            from datetime import datetime, timezone
            import dateutil.parser
            
            # Parse de la date
            doc_date = dateutil.parser.parse(created_at)
            now = datetime.now(timezone.utc)
            
            # Calcul de l'âge en jours
            age_days = (now - doc_date).days
            
            # Boost dégressif selon l'âge
            if age_days <= 30:  # Moins d'un mois
                return 1.3
            elif age_days <= 90:  # Moins de 3 mois
                return 1.2
            elif age_days <= 365:  # Moins d'un an
                return 1.1
            else:
                return 1.0
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul du boost de récence: {e}")
            return 1.0
    
    def _calculate_content_quality_boost(self, content: Optional[str]) -> float:
        """
        Calcule un boost basé sur la qualité du contenu
        
        Args:
            content: Contenu du document
            
        Returns:
            Facteur de boost (0.8 à 1.2)
        """
        try:
            if not content:
                return 0.8
            
            content_length = len(content)
            
            # Boost basé sur la longueur
            if content_length < 100:  # Contenu très court
                return 0.8
            elif content_length < 500:  # Contenu court
                return 0.9
            elif content_length < 2000:  # Contenu optimal
                return 1.2
            elif content_length < 5000:  # Contenu long
                return 1.1
            else:  # Contenu très long
                return 1.0
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul du boost de qualité: {e}")
            return 1.0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de traitement
        
        Returns:
            Statistiques du processeur
        """
        return {
            "max_snippet_length": self.max_snippet_length,
            "max_highlights_per_field": self.max_highlights_per_field,
            "min_score_threshold": self.min_score_threshold,
            "deduplication_cache_size": len(self.deduplication_cache)
        }
    
    def clear_cache(self):
        """
        Vide le cache de déduplication
        """
        self.deduplication_cache.clear()
        logger.info("🧹 Cache de déduplication vidé")