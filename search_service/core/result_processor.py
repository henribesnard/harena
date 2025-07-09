"""
Processeur de résultats Elasticsearch pour le Search Service.

Traite, formate et enrichit les résultats Elasticsearch selon
les contrats standardisés et optimise les performances.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import re

from ..models.service_contracts import (
    SearchResultItem, AggregationResult, AggregationBucket,
    SearchServiceQuery, QueryType, IntentType
)
from ..config.settings import SearchServiceSettings, get_settings
from ..utils.metrics import ResultProcessingMetrics


logger = logging.getLogger(__name__)


class ResultQuality(str, Enum):
    """Niveaux de qualité des résultats."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class DuplicationStrategy(str, Enum):
    """Stratégies de déduplication."""
    NONE = "none"                    # Pas de déduplication
    STRICT = "strict"                # Déduplication stricte par transaction_id
    FUZZY = "fuzzy"                  # Déduplication floue par similarité
    SMART = "smart"                  # Déduplication intelligente contextuelle


class ResultProcessor:
    """
    Processeur de résultats Elasticsearch haute performance.
    
    Responsabilités:
    - Formatage des résultats selon les contrats
    - Déduplication intelligente
    - Enrichissement contextuel
    - Highlighting optimisé
    - Traitement des agrégations
    - Validation et nettoyage des données
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        self.settings = settings or get_settings()
        
        # Métriques de traitement
        self.metrics = ResultProcessingMetrics()
        
        # Configuration déduplication
        self.deduplication_enabled = True
        self.deduplication_strategy = DuplicationStrategy.SMART
        
        # Cache de patterns de highlighting
        self.highlight_patterns = {}
        
        # Compteurs de performance
        self.results_processed = 0
        self.total_processing_time = 0.0
        self.duplicates_removed = 0
        self.enrichments_applied = 0
        
        logger.info("Result processor initialized with optimized configuration")
    
    def process_search_results(
        self,
        elasticsearch_response: Dict[str, Any],
        query: SearchServiceQuery,
        include_highlights: bool = True,
        include_explanations: bool = False
    ) -> List[SearchResultItem]:
        """
        Traite les résultats de recherche Elasticsearch.
        
        Args:
            elasticsearch_response: Réponse brute Elasticsearch
            query: Requête originale pour contexte
            include_highlights: Inclure le highlighting
            include_explanations: Inclure les explications de score
            
        Returns:
            List[SearchResultItem]: Résultats formatés selon le contrat
        """
        start_time = datetime.utcnow()
        
        try:
            # Extraction des hits
            hits = elasticsearch_response.get("hits", {}).get("hits", [])
            
            if not hits:
                logger.debug("Aucun résultat trouvé")
                return []
            
            # Traitement des résultats individuels
            processed_results = []
            for hit in hits:
                try:
                    result_item = self._process_single_hit(
                        hit, query, include_highlights, include_explanations
                    )
                    if result_item:
                        processed_results.append(result_item)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur traitement hit {hit.get('_id', 'unknown')}: {str(e)}")
                    continue
            
            # Déduplication si activée
            if self.deduplication_enabled and len(processed_results) > 1:
                processed_results = self._deduplicate_results(processed_results, query)
            
            # Enrichissement contextuel
            processed_results = self._enrich_results(processed_results, query)
            
            # Tri final selon les critères de qualité
            processed_results = self._apply_final_sorting(processed_results, query)
            
            # Mise à jour des métriques
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_processing_metrics(len(processed_results), processing_time)
            
            logger.debug(f"✅ {len(processed_results)} résultats traités en {processing_time:.2f}ms")
            
            return processed_results
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"❌ Erreur traitement résultats: {str(e)}")
            self._update_processing_metrics(0, processing_time, error=True)
            return []
    
    def process_aggregations(
        self,
        elasticsearch_aggregations: Dict[str, Any],
        query: SearchServiceQuery
    ) -> Optional[AggregationResult]:
        """
        Traite les agrégations Elasticsearch.
        
        Args:
            elasticsearch_aggregations: Agrégations brutes
            query: Requête originale pour contexte
            
        Returns:
            Optional[AggregationResult]: Agrégations formatées
        """
        if not elasticsearch_aggregations:
            return None
        
        try:
            # Métriques globales
            total_amount = self._extract_metric_value(elasticsearch_aggregations, "total_amount")
            total_amount_abs = self._extract_metric_value(elasticsearch_aggregations, "total_amount_abs")
            avg_amount = self._extract_metric_value(elasticsearch_aggregations, "avg_amount")
            min_amount = self._extract_metric_value(elasticsearch_aggregations, "min_amount")
            max_amount = self._extract_metric_value(elasticsearch_aggregations, "max_amount")
            
            # Comptage de documents
            doc_count = self._extract_doc_count(elasticsearch_aggregations)
            
            # Agrégations par groupement
            by_month = self._process_terms_aggregation(
                elasticsearch_aggregations.get("by_month"), "month"
            )
            by_category = self._process_terms_aggregation(
                elasticsearch_aggregations.get("by_category"), "category"
            )
            by_merchant = self._process_terms_aggregation(
                elasticsearch_aggregations.get("by_merchant"), "merchant"
            )
            
            # Statistiques détaillées
            statistics = self._extract_detailed_statistics(elasticsearch_aggregations)
            
            result = AggregationResult(
                total_amount=total_amount,
                transaction_count=doc_count,
                average_amount=avg_amount,
                by_month=by_month,
                by_category=by_category,
                by_merchant=by_merchant,
                statistics=statistics
            )
            
            logger.debug(f"✅ Agrégations traitées: {doc_count} transactions")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement agrégations: {str(e)}")
            return None
    
    def _process_single_hit(
        self,
        hit: Dict[str, Any],
        query: SearchServiceQuery,
        include_highlights: bool,
        include_explanations: bool
    ) -> Optional[SearchResultItem]:
        """
        Traite un hit Elasticsearch individuel.
        
        Args:
            hit: Hit Elasticsearch
            query: Requête originale
            include_highlights: Inclure highlighting
            include_explanations: Inclure explications
            
        Returns:
            Optional[SearchResultItem]: Item formaté ou None si erreur
        """
        try:
            source = hit.get("_source", {})
            
            # Validation des champs obligatoires
            if not self._validate_required_fields(source):
                logger.debug(f"⚠️ Champs obligatoires manquants dans {hit.get('_id')}")
                return None
            
            # Extraction et nettoyage des données
            cleaned_data = self._clean_and_normalize_data(source)
            
            # Traitement du highlighting
            highlights = None
            if include_highlights and "highlight" in hit:
                highlights = self._process_highlights(hit["highlight"], query)
            
            # Construction de l'item selon le contrat
            result_item = SearchResultItem(
                transaction_id=cleaned_data.get("transaction_id", ""),
                user_id=cleaned_data.get("user_id", 0),
                account_id=cleaned_data.get("account_id", 0),
                amount=float(cleaned_data.get("amount", 0.0)),
                amount_abs=float(cleaned_data.get("amount_abs", 0.0)),
                transaction_type=cleaned_data.get("transaction_type", ""),
                currency_code=cleaned_data.get("currency_code", "EUR"),
                date=cleaned_data.get("date", ""),
                month_year=cleaned_data.get("month_year", ""),
                weekday=cleaned_data.get("weekday"),
                primary_description=cleaned_data.get("primary_description", ""),
                merchant_name=cleaned_data.get("merchant_name"),
                category_name=cleaned_data.get("category_name"),
                operation_type=cleaned_data.get("operation_type"),
                score=float(hit.get("_score", 0.0)),
                highlights=highlights,
                searchable_text=cleaned_data.get("searchable_text")
            )
            
            return result_item
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur traitement hit: {str(e)}")
            return None
    
    def _validate_required_fields(self, source: Dict[str, Any]) -> bool:
        """
        Valide la présence des champs obligatoires.
        
        Args:
            source: Source Elasticsearch
            
        Returns:
            bool: True si valide
        """
        required_fields = [
            "transaction_id", "user_id", "amount", "date", "primary_description"
        ]
        
        for field in required_fields:
            if field not in source or source[field] is None:
                return False
        
        return True
    
    def _clean_and_normalize_data(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nettoie et normalise les données.
        
        Args:
            source: Données source
            
        Returns:
            Dict: Données nettoyées
        """
        cleaned = {}
        
        for key, value in source.items():
            if value is None:
                cleaned[key] = None
                continue
            
            # Nettoyage selon le type de champ
            if key in ["primary_description", "merchant_name", "searchable_text"]:
                # Nettoyage des chaînes de caractères
                cleaned[key] = self._clean_text_field(str(value))
            
            elif key in ["amount", "amount_abs"]:
                # Normalisation des montants
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    cleaned[key] = 0.0
            
            elif key in ["user_id", "account_id"]:
                # Normalisation des IDs
                try:
                    cleaned[key] = int(value)
                except (ValueError, TypeError):
                    cleaned[key] = 0
            
            elif key == "date":
                # Normalisation des dates
                cleaned[key] = self._normalize_date(value)
            
            else:
                # Valeur par défaut
                cleaned[key] = value
        
        return cleaned
    
    def _clean_text_field(self, text: str) -> str:
        """
        Nettoie un champ texte.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""
        
        # Suppression des caractères de contrôle
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limitation de la longueur
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def _normalize_date(self, date_value: Any) -> str:
        """
        Normalise une valeur de date.
        
        Args:
            date_value: Valeur de date
            
        Returns:
            str: Date normalisée (YYYY-MM-DD)
        """
        try:
            if isinstance(date_value, str):
                # Extraction de la partie date si datetime
                if 'T' in date_value:
                    return date_value.split('T')[0]
                return date_value
            
            # Conversion depuis d'autres formats si nécessaire
            return str(date_value)
            
        except Exception:
            return ""
    
    def _process_highlights(
        self,
        highlights: Dict[str, List[str]],
        query: SearchServiceQuery
    ) -> Dict[str, List[str]]:
        """
        Traite et optimise les highlights.
        
        Args:
            highlights: Highlights bruts Elasticsearch
            query: Requête pour contexte
            
        Returns:
            Dict: Highlights optimisés
        """
        processed_highlights = {}
        
        for field, fragments in highlights.items():
            if not fragments:
                continue
            
            # Nettoyage et optimisation des fragments
            cleaned_fragments = []
            for fragment in fragments:
                cleaned = self._clean_highlight_fragment(fragment)
                if cleaned and len(cleaned) > 10:  # Fragments trop courts ignorés
                    cleaned_fragments.append(cleaned)
            
            if cleaned_fragments:
                # Limitation du nombre de fragments
                max_fragments = self.settings.SEARCH_HIGHLIGHT_MAX_FRAGMENTS
                processed_highlights[field] = cleaned_fragments[:max_fragments]
        
        return processed_highlights
    
    def _clean_highlight_fragment(self, fragment: str) -> str:
        """
        Nettoie un fragment de highlighting.
        
        Args:
            fragment: Fragment brut
            
        Returns:
            str: Fragment nettoyé
        """
        if not fragment:
            return ""
        
        # Vérification des balises de highlighting valides
        if "<em>" not in fragment and "</em>" not in fragment:
            return ""
        
        # Nettoyage des caractères de contrôle
        fragment = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fragment)
        
        # Normalisation des espaces autour des balises
        fragment = re.sub(r'\s*<em>\s*', '<em>', fragment)
        fragment = re.sub(r'\s*</em>\s*', '</em>', fragment)
        
        return fragment.strip()
    
    def _deduplicate_results(
        self,
        results: List[SearchResultItem],
        query: SearchServiceQuery
    ) -> List[SearchResultItem]:
        """
        Déduplique les résultats selon la stratégie configurée.
        
        Args:
            results: Résultats à dédupliquer
            query: Requête pour contexte
            
        Returns:
            List[SearchResultItem]: Résultats dédupliqués
        """
        if self.deduplication_strategy == DuplicationStrategy.NONE:
            return results
        
        initial_count = len(results)
        
        if self.deduplication_strategy == DuplicationStrategy.STRICT:
            deduplicated = self._strict_deduplication(results)
        elif self.deduplication_strategy == DuplicationStrategy.FUZZY:
            deduplicated = self._fuzzy_deduplication(results)
        else:  # SMART
            deduplicated = self._smart_deduplication(results, query)
        
        duplicates_removed = initial_count - len(deduplicated)
        if duplicates_removed > 0:
            self.duplicates_removed += duplicates_removed
            logger.debug(f"🧹 {duplicates_removed} doublons supprimés")
        
        return deduplicated
    
    def _strict_deduplication(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """Déduplication stricte par transaction_id."""
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            if result.transaction_id not in seen_ids:
                seen_ids.add(result.transaction_id)
                deduplicated.append(result)
        
        return deduplicated
    
    def _fuzzy_deduplication(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """Déduplication floue par similarité."""
        deduplicated = []
        
        for result in results:
            is_duplicate = False
            
            for existing in deduplicated:
                if self._are_similar_transactions(result, existing):
                    is_duplicate = True
                    # Garder celui avec le meilleur score
                    if result.score > existing.score:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def _smart_deduplication(
        self,
        results: List[SearchResultItem],
        query: SearchServiceQuery
    ) -> List[SearchResultItem]:
        """Déduplication intelligente contextuelle."""
        # Combinaison de strict + fuzzy selon le contexte
        
        # D'abord déduplication stricte
        strict_deduplicated = self._strict_deduplication(results)
        
        # Puis déduplication floue si beaucoup de résultats
        if len(strict_deduplicated) > 20:
            return self._fuzzy_deduplication(strict_deduplicated)
        
        return strict_deduplicated
    
    def _are_similar_transactions(
        self,
        trans1: SearchResultItem,
        trans2: SearchResultItem
    ) -> bool:
        """
        Détermine si deux transactions sont similaires.
        
        Args:
            trans1: Première transaction
            trans2: Deuxième transaction
            
        Returns:
            bool: True si similaires
        """
        # Même utilisateur
        if trans1.user_id != trans2.user_id:
            return False
        
        # Même montant absolu
        if abs(trans1.amount_abs - trans2.amount_abs) > 0.01:
            return False
        
        # Même date
        if trans1.date != trans2.date:
            return False
        
        # Descriptions similaires (Jaccard similarity)
        desc1_words = set(trans1.primary_description.lower().split())
        desc2_words = set(trans2.primary_description.lower().split())
        
        if not desc1_words or not desc2_words:
            return False
        
        jaccard = len(desc1_words & desc2_words) / len(desc1_words | desc2_words)
        
        return jaccard > 0.8  # 80% de similarité
    
    def _enrich_results(
        self,
        results: List[SearchResultItem],
        query: SearchServiceQuery
    ) -> List[SearchResultItem]:
        """
        Enrichit les résultats avec des informations contextuelles.
        
        Args:
            results: Résultats à enrichir
            query: Requête pour contexte
            
        Returns:
            List[SearchResultItem]: Résultats enrichis
        """
        enriched_results = []
        
        for result in results:
            enriched = self._enrich_single_result(result, query)
            enriched_results.append(enriched)
        
        if len(enriched_results) != len(results):
            self.enrichments_applied += len(enriched_results)
        
        return enriched_results
    
    def _enrich_single_result(
        self,
        result: SearchResultItem,
        query: SearchServiceQuery
    ) -> SearchResultItem:
        """
        Enrichit un résultat individuel.
        
        Args:
            result: Résultat à enrichir
            query: Requête pour contexte
            
        Returns:
            SearchResultItem: Résultat enrichi
        """
        # Pour l'instant, retourne le résultat tel quel
        # TODO: Implémenter enrichissements contextuels
        # - Calcul de pertinence contextuelle
        # - Ajout de métadonnées dérivées
        # - Amélioration du score selon l'intention
        
        return result
    
    def _apply_final_sorting(
        self,
        results: List[SearchResultItem],
        query: SearchServiceQuery
    ) -> List[SearchResultItem]:
        """
        Applique un tri final selon les critères de qualité.
        
        Args:
            results: Résultats à trier
            query: Requête pour contexte
            
        Returns:
            List[SearchResultItem]: Résultats triés
        """
        # Tri par défaut : score puis date puis montant
        return sorted(
            results,
            key=lambda r: (-r.score, r.date, -r.amount_abs),
            reverse=False
        )
    
    def _extract_metric_value(
        self,
        aggregations: Dict[str, Any],
        metric_name: str
    ) -> Optional[float]:
        """
        Extrait une valeur métrique des agrégations.
        
        Args:
            aggregations: Agrégations Elasticsearch
            metric_name: Nom de la métrique
            
        Returns:
            Optional[float]: Valeur de la métrique
        """
        metric = aggregations.get(metric_name)
        if not metric:
            return None
        
        return metric.get("value")
    
    def _extract_doc_count(self, aggregations: Dict[str, Any]) -> int:
        """
        Extrait le nombre de documents des agrégations.
        
        Args:
            aggregations: Agrégations Elasticsearch
            
        Returns:
            int: Nombre de documents
        """
        # Recherche dans différents endroits possibles
        if "doc_count" in aggregations:
            doc_count = aggregations["doc_count"]
            if isinstance(doc_count, dict):
                return doc_count.get("value", 0)
            return doc_count
        
        # Fallback: somme des buckets
        total_docs = 0
        for agg_name, agg_data in aggregations.items():
            if isinstance(agg_data, dict) and "buckets" in agg_data:
                total_docs += sum(bucket.get("doc_count", 0) for bucket in agg_data["buckets"])
        
        return total_docs
    
    def _process_terms_aggregation(
        self,
        aggregation_data: Optional[Dict[str, Any]],
        aggregation_type: str
    ) -> Optional[List[AggregationBucket]]:
        """
        Traite une agrégation de type terms.
        
        Args:
            aggregation_data: Données d'agrégation
            aggregation_type: Type d'agrégation pour logs
            
        Returns:
            Optional[List[AggregationBucket]]: Buckets traités
        """
        if not aggregation_data or "buckets" not in aggregation_data:
            return None
        
        buckets = []
        for bucket_data in aggregation_data["buckets"]:
            bucket = AggregationBucket(
                key=bucket_data.get("key", ""),
                doc_count=bucket_data.get("doc_count", 0),
                total_amount=self._extract_bucket_metric(bucket_data, "total_amount"),
                avg_amount=self._extract_bucket_metric(bucket_data, "avg_amount"),
                min_amount=self._extract_bucket_metric(bucket_data, "min_amount"),
                max_amount=self._extract_bucket_metric(bucket_data, "max_amount")
            )
            buckets.append(bucket)
        
        return buckets
    
    def _extract_bucket_metric(
        self,
        bucket_data: Dict[str, Any],
        metric_name: str
    ) -> Optional[float]:
        """
        Extrait une métrique d'un bucket d'agrégation.
        
        Args:
            bucket_data: Données du bucket
            metric_name: Nom de la métrique
            
        Returns:
            Optional[float]: Valeur de la métrique
        """
        metric = bucket_data.get(metric_name)
        if not metric:
            return None
        
        if isinstance(metric, dict):
            return metric.get("value")
        
        return float(metric) if metric is not None else None
    
    def _extract_detailed_statistics(
        self,
        aggregations: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        Extrait les statistiques détaillées.
        
        Args:
            aggregations: Agrégations Elasticsearch
            
        Returns:
            Optional[Dict]: Statistiques détaillées
        """
        stats_agg = aggregations.get("amount_stats")
        if not stats_agg:
            return None
        
        return {
            "count": stats_agg.get("count", 0),
            "min": stats_agg.get("min", 0.0),
            "max": stats_agg.get("max", 0.0),
            "avg": stats_agg.get("avg", 0.0),
            "sum": stats_agg.get("sum", 0.0)
        }
    
    def _update_processing_metrics(
        self,
        results_count: int,
        processing_time_ms: float,
        error: bool = False
    ) -> None:
        """
        Met à jour les métriques de traitement.
        
        Args:
            results_count: Nombre de résultats traités
            processing_time_ms: Temps de traitement
            error: True si erreur
        """
        self.results_processed += results_count
        self.total_processing_time += processing_time_ms
        
        self.metrics.record_processing(
            results_count=results_count,
            processing_time_ms=processing_time_ms,
            error=error
        )
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de traitement.
        
        Returns:
            Dict: Métriques détaillées
        """
        avg_processing_time = (
            self.total_processing_time / max(self.results_processed, 1)
        )
        
        return {
            "results_processed": self.results_processed,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": avg_processing_time,
            "duplicates_removed": self.duplicates_removed,
            "enrichments_applied": self.enrichments_applied,
            "deduplication_enabled": self.deduplication_enabled,
            "deduplication_strategy": self.deduplication_strategy.value,
            "detailed_metrics": self.metrics.get_metrics()
        }
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques."""
        self.results_processed = 0
        self.total_processing_time = 0.0
        self.duplicates_removed = 0
        self.enrichments_applied = 0
        self.metrics.reset()
        
        logger.info("🔄 Métriques de traitement réinitialisées")


# === HELPER FUNCTIONS ===

def create_result_processor(
    settings: Optional[SearchServiceSettings] = None
) -> ResultProcessor:
    """
    Factory pour créer un processeur de résultats.
    
    Args:
        settings: Configuration
        
    Returns:
        ResultProcessor: Processeur configuré
    """
    return ResultProcessor(settings=settings or get_settings())


def calculate_result_quality(results: List[SearchResultItem]) -> ResultQuality:
    """
    Calcule la qualité globale d'un ensemble de résultats.
    
    Args:
        results: Résultats à évaluer
        
    Returns:
        ResultQuality: Niveau de qualité
    """
    if not results:
        return ResultQuality.POOR
    
    # Calcul du score moyen
    avg_score = sum(r.score for r in results) / len(results)
    
    # Nombre de résultats
    result_count = len(results)
    
    # Évaluation basée sur score et nombre
    if avg_score > 0.9 and result_count >= 5:
        return ResultQuality.EXCELLENT
    elif avg_score > 0.7 and result_count >= 3:
        return ResultQuality.GOOD
    elif avg_score > 0.5 and result_count >= 1:
        return ResultQuality.FAIR
    else:
        return ResultQuality.POOR