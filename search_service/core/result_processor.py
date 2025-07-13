"""
Processeur de résultats Elasticsearch avancé
Responsable du formatage, enrichissement et optimisation des résultats de recherche
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import math

from search_service.models.responses import (
    InternalSearchResponse, RawTransaction, InternalAggregationResult,
    QualityIndicator
)

from search_service.utils.metrics import MetricsCollector, ResultMetrics


logger = logging.getLogger(__name__)


class ProcessingStrategy(str, Enum):
    """Stratégies de traitement des résultats"""
    BASIC = "basic"                     # Traitement minimal
    ENHANCED = "enhanced"               # Enrichissement standard
    INTELLIGENT = "intelligent"        # Traitement intelligent avec ML
    CONVERSATIONAL = "conversational"  # Optimisé pour agents conversationnels


class RelevanceAlgorithm(str, Enum):
    """Algorithmes de calcul de pertinence"""
    ELASTICSEARCH_SCORE = "elasticsearch_score"    # Score ES natif
    BM25_ENHANCED = "bm25_enhanced"                # BM25 amélioré
    FINANCIAL_WEIGHTED = "financial_weighted"      # Pondération financière
    CONTEXTUAL_BOOST = "contextual_boost"          # Boost contextuel
    HYBRID_SCORING = "hybrid_scoring"              # Scoring hybride


class HighlightingMode(str, Enum):
    """Modes de mise en évidence"""
    NONE = "none"
    BASIC = "basic"
    INTELLIGENT = "intelligent"
    CONTEXTUAL = "contextual"


@dataclass
class ProcessingContext:
    """Contexte pour le traitement des résultats"""
    user_id: int
    query_text: Optional[str] = None
    search_intention: Optional[str] = None
    agent_context: Dict[str, Any] = field(default_factory=dict)
    processing_strategy: ProcessingStrategy = ProcessingStrategy.ENHANCED
    relevance_algorithm: RelevanceAlgorithm = RelevanceAlgorithm.FINANCIAL_WEIGHTED
    highlighting_mode: HighlightingMode = HighlightingMode.INTELLIGENT
    include_debug_info: bool = False
    max_suggestions: int = 5
    deduplication_enabled: bool = True
    score_normalization: bool = True


@dataclass
class EnhancedResult:
    """Résultat enrichi avec métadonnées avancées"""
    original_result: RawTransaction
    processed_score: float
    relevance_factors: Dict[str, float]
    highlights: Dict[str, List[str]]
    category_confidence: float
    merchant_confidence: float
    semantic_tags: List[str]
    financial_insights: Dict[str, Any]
    duplicate_group: Optional[str] = None
    processing_notes: List[str] = field(default_factory=list)


class FinancialResultProcessor:
    """Processeur spécialisé pour les résultats de transactions financières"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        # Créer un MetricsCollector par défaut si non fourni
        if metrics_collector is None:
            self._metrics_collector = MetricsCollector()
        else:
            self._metrics_collector = metrics_collector
            
        self.metrics = ResultMetrics(self._metrics_collector)
        self.category_patterns = self._load_category_patterns()
        self.merchant_normalizer = self._load_merchant_normalizer()
        self.financial_terms = self._load_financial_terms()
        
        logger.info("FinancialResultProcessor initialisé")
    
    def process_search_results(self, 
                              response: InternalSearchResponse,
                              context: ProcessingContext) -> InternalSearchResponse:
        """
        Traite et enrichit les résultats de recherche
        """
        start_time = datetime.now()
        
        try:
            # 1. Validation et préparation
            if not self._validate_processing_input(response, context):
                logger.warning("Validation échouée, retour des résultats originaux")
                return response
            
            # 2. Préparation des résultats pour traitement
            enhanced_results = self._prepare_results_for_processing(response.raw_results, context)
            
            # 3. Calcul des scores de pertinence
            enhanced_results = self._calculate_relevance_scores(enhanced_results, context)
            
            # 4. Déduplication si activée
            if context.deduplication_enabled:
                enhanced_results = self._deduplicate_results(enhanced_results, context)
            
            # 5. Highlighting intelligent
            enhanced_results = self._apply_intelligent_highlighting(enhanced_results, context)
            
            # 6. Enrichissement contextuel
            enhanced_results = self._enrich_with_context(enhanced_results, context)
            
            # 7. Tri final et limitation
            enhanced_results = self._apply_final_ranking(enhanced_results, context)
            
            # 8. Conversion vers format interne
            processed_results = self._convert_to_raw_transactions(enhanced_results)
            
            # 9. Mise à jour de la réponse
            updated_response = self._update_response_with_processed_results(
                response, processed_results, enhanced_results, context
            )
            
            # 10. Métriques et logging
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._record_processing_metrics(response, updated_response, processing_time, context)
            
            return updated_response
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des résultats: {str(e)}")
            # Retourner les résultats originaux en cas d'erreur
            return response
    
    def _prepare_results_for_processing(self, 
                                       raw_results: List[RawTransaction],
                                       context: ProcessingContext) -> List[EnhancedResult]:
        """Prépare les résultats pour le traitement avancé"""
        
        enhanced_results = []
        
        for result in raw_results:
            enhanced = EnhancedResult(
                original_result=result,
                processed_score=result.score,
                relevance_factors={},
                highlights={},
                category_confidence=0.0,
                merchant_confidence=0.0,
                semantic_tags=[],
                financial_insights={}
            )
            enhanced_results.append(enhanced)
        
        return enhanced_results
    
    def _calculate_relevance_scores(self, 
                                   enhanced_results: List[EnhancedResult],
                                   context: ProcessingContext) -> List[EnhancedResult]:
        """Calcule les scores de pertinence selon l'algorithme sélectionné"""
        
        if context.relevance_algorithm == RelevanceAlgorithm.ELASTICSEARCH_SCORE:
            return self._apply_elasticsearch_scoring(enhanced_results, context)
        elif context.relevance_algorithm == RelevanceAlgorithm.BM25_ENHANCED:
            return self._apply_enhanced_bm25(enhanced_results, context)
        elif context.relevance_algorithm == RelevanceAlgorithm.FINANCIAL_WEIGHTED:
            return self._apply_financial_weighting(enhanced_results, context)
        elif context.relevance_algorithm == RelevanceAlgorithm.CONTEXTUAL_BOOST:
            return self._apply_contextual_boost(enhanced_results, context)
        elif context.relevance_algorithm == RelevanceAlgorithm.HYBRID_SCORING:
            return self._apply_hybrid_scoring(enhanced_results, context)
        else:
            return enhanced_results
    
    def _apply_financial_weighting(self, 
                                  enhanced_results: List[EnhancedResult],
                                  context: ProcessingContext) -> List[EnhancedResult]:
        """Applique une pondération spécialisée pour les données financières"""
        
        for enhanced in enhanced_results:
            result = enhanced.original_result
            factors = {}
            
            # 1. Facteur de récence (transactions récentes plus pertinentes)
            if hasattr(result, 'date') and result.date:
                try:
                    if isinstance(result.date, str):
                        transaction_date = datetime.fromisoformat(result.date.replace('Z', '+00:00'))
                    else:
                        transaction_date = result.date
                    
                    days_ago = (datetime.now() - transaction_date.replace(tzinfo=None)).days
                    recency_factor = max(0.1, 1.0 - (days_ago / 365))  # Décroissance sur 1 an
                    factors['recency'] = recency_factor
                except:
                    factors['recency'] = 0.5  # Facteur neutre si date invalide
            else:
                factors['recency'] = 0.5
            
            # 2. Facteur de montant (transactions importantes plus visibles)
            if hasattr(result, 'amount_abs') and result.amount_abs:
                # Normalisation logarithmique pour éviter la domination des gros montants
                amount_factor = min(1.0, math.log10(max(1, result.amount_abs)) / 3)
                factors['amount'] = amount_factor
            else:
                factors['amount'] = 0.5
            
            # 3. Facteur de correspondance textuelle
            if context.query_text:
                text_match_score = self._calculate_text_match_score(result, context.query_text)
                factors['text_match'] = text_match_score
            else:
                factors['text_match'] = 1.0
            
            # 4. Facteur de confiance catégorie
            category_confidence = self._calculate_category_confidence(result)
            factors['category_confidence'] = category_confidence
            enhanced.category_confidence = category_confidence
            
            # 5. Facteur de confiance marchand
            merchant_confidence = self._calculate_merchant_confidence(result)
            factors['merchant_confidence'] = merchant_confidence
            enhanced.merchant_confidence = merchant_confidence
            
            # 6. Calcul du score final pondéré
            weighted_score = (
                enhanced.original_result.score * 0.3 +           # Score ES original (30%)
                factors['text_match'] * 0.25 +                   # Correspondance textuelle (25%)
                factors['recency'] * 0.15 +                      # Récence (15%)
                factors['amount'] * 0.1 +                        # Importance montant (10%)
                factors['category_confidence'] * 0.1 +           # Confiance catégorie (10%)
                factors['merchant_confidence'] * 0.1             # Confiance marchand (10%)
            )
            
            enhanced.processed_score = weighted_score
            enhanced.relevance_factors = factors
        
        return enhanced_results
    
    def _calculate_text_match_score(self, result: RawTransaction, query_text: str) -> float:
        """Calcule le score de correspondance textuelle"""
        
        query_terms = set(query_text.lower().split())
        if not query_terms:
            return 1.0
        
        # Collecter tout le texte disponible
        searchable_fields = [
            getattr(result, 'searchable_text', ''),
            getattr(result, 'primary_description', ''),
            getattr(result, 'merchant_name', ''),
            getattr(result, 'category_name', '')
        ]
        
        all_text = ' '.join(filter(None, searchable_fields)).lower()
        
        # Calculer les correspondances
        matches = sum(1 for term in query_terms if term in all_text)
        exact_phrase_bonus = 0.2 if query_text.lower() in all_text else 0
        
        # Score basé sur la couverture des termes + bonus phrase exacte
        base_score = matches / len(query_terms) if query_terms else 0
        final_score = min(1.0, base_score + exact_phrase_bonus)
        
        return final_score
    
    def _calculate_category_confidence(self, result: RawTransaction) -> float:
        """Calcule la confiance dans la catégorisation"""
        
        if not hasattr(result, 'category_name') or not result.category_name:
            return 0.0
        
        category = result.category_name.lower()
        
        # Vérifier la cohérence avec les patterns connus
        description = getattr(result, 'primary_description', '').lower()
        merchant = getattr(result, 'merchant_name', '').lower()
        
        confidence = 0.5  # Base
        
        # Augmenter la confiance si des mots-clés de catégorie sont présents
        if category in self.category_patterns:
            patterns = self.category_patterns[category]
            for pattern in patterns:
                if pattern in description or pattern in merchant:
                    confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_merchant_confidence(self, result: RawTransaction) -> float:
        """Calcule la confiance dans l'identification du marchand"""
        
        if not hasattr(result, 'merchant_name') or not result.merchant_name:
            return 0.0
        
        merchant = result.merchant_name.lower()
        
        # Confiance basée sur la longueur et la structure
        confidence = 0.5
        
        # Bonus pour les marchands bien formatés
        if len(merchant) > 3 and merchant.replace(' ', '').isalnum():
            confidence += 0.2
        
        # Bonus pour les marchands connus
        if merchant in self.merchant_normalizer:
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def _apply_elasticsearch_scoring(self, enhanced_results: List[EnhancedResult],
                                   context: ProcessingContext) -> List[EnhancedResult]:
        """Conserve le scoring Elasticsearch original"""
        return enhanced_results
    
    def _apply_enhanced_bm25(self, enhanced_results: List[EnhancedResult],
                           context: ProcessingContext) -> List[EnhancedResult]:
        """Applique une version améliorée du scoring BM25"""
        
        # Implementation simplifiée - en production, utiliser une vraie implémentation BM25
        for enhanced in enhanced_results:
            # Facteurs BM25 ajustés pour les données financières
            k1 = 1.5  # Saturation term frequency
            b = 0.75  # Field length normalization
            
            # Calculer un score BM25 simplifié
            original_score = enhanced.original_result.score
            bm25_adjusted = original_score * (k1 + 1) / (original_score + k1)
            
            enhanced.processed_score = bm25_adjusted
            enhanced.relevance_factors['bm25_adjustment'] = bm25_adjusted / original_score if original_score > 0 else 1.0
        
        return enhanced_results
    
    def _apply_contextual_boost(self, enhanced_results: List[EnhancedResult],
                              context: ProcessingContext) -> List[EnhancedResult]:
        """Applique des boosts basés sur le contexte"""
        
        intention = context.search_intention
        agent_context = context.agent_context
        
        for enhanced in enhanced_results:
            boost_factors = {}
            
            # Boost basé sur l'intention de recherche
            if intention:
                if intention == "SEARCH_BY_CATEGORY" and hasattr(enhanced.original_result, 'category_name'):
                    boost_factors['intention_match'] = 1.2
                elif intention == "SEARCH_BY_MERCHANT" and hasattr(enhanced.original_result, 'merchant_name'):
                    boost_factors['intention_match'] = 1.2
                elif intention == "SEARCH_BY_AMOUNT" and hasattr(enhanced.original_result, 'amount_abs'):
                    boost_factors['intention_match'] = 1.1
            
            # Boost basé sur le contexte agent
            if agent_context.get('preferred_categories'):
                preferred = agent_context['preferred_categories']
                if hasattr(enhanced.original_result, 'category_name') and enhanced.original_result.category_name in preferred:
                    boost_factors['agent_preference'] = 1.15
            
            # Appliquer les boosts
            total_boost = 1.0
            for factor_name, boost_value in boost_factors.items():
                total_boost *= boost_value
            
            enhanced.processed_score = enhanced.original_result.score * total_boost
            enhanced.relevance_factors.update(boost_factors)
        
        return enhanced_results
    
    def _apply_hybrid_scoring(self, enhanced_results: List[EnhancedResult],
                            context: ProcessingContext) -> List[EnhancedResult]:
        """Combine plusieurs algorithmes de scoring"""
        
        # Appliquer séquentiellement plusieurs algorithmes
        enhanced_results = self._apply_financial_weighting(enhanced_results, context)
        enhanced_results = self._apply_contextual_boost(enhanced_results, context)
        
        # Normaliser les scores finaux
        if context.score_normalization:
            enhanced_results = self._normalize_scores(enhanced_results)
        
        return enhanced_results
    
    def _normalize_scores(self, enhanced_results: List[EnhancedResult]) -> List[EnhancedResult]:
        """Normalise les scores sur une échelle 0-1"""
        
        if not enhanced_results:
            return enhanced_results
        
        scores = [e.processed_score for e in enhanced_results]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        
        # Éviter la division par zéro
        if max_score == min_score:
            for enhanced in enhanced_results:
                enhanced.processed_score = 1.0
        else:
            for enhanced in enhanced_results:
                normalized = (enhanced.processed_score - min_score) / (max_score - min_score)
                enhanced.processed_score = normalized
        
        return enhanced_results
    
    def _deduplicate_results(self, enhanced_results: List[EnhancedResult],
                           context: ProcessingContext) -> List[EnhancedResult]:
        """Supprime ou groupe les résultats en double"""
        
        if not enhanced_results:
            return enhanced_results
        
        # Grouper par similarité
        groups = defaultdict(list)
        
        for enhanced in enhanced_results:
            # Créer une signature pour la déduplication
            signature = self._create_deduplication_signature(enhanced.original_result)
            groups[signature].append(enhanced)
        
        # Garder le meilleur de chaque groupe
        deduplicated = []
        
        for group_id, group_results in groups.items():
            if len(group_results) == 1:
                deduplicated.append(group_results[0])
            else:
                # Garder celui avec le meilleur score
                best_result = max(group_results, key=lambda x: x.processed_score)
                best_result.duplicate_group = group_id
                best_result.processing_notes.append(f"Déduplication: {len(group_results)} doublons supprimés")
                deduplicated.append(best_result)
        
        return deduplicated
    
    def _create_deduplication_signature(self, result: RawTransaction) -> str:
        """Crée une signature pour identifier les doublons"""
        
        # Utiliser plusieurs critères pour identifier les doublons
        signature_parts = []
        
        # Date (jour seulement)
        if hasattr(result, 'date') and result.date:
            try:
                if isinstance(result.date, str):
                    date_obj = datetime.fromisoformat(result.date.replace('Z', '+00:00'))
                else:
                    date_obj = result.date
                signature_parts.append(date_obj.strftime('%Y-%m-%d'))
            except:
                signature_parts.append('unknown_date')
        
        # Montant (arrondi à l'euro)
        if hasattr(result, 'amount_abs') and result.amount_abs:
            signature_parts.append(str(int(result.amount_abs)))
        
        # Marchand (normalisé)
        if hasattr(result, 'merchant_name') and result.merchant_name:
            normalized_merchant = re.sub(r'\W+', '', result.merchant_name.lower())
            signature_parts.append(normalized_merchant[:10])  # Premiers 10 caractères
        
        return '|'.join(signature_parts)
    
    def _apply_intelligent_highlighting(self, enhanced_results: List[EnhancedResult],
                                      context: ProcessingContext) -> List[EnhancedResult]:
        """Applique un highlighting intelligent"""
        
        if context.highlighting_mode == HighlightingMode.NONE:
            return enhanced_results
        
        if not context.query_text:
            return enhanced_results
        
        query_terms = context.query_text.lower().split()
        
        for enhanced in enhanced_results:
            highlights = {}
            result = enhanced.original_result
            
            # Highlighting pour chaque champ textuel
            text_fields = {
                'searchable_text': getattr(result, 'searchable_text', ''),
                'primary_description': getattr(result, 'primary_description', ''),
                'merchant_name': getattr(result, 'merchant_name', '')
            }
            
            for field_name, field_value in text_fields.items():
                if field_value:
                    highlighted = self._highlight_text(field_value, query_terms, context.highlighting_mode)
                    if highlighted != field_value:  # Seulement si des highlights ont été ajoutés
                        highlights[field_name] = [highlighted]
            
            enhanced.highlights = highlights
        
        return enhanced_results
    
    def _highlight_text(self, text: str, query_terms: List[str], mode: HighlightingMode) -> str:
        """Applique le highlighting sur un texte"""
        
        if mode == HighlightingMode.BASIC:
            # Highlighting simple
            highlighted = text
            for term in query_terms:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(f'<mark>{term}</mark>', highlighted)
            return highlighted
        
        elif mode == HighlightingMode.INTELLIGENT:
            # Highlighting intelligent avec gestion des variations
            highlighted = text
            for term in query_terms:
                # Chercher des variations (pluriels, conjugaisons simples)
                patterns = [
                    term,
                    term + 's',
                    term + 'es',
                    term[:-1] if term.endswith('s') else term + 's'
                ]
                
                for pattern in patterns:
                    regex = re.compile(r'\b' + re.escape(pattern) + r'\b', re.IGNORECASE)
                    highlighted = regex.sub(f'<mark>\\g<0></mark>', highlighted)
            
            return highlighted
        
        elif mode == HighlightingMode.CONTEXTUAL:
            # Highlighting contextuel (plus sophistiqué)
            return self._apply_contextual_highlighting(text, query_terms)
        
        return text
    
    def _apply_contextual_highlighting(self, text: str, query_terms: List[str]) -> str:
        """Applique un highlighting contextuel avancé"""
        
        # Identifier les concepts financiers et les mettre en valeur différemment
        financial_patterns = {
            r'\b\d+[.,]\d{2}\s*€?\b': 'amount',
            r'\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b': 'date',
            r'\b(restaurant|supermarché|carburant|transport|santé|loisirs)\b': 'category'
        }
        
        highlighted = text
        
        # Highlighting des termes de requête
        for term in query_terms:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f'<mark class="query-term">\\g<0></mark>', highlighted)
        
        # Highlighting des concepts financiers
        for pattern, concept_type in financial_patterns.items():
            regex = re.compile(pattern, re.IGNORECASE)
            highlighted = regex.sub(f'<mark class="{concept_type}">\\g<0></mark>', highlighted)
        
        return highlighted
    
    def _enrich_with_context(self, enhanced_results: List[EnhancedResult],
                           context: ProcessingContext) -> List[EnhancedResult]:
        """Enrichit les résultats avec du contexte additionnel"""
        
        for enhanced in enhanced_results:
            # Ajouter des tags sémantiques
            enhanced.semantic_tags = self._generate_semantic_tags(enhanced.original_result)
            
            # Ajouter des insights financiers
            enhanced.financial_insights = self._generate_financial_insights(enhanced.original_result)
            
            # Ajouter des notes de traitement si en mode debug
            if context.include_debug_info:
                enhanced.processing_notes.append(f"Score original: {enhanced.original_result.score:.3f}")
                enhanced.processing_notes.append(f"Score traité: {enhanced.processed_score:.3f}")
                enhanced.processing_notes.extend([f"{k}: {v:.3f}" for k, v in enhanced.relevance_factors.items()])
        
        return enhanced_results
    
    def _generate_semantic_tags(self, result: RawTransaction) -> List[str]:
        """Génère des tags sémantiques pour un résultat"""
        
        tags = []
        
        # Tags basés sur la catégorie
        if hasattr(result, 'category_name') and result.category_name:
            category = result.category_name.lower()
            if category in ['restaurant', 'alimentation']:
                tags.extend(['food', 'dining'])
            elif category in ['transport', 'carburant']:
                tags.extend(['transport', 'mobility'])
            elif category in ['loisirs', 'divertissement']:
                tags.extend(['entertainment', 'leisure'])
        
        # Tags basés sur le montant
        if hasattr(result, 'amount_abs') and result.amount_abs:
            if result.amount_abs > 100:
                tags.append('large_amount')
            elif result.amount_abs < 10:
                tags.append('small_amount')
        
        # Tags basés sur la récence
        if hasattr(result, 'date') and result.date:
            try:
                if isinstance(result.date, str):
                    date_obj = datetime.fromisoformat(result.date.replace('Z', '+00:00'))
                else:
                    date_obj = result.date
                
                days_ago = (datetime.now() - date_obj.replace(tzinfo=None)).days
                if days_ago <= 7:
                    tags.append('recent')
                elif days_ago <= 30:
                    tags.append('this_month')
                elif days_ago > 365:
                    tags.append('old')
            except:
                pass
        
        return list(set(tags))  # Supprimer les doublons
    
    def _generate_financial_insights(self, result: RawTransaction) -> Dict[str, Any]:
        """Génère des insights financiers pour un résultat"""
        
        insights = {}
        
        # Insight sur le type de transaction
        if hasattr(result, 'transaction_type'):
            insights['transaction_type'] = result.transaction_type
        
        # Insight sur la fréquence (nécessiterait une analyse historique)
        if hasattr(result, 'merchant_name') and result.merchant_name:
            insights['merchant_familiarity'] = 'known'  # Simplifié
        
        # Insight sur l'anomalie de montant (nécessiterait des stats utilisateur)
        if hasattr(result, 'amount_abs') and result.amount_abs:
            if result.amount_abs > 500:
                insights['amount_significance'] = 'high'
            elif result.amount_abs < 5:
                insights['amount_significance'] = 'low'
            else:
                insights['amount_significance'] = 'normal'
        
        # Insight temporel
        if hasattr(result, 'weekday'):
            insights['temporal_pattern'] = getattr(result, 'weekday', 'unknown')
        
        return insights
    
    def _apply_final_ranking(self, enhanced_results: List[EnhancedResult],
                           context: ProcessingContext) -> List[EnhancedResult]:
        """Applique le tri final et limite les résultats"""
        
        # Tri par score traité décroissant
        enhanced_results.sort(key=lambda x: x.processed_score, reverse=True)
        
        # Limiter le nombre de résultats si nécessaire
        max_results = getattr(context, 'max_results', None)
        if max_results and len(enhanced_results) > max_results:
            enhanced_results = enhanced_results[:max_results]
        
        return enhanced_results
    
    def _convert_to_raw_transactions(self, enhanced_results: List[EnhancedResult]) -> List[RawTransaction]:
        """Convertit les résultats enrichis vers le format RawTransaction"""
        
        raw_results = []
        
        for enhanced in enhanced_results:
            # Copier le résultat original
            raw_result = enhanced.original_result
            
            # Mettre à jour le score
            raw_result.score = enhanced.processed_score
            
            # Ajouter les highlights si présents
            if enhanced.highlights:
                raw_result._highlights = enhanced.highlights
            
            # Ajouter les métadonnées d'enrichissement
            raw_result._semantic_tags = enhanced.semantic_tags
            raw_result._financial_insights = enhanced.financial_insights
            raw_result._processing_notes = enhanced.processing_notes
            
            raw_results.append(raw_result)
        
        return raw_results
    
    def _update_response_with_processed_results(self,
                                              original_response: InternalSearchResponse,
                                              processed_results: List[RawTransaction],
                                              enhanced_results: List[EnhancedResult],
                                              context: ProcessingContext) -> InternalSearchResponse:
        """Met à jour la réponse avec les résultats traités"""
        
        # Créer une nouvelle réponse basée sur l'originale
        updated_response = InternalSearchResponse(
            request_id=original_response.request_id,
            user_id=original_response.user_id,
            total_hits=original_response.total_hits,
            returned_hits=len(processed_results),
            raw_results=processed_results,
            aggregations=original_response.aggregations,
            execution_metrics=original_response.execution_metrics,
            quality_score=self._calculate_enhanced_quality_score(enhanced_results),
            quality_indicator=self._determine_quality_indicator(enhanced_results),
            elasticsearch_response=original_response.elasticsearch_response,
            served_from_cache=original_response.served_from_cache,
            cache_key=original_response.cache_key,
            suggested_followups=self._generate_followup_suggestions(enhanced_results, context),
            related_categories=self._extract_related_categories(enhanced_results)
        )
        
        return updated_response
    
    def _calculate_enhanced_quality_score(self, enhanced_results: List[EnhancedResult]) -> float:
        """Calcule un score de qualité amélioré"""
        
        if not enhanced_results:
            return 0.0
        
        quality_factors = []
        
        # 1. Score moyen des résultats
        avg_score = sum(e.processed_score for e in enhanced_results) / len(enhanced_results)
        quality_factors.append(avg_score)
        
        # 2. Distribution des scores (éviter trop de scores faibles)
        high_score_count = sum(1 for e in enhanced_results if e.processed_score > 0.7)
        distribution_score = high_score_count / len(enhanced_results)
        quality_factors.append(distribution_score)
        
        # 3. Confiance moyenne des catégories
        avg_category_confidence = sum(e.category_confidence for e in enhanced_results) / len(enhanced_results)
        quality_factors.append(avg_category_confidence)
        
        # 4. Confiance moyenne des marchands
        avg_merchant_confidence = sum(e.merchant_confidence for e in enhanced_results) / len(enhanced_results)
        quality_factors.append(avg_merchant_confidence)
        
        # 5. Diversité des résultats (éviter trop de similarité)
        diversity_score = self._calculate_diversity_score(enhanced_results)
        quality_factors.append(diversity_score)
        
        # Score final pondéré
        final_score = (
            quality_factors[0] * 0.35 +    # Score moyen (35%)
            quality_factors[1] * 0.25 +    # Distribution (25%)
            quality_factors[2] * 0.15 +    # Confiance catégorie (15%)
            quality_factors[3] * 0.15 +    # Confiance marchand (15%)
            quality_factors[4] * 0.1       # Diversité (10%)
        )
        
        return min(1.0, final_score)
    
    def _calculate_diversity_score(self, enhanced_results: List[EnhancedResult]) -> float:
        """Calcule un score de diversité des résultats"""
        
        if len(enhanced_results) <= 1:
            return 1.0
        
        # Compter les catégories uniques
        categories = set()
        merchants = set()
        
        for result in enhanced_results:
            if hasattr(result.original_result, 'category_name') and result.original_result.category_name:
                categories.add(result.original_result.category_name)
            if hasattr(result.original_result, 'merchant_name') and result.original_result.merchant_name:
                merchants.add(result.original_result.merchant_name)
        
        # Score basé sur la diversité des catégories et marchands
        category_diversity = len(categories) / min(len(enhanced_results), 5)  # Max 5 catégories attendues
        merchant_diversity = len(merchants) / len(enhanced_results)
        
        return (category_diversity + merchant_diversity) / 2
    
    def _determine_quality_indicator(self, enhanced_results: List[EnhancedResult]) -> QualityIndicator:
        """Détermine l'indicateur de qualité global"""
        
        quality_score = self._calculate_enhanced_quality_score(enhanced_results)
        
        if quality_score >= 0.9:
            return QualityIndicator.EXCELLENT
        elif quality_score >= 0.7:
            return QualityIndicator.GOOD
        elif quality_score >= 0.5:
            return QualityIndicator.AVERAGE
        elif quality_score >= 0.3:
            return QualityIndicator.POOR
        else:
            return QualityIndicator.VERY_POOR
    
    def _generate_followup_suggestions(self, enhanced_results: List[EnhancedResult],
                                     context: ProcessingContext) -> List[str]:
        """Génère des suggestions de questions de suivi"""
        
        suggestions = []
        
        if not enhanced_results:
            return ["Essayez une recherche plus générale", "Vérifiez l'orthographe de votre requête"]
        
        # Analyser les résultats pour générer des suggestions pertinentes
        categories = set()
        merchants = set()
        date_ranges = []
        amounts = []
        
        for result in enhanced_results[:5]:  # Analyser les 5 premiers
            original = result.original_result
            
            if hasattr(original, 'category_name') and original.category_name:
                categories.add(original.category_name)
            if hasattr(original, 'merchant_name') and original.merchant_name:
                merchants.add(original.merchant_name)
            if hasattr(original, 'amount_abs') and original.amount_abs:
                amounts.append(original.amount_abs)
        
        # Suggestions basées sur les catégories trouvées
        if categories:
            most_common_category = list(categories)[0]  # Simplification
            suggestions.append(f"Voir plus de transactions {most_common_category}")
            if len(categories) > 1:
                suggestions.append("Comparer les dépenses par catégorie")
        
        # Suggestions basées sur les marchands
        if merchants:
            most_common_merchant = list(merchants)[0]  # Simplification
            suggestions.append(f"Historique complet chez {most_common_merchant}")
        
        # Suggestions basées sur les montants
        if amounts:
            avg_amount = sum(amounts) / len(amounts)
            if avg_amount > 50:
                suggestions.append("Transactions importantes (>50€)")
            suggestions.append("Évolution des dépenses ce mois")
        
        # Suggestions temporelles
        suggestions.append("Transactions de la semaine dernière")
        suggestions.append("Résumé mensuel des dépenses")
        
        # Limiter à max_suggestions
        return suggestions[:context.max_suggestions]
    
    def _extract_related_categories(self, enhanced_results: List[EnhancedResult]) -> List[str]:
        """Extrait les catégories liées des résultats"""
        
        categories = set()
        
        for result in enhanced_results:
            if hasattr(result.original_result, 'category_name') and result.original_result.category_name:
                categories.add(result.original_result.category_name)
        
        # Ajouter des catégories liées basées sur des règles métier
        related = set(categories)
        
        category_relationships = {
            'restaurant': ['alimentation', 'loisirs'],
            'alimentation': ['restaurant', 'santé'],
            'transport': ['carburant', 'loisirs'],
            'carburant': ['transport', 'automobile'],
            'loisirs': ['restaurant', 'culture', 'sport'],
            'santé': ['pharmacie', 'bien-être'],
            'shopping': ['vêtements', 'maison', 'électronique']
        }
        
        for category in categories:
            if category.lower() in category_relationships:
                related.update(category_relationships[category.lower()])
        
        return list(related)
    
    def _record_processing_metrics(self, original_response: InternalSearchResponse,
                                 updated_response: InternalSearchResponse,
                                 processing_time_ms: float,
                                 context: ProcessingContext):
        """Enregistre les métriques de traitement"""
        
        self.metrics.record_processing_result(
            processing_type=context.processing_strategy.value,
            duration_ms=processing_time_ms,
            input_count=len(original_response.raw_results),
            output_count=len(updated_response.raw_results),
            success=True
        )
    
    def _validate_processing_input(self, response: InternalSearchResponse,
                                 context: ProcessingContext) -> bool:
        """Valide les entrées pour le traitement"""
        
        if not response:
            logger.error("Réponse vide fournie pour traitement")
            return False
        
        if not context:
            logger.error("Contexte de traitement manquant")
            return False
        
        if context.user_id <= 0:
            logger.error("user_id invalide dans le contexte")
            return False
        
        return True
    
    # === MÉTHODES DE CONFIGURATION ===
    
    def _load_category_patterns(self) -> Dict[str, List[str]]:
        """Charge les patterns de catégorisation"""
        
        # En production, ceci serait chargé depuis une base de données ou un fichier
        return {
            'restaurant': ['restaurant', 'cafe', 'bar', 'bistrot', 'brasserie', 'pizzeria'],
            'alimentation': ['supermarche', 'marche', 'epicerie', 'boulangerie', 'carrefour', 'leclerc'],
            'transport': ['sncf', 'ratp', 'uber', 'taxi', 'metro', 'bus', 'tramway'],
            'carburant': ['station', 'essence', 'diesel', 'total', 'shell', 'bp'],
            'pharmacie': ['pharmacie', 'parapharmacie', 'medicament'],
            'loisirs': ['cinema', 'theatre', 'concert', 'sport', 'fitness', 'piscine'],
            'shopping': ['amazon', 'fnac', 'zara', 'h&m', 'decathlon', 'ikea'],
            'banque': ['virement', 'retrait', 'commission', 'frais', 'agios']
        }
    
    def _load_merchant_normalizer(self) -> Dict[str, str]:
        """Charge le normaliseur de marchands"""
        
        return {
            'amazon.fr': 'Amazon',
            'amazon.com': 'Amazon',
            'carrefour market': 'Carrefour',
            'carrefour express': 'Carrefour',
            'leclerc drive': 'Leclerc',
            'sncf connect': 'SNCF',
            'sncf oui': 'SNCF',
            'total energies': 'Total',
            'shell express': 'Shell'
        }
    
    def _load_financial_terms(self) -> Set[str]:
        """Charge les termes financiers"""
        
        return {
            'virement', 'prelevement', 'retrait', 'depot', 'commission', 'frais',
            'agios', 'cotisation', 'abonnement', 'facture', 'remboursement',
            'cashback', 'bonus', 'promotion', 'reduction', 'tva', 'carte'
        }


class AggregationResultProcessor:
    """Processeur spécialisé pour les résultats d'agrégation"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        # Créer un MetricsCollector par défaut si non fourni
        if metrics_collector is None:
            self._metrics_collector = MetricsCollector()
        else:
            self._metrics_collector = metrics_collector
            
        self.metrics = ResultMetrics(self._metrics_collector)
    
    def process_aggregation_results(self, aggregations: List[InternalAggregationResult],
                                  context: ProcessingContext) -> List[InternalAggregationResult]:
        """Traite et enrichit les résultats d'agrégation"""
        
        if not aggregations:
            return aggregations
        
        processed_aggregations = []
        
        for agg in aggregations:
            processed_agg = self._process_single_aggregation(agg, context)
            processed_aggregations.append(processed_agg)
        
        return processed_aggregations
    
    def _process_single_aggregation(self, aggregation: InternalAggregationResult,
                                  context: ProcessingContext) -> InternalAggregationResult:
        """Traite une agrégation individuelle"""
        
        # Enrichir les buckets avec des métadonnées
        enriched_buckets = []
        
        for bucket in aggregation.buckets:
            enriched_bucket = self._enrich_aggregation_bucket(bucket, aggregation, context)
            enriched_buckets.append(enriched_bucket)
        
        # Créer l'agrégation enrichie
        processed_agg = InternalAggregationResult(
            name=aggregation.name,
            aggregation_type=aggregation.aggregation_type,
            buckets=enriched_buckets,
            total_value=aggregation.total_value,
            metadata=self._generate_aggregation_metadata(aggregation, context)
        )
        
        processed_agg.calculate_totals()
        return processed_agg
    
    def _enrich_aggregation_bucket(self, bucket, aggregation: InternalAggregationResult,
                                 context: ProcessingContext):
        """Enrichit un bucket d'agrégation"""
        
        # Calculer des métriques additionnelles
        bucket_metadata = {
            'percentage_of_total': 0.0,
            'rank': 0,
            'category_info': {},
            'temporal_info': {}
        }
        
        # Pourcentage du total
        if aggregation.total_value and aggregation.total_value > 0:
            bucket_value = getattr(bucket, 'total_amount', 0) or getattr(bucket, 'doc_count', 0)
            bucket_metadata['percentage_of_total'] = (bucket_value / aggregation.total_value) * 100
        
        # Ajouter des informations contextuelles selon le type d'agrégation
        if aggregation.name == 'by_category':
            bucket_metadata['category_info'] = self._get_category_insights(bucket.key)
        elif aggregation.name == 'by_month':
            bucket_metadata['temporal_info'] = self._get_temporal_insights(bucket.key)
        
        # Ajouter les métadonnées au bucket
        bucket._metadata = bucket_metadata
        
        return bucket
    
    def _get_category_insights(self, category: str) -> Dict[str, Any]:
        """Génère des insights pour une catégorie"""
        
        category_insights = {
            'restaurant': {
                'icon': '🍽️',
                'description': 'Restaurants et établissements de restauration',
                'typical_frequency': 'weekly',
                'budget_category': 'variable'
            },
            'alimentation': {
                'icon': '🛒',
                'description': 'Courses alimentaires et supermarché',
                'typical_frequency': 'weekly',
                'budget_category': 'essential'
            },
            'transport': {
                'icon': '🚌',
                'description': 'Transports en commun et déplacements',
                'typical_frequency': 'daily',
                'budget_category': 'essential'
            },
            'carburant': {
                'icon': '⛽',
                'description': 'Carburant et stations-service',
                'typical_frequency': 'weekly',
                'budget_category': 'essential'
            }
        }
        
        return category_insights.get(category.lower(), {
            'icon': '💳',
            'description': 'Autres dépenses',
            'typical_frequency': 'variable',
            'budget_category': 'other'
        })
    
    def _get_temporal_insights(self, period: str) -> Dict[str, Any]:
        """Génère des insights temporels"""
        
        try:
            # Parser différents formats de période
            if len(period) == 7 and '-' in period:  # Format YYYY-MM
                year, month = period.split('-')
                month_names = {
                    '01': 'Janvier', '02': 'Février', '03': 'Mars', '04': 'Avril',
                    '05': 'Mai', '06': 'Juin', '07': 'Juillet', '08': 'Août',
                    '09': 'Septembre', '10': 'Octobre', '11': 'Novembre', '12': 'Décembre'
                }
                
                return {
                    'display_name': f"{month_names.get(month, month)} {year}",
                    'season': self._get_season(int(month)),
                    'is_current_month': period == datetime.now().strftime('%Y-%m'),
                    'quarter': f"Q{(int(month) - 1) // 3 + 1}"
                }
        except:
            pass
        
        return {
            'display_name': period,
            'season': 'unknown',
            'is_current_month': False,
            'quarter': 'unknown'
        }
    
    def _get_season(self, month: int) -> str:
        """Détermine la saison selon le mois"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        return 'unknown'
    
    def _generate_aggregation_metadata(self, aggregation: InternalAggregationResult,
                                     context: ProcessingContext) -> Dict[str, Any]:
        """Génère les métadonnées pour une agrégation"""
        
        return {
            'processing_strategy': context.processing_strategy.value,
            'bucket_count': len(aggregation.buckets),
            'total_value': aggregation.total_value,
            'aggregation_quality': self._assess_aggregation_quality(aggregation),
            'suggested_visualizations': self._suggest_visualizations(aggregation),
            'insights': self._generate_aggregation_insights(aggregation)
        }
    
    def _assess_aggregation_quality(self, aggregation: InternalAggregationResult) -> str:
        """Évalue la qualité d'une agrégation"""
        
        bucket_count = len(aggregation.buckets)
        
        if bucket_count == 0:
            return 'empty'
        elif bucket_count < 3:
            return 'limited'
        elif bucket_count < 10:
            return 'good'
        else:
            return 'comprehensive'
    
    def _suggest_visualizations(self, aggregation: InternalAggregationResult) -> List[str]:
        """Suggère des types de visualisation"""
        
        suggestions = []
        
        if aggregation.name.startswith('by_'):
            if 'month' in aggregation.name or 'date' in aggregation.name:
                suggestions.extend(['line_chart', 'area_chart', 'bar_chart'])
            elif 'category' in aggregation.name or 'merchant' in aggregation.name:
                suggestions.extend(['pie_chart', 'donut_chart', 'horizontal_bar'])
            else:
                suggestions.extend(['bar_chart', 'pie_chart'])
        
        if len(aggregation.buckets) > 10:
            suggestions.append('table')
        
        return suggestions
    
    def _generate_aggregation_insights(self, aggregation: InternalAggregationResult) -> List[str]:
        """Génère des insights pour une agrégation"""
        
        insights = []
        
        if not aggregation.buckets:
            insights.append("Aucune donnée trouvée pour cette période")
            return insights
        
        # Insights généraux
        total_buckets = len(aggregation.buckets)
        insights.append(f"{total_buckets} éléments trouvés")
        
        if aggregation.total_value:
            insights.append(f"Total: {aggregation.total_value:.2f}€")
        
        # Insights spécifiques selon le type
        if total_buckets > 0:
            # Élément dominant
            top_bucket = max(aggregation.buckets, key=lambda b: getattr(b, 'total_amount', 0) or getattr(b, 'doc_count', 0))
            if hasattr(top_bucket, 'total_amount') and top_bucket.total_amount:
                percentage = (top_bucket.total_amount / aggregation.total_value) * 100 if aggregation.total_value else 0
                insights.append(f"'{top_bucket.key}' représente {percentage:.1f}% du total")
        
        return insights


class ResultProcessorManager:
    """Gestionnaire principal pour le traitement des résultats"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        # Créer ou utiliser le MetricsCollector fourni
        if metrics_collector is None:
            self._metrics_collector = MetricsCollector()
        else:
            self._metrics_collector = metrics_collector
            
        self.financial_processor = FinancialResultProcessor(self._metrics_collector)
        self.aggregation_processor = AggregationResultProcessor(self._metrics_collector)
        self._initialized = True
    
    def process_complete_response(self, response: InternalSearchResponse,
                                context: ProcessingContext) -> InternalSearchResponse:
        """Traite une réponse complète (résultats + agrégations)"""
        
        if not self._initialized:
            logger.warning("ResultProcessorManager non initialisé")
            return response
        
        try:
            # 1. Traiter les résultats de recherche
            if response.raw_results:
                response = self.financial_processor.process_search_results(response, context)
            
            # 2. Traiter les agrégations
            if response.aggregations:
                processed_aggregations = self.aggregation_processor.process_aggregation_results(
                    response.aggregations, context
                )
                response.aggregations = processed_aggregations
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement complet: {str(e)}")
            return response
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de traitement"""
        
        return {
            'financial_processor': self.financial_processor.metrics.get_summary(),
            'aggregation_processor': self.aggregation_processor.metrics.get_summary(),
            'system_status': 'active' if self._initialized else 'inactive'
        }


# === INSTANCE GLOBALE ===
# Créer l'instance globale avec un MetricsCollector partagé
_global_metrics_collector = MetricsCollector()
result_processor_manager = ResultProcessorManager(_global_metrics_collector)


# === FONCTIONS D'UTILITÉ PRINCIPALES ===

def process_search_response(response: InternalSearchResponse,
                          user_id: int,
                          query_text: Optional[str] = None,
                          search_intention: Optional[str] = None,
                          processing_strategy: ProcessingStrategy = ProcessingStrategy.ENHANCED) -> InternalSearchResponse:
    """
    Fonction principale pour traiter une réponse de recherche
    """
    context = ProcessingContext(
        user_id=user_id,
        query_text=query_text,
        search_intention=search_intention,
        processing_strategy=processing_strategy
    )
    
    return result_processor_manager.process_complete_response(response, context)


def create_processing_context(user_id: int, **kwargs) -> ProcessingContext:
    """
    Créer un contexte de traitement avec des paramètres personnalisés
    """
    return ProcessingContext(user_id=user_id, **kwargs)


def get_processing_metrics() -> Dict[str, Any]:
    """
    Récupère les métriques de traitement
    """
    return result_processor_manager.get_processing_statistics()


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === CLASSES PRINCIPALES ===
    "FinancialResultProcessor",
    "AggregationResultProcessor", 
    "ResultProcessorManager",
    
    # === ENUMS ===
    "ProcessingStrategy",
    "RelevanceAlgorithm",
    "HighlightingMode",
    
    # === MODÈLES ===
    "ProcessingContext",
    "EnhancedResult",
    
    # === FONCTIONS PRINCIPALES ===
    "process_search_response",
    "create_processing_context",
    "get_processing_metrics",
    
    # === INSTANCE GLOBALE ===
    "result_processor_manager"
]


# === HELPERS ET UTILITAIRES ===

def get_processor_components():
    """Retourne les composants du processeur"""
    return {
        "manager": result_processor_manager,
        "financial_processor": result_processor_manager.financial_processor,
        "aggregation_processor": result_processor_manager.aggregation_processor
    }


def configure_processing_defaults(strategy: ProcessingStrategy = ProcessingStrategy.ENHANCED,
                                algorithm: RelevanceAlgorithm = RelevanceAlgorithm.FINANCIAL_WEIGHTED,
                                highlighting: HighlightingMode = HighlightingMode.INTELLIGENT):
    """Configure les paramètres par défaut du traitement"""
    
    # Cette fonction pourrait être étendue pour configurer des defaults globaux
    logger.info(f"Configuration par défaut: {strategy.value}, {algorithm.value}, {highlighting.value}")


def analyze_result_quality(response: InternalSearchResponse) -> Dict[str, Any]:
    """Analyse la qualité des résultats"""
    
    if not response or not response.raw_results:
        return {
            "overall_quality": "empty",
            "score": 0.0,
            "recommendations": ["Élargir les critères de recherche"]
        }
    
    scores = [r.score for r in response.raw_results]
    avg_score = sum(scores) / len(scores)
    high_quality_count = sum(1 for s in scores if s > 0.7)
    
    quality_analysis = {
        "overall_quality": "excellent" if avg_score > 0.8 else "good" if avg_score > 0.6 else "average",
        "score": avg_score,
        "total_results": len(response.raw_results),
        "high_quality_results": high_quality_count,
        "diversity_score": len(set(getattr(r, 'category_name', 'unknown') for r in response.raw_results[:10])) / min(10, len(response.raw_results)),
        "recommendations": []
    }
    
    # Générer des recommandations
    if avg_score < 0.5:
        quality_analysis["recommendations"].append("Considérer des termes de recherche différents")
    if high_quality_count < 3:
        quality_analysis["recommendations"].append("Affiner les filtres pour de meilleurs résultats")
    if quality_analysis["diversity_score"] < 0.3:
        quality_analysis["recommendations"].append("Élargir la recherche pour plus de diversité")
    
    return quality_analysis


# === CONFIGURATION ET INITIALISATION ===

logger.info("ResultProcessor initialisé avec succès")