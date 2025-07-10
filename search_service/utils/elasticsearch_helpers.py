"""
Helpers Elasticsearch pour le Search Service - Module Principal.

Ce module centralise tous les utilitaires Elasticsearch et expose une API unifiée
pour la construction, optimisation et traitement des requêtes dans le contexte financier.

ARCHITECTURE CONSOLIDÉE:
- Toutes les classes et fonctions dans ce fichier pour éviter les imports circulaires
- QueryBuilder pour construction de requêtes complexes
- ElasticsearchHelpers pour opérations courantes
- ResultFormatter pour formatage des résultats
- Fonctions utilitaires pour optimisation

USAGE SIMPLIFIÉ:
    from search_service.utils.elasticsearch_helpers import (
        QueryBuilder, ElasticsearchHelpers, format_search_results
    )
    
    # Construction rapide
    query = ElasticsearchHelpers.build_financial_query(
        query="virement café", user_id=123
    )
    
    # Builder avancé
    builder = QueryBuilder()
    query = (builder
        .with_user_filter(123)
        .with_financial_search("virement café")
        .build())
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Set

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ET CONSTANTES ====================

class QueryStrategy(str, Enum):
    """Stratégies de requête Elasticsearch."""
    EXACT = "exact"                # Correspondance exacte uniquement
    FUZZY = "fuzzy"               # Recherche floue avec tolérance d'erreurs
    WILDCARD = "wildcard"         # Recherche partielle avec patterns
    SEMANTIC = "semantic"         # Recherche sémantique avec synonymes
    HYBRID = "hybrid"             # Combinaison optimale de toutes les stratégies

class SortStrategy(str, Enum):
    """Stratégies de tri des résultats."""
    RELEVANCE = "relevance"       # Par score de pertinence (défaut)
    DATE_DESC = "date_desc"       # Par date décroissante
    DATE_ASC = "date_asc"         # Par date croissante
    AMOUNT_DESC = "amount_desc"   # Par montant décroissant
    AMOUNT_ASC = "amount_asc"     # Par montant croissant
    MERCHANT_ASC = "merchant_asc" # Par nom de marchand alphabétique

class BoostType(str, Enum):
    """Types de boost de scoring."""
    EXACT_PHRASE = "exact_phrase"
    MERCHANT_NAME = "merchant_name"
    RECENT_TRANSACTION = "recent_transaction"
    FREQUENT_MERCHANT = "frequent_merchant"
    AMOUNT_RELEVANCE = "amount_relevance"

class AggregationType(str, Enum):
    """Types d'agrégations financières."""
    MERCHANTS = "merchants"
    CATEGORIES = "categories"
    AMOUNTS = "amounts"
    DATE_HISTOGRAM = "date_histogram"
    CURRENCY = "currency"

# Synonymes financiers pour expansion de requêtes
FINANCIAL_SYNONYMS: Dict[str, List[str]] = {
    "virement": ["transfer", "transfert", "wire", "transfer bancaire", "vir"],
    "carte": ["card", "cb", "credit card", "debit card", "visa", "mastercard", "carte bancaire"],
    "retrait": ["withdrawal", "cash", "atm", "distributeur", "retrait especes"],
    "depot": ["deposit", "dépôt", "versement", "depot especes"],
    "prelevement": ["direct debit", "prélèvement automatique", "debit", "prelev"],
    "cheque": ["check", "chèque", "cheque bancaire"],
    "cafe": ["café", "coffee", "restaurant", "bar"],
    "essence": ["carburant", "fuel", "station service", "gas"],
    "supermarche": ["supermarket", "grocery", "courses", "alimentaire"],
}

# Champs de recherche financière
FINANCIAL_SEARCH_FIELDS: List[str] = [
    "searchable_text^3",
    "merchant_name^2.5", 
    "clean_description^2",
    "primary_description^1.5",
    "provider_description^1"
]

# Champs pour highlighting
HIGHLIGHT_FIELDS: Dict[str, Dict[str, Any]] = {
    "searchable_text": {
        "fragment_size": 150,
        "number_of_fragments": 3
    },
    "merchant_name": {
        "fragment_size": 100,
        "number_of_fragments": 1
    },
    "clean_description": {
        "fragment_size": 200,
        "number_of_fragments": 2
    }
}

# Valeurs de boost par défaut
DEFAULT_BOOST_VALUES: Dict[str, float] = {
    "exact_match": 3.0,
    "phrase_match": 2.0,
    "recent_transaction": 1.5,
    "frequent_merchant": 1.3,
    "amount_relevance": 1.2
}

# Buckets pour agrégations de montants
AMOUNT_AGGREGATION_BUCKETS = [
    {"to": 10},
    {"from": 10, "to": 50},
    {"from": 50, "to": 100},
    {"from": 100, "to": 500},
    {"from": 500}
]

# ==================== STRUCTURES DE DONNÉES ====================

@dataclass
class QueryContext:
    """Contexte pour la construction de requêtes."""
    user_id: int
    query_text: str = ""
    strategy: QueryStrategy = QueryStrategy.HYBRID
    sort_strategy: SortStrategy = SortStrategy.RELEVANCE
    filters: Dict[str, Any] = field(default_factory=dict)
    boost_recent: bool = True
    include_synonyms: bool = True
    max_results: int = 20
    highlight: bool = True

@dataclass 
class FormattedHit:
    """Résultat formaté d'une recherche."""
    id: str
    score: float
    source: Dict[str, Any]
    highlights: Dict[str, List[str]] = field(default_factory=dict)
    explanation: Optional[Dict[str, Any]] = None

@dataclass
class AggregationResult:
    """Résultat d'agrégation formaté."""
    name: str
    buckets: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    other_doc_count: int = 0

# ==================== QUERY BUILDER ====================

class QueryBuilder:
    """
    Builder pour construire des requêtes Elasticsearch optimisées.
    
    Permet la construction fluide de requêtes complexes avec validation
    et optimisation automatique pour le domaine financier.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'QueryBuilder':
        """Remet à zéro le builder."""
        self._query_body = {
            "query": {"match_all": {}},
            "size": 20,
            "from": 0
        }
        self._must_clauses = []
        self._should_clauses = []
        self._filter_clauses = []
        self._must_not_clauses = []
        return self
    
    def with_user_filter(self, user_id: int) -> 'QueryBuilder':
        """Ajoute un filtre utilisateur obligatoire."""
        if user_id and user_id > 0:
            self._filter_clauses.append({"term": {"user_id": user_id}})
        return self
    
    def with_text_search(self, query_text: str, fields: List[str] = None) -> 'QueryBuilder':
        """Ajoute une recherche textuelle."""
        if not query_text or not query_text.strip():
            return self
        
        fields = fields or FINANCIAL_SEARCH_FIELDS
        
        # Recherche multi_match principale
        self._must_clauses.append({
            "multi_match": {
                "query": query_text.strip(),
                "fields": fields,
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        })
        
        return self
    
    def with_financial_search(self, query_text: str) -> 'QueryBuilder':
        """Ajoute une recherche financière optimisée avec synonymes."""
        if not query_text or not query_text.strip():
            return self
        
        query_text = query_text.strip().lower()
        
        # Recherche principale
        self.with_text_search(query_text, FINANCIAL_SEARCH_FIELDS)
        
        # Ajout des synonymes
        synonyms = self._get_synonyms(query_text)
        for synonym in synonyms:
            self._should_clauses.append({
                "multi_match": {
                    "query": synonym,
                    "fields": FINANCIAL_SEARCH_FIELDS,
                    "type": "phrase",
                    "boost": 0.8
                }
            })
        
        # Boost pour correspondance exacte dans merchant_name
        self._should_clauses.append({
            "match_phrase": {
                "merchant_name": {
                    "query": query_text,
                    "boost": DEFAULT_BOOST_VALUES["exact_match"]
                }
            }
        })
        
        return self
    
    def with_amount_filter(self, min_amount: float = None, max_amount: float = None) -> 'QueryBuilder':
        """Ajoute un filtre de montant."""
        if min_amount is not None or max_amount is not None:
            range_filter = {"range": {"amount": {}}}
            if min_amount is not None:
                range_filter["range"]["amount"]["gte"] = min_amount
            if max_amount is not None:
                range_filter["range"]["amount"]["lte"] = max_amount
            self._filter_clauses.append(range_filter)
        return self
    
    def with_date_filter(self, start_date: str = None, end_date: str = None) -> 'QueryBuilder':
        """Ajoute un filtre de date."""
        if start_date or end_date:
            range_filter = {"range": {"transaction_date": {}}}
            if start_date:
                range_filter["range"]["transaction_date"]["gte"] = start_date
            if end_date:
                range_filter["range"]["transaction_date"]["lte"] = end_date
            self._filter_clauses.append(range_filter)
        return self
    
    def with_category_filter(self, categories: List[str]) -> 'QueryBuilder':
        """Ajoute un filtre de catégories."""
        if categories:
            self._filter_clauses.append({
                "terms": {"category_id": categories}
            })
        return self
    
    def with_merchant_filter(self, merchants: List[str]) -> 'QueryBuilder':
        """Ajoute un filtre de marchands."""
        if merchants:
            self._filter_clauses.append({
                "terms": {"merchant_name.keyword": merchants}
            })
        return self
    
    def with_recency_boost(self, boost_factor: float = 1.5) -> 'QueryBuilder':
        """Ajoute un boost pour les transactions récentes."""
        if boost_factor > 1.0:
            self._should_clauses.append({
                "range": {
                    "transaction_date": {
                        "gte": "now-30d",
                        "boost": boost_factor
                    }
                }
            })
        return self
    
    def with_sort(self, strategy: SortStrategy = SortStrategy.RELEVANCE) -> 'QueryBuilder':
        """Configure le tri des résultats."""
        sort_configs = {
            SortStrategy.RELEVANCE: [{"_score": {"order": "desc"}}],
            SortStrategy.DATE_DESC: [{"transaction_date": {"order": "desc"}}],
            SortStrategy.DATE_ASC: [{"transaction_date": {"order": "asc"}}],
            SortStrategy.AMOUNT_DESC: [{"amount": {"order": "desc"}}],
            SortStrategy.AMOUNT_ASC: [{"amount": {"order": "asc"}}],
            SortStrategy.MERCHANT_ASC: [{"merchant_name.keyword": {"order": "asc"}}]
        }
        
        self._query_body["sort"] = sort_configs.get(strategy, sort_configs[SortStrategy.RELEVANCE])
        return self
    
    def with_pagination(self, size: int = 20, from_: int = 0) -> 'QueryBuilder':
        """Configure la pagination."""
        self._query_body["size"] = max(0, min(size, 1000))
        self._query_body["from"] = max(0, min(from_, 10000))
        return self
    
    def with_highlighting(self, enable: bool = True) -> 'QueryBuilder':
        """Active ou désactive le highlighting."""
        if enable:
            self._query_body["highlight"] = {
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
                "fields": HIGHLIGHT_FIELDS
            }
        elif "highlight" in self._query_body:
            del self._query_body["highlight"]
        return self
    
    def with_aggregations(self, agg_types: List[AggregationType] = None) -> 'QueryBuilder':
        """Ajoute des agrégations."""
        if not agg_types:
            return self
        
        aggs = {}
        
        for agg_type in agg_types:
            if agg_type == AggregationType.MERCHANTS:
                aggs["merchants"] = {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": 10
                    }
                }
            elif agg_type == AggregationType.CATEGORIES:
                aggs["categories"] = {
                    "terms": {
                        "field": "category_id",
                        "size": 20
                    }
                }
            elif agg_type == AggregationType.AMOUNTS:
                aggs["amounts"] = {
                    "range": {
                        "field": "amount",
                        "ranges": AMOUNT_AGGREGATION_BUCKETS
                    }
                }
            elif agg_type == AggregationType.DATE_HISTOGRAM:
                aggs["dates"] = {
                    "date_histogram": {
                        "field": "transaction_date",
                        "calendar_interval": "month"
                    }
                }
        
        if aggs:
            self._query_body["aggs"] = aggs
        
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construit la requête finale."""
        # Construction de la clause bool
        bool_query = {}
        
        if self._must_clauses:
            bool_query["must"] = self._must_clauses
        
        if self._should_clauses:
            bool_query["should"] = self._should_clauses
            bool_query["minimum_should_match"] = 0
        
        if self._filter_clauses:
            bool_query["filter"] = self._filter_clauses
        
        if self._must_not_clauses:
            bool_query["must_not"] = self._must_not_clauses
        
        # Si on a des clauses, utiliser bool, sinon match_all
        if bool_query:
            self._query_body["query"] = {"bool": bool_query}
        
        return self._query_body.copy()
    
    def _get_synonyms(self, query_text: str) -> List[str]:
        """Récupère les synonymes pour un texte de requête."""
        words = query_text.lower().split()
        synonyms = []
        
        for word in words:
            if word in FINANCIAL_SYNONYMS:
                synonyms.extend(FINANCIAL_SYNONYMS[word][:3])  # Limite à 3 synonymes par mot
        
        return synonyms

# ==================== ELASTICSEARCH HELPERS ====================

class ElasticsearchHelpers:
    """
    Classe utilitaire pour les opérations Elasticsearch courantes.
    
    Fournit des méthodes statiques pour les cas d'usage fréquents
    sans nécessiter de configuration complexe.
    """
    
    @staticmethod
    def build_financial_query(query: str, user_id: int, **kwargs) -> Dict[str, Any]:
        """
        Construit rapidement une requête financière optimisée.
        
        Args:
            query: Texte de recherche
            user_id: ID utilisateur
            **kwargs: Options supplémentaires (filters, sort, size, etc.)
            
        Returns:
            Dictionnaire de requête Elasticsearch
        """
        builder = QueryBuilder()
        
        # Configuration de base
        builder.with_user_filter(user_id)
        builder.with_financial_search(query)
        
        # Application des options
        if 'amount_min' in kwargs or 'amount_max' in kwargs:
            builder.with_amount_filter(kwargs.get('amount_min'), kwargs.get('amount_max'))
        
        if 'date_start' in kwargs or 'date_end' in kwargs:
            builder.with_date_filter(kwargs.get('date_start'), kwargs.get('date_end'))
        
        if 'categories' in kwargs:
            builder.with_category_filter(kwargs['categories'])
        
        if 'merchants' in kwargs:
            builder.with_merchant_filter(kwargs['merchants'])
        
        # Configuration avancée
        sort_strategy = kwargs.get('sort', SortStrategy.RELEVANCE)
        builder.with_sort(sort_strategy)
        
        size = kwargs.get('size', 20)
        from_ = kwargs.get('from', 0)
        builder.with_pagination(size, from_)
        
        if kwargs.get('highlight', True):
            builder.with_highlighting()
        
        if kwargs.get('aggregations'):
            builder.with_aggregations(kwargs['aggregations'])
        
        if kwargs.get('boost_recent', True):
            builder.with_recency_boost()
        
        return builder.build()
    
    @staticmethod
    def build_simple_search(query: str, user_id: int, size: int = 20) -> Dict[str, Any]:
        """
        Construit une requête de recherche simple.
        
        Args:
            query: Texte de recherche
            user_id: ID utilisateur
            size: Nombre de résultats
            
        Returns:
            Dictionnaire de requête Elasticsearch
        """
        return {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                        {"multi_match": {
                            "query": query,
                            "fields": FINANCIAL_SEARCH_FIELDS,
                            "type": "best_fields"
                        }}
                    ]
                }
            },
            "size": size,
            "highlight": {
                "fields": HIGHLIGHT_FIELDS
            }
        }
    
    @staticmethod
    def build_filter_query(user_id: int, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construit une requête basée uniquement sur des filtres.
        
        Args:
            user_id: ID utilisateur
            filters: Dictionnaire de filtres
            
        Returns:
            Dictionnaire de requête Elasticsearch
        """
        filter_clauses = [{"term": {"user_id": user_id}}]
        
        for field, value in filters.items():
            if field == "amount_range":
                if isinstance(value, dict):
                    range_filter = {"range": {"amount": {}}}
                    if "min" in value:
                        range_filter["range"]["amount"]["gte"] = value["min"]
                    if "max" in value:
                        range_filter["range"]["amount"]["lte"] = value["max"]
                    filter_clauses.append(range_filter)
            elif field == "date_range":
                if isinstance(value, dict):
                    range_filter = {"range": {"transaction_date": {}}}
                    if "start" in value:
                        range_filter["range"]["transaction_date"]["gte"] = value["start"]
                    if "end" in value:
                        range_filter["range"]["transaction_date"]["lte"] = value["end"]
                    filter_clauses.append(range_filter)
            elif field in ["category_id", "merchant_name", "transaction_type"]:
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})
        
        return {
            "query": {
                "bool": {
                    "filter": filter_clauses
                }
            },
            "sort": [{"transaction_date": {"order": "desc"}}]
        }
    
    @staticmethod
    def build_aggregation_query(user_id: int, agg_types: List[str]) -> Dict[str, Any]:
        """
        Construit une requête d'agrégation.
        
        Args:
            user_id: ID utilisateur
            agg_types: Types d'agrégations demandées
            
        Returns:
            Dictionnaire de requête Elasticsearch
        """
        builder = QueryBuilder()
        builder.with_user_filter(user_id)
        
        # Conversion des types d'agrégation
        agg_enum_types = []
        for agg_type in agg_types:
            try:
                agg_enum_types.append(AggregationType(agg_type))
            except ValueError:
                logger.warning(f"Type d'agrégation non reconnu: {agg_type}")
        
        builder.with_aggregations(agg_enum_types)
        builder.with_pagination(size=0)  # Pas de résultats, juste agrégations
        
        return builder.build()

# ==================== RESULT FORMATTER ====================

class ResultFormatter:
    """
    Classe pour le formatage des résultats Elasticsearch.
    
    Transforme les réponses brutes d'Elasticsearch en objets
    structurés plus faciles à manipuler.
    """
    
    def __init__(self, include_score: bool = True, include_highlights: bool = True):
        self.include_score = include_score
        self.include_highlights = include_highlights
    
    def format_search_results(self, es_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formate les résultats d'une recherche Elasticsearch.
        
        Args:
            es_response: Réponse brute d'Elasticsearch
            
        Returns:
            Résultats formatés
        """
        formatted = {
            "total": self._extract_total(es_response),
            "hits": [],
            "aggregations": {},
            "took": es_response.get("took", 0),
            "max_score": es_response.get("hits", {}).get("max_score", 0)
        }
        
        # Formatage des hits
        hits = es_response.get("hits", {}).get("hits", [])
        for hit in hits:
            formatted_hit = self._format_hit(hit)
            formatted["hits"].append(formatted_hit)
        
        # Formatage des agrégations
        if "aggregations" in es_response:
            formatted["aggregations"] = self._format_aggregations(es_response["aggregations"])
        
        return formatted
    
    def format_hit(self, hit: Dict[str, Any]) -> FormattedHit:
        """
        Formate un hit individuel.
        
        Args:
            hit: Hit brut d'Elasticsearch
            
        Returns:
            Hit formaté
        """
        formatted_hit = FormattedHit(
            id=hit.get("_id", ""),
            score=hit.get("_score", 0.0) if self.include_score else 0.0,
            source=hit.get("_source", {})
        )
        
        if self.include_highlights and "highlight" in hit:
            formatted_hit.highlights = hit["highlight"]
        
        if "explain" in hit:
            formatted_hit.explanation = hit["explain"]
        
        return formatted_hit
    
    def _format_hit(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Formate un hit en dictionnaire."""
        formatted = {
            "id": hit.get("_id", ""),
            "source": hit.get("_source", {})
        }
        
        if self.include_score:
            formatted["score"] = hit.get("_score", 0.0)
        
        if self.include_highlights and "highlight" in hit:
            formatted["highlights"] = hit["highlight"]
        
        return formatted
    
    def _extract_total(self, es_response: Dict[str, Any]) -> int:
        """Extrait le nombre total de résultats."""
        hits = es_response.get("hits", {})
        total = hits.get("total", 0)
        
        if isinstance(total, dict):
            return total.get("value", 0)
        return total
    
    def _format_aggregations(self, aggregations: Dict[str, Any]) -> Dict[str, Any]:
        """Formate les agrégations."""
        formatted_aggs = {}
        
        for agg_name, agg_data in aggregations.items():
            if "buckets" in agg_data:
                formatted_aggs[agg_name] = {
                    "buckets": agg_data["buckets"],
                    "total_count": agg_data.get("doc_count", 0),
                    "other_doc_count": agg_data.get("sum_other_doc_count", 0)
                }
            elif "value" in agg_data:
                formatted_aggs[agg_name] = {
                    "value": agg_data["value"]
                }
        
        return formatted_aggs

# ==================== SCORE CALCULATOR ====================

class ScoreCalculator:
    """
    Calculateur de scores personnalisés pour les résultats de recherche.
    
    Combine différents facteurs pour calculer un score de pertinence
    adapté au domaine financier.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "elasticsearch_score": 0.7,
            "recency_factor": 0.2,
            "user_preference": 0.1
        }
    
    def calculate_composite_score(self, elasticsearch_score: float, 
                                recency_factor: float = 1.0,
                                user_preference_factor: float = 1.0) -> float:
        """
        Calcule un score composite basé sur plusieurs facteurs.
        
        Args:
            elasticsearch_score: Score Elasticsearch de base
            recency_factor: Facteur de récence (0.0-2.0)
            user_preference_factor: Facteur de préférence utilisateur (0.0-2.0)
            
        Returns:
            Score composite normalisé
        """
        if elasticsearch_score <= 0:
            return 0.0
        
        composite_score = (
            elasticsearch_score * self.weights["elasticsearch_score"] +
            elasticsearch_score * recency_factor * self.weights["recency_factor"] +
            elasticsearch_score * user_preference_factor * self.weights["user_preference"]
        )
        
        return round(composite_score, 2)
    
    def calculate_recency_factor(self, transaction_date: str) -> float:
        """
        Calcule un facteur de récence basé sur la date de transaction.
        
        Args:
            transaction_date: Date de transaction (ISO format)
            
        Returns:
            Facteur de récence (0.0-2.0)
        """
        try:
            trans_date = datetime.fromisoformat(transaction_date.replace('Z', '+00:00'))
            now = datetime.now()
            days_ago = (now - trans_date).days
            
            if days_ago <= 7:
                return 2.0  # Très récent
            elif days_ago <= 30:
                return 1.5  # Récent
            elif days_ago <= 90:
                return 1.2  # Modérément récent
            else:
                return 1.0  # Normal
        except (ValueError, AttributeError):
            return 1.0

# ==================== HIGHLIGHT PROCESSOR ====================

class HighlightProcessor:
    """
    Processeur pour les highlights Elasticsearch.
    
    Traite et améliore les highlights retournés par Elasticsearch
    pour une meilleure présentation à l'utilisateur.
    """
    
    def __init__(self, max_fragment_length: int = 200):
        self.max_fragment_length = max_fragment_length
    
    def process_highlights(self, highlights: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Traite les highlights bruts d'Elasticsearch.
        
        Args:
            highlights: Highlights bruts
            
        Returns:
            Highlights traités
        """
        processed = {}
        
        for field, fragments in highlights.items():
            processed_fragments = []
            for fragment in fragments:
                # Nettoyage et troncature
                cleaned = self._clean_fragment(fragment)
                if len(cleaned) > self.max_fragment_length:
                    cleaned = cleaned[:self.max_fragment_length] + "..."
                processed_fragments.append(cleaned)
            
            processed[field] = processed_fragments
        
        return processed
    
    def _clean_fragment(self, fragment: str) -> str:
        """Nettoie un fragment de highlight."""
        # Suppression des espaces multiples
        cleaned = re.sub(r'\s+', ' ', fragment)
        return cleaned.strip()

# ==================== FONCTIONS UTILITAIRES ====================

def format_search_results(es_response: Dict[str, Any], 
                         include_score: bool = True,
                         include_highlights: bool = True) -> Dict[str, Any]:
    """
    Fonction utilitaire pour formater les résultats de recherche.
    
    Args:
        es_response: Réponse Elasticsearch
        include_score: Inclure les scores
        include_highlights: Inclure les highlights
        
    Returns:
        Résultats formatés
    """
    formatter = ResultFormatter(include_score, include_highlights)
    return formatter.format_search_results(es_response)

def extract_highlights(hit: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extrait les highlights d'un hit Elasticsearch.
    
    Args:
        hit: Hit Elasticsearch
        
    Returns:
        Dictionnaire des highlights
    """
    return hit.get("highlight", {})

def calculate_relevance_score(elasticsearch_score: float, 
                            recency_factor: float = 1.0,
                            user_preference_factor: float = 1.0) -> float:
    """
    Calcule un score de pertinence composite.
    
    Args:
        elasticsearch_score: Score Elasticsearch de base
        recency_factor: Facteur de récence (0.0-2.0)
        user_preference_factor: Facteur de préférence utilisateur (0.0-2.0)
        
    Returns:
        Score composite normalisé
    """
    calculator = ScoreCalculator()
    return calculator.calculate_composite_score(
        elasticsearch_score, recency_factor, user_preference_factor
    )

def optimize_query_for_performance(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimise une requête pour améliorer les performances.
    
    Args:
        query: Requête Elasticsearch
        
    Returns:
        Requête optimisée
    """
    optimized = query.copy()
    
    # Limitation de la taille des résultats
    if optimized.get("size", 0) > 100:
        optimized["size"] = 100
        logger.warning("Taille des résultats limitée à 100 pour optimisation")
    
    # Limitation de l'offset
    if optimized.get("from", 0) > 1000:
        optimized["from"] = 1000
        logger.warning("Offset limité à 1000 pour optimisation")
    
    # Ajout d'un timeout
    if "timeout" not in optimized:
        optimized["timeout"] = "10s"
    
    return optimized

def build_suggestion_query(query: str, user_id: int, size: int = 5) -> Dict[str, Any]:
    """
    Construit une requête pour les suggestions de recherche.
    
    Args:
        query: Texte de recherche partiel
        user_id: ID utilisateur
        size: Nombre de suggestions
        
    Returns:
        Requête de suggestion
    """
    return {
        "suggest": {
            "merchant_suggestion": {
                "prefix": query,
                "completion": {
                    "field": "merchant_name.suggest",
                    "size": size,
                    "contexts": {
                        "user_id": [str(user_id)]
                    }
                }
            }
        }
    }

def validate_query_structure(query: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valide la structure d'une requête Elasticsearch.
    
    Args:
        query: Requête à valider
        
    Returns:
        Tuple (is_valid, errors)
    """
    errors = []
    
    if not isinstance(query, dict):
        errors.append("Query must be a dictionary")
        return False, errors
    
    if "query" not in query:
        errors.append("Missing 'query' field")
    
    if "size" in query:
        size = query["size"]
        if not isinstance(size, int) or size < 0:
            errors.append("'size' must be a non-negative integer")
        elif size > 1000:
            errors.append("'size' is too large (max 1000)")
    
    if "from" in query:
        from_val = query["from"]
        if not isinstance(from_val, int) or from_val < 0:
            errors.append("'from' must be a non-negative integer")
        elif from_val > 10000:
            errors.append("'from' is too large (max 10000)")
    
    return len(errors) == 0, errors

# ==================== FACTORY FUNCTIONS ====================

def create_query_builder() -> QueryBuilder:
    """Crée un QueryBuilder configuré."""
    return QueryBuilder()

def create_result_formatter(**kwargs) -> ResultFormatter:
    """Crée un ResultFormatter configuré."""
    return ResultFormatter(**kwargs)

def create_score_calculator(**kwargs) -> ScoreCalculator:
    """Crée un ScoreCalculator configuré."""
    return ScoreCalculator(**kwargs)

def create_highlight_processor(**kwargs) -> HighlightProcessor:
    """Crée un HighlightProcessor configuré."""
    return HighlightProcessor(**kwargs)

# ==================== EXPORTS PRINCIPAUX ====================

__all__ = [
    # Classes principales
    'QueryBuilder',
    'ElasticsearchHelpers',
    'ResultFormatter', 
    'ScoreCalculator',
    'HighlightProcessor',
    
    # Structures de données
    'QueryContext',
    'FormattedHit',
    'AggregationResult',
    'QueryStrategy',
    'SortStrategy',
    'BoostType',
    'AggregationType',
    
    # Fonctions utilitaires
    'format_search_results',
    'extract_highlights',
    'calculate_relevance_score',
    'optimize_query_for_performance',
    'build_suggestion_query',
    'validate_query_structure',
    
    # Factory functions
    'create_query_builder',
    'create_result_formatter',
    'create_score_calculator',
    'create_highlight_processor',
    
    # Configuration
    'FINANCIAL_SYNONYMS',
    'FINANCIAL_SEARCH_FIELDS',
    'HIGHLIGHT_FIELDS',
    'DEFAULT_BOOST_VALUES',
    'AMOUNT_AGGREGATION_BUCKETS'
]