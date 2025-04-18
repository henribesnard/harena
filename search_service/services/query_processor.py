"""
Processeur de requêtes pour le service de recherche.

Ce module est responsable de l'analyse, l'expansion et la structuration
des requêtes de recherche.
"""
import logging
from typing import Optional, Dict, Any, List
import asyncio
import json

from search_service.schemas.query import (
    SearchQuery, SearchParameters, FilterSet, DateRange,
    AggregationType, GroupBy, SearchType
)
from search_service.core.config import settings
from search_service.utils.text_processing import normalize_text, extract_entities
from search_service.storage.cache import get_cache, set_cache

logger = logging.getLogger(__name__)

async def process_query(
    query: SearchQuery,
    user_id: int,
    db = None
) -> SearchQuery:
    """
    Traite et enrichit une requête de recherche.
    
    Args:
        query: Requête de recherche originale
        user_id: ID de l'utilisateur
        db: Session de base de données (optionnelle)
        
    Returns:
        Requête enrichie et structurée
    """
    # Logger la requête entrante
    logger.info(f"Traitement de la requête: {query.query.text} pour user_id={user_id}")
    
    # Normaliser le texte de la requête
    normalized_text = normalize_text(query.query.text)
    
    # Définir les paramètres par défaut si non spécifiés
    if not query.search_params:
        query.search_params = SearchParameters(
            lexical_weight=settings.DEFAULT_LEXICAL_WEIGHT,
            semantic_weight=settings.DEFAULT_SEMANTIC_WEIGHT,
            top_k_initial=settings.DEFAULT_TOP_K_INITIAL,
            top_k_final=settings.DEFAULT_TOP_K_FINAL
        )
    
    # Expansion de la requête si non déjà fournie
    if not query.query.expanded_text:
        expanded_text = await expand_query(normalized_text, user_id)
        query.query.expanded_text = expanded_text
    
    # Extraire et structurer les filtres implicites du texte
    extracted_filters = await extract_filters_from_text(normalized_text)
    
    # Fusionner avec les filtres explicites
    merged_filters = merge_filters(query.filters, extracted_filters)
    query.filters = merged_filters
    
    # Détecter le type d'agrégation nécessaire si non spécifié
    if not query.aggregation:
        detected_aggregation = await detect_aggregation_type(normalized_text)
        query.aggregation = detected_aggregation
    
    logger.debug(f"Requête traitée: {query.query.expanded_text or query.query.text}")
    logger.debug(f"Filtres: {json.dumps(query.filters.dict() if query.filters else {})}")
    
    return query

async def expand_query(text: str, user_id: int) -> str:
    """
    Enrichit la requête avec des termes supplémentaires pour améliorer la recherche.
    
    Args:
        text: Texte de la requête originale
        user_id: ID de l'utilisateur
        
    Returns:
        Texte de requête enrichi
    """
    # Vérifier le cache
    cache_key = f"expand_query:{user_id}:{text}"
    cached_expansion = await get_cache(cache_key)
    
    if cached_expansion:
        logger.debug(f"Expansion de requête récupérée du cache: {text}")
        return cached_expansion
    
    # Exemple simple d'expansion de requête basé sur des règles
    # Dans une implémentation réelle, on pourrait utiliser un LLM ou un modèle dédié
    expanded_terms = []
    
    # Expansion basée sur des règles financières simples
    if "dépenses" in text.lower():
        expanded_terms.extend(["paiement", "achat", "débit"])
    
    if "revenus" in text.lower():
        expanded_terms.extend(["salaire", "virement", "crédit"])
    
    if "amazon" in text.lower():
        expanded_terms.extend(["amazon.fr", "amazon prime", "amzn"])
    
    # Ajout des variations orthographiques courantes
    if "abonnement" in text.lower():
        expanded_terms.append("souscription")
    
    if "banque" in text.lower():
        expanded_terms.extend(["bancaire", "crédit"])
    
    # Construction de la requête enrichie
    if expanded_terms:
        expanded_text = f"{text} {' '.join(expanded_terms)}"
    else:
        expanded_text = text
    
    # Mettre en cache
    await set_cache(cache_key, expanded_text, ttl=86400)  # Cache pour 24h
    
    return expanded_text

async def extract_filters_from_text(text: str) -> Optional[FilterSet]:
    """
    Extrait des filtres structurés à partir du texte de la requête.
    
    Args:
        text: Texte de la requête
        
    Returns:
        Filtres structurés extraits ou None
    """
    # Extraction d'entités et expressions temporelles
    entities = extract_entities(text)
    
    # Initialisation des filtres
    filters = FilterSet()
    
    # Traitement des expressions temporelles
    if "date_expressions" in entities:
        date_range = DateRange()
        
        for expr in entities["date_expressions"]:
            if expr["type"] == "relative":
                # Expressions relatives comme "le mois dernier", "la semaine dernière"
                date_range.relative = expr["value"]
            elif expr["type"] == "absolute":
                # Dates absolues extraites
                if expr["position"] == "start":
                    date_range.start = expr["value"]
                elif expr["position"] == "end":
                    date_range.end = expr["value"]
        
        if date_range.relative or date_range.start or date_range.end:
            filters.date_range = date_range
    
    # Traitement des montants
    if "amounts" in entities:
        from search_service.schemas.query import AmountRange
        amount_range = AmountRange()
        
        for amount in entities["amounts"]:
            if amount["type"] == "min":
                amount_range.min = amount["value"]
            elif amount["type"] == "max":
                amount_range.max = amount["value"]
        
        if amount_range.min is not None or amount_range.max is not None:
            filters.amount_range = amount_range
    
    # Extraction des catégories mentionnées
    if "categories" in entities:
        filters.categories = entities["categories"]
    
    # Extraction des marchands mentionnés
    if "merchants" in entities:
        filters.merchants = entities["merchants"]
    
    # Détection du type d'opération
    if "dépenses" in text.lower() or "dépensé" in text.lower() or "payé" in text.lower():
        filters.operation_types = ["debit"]
    elif "revenus" in text.lower() or "reçu" in text.lower() or "gagné" in text.lower():
        filters.operation_types = ["credit"]
    
    # Si aucun filtre n'a été extrait, retourner None
    if (not filters.date_range and not filters.amount_range and 
            not filters.categories and not filters.merchants and 
            not filters.operation_types):
        return None
    
    return filters

def merge_filters(explicit_filters: Optional[FilterSet], extracted_filters: Optional[FilterSet]) -> FilterSet:
    """
    Fusionne les filtres explicites et extraits.
    
    Args:
        explicit_filters: Filtres fournis explicitement dans la requête
        extracted_filters: Filtres extraits du texte
        
    Returns:
        Filtres fusionnés
    """
    if not explicit_filters and not extracted_filters:
        return FilterSet()
    
    if explicit_filters and not extracted_filters:
        return explicit_filters
    
    if not explicit_filters and extracted_filters:
        return extracted_filters
    
    # Fusion des filtres en privilégiant les filtres explicites
    merged = FilterSet()
    
    # Date range
    if explicit_filters.date_range:
        merged.date_range = explicit_filters.date_range
    elif extracted_filters.date_range:
        merged.date_range = extracted_filters.date_range
    
    # Amount range
    if explicit_filters.amount_range:
        merged.amount_range = explicit_filters.amount_range
    elif extracted_filters.amount_range:
        merged.amount_range = extracted_filters.amount_range
    
    # Categories (union)
    if explicit_filters.categories or extracted_filters.categories:
        categories_set = set()
        if explicit_filters.categories:
            categories_set.update(explicit_filters.categories)
        if extracted_filters.categories:
            categories_set.update(extracted_filters.categories)
        merged.categories = list(categories_set)
    
    # Merchants (union)
    if explicit_filters.merchants or extracted_filters.merchants:
        merchants_set = set()
        if explicit_filters.merchants:
            merchants_set.update(explicit_filters.merchants)
        if extracted_filters.merchants:
            merchants_set.update(extracted_filters.merchants)
        merged.merchants = list(merchants_set)
    
    # Operation types (explicit priorité)
    if explicit_filters.operation_types:
        merged.operation_types = explicit_filters.operation_types
    elif extracted_filters.operation_types:
        merged.operation_types = extracted_filters.operation_types
    
    # Custom filters (explicit seulement)
    if explicit_filters.custom_filters:
        merged.custom_filters = explicit_filters.custom_filters
    
    return merged

async def detect_aggregation_type(text: str) -> Optional[Dict[str, Any]]:
    """
    Détecte le type d'agrégation nécessaire à partir de la requête.
    
    Args:
        text: Texte de la requête
        
    Returns:
        Configuration d'agrégation ou None
    """
    from search_service.schemas.query import AggregationRequest, AggregationType, GroupBy
    
    # Détection de mots-clés pour les agrégations
    text_lower = text.lower()
    
    # Détection de somme
    if any(keyword in text_lower for keyword in ["combien", "total", "somme", "montant"]):
        return AggregationRequest(type=AggregationType.SUM, field="amount")
    
    # Détection de moyenne
    if any(keyword in text_lower for keyword in ["moyenne", "moyen", "en moyenne"]):
        return AggregationRequest(type=AggregationType.AVG, field="amount")
    
    # Détection de ratio
    if any(keyword in text_lower for keyword in ["ratio", "proportion", "pourcentage"]):
        return AggregationRequest(type=AggregationType.RATIO, field="amount")
    
    # Détection de groupage par catégorie
    if any(keyword in text_lower for keyword in ["par catégorie", "catégories"]):
        if any(kw in text_lower for kw in ["combien", "total", "somme"]):
            return AggregationRequest(
                type=AggregationType.SUM,
                field="amount",
                group_by=GroupBy.CATEGORY
            )
    
    # Détection de groupage par période
    for period, group_by in [
        (["par mois", "mensuel"], GroupBy.MONTH),
        (["par semaine", "hebdomadaire"], GroupBy.WEEK),
        (["par jour", "quotidien"], GroupBy.DAY),
        (["par an", "annuel"], GroupBy.YEAR)
    ]:
        if any(keyword in text_lower for keyword in period):
            if any(kw in text_lower for kw in ["combien", "total", "somme"]):
                return AggregationRequest(
                    type=AggregationType.SUM,
                    field="amount",
                    group_by=group_by
                )
    
    # Aucune agrégation détectée
    return None