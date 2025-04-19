"""
Utilitaires de gestion des poids de champs pour BM25F.

Ce module fournit des fonctions pour calculer et optimiser les poids
des différents champs dans les requêtes de recherche.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

from config_service.config import settings

logger = logging.getLogger(__name__)

# Configuration des poids par défaut pour les différents champs
DEFAULT_FIELD_WEIGHTS = {
    "description": 3.0,
    "clean_description": 3.5,
    "merchant_name": 4.0,
    "category": 2.0
}

def get_default_field_weights() -> Dict[str, float]:
    """
    Récupère les poids par défaut pour les champs.
    
    Returns:
        Dictionnaire des poids par défaut
    """
    # Charger depuis la configuration si disponible
    config_weights = getattr(settings, "FIELD_WEIGHTS", None)
    if config_weights and isinstance(config_weights, dict):
        return config_weights
    
    return DEFAULT_FIELD_WEIGHTS

def adjust_weights_for_query(query: str, field_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Ajuste les poids des champs en fonction du contexte de la requête.
    
    Args:
        query: Requête de recherche
        field_weights: Poids des champs de base (utilise les poids par défaut si None)
        
    Returns:
        Poids des champs ajustés
    """
    # Utiliser les poids par défaut si non spécifiés
    if field_weights is None:
        field_weights = get_default_field_weights()
    
    adjusted_weights = field_weights.copy()
    
    # Analyse simple de la requête pour ajuster les poids
    query_lower = query.lower()
    
    # Si la requête semble concerner un marchand spécifique
    if any(term in query_lower for term in ["chez", "achat", "magasin", "boutique", "store", "shop"]):
        adjusted_weights["merchant_name"] *= 1.5
        logger.debug(f"Augmentation du poids du champ merchant_name pour la requête: {query}")
    
    # Si la requête semble concerner une catégorie
    if any(term in query_lower for term in ["catégorie", "type", "dépense", "category"]):
        adjusted_weights["category"] *= 1.5
        logger.debug(f"Augmentation du poids du champ category pour la requête: {query}")
    
    # Si la requête contient des mots-clés spécifiques liés à la description
    if any(term in query_lower for term in ["description", "détail", "libellé", "detail"]):
        adjusted_weights["description"] *= 1.5
        adjusted_weights["clean_description"] *= 1.5
        logger.debug(f"Augmentation du poids des champs de description pour la requête: {query}")
    
    return adjusted_weights

def optimize_weights_from_feedback(
    feedback_data: List[Dict[str, Any]],
    current_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Optimise les poids des champs en utilisant les données de feedback utilisateur.
    
    Args:
        feedback_data: Liste des entrées de feedback (requêtes, résultats, évaluations)
        current_weights: Poids actuels des champs
        
    Returns:
        Poids optimisés des champs
    """
    # Utiliser les poids actuels ou par défaut
    if current_weights is None:
        current_weights = get_default_field_weights()
    
    # Si pas de données de feedback, retourner les poids actuels
    if not feedback_data:
        return current_weights
    
    optimized_weights = current_weights.copy()
    
    # Algorithme simple d'optimisation basé sur le feedback
    # En production, utiliser des techniques plus sophistiquées comme la descente de gradient
    
    # Compteur pour chaque champ (combien de fois il semble être important)
    field_relevance_count = {field: 0 for field in optimized_weights}
    total_relevant_samples = 0
    
    for entry in feedback_data:
        # Ne considérer que les entrées avec une évaluation de pertinence élevée
        relevance_score = entry.get("relevance_score", 0)
        if relevance_score < 4:  # Sur une échelle de 1-5
            continue
        
        total_relevant_samples += 1
        query = entry.get("query", "")
        result = entry.get("result", {})
        
        # Analyser pour quels champs il y a une correspondance forte
        for field in optimized_weights:
            field_value = result.get("content", {}).get(field, "")
            if not field_value or not isinstance(field_value, str):
                continue
            
            # Vérifier si des termes de la requête sont présents dans le champ
            query_terms = query.lower().split()
            field_terms = field_value.lower().split()
            
            matches = sum(1 for term in query_terms if term in field_terms)
            if matches > 0:
                field_relevance_count[field] += 1
    
    # Ajuster les poids en fonction des statistiques
    if total_relevant_samples > 0:
        # Trouver le champ le plus pertinent pour normaliser
        max_relevance = max(field_relevance_count.values())
        if max_relevance > 0:
            for field, count in field_relevance_count.items():
                # Ajuster le poids en fonction de la fréquence relative
                if count > 0:
                    relevance_ratio = count / max_relevance
                    # Ajustement progressif (éviter les changements trop brusques)
                    optimized_weights[field] = optimized_weights[field] * 0.8 + (optimized_weights[field] * relevance_ratio * 0.2)
    
    logger.info(f"Poids optimisés basés sur {total_relevant_samples} échantillons pertinents: {optimized_weights}")
    return optimized_weights

def get_field_boost_query(query_text: str, field_weights: Dict[str, float]) -> str:
    """
    Génère une requête boostée pour Whoosh.
    
    Args:
        query_text: Texte de la requête
        field_weights: Poids des champs
        
    Returns:
        Requête formatée avec les boosts
    """
    # Pour Whoosh, on peut utiliser la syntaxe "field:term^boost"
    terms = query_text.split()
    boosted_query_parts = []
    
    for term in terms:
        term_boosted_parts = []
        for field, weight in field_weights.items():
            term_boosted_parts.append(f"{field}:{term}^{weight}")
        
        # Joindre les parties de ce terme avec OR
        if term_boosted_parts:
            boosted_query_parts.append(f"({' OR '.join(term_boosted_parts)})")
    
    # Joindre tous les termes avec AND
    if boosted_query_parts:
        return " AND ".join(boosted_query_parts)
    
    return query_text  # Fallback à la requête originale si pas de termes