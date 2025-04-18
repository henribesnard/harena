"""
Utilitaires pour le traitement de texte.

Ce module fournit des fonctionnalités pour normaliser le texte
et extraire des entités significatives des requêtes.
"""
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Expressions régulières pour l'extraction d'entités
DATE_EXPRESSIONS = {
    # Expressions relatives
    r"le mois dernier|mois précédent": "last_month",
    r"la semaine dernière|semaine précédente": "last_week",
    r"cette semaine|semaine en cours": "this_week",
    r"ce mois-ci|mois en cours": "this_month",
    r"l'année dernière|année précédente|l'an dernier": "last_year",
    r"cette année|année en cours": "this_year",
    r"hier": "yesterday",
    r"aujourd'hui": "today",
    r"les 30 derniers jours": "last_30_days",
    r"les trois derniers mois": "last_90_days",
    # On pourrait ajouter plus d'expressions
}

# Expressions régulières pour l'extraction de montants
AMOUNT_EXPRESSIONS = {
    r"(?:plus|supérieur) (?:de|à) (\d+(?:[.,]\d+)?)\s*(?:€|euros|EUR)": {"type": "min", "group": 1},
    r"(?:moins|inférieur) (?:de|à) (\d+(?:[.,]\d+)?)\s*(?:€|euros|EUR)": {"type": "max", "group": 1},
    r"(\d+(?:[.,]\d+)?)\s*(?:€|euros|EUR) (?:et plus|minimum)": {"type": "min", "group": 1},
    r"(\d+(?:[.,]\d+)?)\s*(?:€|euros|EUR) (?:maximum|au maximum)": {"type": "max", "group": 1},
    r"entre (\d+(?:[.,]\d+)?)\s*(?:€|euros|EUR)? et (\d+(?:[.,]\d+)?)\s*(?:€|euros|EUR)": {"type": "range", "min_group": 1, "max_group": 2}
}

# Catégories financières communes (exemple simplifié)
COMMON_CATEGORIES = [
    "alimentation", "courses", "restaurant", "transport", "essence", "carburant",
    "logement", "loyer", "électricité", "eau", "gaz", "internet", "téléphone",
    "loisirs", "voyage", "shopping", "vêtements", "santé", "assurance", "banque",
    "épargne", "salaire", "revenu", "impôts", "taxes", "éducation", "abonnement"
]

# Marchands communs (exemple)
COMMON_MERCHANTS = [
    "amazon", "carrefour", "leclerc", "auchan", "intermarché", "netflix", 
    "spotify", "apple", "google", "uber", "orange", "sfr", "bouygues", 
    "free", "sncf", "ratp", "edf", "engie", "veolia", "la poste", "paypal"
]

def normalize_text(text: str) -> str:
    """
    Normalise le texte pour le traitement.
    
    Args:
        text: Texte à normaliser
        
    Returns:
        Texte normalisé
    """
    if not text:
        return ""
    
    # Convertir en minuscules
    normalized = text.lower()
    
    # Supprimer les accents (simplification)
    accents = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c',
        'ÿ': 'y'
    }
    for accent, replacement in accents.items():
        normalized = normalized.replace(accent, replacement)
    
    # Supprimer la ponctuation non significative
    normalized = re.sub(r'[^\w\s€,.]', ' ', normalized)
    
    # Normaliser les espaces (multiples → simple)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def extract_entities(text: str) -> Dict[str, List[Any]]:
    """
    Extrait des entités significatives d'un texte (dates, montants, catégories, etc.).
    
    Args:
        text: Texte à analyser
        
    Returns:
        Dictionnaire d'entités extraites
    """
    if not text:
        return {}
    
    normalized_text = normalize_text(text)
    entities = {}
    
    # Extraire les expressions de date
    date_expressions = []
    for pattern, value in DATE_EXPRESSIONS.items():
        if re.search(pattern, normalized_text):
            date_expressions.append({"type": "relative", "value": value})
    
    if date_expressions:
        entities["date_expressions"] = date_expressions
    
    # Extraire les montants
    amounts = []
    for pattern, config in AMOUNT_EXPRESSIONS.items():
        matches = re.search(pattern, normalized_text)
        if matches:
            if config["type"] == "range":
                min_val = float(matches.group(config["min_group"]).replace(',', '.'))
                max_val = float(matches.group(config["max_group"]).replace(',', '.'))
                amounts.append({"type": "min", "value": min_val})
                amounts.append({"type": "max", "value": max_val})
            else:
                val = float(matches.group(config["group"]).replace(',', '.'))
                amounts.append({"type": config["type"], "value": val})
    
    if amounts:
        entities["amounts"] = amounts
    
    # Extraire les catégories
    categories = []
    for category in COMMON_CATEGORIES:
        if category in normalized_text.split():
            categories.append(category)
    
    if categories:
        entities["categories"] = categories
    
    # Extraire les marchands
    merchants = []
    for merchant in COMMON_MERCHANTS:
        if merchant in normalized_text:
            merchants.append(merchant)
    
    if merchants:
        entities["merchants"] = merchants
    
    return entities