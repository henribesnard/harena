"""
Normalisation des entités financières pour compatibilité Elasticsearch
Mappings des termes français vers les valeurs Elasticsearch standardisées
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger("conversation_service.entity_normalization")

# Mappings des types d'opérations
OPERATION_TYPE_MAPPINGS = {
    # Termes français vers valeurs Elasticsearch
    "virement": "transfer",
    "virements": "transfer", 
    "vir": "transfer",
    "transfer": "transfer",
    "transfert": "transfer",
    
    "prelevement": "direct_debit",
    "prélèvement": "direct_debit",
    "prelevements": "direct_debit",
    "prélèvements": "direct_debit",
    "prélev": "direct_debit",
    "direct_debit": "direct_debit",
    
    "carte": "card",
    "cb": "card",
    "card": "card",
    "paiement carte": "card",
    "achat carte": "card",
    
    "retrait": "card",
    "retraits": "card",
    "retrait dab": "card",
    "retrait atm": "card",
    "retrait espèces": "card",
    "retrait especes": "card",

    "cheque": "check",
    "chèque": "check",
    "chéque": "check",
    "chq": "check",
    "check": "check",
    
    "inconnu": "unknown",
    "unknown": "unknown",
    "autre": "unknown",
}

# Types de transactions possibles
TRANSACTION_TYPE_MAPPINGS = {
    "debit": "debit",
    "débit": "debit", 
    "sortie": "debit",
    "paiement": "debit",
    "dépense": "debit",
    "depense": "debit",
    
    "credit": "credit",
    "crédit": "credit",
    "entrée": "credit",
    "entree": "credit",
    "recette": "credit",
    "versement": "credit",
    "depot": "credit",
    "dépôt": "credit",
}

# Types de comptes
ACCOUNT_TYPE_MAPPINGS = {
    "carte": "card",
    "card": "card",
    "cb": "card",
    
    "courant": "checking",
    "checking": "checking",
    "compte courant": "checking",
    "compte": "checking",
    "epargne": "checking",  # Pas de type épargne dans Elasticsearch, on met checking
    "épargne": "checking",
}

# Catégories courantes (pour mapping futur)
CATEGORY_MAPPINGS = {
    "restaurant": "restaurants",
    "restaurants": "restaurants",
    "restau": "restaurants",
    
    "transport": "transport",
    "transports": "transport",
    "essence": "transport",
    "carburant": "transport",
    
    "courses": "grocery",
    "alimentation": "grocery",
    "supermarché": "grocery",
    "supermarche": "grocery",
    
    "santé": "health",
    "sante": "health",
    "médecin": "health",
    "medecin": "health",
    "pharmacie": "health",
}

def normalize_operation_types(operation_types: List[str]) -> List[str]:
    """
    Normalise une liste de types d'opérations vers les valeurs Elasticsearch
    """
    normalized = []
    
    for op_type in operation_types:
        if not isinstance(op_type, str):
            continue
            
        # Nettoyage et normalisation
        clean_op = op_type.lower().strip()
        
        # Recherche exacte d'abord
        if clean_op in OPERATION_TYPE_MAPPINGS:
            normalized_value = OPERATION_TYPE_MAPPINGS[clean_op]
            if normalized_value not in normalized:
                normalized.append(normalized_value)
            continue
        
        # Recherche partielle pour les termes composés
        found = False
        for french_term, es_value in OPERATION_TYPE_MAPPINGS.items():
            if french_term in clean_op or clean_op in french_term:
                if es_value not in normalized:
                    normalized.append(es_value)
                found = True
                break
        
        # Si aucune correspondance, log et garde la valeur originale si elle semble valide
        if not found:
            if clean_op in ["card", "check", "transfer", "direct_debit", "unknown"]:
                # C'est déjà une valeur Elasticsearch valide
                if clean_op not in normalized:
                    normalized.append(clean_op)
            else:
                logger.warning(f"Type d'opération non mappé: '{op_type}' -> 'unknown'")
                if "unknown" not in normalized:
                    normalized.append("unknown")
    
    return normalized

def normalize_transaction_types(transaction_types: List[str]) -> List[str]:
    """
    Normalise les types de transaction (debit/credit)
    """
    normalized = []
    
    for tx_type in transaction_types:
        if not isinstance(tx_type, str):
            continue
            
        clean_type = tx_type.lower().strip()
        
        if clean_type in TRANSACTION_TYPE_MAPPINGS:
            normalized_value = TRANSACTION_TYPE_MAPPINGS[clean_type]
            if normalized_value not in normalized:
                normalized.append(normalized_value)
        elif clean_type in ["debit", "credit"]:
            # Déjà normalisé
            if clean_type not in normalized:
                normalized.append(clean_type)
        else:
            logger.warning(f"Type de transaction non mappé: '{tx_type}'")
    
    return normalized

def normalize_account_types(account_types: List[str]) -> List[str]:
    """
    Normalise les types de comptes
    """
    normalized = []
    
    for acc_type in account_types:
        if not isinstance(acc_type, str):
            continue
            
        clean_type = acc_type.lower().strip()
        
        if clean_type in ACCOUNT_TYPE_MAPPINGS:
            normalized_value = ACCOUNT_TYPE_MAPPINGS[clean_type]
            if normalized_value not in normalized:
                normalized.append(normalized_value)
        elif clean_type in ["card", "checking"]:
            # Déjà normalisé
            if clean_type not in normalized:
                normalized.append(clean_type)
        else:
            logger.warning(f"Type de compte non mappé: '{acc_type}' -> 'checking'")
            if "checking" not in normalized:
                normalized.append("checking")
    
    return normalized

def normalize_amounts(amounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalise les montants (format, devise, etc.)
    """
    normalized = []
    
    for amount_data in amounts:
        if not isinstance(amount_data, dict):
            continue
            
        normalized_amount = amount_data.copy()
        
        # Normalisation devise par défaut
        if "currency" not in normalized_amount or not normalized_amount["currency"]:
            normalized_amount["currency"] = "EUR"
        
        # Normalisation format montant
        if "value" in normalized_amount:
            try:
                # Conversion en float si c'est une string
                if isinstance(normalized_amount["value"], str):
                    # Nettoie les caractères non numériques sauf . et -
                    clean_value = re.sub(r'[^\d.,-]', '', normalized_amount["value"])
                    clean_value = clean_value.replace(',', '.')
                    normalized_amount["value"] = float(clean_value)
                elif not isinstance(normalized_amount["value"], (int, float)):
                    logger.warning(f"Montant invalide ignoré: {normalized_amount['value']}")
                    continue
            except (ValueError, TypeError):
                logger.warning(f"Erreur conversion montant: {normalized_amount['value']}")
                continue
        
        normalized.append(normalized_amount)
    
    return normalized

def normalize_dates(dates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalise les dates (format ISO, périodes, etc.)
    """
    normalized = []
    
    for date_data in dates:
        if not isinstance(date_data, dict):
            continue
            
        normalized_date = date_data.copy()
        
        # Normalisation format date si nécessaire
        if "value" in normalized_date:
            value = normalized_date["value"]
            
            # Si c'est déjà au format ISO, on garde
            if isinstance(value, str) and re.match(r'\d{4}-\d{2}(-\d{2})?', value):
                normalized.append(normalized_date)
                continue
            
            # Normalisation des périodes françaises vers ISO
            if isinstance(value, str):
                value_lower = value.lower().strip()
                
                # Gestion des termes relatifs
                current_date = datetime.now()
                
                if any(term in value_lower for term in ["ce mois", "mois courant", "mois actuel"]):
                    normalized_date["value"] = current_date.strftime("%Y-%m")
                    normalized_date["type"] = "period"
                    logger.debug(f"Période relative 'ce mois' → {normalized_date['value']}")
                    normalized.append(normalized_date)
                    continue
                elif any(term in value_lower for term in ["mois dernier", "mois précédent", "mois precedent"]):
                    if current_date.month > 1:
                        last_month = current_date.replace(month=current_date.month-1)
                    else:
                        last_month = current_date.replace(year=current_date.year-1, month=12)
                    normalized_date["value"] = last_month.strftime("%Y-%m")
                    normalized_date["type"] = "period"
                    logger.debug(f"Période relative 'mois dernier' → {normalized_date['value']}")
                    normalized.append(normalized_date)
                    continue
                elif any(term in value_lower for term in ["mois prochain", "mois suivant"]):
                    if current_date.month < 12:
                        next_month = current_date.replace(month=current_date.month+1)
                    else:
                        next_month = current_date.replace(year=current_date.year+1, month=1)
                    normalized_date["value"] = next_month.strftime("%Y-%m")
                    normalized_date["type"] = "period"
                    logger.debug(f"Période relative 'mois prochain' → {normalized_date['value']}")
                    normalized.append(normalized_date)
                    continue
                
                # Mapping mois français
                month_mappings = {
                    "janvier": "01", "jan": "01",
                    "février": "02", "fevrier": "02", "fév": "02", "fev": "02",
                    "mars": "03", "mar": "03",
                    "avril": "04", "avr": "04",
                    "mai": "05",
                    "juin": "06", "jun": "06",
                    "juillet": "07", "juil": "07", "jul": "07",
                    "août": "08", "aout": "08", "aug": "08",
                    "septembre": "09", "sept": "09", "sep": "09",
                    "octobre": "10", "oct": "10",
                    "novembre": "11", "nov": "11",
                    "décembre": "12", "decembre": "12", "déc": "12", "dec": "12"
                }
                
                # Recherche mois français
                current_date = datetime.now()
                current_year = current_date.year
                current_month = current_date.month
                
                for french_month, month_num in month_mappings.items():
                    if french_month in value_lower:
                        # Extraction année si présente explicitement
                        year_match = re.search(r'\b(20\d{2})\b', value)
                        
                        if year_match:
                            # Année explicitement mentionnée
                            year = year_match.group(1)
                        else:
                            # Logique intelligente pour déterminer l'année
                            month_int = int(month_num)
                            
                            if month_int <= current_month:
                                # Le mois est passé ou en cours cette année → année courante
                                year = str(current_year)
                            else:
                                # Le mois est dans le futur → année précédente
                                year = str(current_year - 1)
                        
                        normalized_date["value"] = f"{year}-{month_num}"
                        normalized_date["type"] = "period"
                        
                        # Log de la logique appliquée pour debugging
                        if not year_match:
                            logger.debug(f"Date contextualisée: '{value}' → {year}-{month_num} (mois {month_int} vs courant {current_month})")
                        
                        break
                else:
                    # Aucun mois français trouvé, garde la valeur originale
                    pass
        
        normalized.append(normalized_date)
    
    return normalized

def normalize_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise toutes les entités d'un résultat d'extraction
    """
    if not isinstance(entities, dict):
        logger.warning("Format d'entités invalide pour normalisation")
        return entities
    
    # Copie pour éviter de modifier l'original
    normalized = entities.copy()
    
    try:
        # Normalisation des types d'opérations
        if "operation_types" in normalized and isinstance(normalized["operation_types"], list):
            normalized["operation_types"] = normalize_operation_types(normalized["operation_types"])
        
        # Normalisation des types de transactions
        if "transaction_types" in normalized and isinstance(normalized["transaction_types"], list):
            normalized["transaction_types"] = normalize_transaction_types(normalized["transaction_types"])
        
        # Normalisation des types de comptes
        if "account_types" in normalized and isinstance(normalized["account_types"], list):
            normalized["account_types"] = normalize_account_types(normalized["account_types"])
        
        # Normalisation des montants
        if "amounts" in normalized and isinstance(normalized["amounts"], list):
            normalized["amounts"] = normalize_amounts(normalized["amounts"])
        
        # Normalisation des dates
        if "dates" in normalized and isinstance(normalized["dates"], list):
            normalized["dates"] = normalize_dates(normalized["dates"])
        
        # Log résultat normalisation
        changes = []
        if "operation_types" in entities and entities["operation_types"] != normalized.get("operation_types", []):
            changes.append(f"operation_types: {entities['operation_types']} -> {normalized['operation_types']}")
        
        if changes:
            logger.info(f"Entités normalisées: {', '.join(changes)}")
    
    except Exception as e:
        logger.error(f"Erreur normalisation entités: {str(e)}")
        return entities  # Retourne les entités originales en cas d'erreur
    
    return normalized

def get_elasticsearch_field_mappings() -> Dict[str, str]:
    """
    Retourne les mappings des champs pour les requêtes Elasticsearch
    """
    return {
        "operation_types": "operation_type",
        "transaction_types": "transaction_type", 
        "account_types": "account_type",
        "amounts": "amount",
        "dates": "date",
        "merchants": "searchable_text",  # Les marchands sont dans le texte de recherche
        "categories": "category_name",
        "text_search": "searchable_text"
    }

def get_supported_elasticsearch_values() -> Dict[str, List[str]]:
    """
    Retourne les valeurs supportées par Elasticsearch pour chaque champ
    """
    return {
        "operation_type": ["card", "check", "direct_debit", "transfer", "unknown"],
        "transaction_type": ["debit", "credit"],
        "account_type": ["card", "checking"]
    }