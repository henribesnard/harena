"""
Module de correction pour l'expansion de requêtes.
Corrige le bug 'dict' object has no attribute 'lower'

Ce module fournit des fonctions sécurisées pour l'expansion et la validation
des requêtes de recherche, avec une attention particulière aux types de données.
"""
import logging
import re
from typing import List, Set, Union, Any

logger = logging.getLogger("search_service.query_expansion")


def validate_query_input(query: Any) -> str:
    """
    Valide et normalise l'entrée de requête.
    
    Args:
        query: L'entrée de requête (peut être de n'importe quel type)
        
    Returns:
        str: Requête validée et normalisée
        
    Raises:
        ValueError: Si la requête ne peut pas être convertie en string valide
    """
    if query is None:
        logger.warning("Query est None, conversion en string vide")
        return ""
    
    if isinstance(query, str):
        # Déjà une string, nettoyer seulement
        cleaned = query.strip()
        if not cleaned:
            logger.warning("Query string vide après nettoyage")
        return cleaned
    
    if isinstance(query, (dict, list)):
        logger.error(f"Query est un {type(query).__name__}, impossible de convertir: {query}")
        raise ValueError(f"Query cannot be a {type(query).__name__} object")
    
    # Tentative de conversion pour les autres types
    try:
        converted = str(query).strip()
        logger.warning(f"Query convertie de {type(query).__name__} vers string: '{converted}'")
        
        if not converted or converted.lower() in ['none', 'null', 'undefined']:
            logger.warning("Query convertie est vide ou invalide")
            return ""
        
        return converted
        
    except Exception as e:
        logger.error(f"Impossible de convertir query {type(query).__name__} en string: {e}")
        raise ValueError(f"Cannot convert query of type {type(query).__name__} to string")


def clean_and_tokenize(query: str) -> List[str]:
    """
    Nettoie et tokenise une requête string.
    
    Args:
        query (str): Requête à tokeniser
        
    Returns:
        List[str]: Liste des tokens nettoyés
    """
    if not isinstance(query, str):
        logger.error(f"clean_and_tokenize reçu un {type(query).__name__}, attendait string")
        query = str(query)
    
    # Nettoyer la requête
    # Supprimer les caractères spéciaux mais garder les traits d'union
    cleaned = re.sub(r'[^\w\s\-àâäéèêëïîôöùûüÿç]', ' ', query.lower())
    
    # Tokeniser en respectant les mots composés avec trait d'union
    tokens = []
    for token in cleaned.split():
        token = token.strip()
        if token and len(token) > 0:
            tokens.append(token)
    
    return tokens


def expand_financial_terms(term: str) -> Set[str]:
    """
    Expanse les termes financiers français spécifiques.
    
    Args:
        term (str): Terme à expanser
        
    Returns:
        Set[str]: Ensemble des termes expandus
    """
    if not isinstance(term, str):
        logger.warning(f"expand_financial_terms reçu {type(term).__name__}, conversion en string")
        term = str(term)
    
    term = term.lower().strip()
    expanded = {term}  # Toujours inclure le terme original
    
    # Règles d'expansion pour les termes financiers français
    financial_expansions = {
        'vir': {'virs', 'virement', 'virements', 'vir', 'paiement', 'transfer', 'transfert'},
        'virs': {'vir', 'virement', 'virements', 'paiement', 'transfer', 'transfert'},
        'virement': {'vir', 'virs', 'virements', 'paiement', 'transfer', 'transfert'},
        'virements': {'vir', 'virs', 'virement', 'paiement', 'transfer', 'transfert'},
        
        'sepa': {'sepas', 'sepa', 'virement', 'paiement', 'transfer', 'transfert', 'europeen'},
        'sepas': {'sepa', 'virement', 'paiement', 'transfer', 'transfert', 'europeen'},
        
        'cb': {'carte', 'bancaire', 'paiement', 'achat', 'debit'},
        'carte': {'cb', 'bancaire', 'paiement', 'achat', 'debit'},
        'bancaire': {'cb', 'carte', 'paiement', 'achat', 'banque'},
        
        'cheque': {'chèque', 'cheques', 'chèques', 'paiement'},
        'chèque': {'cheque', 'cheques', 'chèques', 'paiement'},
        'cheques': {'cheque', 'chèque', 'chèques', 'paiement'},
        'chèques': {'cheque', 'chèque', 'cheques', 'paiement'},
        
        'prelevement': {'prélèvement', 'prelevements', 'prélèvements', 'auto', 'automatique'},
        'prélèvement': {'prelevement', 'prelevements', 'prélèvements', 'auto', 'automatique'},
        'prelevements': {'prelevement', 'prélèvement', 'prélèvements', 'auto', 'automatique'},
        'prélèvements': {'prelevement', 'prélèvement', 'prelevements', 'auto', 'automatique'},
        
        # Termes temporels
        'mois-ci': {'mois-ci', 'mois-cis', 'ce-mois', 'mois', 'actuel'},
        'mois-cis': {'mois-ci', 'ce-mois', 'mois', 'actuel'},
        'ce-mois': {'mois-ci', 'mois-cis', 'mois', 'actuel'},
        
        'semaine': {'sem', 'hebdo', 'hebdomadaire', 'cette-semaine'},
        'sem': {'semaine', 'hebdo', 'hebdomadaire', 'cette-semaine'},
        
        'jour': {'journalier', 'quotidien', 'aujourd', 'aujourdhui'},
        'aujourd': {'aujourdhui', 'jour', 'journalier', 'quotidien'},
        'aujourdhui': {'aujourd', 'jour', 'journalier', 'quotidien'},
        
        # Montants et devises
        'euro': {'euros', 'eur', '€'},
        'euros': {'euro', 'eur', '€'},
        'eur': {'euro', 'euros', '€'},
        
        # Commerçants et catégories communes
        'supermarche': {'supermarché', 'hypermarche', 'hypermarché', 'grande-surface', 'courses'},
        'supermarché': {'supermarche', 'hypermarche', 'hypermarché', 'grande-surface', 'courses'},
        'hypermarche': {'hypermarché', 'supermarche', 'supermarché', 'grande-surface', 'courses'},
        'hypermarché': {'hypermarche', 'supermarche', 'supermarché', 'grande-surface', 'courses'},
        
        'restaurant': {'resto', 'restauration', 'repas', 'diner', 'déjeuner'},
        'resto': {'restaurant', 'restauration', 'repas', 'diner', 'déjeuner'},
        
        'essence': {'carburant', 'station-service', 'petrole', 'diesel', 'sp95', 'sp98'},
        'carburant': {'essence', 'station-service', 'petrole', 'diesel', 'sp95', 'sp98'},
        'diesel': {'carburant', 'essence', 'station-service', 'petrole'},
        
        'transport': {'transports', 'metro', 'métro', 'bus', 'train', 'sncf', 'ratp'},
        'transports': {'transport', 'metro', 'métro', 'bus', 'train', 'sncf', 'ratp'},
        'metro': {'métro', 'transport', 'transports', 'ratp'},
        'métro': {'metro', 'transport', 'transports', 'ratp'},
    }
    
    # Appliquer les expansions
    if term in financial_expansions:
        expanded.update(financial_expansions[term])
    
    # Expansions par pattern matching
    if 'vir' in term and len(term) > 3:
        expanded.update(['virement', 'paiement', 'transfer'])
    
    if 'sepa' in term:
        expanded.update(['virement', 'europeen', 'paiement'])
    
    if 'mois' in term:
        expanded.update(['mensuel', 'monthly'])
    
    if any(x in term for x in ['carte', 'cb']):
        expanded.update(['paiement', 'achat'])
    
    return expanded


def expand_query_terms(query: Any) -> List[str]:
    """
    Expand query terms avec validation stricte des types.
    
    Args:
        query: Requête utilisateur (peut être de n'importe quel type)
        
    Returns:
        List[str]: Liste des termes expandus et validés
    """
    # Étape 1: Validation et normalisation de l'entrée
    try:
        validated_query = validate_query_input(query)
    except ValueError as e:
        logger.error(f"Erreur validation query: {e}")
        return []
    
    if not validated_query:
        logger.warning("Query vide après validation")
        return []
    
    # Étape 2: Tokenisation
    base_tokens = clean_and_tokenize(validated_query)
    
    if not base_tokens:
        logger.warning("Aucun token après tokenisation")
        return []
    
    # Étape 3: Expansion des termes
    all_terms = set()
    
    for token in base_tokens:
        # Ajouter le token original
        all_terms.add(token)
        
        # Expanser le token
        try:
            expanded_terms = expand_financial_terms(token)
            all_terms.update(expanded_terms)
        except Exception as e:
            logger.warning(f"Erreur expansion pour token '{token}': {e}")
            # Continuer avec les autres tokens
    
    # Étape 4: Filtrage et validation finale
    final_terms = []
    for term in all_terms:
        # S'assurer que chaque terme est une string valide
        if isinstance(term, str) and len(term.strip()) > 0:
            cleaned_term = term.strip().lower()
            if cleaned_term not in final_terms:  # Éviter les doublons
                final_terms.append(cleaned_term)
        else:
            logger.warning(f"Term ignoré (invalide): {type(term).__name__} = {term}")
    
    # Trier pour la cohérence (optionnel)
    final_terms.sort()
    
    logger.info(f"Expansion terminée: '{validated_query}' -> {len(final_terms)} termes")
    logger.debug(f"Termes expandus: {final_terms}")
    
    return final_terms


def build_elasticsearch_query_string(terms: List[str]) -> str:
    """
    Construit une chaîne de requête pour Elasticsearch.
    
    Args:
        terms (List[str]): Liste des termes
        
    Returns:
        str: Chaîne de requête formatée
    """
    if not terms:
        return ""
    
    # Valider que tous les termes sont des strings
    validated_terms = []
    for term in terms:
        if isinstance(term, str) and term.strip():
            validated_terms.append(term.strip())
        else:
            logger.warning(f"Term ignoré dans query string: {type(term).__name__} = {term}")
    
    if not validated_terms:
        return ""
    
    # Joindre les termes avec des espaces
    query_string = " ".join(validated_terms)
    
    logger.debug(f"Query string construite: '{query_string}'")
    return query_string


def validate_terms_list(terms: List[Any]) -> List[str]:
    """
    Valide une liste de termes pour s'assurer qu'ils sont tous des strings.
    
    Args:
        terms (List[Any]): Liste de termes à valider
        
    Returns:
        List[str]: Liste de termes validés (strings uniquement)
    """
    if not isinstance(terms, list):
        logger.error(f"validate_terms_list reçu {type(terms).__name__}, attendait list")
        return []
    
    validated = []
    for i, term in enumerate(terms):
        if isinstance(term, str):
            cleaned = term.strip()
            if cleaned:
                validated.append(cleaned)
        else:
            logger.warning(f"Term [{i}] ignoré (pas string): {type(term).__name__} = {term}")
            # Tentative de conversion
            try:
                converted = str(term).strip()
                if converted and converted.lower() not in ['none', 'null']:
                    validated.append(converted)
                    logger.info(f"Term [{i}] converti: {type(term).__name__} -> '{converted}'")
            except Exception as e:
                logger.error(f"Impossible de convertir term [{i}]: {e}")
    
    return validated


# Fonctions utilitaires pour les tests et le debug
def debug_query_expansion(query: Any) -> dict:
    """
    Version debug de l'expansion de requête avec informations détaillées.
    
    Args:
        query: Requête à analyser
        
    Returns:
        dict: Informations détaillées sur l'expansion
    """
    result = {
        "original_query": query,
        "original_type": type(query).__name__,
        "steps": [],
        "errors": [],
        "final_terms": [],
        "final_query_string": ""
    }
    
    try:
        # Étape 1: Validation
        result["steps"].append("validation")
        validated_query = validate_query_input(query)
        result["validated_query"] = validated_query
        result["validated_type"] = type(validated_query).__name__
        
        if not validated_query:
            result["errors"].append("Query vide après validation")
            return result
        
        # Étape 2: Tokenisation
        result["steps"].append("tokenization")
        tokens = clean_and_tokenize(validated_query)
        result["tokens"] = tokens
        result["token_count"] = len(tokens)
        
        if not tokens:
            result["errors"].append("Aucun token après tokenisation")
            return result
        
        # Étape 3: Expansion
        result["steps"].append("expansion")
        expanded_terms = expand_query_terms(query)
        result["final_terms"] = expanded_terms
        result["expanded_count"] = len(expanded_terms)
        
        # Étape 4: Construction query string
        result["steps"].append("query_string_building")
        query_string = build_elasticsearch_query_string(expanded_terms)
        result["final_query_string"] = query_string
        
        result["success"] = True
        
    except Exception as e:
        result["errors"].append(f"Erreur générale: {str(e)}")
        result["success"] = False
    
    return result


# Constantes pour la configuration
DEFAULT_MAX_TERMS = 50  # Limite le nombre de termes expandus
DEFAULT_MIN_TERM_LENGTH = 2  # Longueur minimale d'un terme

# Export des fonctions principales
__all__ = [
    'expand_query_terms',
    'validate_query_input',
    'clean_and_tokenize',
    'expand_financial_terms',
    'build_elasticsearch_query_string',
    'validate_terms_list',
    'debug_query_expansion'
]