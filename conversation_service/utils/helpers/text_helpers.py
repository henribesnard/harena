"""
📝 Utilitaires Texte - Helpers de traitement texte

Fonctions utilitaires pour nettoyage, normalisation et analyse de texte
optimisées pour le français et le domaine financier.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


# Patterns regex pré-compilés pour performance
FRENCH_ACCENTS_MAP = {
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
    'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
    'ý': 'y', 'ÿ': 'y',
    'ñ': 'n', 'ç': 'c'
}

# Patterns pour nettoyage
EXTRA_WHITESPACE_PATTERN = re.compile(r'\s+')
PUNCTUATION_PATTERN = re.compile(r'[^\w\s\-\.]', re.UNICODE)
MULTIPLE_PUNCT_PATTERN = re.compile(r'[.]{2,}')
CURRENCY_PATTERN = re.compile(r'\b(euros?|€|eur)\b', re.IGNORECASE)
NUMBER_PATTERN = re.compile(r'\b\d+(?:[,\.]\d+)?\b')

# Mots vides français financiers
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais',
    'donc', 'car', 'ni', 'or', 'à', 'au', 'aux', 'avec', 'dans', 'pour',
    'sur', 'sous', 'vers', 'par', 'sans', 'selon', 'entre', 'chez',
    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'mon', 'ma',
    'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'votre', 'leur',
    'ce', 'cette', 'ces', 'cet', 'qui', 'que', 'dont', 'où', 'quoi', 'quel',
    'quelle', 'quels', 'quelles', 'est', 'sont', 'était', 'étaient', 'être',
    'avoir', 'ai', 'as', 'a', 'avons', 'avez', 'ont', 'suis', 'es', 'sommes',
    'êtes', 'très', 'plus', 'moins', 'tout', 'tous', 'toute', 'toutes'
}

# Mots financiers à préserver (ne pas traiter comme stopwords)
FINANCIAL_KEYWORDS = {
    'solde', 'compte', 'virement', 'carte', 'euros', 'argent', 'dépenses',
    'budget', 'transaction', 'paiement', 'crédit', 'débit', 'banque',
    'facture', 'montant', 'somme', 'coût', 'prix', 'total', 'reste',
    'disponible', 'économies', 'épargne', 'livret', 'prêt', 'emprunt'
}


def clean_text(text, preserve_accents=True):
    """
    Nettoie et normalise un texte pour traitement
    
    Args:
        text: Texte à nettoyer
        preserve_accents: Garder les accents français
        
    Returns:
        Texte nettoyé et normalisé
    """
    if not text:
        return ""
    
    # Normalisation Unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Conversion minuscules
    text = text.lower().strip()
    
    # Suppression accents si demandé
    if not preserve_accents:
        text = remove_french_accents(text)
    
    # Nettoyage ponctuation excessive
    text = MULTIPLE_PUNCT_PATTERN.sub('.', text)
    
    # Normalisation espaces
    text = EXTRA_WHITESPACE_PATTERN.sub(' ', text).strip()
    
    return text


def remove_french_accents(text):
    """
    Supprime les accents français d'un texte
    
    Args:
        text: Texte avec accents
        
    Returns:
        Texte sans accents
    """
    if not text:
        return ""
    
    # Mapping direct des caractères accentués
    for accented, plain in FRENCH_ACCENTS_MAP.items():
        text = text.replace(accented, plain)
        text = text.replace(accented.upper(), plain.upper())
    
    return text


def extract_keywords(text, min_length=3, exclude_stopwords=True):
    """
    Extrait les mots-clés d'un texte
    
    Args:
        text: Texte source
        min_length: Longueur minimale des mots
        exclude_stopwords: Exclure les mots vides
        
    Returns:
        Liste des mots-clés extraits
    """
    if not text:
        return []
    
    # Nettoyage de base
    cleaned = clean_text(text)
    
    # Tokenisation simple
    words = re.findall(r'\b[a-zA-Zàâäéèêëïîôöùûüÿç]+\b', cleaned, re.UNICODE)
    
    # Filtrage
    keywords = []
    for word in words:
        word_lower = word.lower()
        
        # Vérifications de validité
        if len(word) < min_length:
            continue
            
        if exclude_stopwords and word_lower in FRENCH_STOPWORDS:
            # Exception pour mots financiers importants
            if word_lower not in FINANCIAL_KEYWORDS:
                continue
        
        keywords.append(word_lower)
    
    return keywords


def detect_language(text):
    """
    Détection basique de langue (français/anglais)
    
    Args:
        text: Texte à analyser
        
    Returns:
        Code langue détectée ('fr', 'en', 'unknown')
    """
    if not text or len(text) < 10:
        return 'unknown'
    
    # Indicateurs français
    french_indicators = [
        'bonjour', 'salut', 'merci', 'au revoir', 'comment', 'pourquoi',
        'quand', 'où', 'que', 'qui', 'avec', 'dans', 'pour', 'sur',
        'être', 'avoir', 'faire', 'aller', 'venir', 'voir', 'savoir',
        'pouvoir', 'vouloir', 'devoir', 'prendre', 'donner', 'mettre'
    ]
    
    # Indicateurs anglais
    english_indicators = [
        'hello', 'hi', 'thank', 'goodbye', 'how', 'why', 'when', 'where',
        'what', 'who', 'with', 'from', 'about', 'after', 'before',
        'the', 'and', 'or', 'but', 'if', 'then', 'this', 'that'
    ]
    
    text_lower = text.lower()
    
    french_score = sum(1 for indicator in french_indicators if indicator in text_lower)
    english_score = sum(1 for indicator in english_indicators if indicator in text_lower)
    
    # Bonus pour caractères accentués français
    accent_bonus = sum(1 for char in text if char in 'àâäéèêëïîôöùûüÿç')
    french_score += accent_bonus * 0.5
    
    if french_score > english_score:
        return 'fr'
    elif english_score > french_score:
        return 'en'
    else:
        return 'unknown'


def extract_numbers(text):
    """
    Extrait les nombres d'un texte avec contexte
    
    Args:
        text: Texte source
        
    Returns:
        Liste des nombres trouvés avec métadonnées
    """
    numbers = []
    
    for match in NUMBER_PATTERN.finditer(text):
        number_str = match.group()
        start_pos = match.start()
        end_pos = match.end()
        
        # Contexte avant/après
        context_start = max(0, start_pos - 20)
        context_end = min(len(text), end_pos + 20)
        context = text[context_start:context_end]
        
        # Conversion numérique
        try:
            # Gestion format français (virgule décimale)
            number_normalized = number_str.replace(',', '.')
            value = float(number_normalized)
        except ValueError:
            continue
        
        # Détection devise
        currency = None
        if re.search(CURRENCY_PATTERN, context):
            currency = 'EUR'
        
        numbers.append({
            'value': value,
            'original_text': number_str,
            'position': (start_pos, end_pos),
            'context': context.strip(),
            'currency': currency
        })
    
    return numbers


def normalize_query(query):
    """
    Normalise une requête pour comparaison/cache
    
    Args:
        query: Requête utilisateur
        
    Returns:
        Requête normalisée
    """
    if not query:
        return ""
    
    # Nettoyage standard
    normalized = clean_text(query, preserve_accents=False)
    
    # Suppression ponctuation non essentielle
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    
    # Normalisation espaces multiples
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def calculate_text_similarity(text1, text2):
    """
    Calcule similarité basique entre deux textes
    
    Args:
        text1, text2: Textes à comparer
        
    Returns:
        Score de similarité [0-1]
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalisation
    norm1 = set(extract_keywords(text1, min_length=2))
    norm2 = set(extract_keywords(text2, min_length=2))
    
    if not norm1 or not norm2:
        return 0.0
    
    # Similarité Jaccard
    intersection = len(norm1.intersection(norm2))
    union = len(norm1.union(norm2))
    
    return intersection / union if union > 0 else 0.0


def truncate_text(text, max_length=500, suffix="..."):
    """
    Tronque un texte intelligemment
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe si tronqué
        
    Returns:
        Texte tronqué
    """
    if not text or len(text) <= max_length:
        return text
    
    # Tentative de troncature sur mot complet
    truncated = text[:max_length - len(suffix)]
    
    # Cherche dernier espace pour éviter coupure au milieu d'un mot
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # Si l'espace n'est pas trop loin
        truncated = truncated[:last_space]
    
    return truncated + suffix


def validate_query_length(query, min_length=1, max_length=500):
    """
    Valide la longueur d'une requête
    
    Args:
        query: Requête à valider
        min_length: Longueur minimale
        max_length: Longueur maximale
        
    Returns:
        (is_valid, error_message)
    """
    if not query:
        return False, "Requête vide"
    
    query_length = len(query.strip())
    
    if query_length < min_length:
        return False, f"Requête trop courte (min {min_length} caractères)"
    
    if query_length > max_length:
        return False, f"Requête trop longue (max {max_length} caractères)"
    
    return True, ""


def extract_quoted_text(text):
    """
    Extrait le texte entre guillemets
    
    Args:
        text: Texte source
        
    Returns:
        Liste des textes entre guillemets
    """
    patterns = [
        r'"([^"]*)"',       # Guillemets anglais
        r'«([^»]*)»',       # Guillemets français
        r"'([^']*)'",       # Apostrophes
    ]
    
    quoted_texts = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        quoted_texts.extend(matches)
    
    return [q.strip() for q in quoted_texts if q.strip()]


def is_question(text):
    """
    Détermine si un texte est une question
    
    Args:
        text: Texte à analyser
        
    Returns:
        True si question détectée
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Détection par ponctuation
    if text.endswith('?'):
        return True
    
    # Détection par mots interrogatifs français
    question_words = [
        'comment', 'combien', 'pourquoi', 'quand', 'où', 'que', 'qui',
        'quoi', 'quel', 'quelle', 'quels', 'quelles', 'est-ce que',
        'peux-tu', 'pouvez-vous', 'puis-je'
    ]
    
    text_lower = text.lower()
    for word in question_words:
        if text_lower.startswith(word + ' ') or ' ' + word + ' ' in text_lower:
            return True
    
    return False


# Fonctions de cache pour patterns compilés
_compiled_patterns_cache = {}


def get_compiled_pattern(pattern, flags=0):
    """
    Récupère un pattern regex compilé depuis le cache
    
    Args:
        pattern: Pattern regex
        flags: Flags de compilation
        
    Returns:
        Pattern compilé
    """
    cache_key = f"{pattern}:{flags}"
    
    if cache_key not in _compiled_patterns_cache:
        try:
            _compiled_patterns_cache[cache_key] = re.compile(pattern, flags)
        except re.error as e:
            logger.error(f"Erreur compilation pattern '{pattern}': {e}")
            raise
    
    return _compiled_patterns_cache[cache_key]


def clear_pattern_cache():
    """Vide le cache des patterns compilés"""
    global _compiled_patterns_cache
    _compiled_patterns_cache.clear()
    logger.debug("Cache patterns regex vidé")


def get_pattern_cache_stats():
    """Retourne statistiques du cache patterns"""
    return {
        "cached_patterns": len(_compiled_patterns_cache),
        "cache_keys": list(_compiled_patterns_cache.keys())
    }


# Fonctions spécialisées pour le domaine financier
def extract_financial_amounts(text):
    """
    Extrait spécifiquement les montants financiers
    
    Args:
        text: Texte source
        
    Returns:
        Liste des montants avec métadonnées financières
    """
    amounts = []
    
    # Patterns spécialisés montants financiers
    financial_patterns = [
        r'(\d+(?:[,\.]\d+)?)\s*euros?',
        r'(\d+(?:[,\.]\d+)?)\s*€',
        r'(\d+(?:[,\.]\d+)?)\s*eur\b',
        r'(\d+(?:[,\.]\d+)?)€',
    ]
    
    for pattern in financial_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                amount_str = match.group(1).replace(',', '.')
                amount_value = float(amount_str)
                
                amounts.append({
                    'value': amount_value,
                    'currency': 'EUR',
                    'original_match': match.group(),
                    'position': match.span(),
                    'confidence': 0.9
                })
            except ValueError:
                continue
    
    return amounts


def extract_french_dates(text):
    """
    Extrait expressions de dates en français
    
    Args:
        text: Texte source
        
    Returns:
        Liste des expressions temporelles trouvées
    """
    date_patterns = [
        # Mois français
        r'\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b',
        # Expressions relatives
        r'\b(ce\s+mois|mois\s+dernier|cette\s+semaine|semaine\s+dernière|aujourd\'hui|hier|avant-hier)\b',
        # Dates numériques
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
        r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            dates.append({
                'value': match.group(),
                'type': 'date_expression',
                'position': match.span(),
                'confidence': 0.8
            })
    
    return dates


def normalize_financial_query(query):
    """
    Normalisation spécialisée pour requêtes financières
    
    Args:
        query: Requête financière
        
    Returns:
        Requête normalisée pour domaine financier
    """
    if not query:
        return ""
    
    # Nettoyage de base
    normalized = clean_text(query)
    
    # Normalisation termes financiers
    financial_normalizations = {
        'cb': 'carte bleue',
        'resto': 'restaurant',
        'course': 'courses',
        'fric': 'argent',
        'tune': 'argent',
        'blé': 'argent',
        'thune': 'argent'
    }
    
    for informal, formal in financial_normalizations.items():
        normalized = re.sub(r'\b' + informal + r'\b', formal, normalized)
    
    return normalized


# Exports publics
__all__ = [
    # Fonctions principales
    "clean_text",
    "remove_french_accents",
    "extract_keywords",
    "detect_language",
    "extract_numbers",
    "normalize_query",
    "calculate_text_similarity",
    "truncate_text",
    "validate_query_length",
    "extract_quoted_text",
    "is_question",
    
    # Gestion patterns compilés
    "get_compiled_pattern",
    "clear_pattern_cache",
    "get_pattern_cache_stats",
    
    # Fonctions spécialisées financières
    "extract_financial_amounts",
    "extract_french_dates",
    "normalize_financial_query",
    
    # Constantes utiles
    "FRENCH_STOPWORDS",
    "FINANCIAL_KEYWORDS",
    "FRENCH_ACCENTS_MAP"
]