"""
🧹 Service Préprocessing - Nettoyage Texte Spécialisé

Service dédié au nettoyage et préprocessing des requêtes utilisateur
pour optimiser la détection d'intention et l'extraction d'entités.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from conversation_service.utils.helpers.text_helpers import (
    clean_text, 
    normalize_query, 
    extract_keywords,
    validate_query_length,
    is_question,
    FRENCH_STOPWORDS,
    FINANCIAL_KEYWORDS
)
from conversation_service.models.exceptions import ValidationError

logger = logging.getLogger(__name__)


class TextCleaner:
    """Service de nettoyage et préprocessing de texte optimisé"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns spécialisés financiers
        self._financial_patterns = {
            # Normalisation montants
            'amount_patterns': [
                (r'\b(\d+(?:[,\.]\d+)?)\s*euros?\b', r'\1 euros'),
                (r'\b(\d+(?:[,\.]\d+)?)\s*€\b', r'\1 euros'),
                (r'\b(\d+(?:[,\.]\d+)?)\s*eur\b', r'\1 euros'),
            ],
            
            # Normalisation dates/périodes
            'date_patterns': [
                (r'\bce\s+mois\b', 'ce mois'),
                (r'\bmois\s+dernier\b', 'mois dernier'),
                (r'\bcette\s+semaine\b', 'cette semaine'),
                (r'\bsemaine\s+dernière\b', 'semaine dernière'),
                (r'\bce\s+jour\b', 'aujourd\'hui'),
                (r'\bhier\b', 'hier'),
                (r'\bavant[- ]?hier\b', 'avant-hier'),
            ],
            
            # Normalisation comptes
            'account_patterns': [
                (r'\bcompte\s+courant\b', 'compte courant'),
                (r'\blivret\s+a\b', 'livret a'),
                (r'\bcompte\s+épargne\b', 'compte épargne'),
                (r'\bcarte\s+bleue\b', 'carte bleue'),
            ],
            
            # Normalisation catégories
            'category_patterns': [
                (r'\b(resto|restaurant)\b', 'restaurant'),
                (r'\b(courses|course)\b', 'courses'),
                (r'\b(essence|carburant)\b', 'essence'),
                (r'\b(vêtements|fringues)\b', 'vêtements'),
                (r'\b(cinéma|ciné)\b', 'cinéma'),
            ]
        }
        
        # Patterns de négation à détecter
        self._negation_patterns = [
            r'\bn\'?(?:est|était|ai|as|a|avons|avez|ont)\s+(?:pas|point|jamais)',
            r'\b(?:pas|jamais|aucun|aucune)\s+(?:de|d\')',
            r'\bne\s+\w+\s+(?:pas|point|jamais|rien)',
            r'\b(?:sans|excepté|sauf)\b'
        ]
        
        # Compilation des patterns pour performance
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile tous les patterns regex pour optimiser performance"""
        self._compiled_financial_patterns = {}
        
        for category, patterns in self._financial_patterns.items():
            self._compiled_financial_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE), replacement)
                for pattern, replacement in patterns
            ]
        
        self._compiled_negation_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self._negation_patterns
        ]
    
    def preprocess_query(self, query, context=None):
        # type: (str, Optional[Dict]) -> Dict[str, any]
        """
        Préprocessing complet d'une requête utilisateur
        
        Args:
            query: Requête utilisateur brute
            context: Contexte conversationnel optionnel
            
        Returns:
            Dict avec requête nettoyée et métadonnées
        """
        if not query:
            raise ValidationError("Query cannot be empty")
        
        # Validation longueur
        is_valid, error_msg = validate_query_length(query)
        if not is_valid:
            raise ValidationError(error_msg, field_name="query", field_value=query)
        
        # 1. Nettoyage de base
        cleaned_query = clean_text(query, preserve_accents=True)
        
        # 2. Normalisation financière
        normalized_query = self._apply_financial_normalization(cleaned_query)
        
        # 3. Détection caractéristiques
        characteristics = self._analyze_query_characteristics(normalized_query)
        
        # 4. Extraction mots-clés avec filtrage intelligent
        keywords = extract_keywords(normalized_query, min_length=2, exclude_stopwords=True)
        meaningful_words = self.extract_meaningful_words(normalized_query)
        
        # 5. Calcul complexité textuelle
        text_complexity = self.calculate_text_complexity(normalized_query)
        
        # 6. Normalisation pour cache/comparaison
        cache_key = normalize_query(normalized_query)
        
        result = {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "normalized_query": normalized_query,
            "cache_key": cache_key,
            "keywords": keywords,
            "meaningful_words": meaningful_words,
            "characteristics": characteristics,
            "preprocessing_metadata": {
                "query_length": len(query),
                "word_count": len(normalized_query.split()),
                "keyword_count": len(keywords),
                "meaningful_word_count": len(meaningful_words),
                "text_complexity": text_complexity,
                "has_financial_terms": any(kw in FINANCIAL_KEYWORDS for kw in keywords),
                "stopword_ratio": round(1 - (len(meaningful_words) / max(1, len(normalized_query.split()))), 3),
                "language": "fr"  # Assumé français pour ce domaine
            }
        }
        
        self.logger.debug(f"Preprocessed query: {query} -> {result}")
        return result
    
    def extract_meaningful_words(self, text):
        # type: (str) -> List[str]
        """
        Extrait les mots significatifs en filtrant les mots vides français
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste des mots significatifs (sans stopwords)
        """
        if not text:
            return []
        
        # Tokenisation simple
        words = re.findall(r'\b[a-zA-Zàâäéèêëïîôöùûüÿç]+\b', text.lower(), re.UNICODE)
        
        # Filtrage mots vides mais préservation mots financiers
        meaningful_words = []
        for word in words:
            if len(word) >= 2:  # Minimum 2 caractères
                if word not in FRENCH_STOPWORDS or word in FINANCIAL_KEYWORDS:
                    meaningful_words.append(word)
        
        return meaningful_words
    
    def calculate_text_complexity(self, text):
        # type: (str) -> float
        """
        Calcule complexité du texte basée sur mots significatifs
        
        Args:
            text: Texte à analyser
            
        Returns:
            Score de complexité (ratio mots significatifs/total)
        """
        if not text:
            return 0.0
        
        all_words = text.split()
        meaningful_words = self.extract_meaningful_words(text)
        
        if len(all_words) == 0:
            return 0.0
        
        complexity_ratio = len(meaningful_words) / len(all_words)
        return round(complexity_ratio, 3)
    
    def clean_for_cache_advanced(self, query):
        # type: (str) -> str
        """
        Nettoyage avancé pour cache utilisant filtrage stopwords
        
        Args:
            query: Requête à nettoyer pour cache
            
        Returns:
            Clé cache optimisée sans mots vides
        """
        if not query:
            return ""
        
        # Extraction mots significatifs seulement
        meaningful_words = self.extract_meaningful_words(query)
        
        # Normalisation et jointure
        normalized_words = []
        for word in meaningful_words:
            # Suppression accents pour uniformité cache
            normalized_word = normalize_query(word)
            if normalized_word and len(normalized_word) > 1:
                normalized_words.append(normalized_word)
        
        return ' '.join(sorted(normalized_words))  # Tri pour cache cohérent
    
    def _apply_financial_normalization(self, text):
        # type: (str) -> str
        """Applique les normalisations spécifiques au domaine financier"""
        normalized = text
        
        # Application des patterns par catégorie
        for category, patterns in self._compiled_financial_patterns.items():
            for pattern, replacement in patterns:
                normalized = pattern.sub(replacement, normalized)
        
        # Nettoyage espaces multiples résultants
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _analyze_query_characteristics(self, text):
        # type: (str) -> Dict[str, any]
        """Analyse les caractéristiques de la requête pour optimiser traitement"""
        characteristics = {
            "is_question": is_question(text),
            "has_negation": self._detect_negation(text),
            "has_amounts": self._detect_amounts(text),
            "has_dates": self._detect_dates(text),
            "has_accounts": self._detect_accounts(text),
            "has_categories": self._detect_categories(text),
            "urgency_indicators": self._detect_urgency(text),
            "politeness_level": self._detect_politeness(text)
        }
        
        return characteristics
    
    def _detect_negation(self, text):
        # type: (str) -> bool
        """Détecte présence de négation dans le texte"""
        for pattern in self._compiled_negation_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _detect_amounts(self, text):
        # type: (str) -> List[Dict[str, any]]
        """Détecte et extrait les montants mentionnés"""
        amount_pattern = re.compile(r'\b(\d+(?:[,\.]\d+)?)\s*(?:euros?|€|eur)?\b', re.IGNORECASE)
        amounts = []
        
        for match in amount_pattern.finditer(text):
            try:
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                amounts.append({
                    "value": value,
                    "original": match.group(),
                    "position": match.span()
                })
            except ValueError:
                continue
        
        return amounts
    
    def _detect_dates(self, text):
        # type: (str) -> List[str]
        """Détecte expressions temporelles"""
        date_keywords = [
            'aujourd\'hui', 'hier', 'avant-hier', 'demain', 'après-demain',
            'ce mois', 'mois dernier', 'mois prochain', 
            'cette semaine', 'semaine dernière', 'semaine prochaine',
            'cette année', 'année dernière', 'année prochaine',
            'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
            'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'
        ]
        
        detected_dates = []
        text_lower = text.lower()
        
        for keyword in date_keywords:
            if keyword in text_lower:
                detected_dates.append(keyword)
        
        return detected_dates
    
    def _detect_accounts(self, text):
        # type: (str) -> List[str]
        """Détecte mentions de comptes/cartes"""
        account_keywords = [
            'compte courant', 'livret a', 'livret', 'épargne', 
            'carte bleue', 'carte', 'cb', 'visa', 'mastercard'
        ]
        
        detected_accounts = []
        text_lower = text.lower()
        
        for keyword in account_keywords:
            if keyword in text_lower:
                detected_accounts.append(keyword)
        
        return detected_accounts
    
    def _detect_categories(self, text):
        # type: (str) -> List[str]
        """Détecte catégories de dépenses mentionnées"""
        category_keywords = [
            'restaurant', 'resto', 'courses', 'alimentation', 'supermarché',
            'transport', 'essence', 'taxi', 'uber', 'métro', 'bus',
            'shopping', 'vêtements', 'achats', 'boutique', 
            'loisirs', 'cinéma', 'sport', 'vacances', 'sortie',
            'santé', 'pharmacie', 'médecin', 'dentiste', 'hôpital'
        ]
        
        detected_categories = []
        text_lower = text.lower()
        
        for keyword in category_keywords:
            if keyword in text_lower:
                detected_categories.append(keyword)
        
        return detected_categories
    
    def _detect_urgency(self, text):
        # type: (str) -> List[str]
        """Détecte indicateurs d'urgence"""
        urgency_keywords = [
            'urgent', 'rapidement', 'vite', 'immédiatement', 'tout de suite',
            'bloquer', 'opposition', 'vol', 'perte', 'problème', 'erreur'
        ]
        
        detected_urgency = []
        text_lower = text.lower()
        
        for keyword in urgency_keywords:
            if keyword in text_lower:
                detected_urgency.append(keyword)
        
        return detected_urgency
    
    def _detect_politeness(self, text):
        # type: (str) -> str
        """Détecte niveau de politesse (formel/informel)"""
        formal_indicators = [
            'bonjour', 'bonsoir', 's\'il vous plaît', 'merci', 'au revoir',
            'pourriez-vous', 'pouvez-vous', 'veuillez', 'monsieur', 'madame'
        ]
        
        informal_indicators = [
            'salut', 'coucou', 'stp', 'peux-tu', 'hey', 'yo'
        ]
        
        text_lower = text.lower()
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"
    
    def clean_for_cache(self, query):
        # type: (str) -> str
        """Nettoyage spécialisé pour génération clé cache"""
        # Utilise la version avancée avec filtrage stopwords
        return self.clean_for_cache_advanced(query)
    
    def extract_intent_hints(self, preprocessed_data):
        # type: (Dict[str, any]) -> Dict[str, any]
        """Extrait des indices pour accélérer détection d'intention"""
        query = preprocessed_data["normalized_query"]
        characteristics = preprocessed_data["characteristics"]
        meaningful_words = preprocessed_data.get("meaningful_words", [])
        
        hints = {
            "likely_financial": bool(characteristics["has_amounts"] or 
                                   characteristics["has_accounts"] or 
                                   characteristics["has_categories"]),
            "likely_greeting": any(word in meaningful_words for word in ['bonjour', 'salut', 'hello']),
            "likely_goodbye": any(word in meaningful_words for word in ['revoir', 'bye', 'ciao']),
            "likely_help": any(word in meaningful_words for word in ['aide', 'help', 'comment']),
            "likely_transfer": any(word in meaningful_words for word in ['virer', 'virement', 'transférer']),
            "likely_balance": any(word in meaningful_words for word in ['solde', 'combien', 'argent']),
            "likely_search": any(word in meaningful_words for word in ['dépenses', 'historique', 'recherche']),
            "urgency_level": "high" if characteristics["urgency_indicators"] else "normal",
            "word_density": len(meaningful_words) / max(1, len(query.split())),  # Densité mots significatifs
            "has_stopwords": len(meaningful_words) < len(query.split())  # Contient des stopwords
        }
        
        return hints


# Factory function pour instance singleton
_text_cleaner_instance = None

def get_text_cleaner():
    """Factory function pour récupérer instance TextCleaner singleton"""
    global _text_cleaner_instance
    if _text_cleaner_instance is None:
        _text_cleaner_instance = TextCleaner()
    return _text_cleaner_instance


# Exports publics
__all__ = [
    "TextCleaner",
    "get_text_cleaner"
]