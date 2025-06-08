"""
Processeur de requêtes pour améliorer la recherche.

Ce module analyse, normalise et enrichit les requêtes de recherche
pour optimiser les résultats.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import unicodedata

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Traite et enrichit les requêtes de recherche."""
    
    def __init__(self):
        # Dictionnaire de synonymes pour les termes financiers
        self.synonyms = {
            "cb": ["carte", "carte bancaire", "paiement"],
            "dab": ["retrait", "distributeur", "cash"],
            "vir": ["virement", "transfer", "paiement"],
            "prlv": ["prélèvement", "prelevement", "débit"],
            "resto": ["restaurant", "repas", "déjeuner", "dîner"],
            "supermarche": ["supermarché", "courses", "alimentation"],
            "essence": ["carburant", "gasoil", "diesel", "sp95", "sp98"],
            "pharmacie": ["médicament", "santé", "parapharmacie"],
            "loyer": ["location", "appartement", "logement"],
            "tel": ["téléphone", "mobile", "forfait"],
            "elec": ["électricité", "edf", "energie"],
            "assur": ["assurance", "mutuelle", "cotisation"]
        }
        
        # Patterns pour extraire des informations temporelles
        self.temporal_patterns = {
            "aujourd'hui": lambda: datetime.now().date(),
            "hier": lambda: (datetime.now() - timedelta(days=1)).date(),
            "cette semaine": lambda: (datetime.now() - timedelta(days=datetime.now().weekday())).date(),
            "semaine dernière": lambda: (datetime.now() - timedelta(days=datetime.now().weekday() + 7)).date(),
            "ce mois": lambda: datetime.now().replace(day=1).date(),
            "mois dernier": lambda: (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1).date(),
            "janvier": lambda: datetime.now().replace(month=1, day=1).date(),
            "février": lambda: datetime.now().replace(month=2, day=1).date(),
            "mars": lambda: datetime.now().replace(month=3, day=1).date(),
            "avril": lambda: datetime.now().replace(month=4, day=1).date(),
            "mai": lambda: datetime.now().replace(month=5, day=1).date(),
            "juin": lambda: datetime.now().replace(month=6, day=1).date(),
            "juillet": lambda: datetime.now().replace(month=7, day=1).date(),
            "août": lambda: datetime.now().replace(month=8, day=1).date(),
            "septembre": lambda: datetime.now().replace(month=9, day=1).date(),
            "octobre": lambda: datetime.now().replace(month=10, day=1).date(),
            "novembre": lambda: datetime.now().replace(month=11, day=1).date(),
            "décembre": lambda: datetime.now().replace(month=12, day=1).date(),
        }
        
        # Mots vides à ignorer
        self.stop_words = {
            "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au", "aux",
            "ce", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
            "notre", "nos", "votre", "vos", "leur", "leurs", "que", "qui", "quoi",
            "pour", "par", "avec", "sans", "sur", "sous", "dans", "en", "entre"
        }
        
        # Patterns pour détecter les montants
        self.amount_pattern = re.compile(r'(\d+(?:[.,]\d{1,2})?)\s*(?:€|eur|euros?)?', re.IGNORECASE)
        
        # Patterns pour détecter les marchands
        self.merchant_patterns = [
            re.compile(r'\b(carrefour|auchan|leclerc|intermarché|lidl|aldi)\b', re.IGNORECASE),
            re.compile(r'\b(mcdo|mcdonald|burger king|kfc|subway)\b', re.IGNORECASE),
            re.compile(r'\b(amazon|fnac|darty|boulanger|cdiscount)\b', re.IGNORECASE),
            re.compile(r'\b(sncf|ratp|uber|blablacar)\b', re.IGNORECASE),
        ]
    
    async def process(self, query: str) -> Dict[str, Any]:
        """
        Traite une requête de recherche.
        
        Args:
            query: Requête brute
            
        Returns:
            Dict: Requête traitée avec métadonnées
        """
        logger.debug(f"Traitement de la requête: {query}")
        
        # Normaliser la requête
        normalized = self._normalize_query(query)
        
        # Extraire les informations
        temporal_info = self._extract_temporal_info(normalized)
        amount_info = self._extract_amount_info(normalized)
        merchant_info = self._extract_merchant_info(normalized)
        
        # Extraire les mots-clés
        keywords = self._extract_keywords(normalized)
        
        # Étendre la requête avec des synonymes
        expanded_query = self._expand_query(normalized, keywords)
        
        # Générer des suggestions
        suggestions = self._generate_suggestions(keywords)
        
        result = {
            "original_query": query,
            "normalized_query": normalized,
            "expanded_query": expanded_query,
            "keywords": keywords,
            "temporal_info": temporal_info,
            "amount_info": amount_info,
            "merchant_info": merchant_info,
            "suggestions": suggestions
        }
        
        logger.debug(f"Requête traitée: {result}")
        return result
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalise une requête (minuscules, accents, espaces).
        
        Args:
            query: Requête brute
            
        Returns:
            str: Requête normalisée
        """
        # Convertir en minuscules
        query = query.lower()
        
        # Supprimer les accents
        query = ''.join(
            c for c in unicodedata.normalize('NFD', query)
            if unicodedata.category(c) != 'Mn'
        )
        
        # Normaliser les espaces
        query = ' '.join(query.split())
        
        return query
    
    def _extract_temporal_info(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extrait les informations temporelles de la requête.
        
        Args:
            query: Requête normalisée
            
        Returns:
            Dict: Informations temporelles ou None
        """
        temporal_info = None
        
        # Chercher des patterns temporels
        for pattern, date_func in self.temporal_patterns.items():
            if pattern in query:
                try:
                    date = date_func()
                    if "mois" in pattern or pattern in ["janvier", "février", "mars", "avril", "mai", "juin",
                                                        "juillet", "août", "septembre", "octobre", "novembre", "décembre"]:
                        # Pour les mois, prendre tout le mois
                        if pattern == "ce mois":
                            date_from = date
                            date_to = datetime.now().date()
                        else:
                            date_from = date
                            # Dernier jour du mois
                            if date.month == 12:
                                date_to = date.replace(year=date.year + 1, month=1, day=1) - timedelta(days=1)
                            else:
                                date_to = date.replace(month=date.month + 1, day=1) - timedelta(days=1)
                    elif "semaine" in pattern:
                        # Pour les semaines
                        date_from = date
                        date_to = date + timedelta(days=6)
                    else:
                        # Pour les jours
                        date_from = date
                        date_to = date
                    
                    temporal_info = {
                        "pattern": pattern,
                        "date_from": date_from,
                        "date_to": date_to
                    }
                    break
                except Exception as e:
                    logger.warning(f"Erreur lors de l'extraction temporelle: {e}")
        
        # Chercher des dates explicites (format JJ/MM/AAAA ou JJ-MM-AAAA)
        date_pattern = re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})')
        matches = date_pattern.findall(query)
        if matches and not temporal_info:
            try:
                day, month, year = matches[0]
                if len(year) == 2:
                    year = "20" + year
                date = datetime(int(year), int(month), int(day)).date()
                temporal_info = {
                    "pattern": "date_explicit",
                    "date_from": date,
                    "date_to": date
                }
            except Exception as e:
                logger.warning(f"Erreur lors du parsing de date: {e}")
        
        return temporal_info
    
    def _extract_amount_info(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extrait les informations de montant de la requête.
        
        Args:
            query: Requête normalisée
            
        Returns:
            Dict: Informations de montant ou None
        """
        amount_info = None
        
        # Chercher des montants
        matches = self.amount_pattern.findall(query)
        if matches:
            amounts = []
            for match in matches:
                try:
                    # Remplacer la virgule par un point
                    amount_str = match.replace(',', '.')
                    amount = float(amount_str)
                    amounts.append(amount)
                except ValueError:
                    pass
            
            if amounts:
                # Détecter si c'est une plage ou un montant unique
                if "entre" in query and len(amounts) >= 2:
                    amount_info = {
                        "type": "range",
                        "min": min(amounts),
                        "max": max(amounts)
                    }
                elif "plus de" in query or "superieur" in query or ">" in query:
                    amount_info = {
                        "type": "min",
                        "min": amounts[0]
                    }
                elif "moins de" in query or "inferieur" in query or "<" in query:
                    amount_info = {
                        "type": "max",
                        "max": amounts[0]
                    }
                else:
                    # Montant exact ou approximatif
                    amount_info = {
                        "type": "exact",
                        "amount": amounts[0]
                    }
        
        return amount_info
    
    def _extract_merchant_info(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extrait les informations de marchand de la requête.
        
        Args:
            query: Requête normalisée
            
        Returns:
            Dict: Informations de marchand ou None
        """
        merchant_info = None
        merchants = []
        
        # Chercher des marchands connus
        for pattern in self.merchant_patterns:
            matches = pattern.findall(query)
            if matches:
                merchants.extend(matches)
        
        if merchants:
            merchant_info = {
                "merchants": list(set(merchants)),  # Éliminer les doublons
                "type": "specific"
            }
        
        return merchant_info
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extrait les mots-clés importants de la requête.
        
        Args:
            query: Requête normalisée
            
        Returns:
            List[str]: Liste des mots-clés
        """
        # Séparer les mots
        words = query.split()
        
        # Filtrer les mots vides
        keywords = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Ajouter des variantes pour certains mots-clés
        expanded_keywords = []
        for keyword in keywords:
            expanded_keywords.append(keyword)
            
            # Ajouter le pluriel/singulier
            if keyword.endswith('s'):
                expanded_keywords.append(keyword[:-1])
            else:
                expanded_keywords.append(keyword + 's')
        
        return list(set(expanded_keywords))
    
    def _expand_query(self, query: str, keywords: List[str]) -> str:
        """
        Étend la requête avec des synonymes.
        
        Args:
            query: Requête normalisée
            keywords: Mots-clés extraits
            
        Returns:
            str: Requête étendue
        """
        expanded_terms = []
        
        # Ajouter les termes originaux
        expanded_terms.extend(keywords)
        
        # Ajouter les synonymes
        for keyword in keywords:
            if keyword in self.synonyms:
                expanded_terms.extend(self.synonyms[keyword])
            
            # Chercher aussi les synonymes partiels
            for syn_key, syn_values in self.synonyms.items():
                if keyword in syn_key or syn_key in keyword:
                    expanded_terms.extend(syn_values)
        
        # Construire la requête étendue
        expanded_query = ' '.join(set(expanded_terms))
        
        return expanded_query if expanded_query else query
    
    def _generate_suggestions(self, keywords: List[str]) -> List[str]:
        """
        Génère des suggestions basées sur les mots-clés.
        
        Args:
            keywords: Mots-clés extraits
            
        Returns:
            List[str]: Suggestions de recherche
        """
        suggestions = []
        
        # Suggestions basées sur les synonymes
        for keyword in keywords[:3]:  # Limiter aux 3 premiers mots-clés
            if keyword in self.synonyms:
                for syn in self.synonyms[keyword][:2]:  # 2 premiers synonymes
                    suggestion = ' '.join([syn] + [k for k in keywords if k != keyword][:2])
                    suggestions.append(suggestion)
        
        # Suggestions de catégories communes
        if any(word in keywords for word in ["restaurant", "resto", "repas"]):
            suggestions.append("restaurants ce mois")
            suggestions.append("restaurants paris")
        
        if any(word in keywords for word in ["courses", "supermarche", "alimentation"]):
            suggestions.append("courses cette semaine")
            suggestions.append("dépenses alimentaires")
        
        return suggestions[:5]  # Limiter à 5 suggestions