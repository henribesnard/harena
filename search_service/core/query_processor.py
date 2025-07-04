"""
Processeur de requêtes intelligent pour la recherche.

Ce module traite et enrichit les requêtes de recherche avant
de les envoyer aux moteurs lexical et sémantique.
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from search_service.models.search_types import FINANCIAL_SYNONYMS

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Résultat de l'analyse d'une requête."""
    original_query: str
    cleaned_query: str
    expanded_query: str
    detected_entities: Dict[str, Any]
    query_type: str
    confidence: float
    suggested_filters: Dict[str, Any]
    processing_notes: List[str]
    
    # ============================================================================
    # PROPRIÉTÉS MANQUANTES AJOUTÉES POUR COMPATIBILITÉ AVEC LES MOTEURS
    # ============================================================================
    
    @property
    def key_terms(self) -> List[str]:
        """
        Retourne les termes clés extraits de la requête.
        
        Cette propriété est utilisée par les moteurs lexical et sémantique
        pour analyser les mots-clés importants.
        
        Returns:
            Liste des mots-clés extraits
        """
        return self.detected_entities.get('keywords', [])
    
    @property
    def has_exact_phrases(self) -> bool:
        """
        Vérifie si la requête contient des phrases exactes.
        
        Cette propriété est utilisée par les moteurs pour déterminer
        si des correspondances de phrase exacte doivent être privilégiées.
        
        Returns:
            True si la requête contient des phrases exactes
        """
        # Détecter les guillemets dans la requête originale
        if '"' in self.original_query:
            return True
        
        # Considérer les requêtes courtes comme des phrases exactes potentielles
        if len(self.cleaned_query.split()) <= 3 and len(self.cleaned_query.split()) > 0:
            return True
        
        # Détecter des patterns de phrases exactes
        exact_phrase_patterns = [
            r'\b\w+\s+\w+\b',  # Deux mots consécutifs
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Noms propres
        ]
        
        for pattern in exact_phrase_patterns:
            if re.search(pattern, self.original_query):
                return True
        
        return False
    
    @property
    def has_financial_entities(self) -> bool:
        """
        Vérifie si la requête contient des entités financières.
        
        Cette propriété est utilisée par les moteurs pour adapter
        leur stratégie de recherche aux requêtes financières.
        
        Returns:
            True si la requête contient des entités financières
        """
        entities = self.detected_entities
        
        # Vérifier la présence d'entités financières extraites
        has_amounts = bool(entities.get('amounts', []))
        has_categories = bool(entities.get('categories', []))
        
        if has_amounts or has_categories:
            return True
        
        # Vérifier la présence de mots-clés financiers dans le texte
        financial_keywords = [
            'euro', '€', 'montant', 'prix', 'coût', 'transaction', 'virement',
            'paiement', 'carte', 'bancaire', 'banque', 'compte', 'solde',
            'débit', 'crédit', 'facture', 'achat', 'vente', 'remboursement',
            'salaire', 'paie', 'pension', 'allocation', 'frais', 'commission'
        ]
        
        query_lower = self.cleaned_query.lower()
        return any(keyword in query_lower for keyword in financial_keywords)
    
    @property
    def is_question(self) -> bool:
        """
        Vérifie si la requête est une question.
        
        Cette propriété est utilisée par les moteurs pour adapter
        leur traitement aux requêtes interrogatives.
        
        Returns:
            True si la requête est une question
        """
        # Vérifier la présence d'un point d'interrogation
        if self.original_query.endswith('?'):
            return True
        
        # Vérifier la présence de mots interrogatifs
        question_words = [
            "quoi", "comment", "pourquoi", "où", "quand", "qui", 
            "quel", "quelle", "quels", "quelles", "combien",
            "qu'est-ce", "est-ce", "y a-t-il", "peut-on", "peut-il"
        ]
        
        query_lower = self.original_query.lower()
        
        # Vérifier si la requête commence par un mot interrogatif
        for word in question_words:
            if query_lower.startswith(word) or f" {word} " in query_lower:
                return True
        
        # Vérifier des patterns interrogatifs
        interrogative_patterns = [
            r'\bcomment\s+',
            r'\bpourquoi\s+',
            r'\bquand\s+',
            r'\boù\s+',
            r'\bque\s+',
            r'\bqui\s+',
            r'\bcombien\s+'
        ]
        
        for pattern in interrogative_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    @property
    def exact_phrases(self) -> List[str]:
        """
        Retourne les phrases exactes détectées dans la requête.
        
        Returns:
            Liste des phrases exactes
        """
        phrases = []
        
        # Extraire les phrases entre guillemets
        quoted_phrases = re.findall(r'"([^"]*)"', self.original_query)
        phrases.extend(quoted_phrases)
        
        # Si pas de guillemets mais requête courte, considérer comme phrase exacte
        if not phrases and self.has_exact_phrases:
            if len(self.cleaned_query.split()) <= 3:
                phrases.append(self.cleaned_query)
        
        return phrases
    
    @property
    def financial_entities_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des entités financières détectées.
        
        Returns:
            Résumé structuré des entités financières
        """
        entities = self.detected_entities
        
        return {
            "amounts_count": len(entities.get('amounts', [])),
            "dates_count": len(entities.get('dates', [])),
            "categories_count": len(entities.get('categories', [])),
            "has_amounts": bool(entities.get('amounts', [])),
            "has_dates": bool(entities.get('dates', [])),
            "has_categories": bool(entities.get('categories', [])),
            "amount_range": self._get_amount_range(),
            "date_range": self._get_date_range(),
            "detected_categories": entities.get('categories', [])
        }
    
    def _get_amount_range(self) -> Optional[Dict[str, float]]:
        """Calcule la fourchette de montants détectés."""
        amounts = self.detected_entities.get('amounts', [])
        if not amounts:
            return None
        
        values = [a['value'] for a in amounts]
        return {
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }
    
    def _get_date_range(self) -> Optional[Dict[str, str]]:
        """Calcule la fourchette de dates détectées."""
        dates = self.detected_entities.get('dates', [])
        if not dates:
            return None
        
        date_objects = []
        for d in dates:
            if isinstance(d.get('date'), str):
                try:
                    date_objects.append(datetime.fromisoformat(d['date']))
                except ValueError:
                    continue
            elif isinstance(d.get('date'), datetime):
                date_objects.append(d['date'])
        
        if not date_objects:
            return None
        
        return {
            "start": min(date_objects).isoformat(),
            "end": max(date_objects).isoformat(),
            "count": len(date_objects)
        }


@dataclass
class EntityExtraction:
    """Entités extraites d'une requête."""
    amounts: List[Dict[str, Any]]
    dates: List[Dict[str, Any]]
    merchants: List[str]
    categories: List[str]
    keywords: List[str]
    negations: List[str]


class QueryProcessor:
    """
    Processeur intelligent de requêtes de recherche.
    
    Fonctionnalités:
    - Nettoyage et normalisation du texte
    - Extraction d'entités financières
    - Expansion avec synonymes
    - Détection de type de requête
    - Suggestion de filtres automatiques
    - Correction orthographique basique
    """
    
    def __init__(self):
        # Patterns de regex pour l'extraction d'entités
        self.amount_patterns = [
            r'(\d+(?:[,.]?\d{3})*(?:[,.]\d{2})?)\s*€?',  # 1000.50€, 1,000.50
            r'€\s*(\d+(?:[,.]?\d{3})*(?:[,.]\d{2})?)',   # €1000.50
            r'(\d+(?:[,.]?\d{3})*(?:[,.]\d{2})?)\s*euros?', # 1000 euros
        ]
        
        self.date_patterns = [
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',      # DD/MM/YYYY
            r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})',      # YYYY/MM/DD
            r'(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})',
        ]
        
        # Mots-clés par catégorie
        self.category_keywords = {
            "restaurant": ["restaurant", "resto", "brasserie", "café", "fast", "food", "mcdo", "burger", "pizza"],
            "supermarché": ["supermarché", "hypermarché", "courses", "carrefour", "leclerc", "auchan", "intermarché"],
            "essence": ["essence", "carburant", "station", "shell", "total", "bp", "esso"],
            "pharmacie": ["pharmacie", "parapharmacie", "médicament", "docteur", "santé"],
            "transport": ["transport", "métro", "bus", "train", "taxi", "uber", "sncf", "ratp"],
            "banque": ["banque", "virement", "retrait", "distributeur", "agios", "frais"],
            "abonnement": ["abonnement", "subscription", "mensuel", "annuel", "netflix", "spotify"],
            "shopping": ["shopping", "vêtement", "chaussure", "magasin", "boutique", "amazon"]
        }
        
        # Mots de négation
        self.negation_words = ["pas", "non", "sans", "sauf", "excepté", "hormis", "ne", "n'"]
        
        # Mots vides spécifiques au domaine financier
        self.stop_words = {
            "le", "la", "les", "un", "une", "des", "du", "de", "d'", "et", "ou", "à", "au", "aux",
            "dans", "sur", "pour", "avec", "par", "chez", "vers", "entre", "depuis", "jusqu",
            "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses"
        }
        
        # Cache des analyses récentes
        self._analysis_cache: Dict[str, QueryAnalysis] = {}
        
        logger.info("Query processor initialized")
    
    def process_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        """
        Traite une requête de recherche complète.
        
        Args:
            query: Requête brute
            user_context: Contexte utilisateur (historique, préférences)
            
        Returns:
            Analyse complète de la requête
        """
        # Vérifier le cache
        cache_key = f"{query}:{hash(str(user_context))}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        processing_notes = []
        
        # 1. Nettoyage initial
        cleaned_query = self._clean_query(query)
        processing_notes.append(f"Query cleaned: '{query}' -> '{cleaned_query}'")
        
        # 2. Extraction d'entités
        entities = self._extract_entities(cleaned_query)
        processing_notes.append(f"Entities extracted: {len(entities.amounts)} amounts, {len(entities.dates)} dates")
        
        # 3. Détection du type de requête
        query_type, confidence = self._detect_query_type(cleaned_query, entities)
        processing_notes.append(f"Query type: {query_type} (confidence: {confidence:.2f})")
        
        # 4. Expansion avec synonymes
        expanded_query = self._expand_query(cleaned_query, query_type)
        processing_notes.append(f"Query expanded with synonyms")
        
        # 5. Suggestion de filtres automatiques
        suggested_filters = self._suggest_filters(entities, query_type, user_context)
        if suggested_filters:
            processing_notes.append(f"Auto-filters suggested: {list(suggested_filters.keys())}")
        
        # 6. Créer l'analyse finale
        analysis = QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            expanded_query=expanded_query,
            detected_entities=self._serialize_entities(entities),
            query_type=query_type,
            confidence=confidence,
            suggested_filters=suggested_filters,
            processing_notes=processing_notes
        )
        
        # Mettre en cache
        self._analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _clean_query(self, query: str) -> str:
        """Nettoie et normalise la requête."""
        if not query:
            return ""
        
        # Supprimer les espaces multiples et normaliser
        cleaned = re.sub(r'\s+', ' ', query.strip().lower())
        
        # Supprimer les caractères spéciaux inutiles
        cleaned = re.sub(r'[^\w\s€.,/-]', ' ', cleaned)
        
        # Normaliser les séparateurs de montants
        cleaned = re.sub(r'(\d),(\d{3})', r'\1\2', cleaned)  # 1,000 -> 1000
        cleaned = re.sub(r'(\d)\.(\d{3})', r'\1\2', cleaned)  # 1.000 -> 1000
        
        # Supprimer les mots vides si la requête est longue
        words = cleaned.split()
        if len(words) > 3:
            words = [w for w in words if w not in self.stop_words]
            cleaned = ' '.join(words)
        
        return cleaned.strip()
    
    def _extract_entities(self, query: str) -> EntityExtraction:
        """Extrait les entités de la requête."""
        entities = EntityExtraction(
            amounts=[], dates=[], merchants=[], categories=[], keywords=[], negations=[]
        )
        
        # Extraction des montants
        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '').replace('.', '')
                try:
                    # Gérer les centimes
                    if '.' in match.group(1) or ',' in match.group(1):
                        amount_value = float(match.group(1).replace(',', ''))
                    else:
                        amount_value = float(amount_str)
                    
                    entities.amounts.append({
                        "value": amount_value,
                        "raw": match.group(0),
                        "position": match.span()
                    })
                except ValueError:
                    continue
        
        # Extraction des dates
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    date_info = self._parse_date_match(match)
                    if date_info:
                        entities.dates.append(date_info)
                except ValueError:
                    continue
        
        # Extraction des catégories
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    if category not in entities.categories:
                        entities.categories.append(category)
        
        # Détection des négations
        for neg_word in self.negation_words:
            if neg_word in query:
                entities.negations.append(neg_word)
        
        # Extraction des mots-clés importants
        words = query.split()
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                entities.keywords.append(word)
        
        return entities
    
    def _parse_date_match(self, match) -> Optional[Dict[str, Any]]:
        """Parse un match de date."""
        groups = match.groups()
        
        # Format DD/MM/YYYY ou DD-MM-YYYY
        if len(groups) == 3 and groups[0].isdigit() and groups[1].isdigit() and groups[2].isdigit():
            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
            
            # Vérifier si c'est DD/MM/YYYY ou YYYY/MM/DD
            if year > 1900 and year < 2100:
                if len(groups[2]) == 4:  # YYYY à la fin
                    date_obj = datetime(year, month, day)
                else:  # YYYY au début
                    date_obj = datetime(int(groups[0]), int(groups[1]), int(groups[2]))
            else:
                return None
            
            return {
                "date": date_obj,
                "raw": match.group(0),
                "position": match.span(),
                "type": "specific"
            }
        
        # Format avec nom de mois
        elif len(groups) == 3:
            month_names = {
                "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
                "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
            }
            
            day = int(groups[0])
            month_name = groups[1].lower()
            year = int(groups[2])
            
            if month_name in month_names:
                month = month_names[month_name]
                date_obj = datetime(year, month, day)
                
                return {
                    "date": date_obj,
                    "raw": match.group(0),
                    "position": match.span(),
                    "type": "named_month"
                }
        
        return None
    
    def _detect_query_type(self, query: str, entities: EntityExtraction) -> Tuple[str, float]:
        """Détecte le type de requête et sa confiance."""
        confidence = 0.0
        
        # Analyse basée sur les entités présentes
        if entities.amounts:
            if len(entities.amounts) == 1:
                return "amount_search", 0.8
            elif len(entities.amounts) == 2:
                return "amount_range", 0.9
            else:
                return "complex_amount", 0.7
        
        if entities.dates:
            if len(entities.dates) == 1:
                return "date_search", 0.8
            else:
                return "date_range", 0.9
        
        # Analyse basée sur les mots-clés
        search_keywords = ["cherche", "trouve", "recherche", "voir", "affiche"]
        if any(keyword in query for keyword in search_keywords):
            confidence += 0.3
        
        # Type basé sur les catégories détectées
        if entities.categories:
            return "category_search", 0.7 + confidence
        
        # Analyse basée sur la structure de la requête
        if any(word in query for word in ["tous", "toutes", "liste", "historique"]):
            return "list_all", 0.6 + confidence
        
        if any(word in query for word in ["similar", "similaire", "comme", "pareil"]):
            return "similarity_search", 0.8
        
        if any(word in query for word in ["récent", "dernier", "nouveau", "aujourd"]):
            return "recent_search", 0.7
        
        # Requête libre par défaut
        return "free_text", 0.5 + confidence
    
    def _expand_query(self, query: str, query_type: str) -> str:
        """Expand la requête avec des synonymes."""
        expanded_terms = set([query])
        query_words = query.split()
        
        # Expansion basée sur les synonymes financiers
        for word in query_words:
            if word in FINANCIAL_SYNONYMS:
                expanded_terms.update(FINANCIAL_SYNONYMS[word])
        
        # Expansion spécifique par type de requête
        if query_type == "category_search":
            # Ajouter des termes de catégorie similaires
            for category, keywords in self.category_keywords.items():
                if any(keyword in query for keyword in keywords):
                    expanded_terms.update(keywords[:3])  # Limiter à 3 synonymes
        
        # Expansion pour améliorer les résultats selon le validateur
        if "restaurant" in query:
            expanded_terms.update(["resto", "brasserie", "restauration", "dining"])
        
        if "carte" in query:
            expanded_terms.update(["cb", "paiement", "card", "bancaire"])
        
        if "virement" in query:
            expanded_terms.update(["transfer", "transfert", "salaire", "paie"])
        
        return " ".join(expanded_terms)
    
    def _suggest_filters(
        self,
        entities: EntityExtraction,
        query_type: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Suggère des filtres automatiques basés sur l'analyse."""
        filters = {}
        
        # Filtres de montant
        if entities.amounts:
            if len(entities.amounts) == 1:
                amount = entities.amounts[0]["value"]
                # Suggérer une fourchette de ±20%
                margin = amount * 0.2
                filters["amount_min"] = max(0, amount - margin)
                filters["amount_max"] = amount + margin
            elif len(entities.amounts) == 2:
                amounts = sorted([a["value"] for a in entities.amounts])
                filters["amount_min"] = amounts[0]
                filters["amount_max"] = amounts[1]
        
        # Filtres de date
        if entities.dates:
            if len(entities.dates) == 1:
                date = entities.dates[0]["date"]
                # Suggérer le jour ou le mois selon le contexte
                if query_type == "recent_search":
                    filters["date_from"] = (date - timedelta(days=7)).strftime("%Y-%m-%d")
                    filters["date_to"] = date.strftime("%Y-%m-%d")
                else:
                    filters["date_from"] = date.strftime("%Y-%m-%d")
                    filters["date_to"] = date.strftime("%Y-%m-%d")
            elif len(entities.dates) == 2:
                dates = sorted([d["date"] for d in entities.dates])
                filters["date_from"] = dates[0].strftime("%Y-%m-%d")
                filters["date_to"] = dates[1].strftime("%Y-%m-%d")
        
        # Filtres de catégorie
        if entities.categories:
            # Note: nécessiterait un mapping catégorie -> category_id
            filters["suggested_categories"] = entities.categories
        
        # Filtres basés sur le type de requête
        if query_type == "recent_search":
            if "date_from" not in filters:
                recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                filters["date_from"] = recent_date
        
        # Filtres basés sur le contexte utilisateur
        if user_context:
            # Exemple: compte préféré de l'utilisateur
            if "preferred_account_id" in user_context:
                filters["account_ids"] = [user_context["preferred_account_id"]]
        
        return filters
    
    def _serialize_entities(self, entities: EntityExtraction) -> Dict[str, Any]:
        """Sérialise les entités pour l'analyse."""
        return {
            "amounts": [
                {
                    "value": a["value"],
                    "raw": a["raw"],
                    "formatted": f"{a['value']:.2f}€"
                }
                for a in entities.amounts
            ],
            "dates": [
                {
                    "date": d["date"].isoformat(),
                    "raw": d["raw"],
                    "type": d["type"],
                    "formatted": d["date"].strftime("%d/%m/%Y")
                }
                for d in entities.dates
            ],
            "categories": entities.categories,
            "keywords": entities.keywords,
            "negations": entities.negations,
            "merchants": entities.merchants
        }
    
    def optimize_for_lexical_search(self, analysis: QueryAnalysis) -> str:
        """Optimise la requête pour la recherche lexicale (Elasticsearch)."""
        # Basé sur les résultats du validateur, prioriser:
        # 1. Les correspondances exactes de phrase
        # 2. Les champs merchant_name
        # 3. Les requêtes multi-champs
        
        optimized_parts = []
        
        # Phrase exacte si courte
        if len(analysis.cleaned_query.split()) <= 3:
            optimized_parts.append(f'"{analysis.cleaned_query}"')
        
        # Requête originale nettoyée
        optimized_parts.append(analysis.cleaned_query)
        
        # Mots-clés importants
        important_keywords = [kw for kw in analysis.key_terms if len(kw) > 3]
        if important_keywords:
            optimized_parts.extend(important_keywords[:3])  # Top 3 mots-clés
        
        # Synonymes financiers pour améliorer la couverture
        if analysis.query_type == "category_search":
            optimized_parts.append(analysis.expanded_query)
        
        return " ".join(set(optimized_parts))
    
    def optimize_for_semantic_search(self, analysis: QueryAnalysis) -> str:
        """Optimise la requête pour la recherche sémantique (Qdrant)."""
        # Pour la recherche sémantique, privilégier:
        # 1. Le contexte et l'intention
        # 2. Les termes enrichis
        # 3. La requête étendue avec synonymes
        
        semantic_parts = []
        
        # Requête originale pour le contexte
        semantic_parts.append(analysis.original_query)
        
        # Ajouter du contexte basé sur les entités détectées
        if analysis.detected_entities["amounts"]:
            amounts_context = " ".join([a["formatted"] for a in analysis.detected_entities["amounts"]])
            semantic_parts.append(f"montant {amounts_context}")
        
        if analysis.detected_entities["categories"]:
            categories_context = " ".join(analysis.detected_entities["categories"])
            semantic_parts.append(f"catégorie {categories_context}")
        
        if analysis.detected_entities["dates"]:
            dates_context = " ".join([d["formatted"] for d in analysis.detected_entities["dates"]])
            semantic_parts.append(f"date {dates_context}")
        
        # Requête étendue pour la richesse sémantique
        semantic_parts.append(analysis.expanded_query)
        
        return " ".join(semantic_parts)
    
    def suggest_corrections(self, query: str) -> List[str]:
        """Suggère des corrections pour les requêtes avec fautes."""
        suggestions = []
        
        # Corrections courantes pour le domaine financier
        corrections = {
            "restorant": "restaurant",
            "restau": "restaurant", 
            "supermaché": "supermarché",
            "pharmaci": "pharmacie",
            "esence": "essence",
            "virment": "virement",
            "bancair": "bancaire",
            "abonemment": "abonnement",
            "transport": "transport"
        }
        
        words = query.lower().split()
        corrected_words = []
        has_corrections = False
        
        for word in words:
            if word in corrections:
                corrected_words.append(corrections[word])
                has_corrections = True
            else:
                # Vérification de distance d'édition simple
                best_match = self._find_closest_match(word)
                if best_match and best_match != word:
                    corrected_words.append(best_match)
                    has_corrections = True
                else:
                    corrected_words.append(word)
        
        if has_corrections:
            suggestions.append(" ".join(corrected_words))
        
        return suggestions
    
    def _find_closest_match(self, word: str) -> Optional[str]:
        """Trouve le mot le plus proche dans le vocabulaire financier."""
        if len(word) < 3:
            return None
        
        # Vocabulaire financier commun
        financial_vocab = set()
        for synonyms in FINANCIAL_SYNONYMS.values():
            financial_vocab.update(synonyms)
        
        for category_keywords in self.category_keywords.values():
            financial_vocab.update(category_keywords)
        
        # Distance d'édition simple (substitution uniquement)
        best_match = None
        min_distance = float('inf')
        
        for vocab_word in financial_vocab:
            if abs(len(word) - len(vocab_word)) <= 2:  # Longueur similaire
                distance = self._simple_edit_distance(word, vocab_word)
                if distance < min_distance and distance <= 2:
                    min_distance = distance
                    best_match = vocab_word
        
        return best_match
    
    def _simple_edit_distance(self, s1: str, s2: str) -> int:
        """Calcule une distance d'édition simplifiée."""
        if len(s1) != len(s2):
            return abs(len(s1) - len(s2)) + 1
        
        differences = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        return differences
    
    def extract_search_intent(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Extrait l'intention de recherche détaillée."""
        intent = {
            "primary_intent": analysis.query_type,
            "confidence": analysis.confidence,
            "search_scope": "user_transactions",  # Par défaut
            "temporal_focus": "all_time",
            "amount_focus": "any",
            "category_focus": "any"
        }
        
        # Analyse temporelle
        if analysis.detected_entities["dates"]:
            if any(d["type"] == "recent" for d in analysis.detected_entities["dates"]):
                intent["temporal_focus"] = "recent"
            elif len(analysis.detected_entities["dates"]) == 1:
                intent["temporal_focus"] = "specific_date"
            else:
                intent["temporal_focus"] = "date_range"
        
        # Analyse des montants
        if analysis.detected_entities["amounts"]:
            if len(analysis.detected_entities["amounts"]) == 1:
                intent["amount_focus"] = "specific_amount"
            else:
                intent["amount_focus"] = "amount_range"
        
        # Analyse des catégories
        if analysis.detected_entities["categories"]:
            if len(analysis.detected_entities["categories"]) == 1:
                intent["category_focus"] = "specific_category"
            else:
                intent["category_focus"] = "multiple_categories"
        
        # Détection d'intentions spéciales
        query_lower = analysis.original_query.lower()
        
        if any(word in query_lower for word in ["similaire", "comme", "pareil"]):
            intent["special_intent"] = "find_similar"
        
        if any(word in query_lower for word in ["récurrent", "régulier", "mensuel"]):
            intent["special_intent"] = "find_recurring"
        
        if any(word in query_lower for word in ["suspect", "inhabituel", "bizarre"]):
            intent["special_intent"] = "find_anomalies"
        
        if any(word in query_lower for word in ["total", "somme", "montant"]):
            intent["special_intent"] = "calculate_sum"
        
        return intent
    
    def generate_alternative_queries(self, analysis: QueryAnalysis) -> List[str]:
        """Génère des requêtes alternatives pour améliorer les résultats."""
        alternatives = []
        
        # Requête simplifiée (mots-clés uniquement)
        keywords = [kw for kw in analysis.key_terms if len(kw) > 2]
        if len(keywords) > 1:
            alternatives.append(" ".join(keywords[:3]))
        
        # Requête par catégorie si détectée
        if analysis.detected_entities["categories"]:
            for category in analysis.detected_entities["categories"]:
                alternatives.append(category)
        
        # Requête avec synonymes
        if analysis.expanded_query != analysis.cleaned_query:
            alternatives.append(analysis.expanded_query)
        
        # Requête sans entités numériques (pour cas où montants/dates perturbent)
        text_only = analysis.cleaned_query
        for amount in analysis.detected_entities["amounts"]:
            text_only = text_only.replace(amount["raw"], "")
        for date in analysis.detected_entities["dates"]:
            text_only = text_only.replace(date["raw"], "")
        text_only = re.sub(r'\s+', ' ', text_only).strip()
        if text_only and text_only != analysis.cleaned_query:
            alternatives.append(text_only)
        
        # Supprimer les doublons et vides
        alternatives = list(set([alt for alt in alternatives if alt.strip()]))
        
        return alternatives[:5]  # Limiter à 5 alternatives
    
    def analyze_search_difficulty(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Analyse la difficulté estimée de la recherche."""
        difficulty = {
            "overall_difficulty": "medium",
            "lexical_difficulty": "medium", 
            "semantic_difficulty": "medium",
            "factors": [],
            "recommendations": []
        }
        
        # Facteurs qui augmentent la difficulté
        if len(analysis.detected_entities["negations"]) > 0:
            difficulty["factors"].append("Contains negations")
            difficulty["overall_difficulty"] = "hard"
        
        if len(analysis.cleaned_query.split()) > 6:
            difficulty["factors"].append("Long query")
            difficulty["semantic_difficulty"] = "hard"
        
        if len(analysis.cleaned_query.split()) == 1:
            difficulty["factors"].append("Single word query")
            difficulty["lexical_difficulty"] = "easy"
        
        if not analysis.key_terms:
            difficulty["factors"].append("No clear keywords")
            difficulty["overall_difficulty"] = "hard"
        
        # Facteurs qui facilitent la recherche
        if analysis.detected_entities["categories"]:
            difficulty["factors"].append("Clear category detected")
            if difficulty["overall_difficulty"] == "medium":
                difficulty["overall_difficulty"] = "easy"
        
        if analysis.detected_entities["amounts"] or analysis.detected_entities["dates"]:
            difficulty["factors"].append("Specific entities detected")
            difficulty["lexical_difficulty"] = "easy"
        
        # Recommandations basées sur la difficulté
        if difficulty["overall_difficulty"] == "hard":
            difficulty["recommendations"].extend([
                "Use expanded query with synonyms",
                "Apply semantic search with lower threshold",
                "Consider query simplification"
            ])
        
        if difficulty["lexical_difficulty"] == "hard":
            difficulty["recommendations"].append("Prioritize semantic search")
        
        if difficulty["semantic_difficulty"] == "hard":
            difficulty["recommendations"].append("Prioritize lexical search")
        
        return difficulty
    
    def clear_cache(self):
        """Vide le cache des analyses."""
        self._analysis_cache.clear()
        logger.info("Query processor cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": len(self._analysis_cache),
            "cache_keys": list(self._analysis_cache.keys())[:10]  # Échantillon
        }


class QueryValidator:
    """Validateur de requêtes pour prévenir les erreurs."""
    
    @staticmethod
    def validate_query(query: str) -> Dict[str, Any]:
        """
        Valide une requête de recherche.
        
        Args:
            query: Requête à valider
            
        Returns:
            Résultat de validation avec erreurs éventuelles
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Vérifications de base
        if not query or not query.strip():
            validation["is_valid"] = False
            validation["errors"].append("Query cannot be empty")
            return validation
        
        if len(query) > 500:
            validation["is_valid"] = False
            validation["errors"].append("Query too long (max 500 characters)")
        
        # Caractères dangereux
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '(', ')', '{', '}']
        if any(char in query for char in dangerous_chars):
            validation["warnings"].append("Query contains special characters that may affect search")
        
        # Trop de mots
        if len(query.split()) > 10:
            validation["warnings"].append("Long queries may return less precise results")
            validation["suggestions"].append("Consider using fewer, more specific terms")
        
        # Mots très courts
        short_words = [w for w in query.split() if len(w) == 1]
        if len(short_words) > 2:
            validation["warnings"].append("Too many single-character terms")
            validation["suggestions"].append("Use more descriptive terms")
        
        # Requête uniquement numérique
        if query.strip().replace(' ', '').replace('.', '').replace(',', '').isdigit():
            validation["warnings"].append("Numeric-only queries may return unexpected results")
            validation["suggestions"].append("Add descriptive terms (e.g., 'montant 100' instead of '100')")
        
        return validation


# Fonctions utilitaires

def normalize_amount(amount_str: str) -> Optional[float]:
    """Normalise une chaîne de montant en float."""
    try:
        # Supprimer les espaces et symboles
        normalized = re.sub(r'[€$\s]', '', amount_str)
        
        # Gérer les séparateurs de milliers et décimales
        if ',' in normalized and '.' in normalized:
            # Format 1,000.50
            normalized = normalized.replace(',', '')
        elif ',' in normalized:
            # Détecter si c'est séparateur de milliers ou décimales
            parts = normalized.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                # Format français: 1000,50
                normalized = normalized.replace(',', '.')
            else:
                # Séparateur de milliers: 1,000
                normalized = normalized.replace(',', '')
        
        return float(normalized)
    except (ValueError, AttributeError):
        return None


def extract_keywords_by_importance(text: str, max_keywords: int = 5) -> List[str]:
    """Extrait les mots-clés les plus importants d'un texte."""
    # Mots vides étendus
    stop_words = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'à', 'au', 'aux',
        'dans', 'sur', 'pour', 'avec', 'par', 'chez', 'vers', 'entre', 'depuis', 'jusqu',
        'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'qui', 'que', 'quoi', 'dont', 'où', 'si', 'comme', 'quand', 'comment', 'pourquoi'
    }
    
    # Nettoyer et diviser
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filtrer et scorer
    keyword_scores = {}
    for word in words:
        if len(word) > 2 and word not in stop_words:
            # Score basé sur la longueur et la fréquence
            keyword_scores[word] = keyword_scores.get(word, 0) + len(word)
    
    # Trier par score et retourner les meilleurs
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:max_keywords]]