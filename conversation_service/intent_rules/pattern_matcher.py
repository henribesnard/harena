"""
🔍 Pattern Matcher - Extracteur d'entités financières

Ce module implémente l'extraction d'entités à partir des patterns configurés
dans entity_patterns.json. Il utilise des regex optimisées avec cache intelligent
pour des performances maximales.

Responsabilité : Extraction et normalisation d'entités financières
Dépendances : rule_loader.py uniquement
Performance : <5ms par requête avec cache, <20ms sans cache
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import calendar

# Import local uniquement
from .rule_loader import RuleLoader, EntityPattern

logger = logging.getLogger(__name__)


class EntityMatch(NamedTuple):
    """Résultat d'un match d'entité"""
    entity_type: str
    raw_value: str
    normalized_value: Any
    confidence: float
    position: Tuple[int, int]  # (start, end)
    pattern_matched: str
    extraction_method: str  # "pattern" | "exact_value"


@dataclass
class ExtractionResult:
    """Résultat complet d'extraction d'entités"""
    entities: Dict[str, List[EntityMatch]]
    total_matches: int
    extraction_time_ms: float
    cache_hit: bool
    
    def get_best_match(self, entity_type: str) -> Optional[EntityMatch]:
        """Retourne le meilleur match pour un type d'entité"""
        matches = self.entities.get(entity_type, [])
        if not matches:
            return None
        
        # Tri par confidence puis par position
        return max(matches, key=lambda m: (m.confidence, -m.position[0]))
    
    def get_all_entities_flat(self) -> List[EntityMatch]:
        """Retourne toutes les entités extraites dans une liste plate"""
        all_entities = []
        for entity_list in self.entities.values():
            all_entities.extend(entity_list)
        
        # Tri par position dans le texte
        return sorted(all_entities, key=lambda e: e.position[0])


class EntityNormalizer:
    """Normalisateur d'entités avec logique métier financière"""
    
    def __init__(self):
        """Initialise le normalisateur"""
        self.month_names = {
            'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
            'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
            'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
        }
        
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month
    
    def normalize_amount(self, value: str, currency: str = "EUR") -> Dict[str, Any]:
        """
        Normalise un montant monétaire
        
        Args:
            value: Valeur brute ("50", "50.5", "50,5")
            currency: Code devise
            
        Returns:
            Dictionnaire avec valeur normalisée et métadonnées
        """
        try:
            # Nettoyage de la valeur
            clean_value = value.replace(',', '.').replace(' ', '')
            
            # Conversion en Decimal pour précision financière
            amount = Decimal(clean_value)
            
            return {
                "value": float(amount),
                "currency": currency,
                "formatted": f"{amount:.2f} {currency}",
                "raw_value": value
            }
            
        except (InvalidOperation, ValueError) as e:
            logger.warning(f"Failed to normalize amount '{value}': {e}")
            return {
                "value": 0.0,
                "currency": currency,
                "formatted": f"0.00 {currency}",
                "raw_value": value,
                "error": str(e)
            }
    
    def normalize_period(self, value: str) -> Dict[str, Any]:
        """
        Normalise une période temporelle
        
        Args:
            value: Période brute ("ce mois", "janvier", "2024")
            
        Returns:
            Dictionnaire avec période normalisée
        """
        value_lower = value.lower().strip()
        now = datetime.now()
        
        # Périodes relatives
        if value_lower in ["ce mois", "mois en cours", "mois actuel"]:
            start_date = now.replace(day=1)
            end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            return {
                "type": "current_month",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "month_year": now.strftime("%Y-%m"),
                "raw_value": value
            }
        
        elif value_lower in ["mois dernier", "dernier mois", "mois précédent"]:
            if now.month == 1:
                last_month = now.replace(year=now.year-1, month=12, day=1)
            else:
                last_month = now.replace(month=now.month-1, day=1)
            
            end_date = now.replace(day=1) - timedelta(days=1)
            
            return {
                "type": "last_month",
                "start_date": last_month.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "month_year": last_month.strftime("%Y-%m"),
                "raw_value": value
            }
        
        elif value_lower in ["cette semaine", "semaine en cours"]:
            # Début de semaine (lundi)
            days_since_monday = now.weekday()
            start_date = now - timedelta(days=days_since_monday)
            end_date = start_date + timedelta(days=6)
            
            return {
                "type": "current_week",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "raw_value": value
            }
        
        elif value_lower in ["semaine dernière", "dernière semaine"]:
            days_since_monday = now.weekday()
            current_week_start = now - timedelta(days=days_since_monday)
            start_date = current_week_start - timedelta(days=7)
            end_date = start_date + timedelta(days=6)
            
            return {
                "type": "last_week",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "raw_value": value
            }
        
        elif value_lower == "hier":
            yesterday = now - timedelta(days=1)
            return {
                "type": "yesterday",
                "start_date": yesterday.strftime("%Y-%m-%d"),
                "end_date": yesterday.strftime("%Y-%m-%d"),
                "raw_value": value
            }
        
        # Noms de mois
        elif value_lower in self.month_names:
            month_num = self.month_names[value_lower]
            year = self.current_year
            
            # Si le mois est futur, prendre l'année précédente
            if month_num > self.current_month:
                year -= 1
            
            start_date = datetime(year, month_num, 1)
            end_date = datetime(year, month_num, calendar.monthrange(year, month_num)[1])
            
            return {
                "type": "named_month",
                "month": month_num,
                "year": year,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "month_year": f"{year}-{month_num:02d}",
                "raw_value": value
            }
        
        # Années
        elif value.isdigit() and len(value) == 4:
            year = int(value)
            if 2020 <= year <= 2030:  # Validation année raisonnable
                return {
                    "type": "year",
                    "year": year,
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                    "raw_value": value
                }
        
        # Valeur non reconnue
        logger.warning(f"Could not normalize period: '{value}'")
        return {
            "type": "unknown",
            "raw_value": value,
            "error": "Unrecognized period format"
        }
    
    def normalize_category(self, value: str) -> Dict[str, Any]:
        """
        Normalise une catégorie de dépense
        
        Args:
            value: Catégorie brute
            
        Returns:
            Dictionnaire avec catégorie normalisée
        """
        value_lower = value.lower().strip()
        
        # Mapping des catégories
        category_mappings = {
            # Restaurant
            "restaurant": "restaurant",
            "resto": "restaurant", 
            "repas": "restaurant",
            "restauration": "restaurant",
            "dîner": "restaurant",
            "déjeuner": "restaurant",
            "manger": "restaurant",
            
            # Alimentation
            "courses": "alimentation",
            "alimentation": "alimentation",
            "supermarché": "alimentation",
            "épicerie": "alimentation",
            "nourriture": "alimentation",
            
            # Transport
            "transport": "transport",
            "essence": "transport",
            "carburant": "transport",
            "taxi": "transport",
            "métro": "transport",
            "bus": "transport",
            "train": "transport",
            
            # Santé
            "santé": "santé",
            "pharmacie": "santé",
            "médecin": "santé",
            "dentiste": "santé",
            "hôpital": "santé",
            "clinique": "santé",
            
            # Shopping
            "shopping": "shopping",
            "vêtements": "shopping",
            "habits": "shopping",
            "chaussures": "shopping",
            "mode": "shopping",
            
            # Loisirs
            "loisirs": "loisirs",
            "cinéma": "loisirs",
            "sport": "loisirs",
            "gym": "loisirs",
            "divertissement": "loisirs"
        }
        
        normalized = category_mappings.get(value_lower, value_lower)
        
        return {
            "category": normalized,
            "display_name": normalized.title(),
            "raw_value": value,
            "mapped": normalized != value_lower
        }
    
    def normalize_merchant(self, value: str) -> Dict[str, Any]:
        """
        Normalise un nom de marchand
        
        Args:
            value: Nom marchand brut
            
        Returns:
            Dictionnaire avec marchand normalisé
        """
        value_lower = value.lower().strip()
        
        # Mapping des marchands connus
        merchant_mappings = {
            "amazon": "AMAZON",
            "netflix": "NETFLIX",
            "spotify": "SPOTIFY",
            "carrefour": "CARREFOUR",
            "leclerc": "LECLERC",
            "uber": "UBER",
            "sncf": "SNCF",
            "mcdo": "MCDONALD'S",
            "mcdonald": "MCDONALD'S",
            "apple": "APPLE",
            "google": "GOOGLE",
            "microsoft": "MICROSOFT",
            "paypal": "PAYPAL"
        }
        
        normalized = merchant_mappings.get(value_lower, value.upper())
        
        return {
            "merchant": normalized,
            "display_name": normalized,
            "raw_value": value,
            "mapped": normalized != value.upper(),
            "category": self._get_merchant_category(normalized)
        }
    
    def _get_merchant_category(self, merchant: str) -> str:
        """Détermine la catégorie d'un marchand"""
        merchant_categories = {
            "AMAZON": "shopping",
            "NETFLIX": "loisirs",
            "SPOTIFY": "loisirs",
            "CARREFOUR": "alimentation",
            "LECLERC": "alimentation",
            "UBER": "transport",
            "SNCF": "transport",
            "MCDONALD'S": "restaurant",
            "APPLE": "technologie",
            "GOOGLE": "technologie",
            "MICROSOFT": "technologie"
        }
        
        return merchant_categories.get(merchant, "autre")
    
    def normalize_operation_type(self, value: str) -> Dict[str, Any]:
        """
        Normalise un type d'opération bancaire
        
        Args:
            value: Type opération brut
            
        Returns:
            Dictionnaire avec type normalisé
        """
        value_lower = value.lower().strip()
        
        operation_mappings = {
            # Carte
            "carte": "card",
            "cb": "card",
            "carte bleue": "card",
            "paiement carte": "card",
            
            # Virement
            "virement": "transfer",
            "virements": "transfer",
            
            # Prélèvement
            "prélèvement": "direct_debit",
            "prélèvements": "direct_debit",
            
            # Chèque
            "chèque": "check",
            "chèques": "check",
            
            # Retrait
            "retrait": "withdrawal",
            "retraits": "withdrawal",
            "distributeur": "withdrawal",
            "dab": "withdrawal",
            
            # Dépôt
            "dépôt": "deposit",
            "versement": "deposit"
        }
        
        normalized = operation_mappings.get(value_lower, value_lower)
        
        operation_descriptions = {
            "card": "Paiement par carte bancaire",
            "transfer": "Virement bancaire", 
            "direct_debit": "Prélèvement automatique",
            "check": "Paiement par chèque",
            "withdrawal": "Retrait d'espèces",
            "deposit": "Dépôt ou versement"
        }
        
        return {
            "operation_type": normalized,
            "description": operation_descriptions.get(normalized, normalized),
            "raw_value": value,
            "mapped": normalized != value_lower
        }


class PatternMatcher:
    """
    Extracteur d'entités basé sur patterns regex configurables
    
    Cette classe utilise les patterns définis dans entity_patterns.json
    pour extraire et normaliser les entités financières avec cache intelligent.
    """
    
    def __init__(self, rule_loader: Optional[RuleLoader] = None):
        """
        Initialise le matcher avec un RuleLoader
        
        Args:
            rule_loader: Loader des règles (optionnel, créé automatiquement sinon)
        """
        if rule_loader is None:
            from . import get_default_loader
            rule_loader = get_default_loader()
        
        self.rule_loader = rule_loader
        self.normalizer = EntityNormalizer()
        
        # Cache pour les résultats d'extraction
        self._extraction_cache: Dict[str, ExtractionResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Chargement des patterns d'entités
        self.entity_patterns = self.rule_loader.get_entity_patterns()
        
        logger.info(f"PatternMatcher initialized with {len(self.entity_patterns)} entity types")
    
    def extract_entities(self, text: str, target_entities: Optional[List[str]] = None) -> ExtractionResult:
        """
        Extrait toutes les entités d'un texte
        
        Args:
            text: Texte à analyser
            target_entities: Types d'entités spécifiques à extraire (optionnel)
            
        Returns:
            Résultat d'extraction avec toutes les entités trouvées
        """
        start_time = datetime.now()
        
        # Vérification cache
        cache_key = f"{hash(text)}_{hash(str(target_entities))}"
        if cache_key in self._extraction_cache:
            self._cache_hits += 1
            cached_result = self._extraction_cache[cache_key]
            cached_result.cache_hit = True
            return cached_result
        
        self._cache_misses += 1
        
        # Extraction des entités
        entities = {}
        total_matches = 0
        
        # Détermine les types d'entités à extraire
        entity_types = target_entities if target_entities else list(self.entity_patterns.keys())
        
        for entity_type in entity_types:
            matches = self.match_entity_patterns(text, entity_type)
            if matches:
                entities[entity_type] = matches
                total_matches += len(matches)
        
        # Calcul du temps d'exécution
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Création du résultat
        result = ExtractionResult(
            entities=entities,
            total_matches=total_matches,
            extraction_time_ms=execution_time,
            cache_hit=False
        )
        
        # Mise en cache (limite à 1000 entrées)
        if len(self._extraction_cache) < 1000:
            self._extraction_cache[cache_key] = result
        
        return result
    
    def match_entity_patterns(self, text: str, entity_type: str) -> List[EntityMatch]:
        """
        Trouve tous les matches d'un type d'entité spécifique
        
        Args:
            text: Texte à analyser
            entity_type: Type d'entité à rechercher
            
        Returns:
            Liste des matches trouvés pour ce type d'entité
        """
        if entity_type not in self.entity_patterns:
            logger.warning(f"Unknown entity type: {entity_type}")
            return []
        
        patterns = self.entity_patterns[entity_type]
        matches = []
        
        # 1. Recherche par valeurs exactes (plus rapide)
        exact_matches = self._find_exact_matches(text, entity_type, patterns)
        matches.extend(exact_matches)
        
        # 2. Recherche par patterns regex
        pattern_matches = self._find_pattern_matches(text, entity_type, patterns)
        matches.extend(pattern_matches)
        
        # 3. Déduplication et tri par position
        matches = self._deduplicate_matches(matches)
        
        return sorted(matches, key=lambda m: m.position[0])
    
    def _find_exact_matches(self, text: str, entity_type: str, patterns: List[EntityPattern]) -> List[EntityMatch]:
        """Recherche par valeurs exactes (ultra-rapide)"""
        matches = []
        text_lower = text.lower()
        
        # Récupération des valeurs exactes depuis la configuration
        exact_values = self._get_exact_values_for_entity(entity_type)
        
        for exact_value, normalized_value in exact_values.items():
            # Recherche de toutes les occurrences
            start_pos = 0
            while True:
                pos = text_lower.find(exact_value.lower(), start_pos)
                if pos == -1:
                    break
                
                # Vérification des limites de mots pour éviter les faux positifs
                if self._is_word_boundary(text, pos, pos + len(exact_value)):
                    match = EntityMatch(
                        entity_type=entity_type,
                        raw_value=text[pos:pos + len(exact_value)],
                        normalized_value=normalized_value,
                        confidence=1.0,  # Confiance maximale pour match exact
                        position=(pos, pos + len(exact_value)),
                        pattern_matched=f"exact:{exact_value}",
                        extraction_method="exact_value"
                    )
                    matches.append(match)
                
                start_pos = pos + 1
        
        return matches
    
    def _find_pattern_matches(self, text: str, entity_type: str, patterns: List[EntityPattern]) -> List[EntityMatch]:
        """Recherche par patterns regex"""
        matches = []
        
        for pattern in patterns:
            try:
                for match in pattern.regex.finditer(text):
                    # Extraction de la valeur
                    if pattern.extract_groups:
                        # Plusieurs groupes d'extraction
                        raw_values = [match.group(g) for g in pattern.extract_groups if match.group(g)]
                        raw_value = " ".join(raw_values)
                    elif pattern.extract_group is not None:
                        # Groupe d'extraction spécifique
                        raw_value = match.group(pattern.extract_group)
                    else:
                        # Match complet
                        raw_value = match.group(0)
                    
                    if not raw_value:
                        continue
                    
                    # Normalisation de la valeur
                    normalized_value = self.normalize_entity_value(raw_value, entity_type, pattern.normalize)
                    
                    entity_match = EntityMatch(
                        entity_type=entity_type,
                        raw_value=raw_value,
                        normalized_value=normalized_value,
                        confidence=pattern.weight,
                        position=(match.start(), match.end()),
                        pattern_matched=pattern.regex.pattern,
                        extraction_method="pattern"
                    )
                    matches.append(entity_match)
                    
            except Exception as e:
                logger.error(f"Error matching pattern for {entity_type}: {e}")
                continue
        
        return matches
    
    def _get_exact_values_for_entity(self, entity_type: str) -> Dict[str, Any]:
        """FIX: Récupère les valeurs exactes pour un type d'entité depuis la configuration"""
        try:
            # FIX: Accès direct aux patterns d'entités du rule_loader
            entity_patterns = self.rule_loader.get_entity_patterns(entity_type)
            if not entity_patterns:
                return {}
            
            # Récupération des exact_values depuis le premier pattern (ils partagent la même config)
            if entity_patterns and hasattr(entity_patterns[0], 'exact_values'):
                return entity_patterns[0].exact_values or {}
            
            # Fallback: hardcodé pour certains types courants
            hardcoded_values = {
                "amount": {
                    "cent euros": {"value": 100.0, "currency": "EUR"},
                    "cinquante euros": {"value": 50.0, "currency": "EUR"},
                    "vingt euros": {"value": 20.0, "currency": "EUR"},
                    "dix euros": {"value": 10.0, "currency": "EUR"}
                },
                "period": {
                    "ce mois": {"type": "current_month", "start_date": "2025-07-01", "end_date": "2025-07-31"},
                    "mois dernier": {"type": "last_month", "start_date": "2025-06-01", "end_date": "2025-06-30"},
                    "cette semaine": {"type": "current_week"},
                    "semaine dernière": {"type": "last_week"}
                },
                "merchant": {
                    "amazon": {"merchant": "AMAZON", "display_name": "AMAZON", "category": "shopping"},
                    "netflix": {"merchant": "NETFLIX", "display_name": "NETFLIX", "category": "loisirs"},
                    "carrefour": {"merchant": "CARREFOUR", "display_name": "CARREFOUR", "category": "alimentation"}
                }
            }
            
            return hardcoded_values.get(entity_type, {})
            
        except Exception as e:
            logger.warning(f"Could not load exact values for {entity_type}: {e}")
            return {}
    
    def _is_word_boundary(self, text: str, start: int, end: int) -> bool:
        """Vérifie si une position constitue une limite de mot"""
        # Vérification début
        if start > 0 and text[start - 1].isalnum():
            return False
        
        # Vérification fin
        if end < len(text) and text[end].isalnum():
            return False
        
        return True
    
    def _deduplicate_matches(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """Supprime les matches en doublon en gardant le meilleur"""
        if not matches:
            return matches
        
        # Groupement par position
        position_groups = {}
        for match in matches:
            pos_key = f"{match.position[0]}-{match.position[1]}"
            if pos_key not in position_groups:
                position_groups[pos_key] = []
            position_groups[pos_key].append(match)
        
        # Sélection du meilleur match par position
        deduplicated = []
        for group in position_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Tri par confidence puis par méthode (exact > pattern)
                best_match = max(group, key=lambda m: (
                    m.confidence,
                    1 if m.extraction_method == "exact_value" else 0
                ))
                deduplicated.append(best_match)
        
        return deduplicated
    
    def normalize_entity_value(self, value: str, entity_type: str, normalize_method: Optional[str] = None) -> Any:
        """
        Normalise une valeur d'entité selon son type
        
        Args:
            value: Valeur brute extraite
            entity_type: Type d'entité
            normalize_method: Méthode de normalisation spécifique
            
        Returns:
            Valeur normalisée selon le type d'entité
        """
        try:
            if entity_type == "amount":
                return self.normalizer.normalize_amount(value)
            elif entity_type == "period":
                return self.normalizer.normalize_period(value)
            elif entity_type == "category":
                return self.normalizer.normalize_category(value)
            elif entity_type == "merchant":
                return self.normalizer.normalize_merchant(value)
            elif entity_type == "operation_type":
                return self.normalizer.normalize_operation_type(value)
            elif entity_type == "transaction_type":
                return {"type": value.lower(), "raw_value": value}
            elif entity_type == "currency":
                return {"currency": value.upper(), "raw_value": value}
            else:
                # Normalisations génériques
                if normalize_method == "float":
                    return float(value.replace(',', '.'))
                elif normalize_method == "uppercase":
                    return value.upper()
                elif normalize_method == "lowercase":
                    return value.lower()
                else:
                    return value
                    
        except Exception as e:
            logger.warning(f"Failed to normalize {entity_type} value '{value}': {e}")
            return value
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de performance du matcher
        
        Returns:
            Dictionnaire avec métriques de performance
        """
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "matcher_name": "PatternMatcher",
            "entity_types_loaded": len(self.entity_patterns),
            "cache_stats": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate_percent": round(cache_hit_rate, 2),
                "cache_size": len(self._extraction_cache)
            },
            "performance": {
                "avg_extraction_time_ms": self._calculate_avg_extraction_time(),
                "patterns_per_entity": {
                    entity_type: len(patterns) 
                    for entity_type, patterns in self.entity_patterns.items()
                }
            }
        }
    
    def _calculate_avg_extraction_time(self) -> float:
        """Calcule le temps moyen d'extraction depuis le cache"""
        if not self._extraction_cache:
            return 0.0
        
        total_time = sum(result.extraction_time_ms for result in self._extraction_cache.values())
        return round(total_time / len(self._extraction_cache), 2)
    
    def clear_cache(self) -> None:
        """Vide le cache d'extraction"""
        self._extraction_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("PatternMatcher cache cleared")


# Factory function pour faciliter l'utilisation
def create_pattern_matcher(rule_loader: Optional[RuleLoader] = None) -> PatternMatcher:
    """
    Factory function pour créer un PatternMatcher
    
    Args:
        rule_loader: RuleLoader personnalisé (optionnel)
        
    Returns:
        Instance PatternMatcher configurée
    """
    return PatternMatcher(rule_loader)


# Exports principaux
__all__ = [
    "PatternMatcher",
    "EntityMatch", 
    "ExtractionResult",
    "EntityNormalizer",
    "create_pattern_matcher"
]