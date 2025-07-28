"""
🔍 Pattern Matcher - Extraction d'entités par regex

Service d'extraction d'entités financières via patterns regex optimisés.
Réutilise la logique éprouvée du fichier original avec améliations modulaires.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Pattern
from dataclasses import dataclass
from conversation_service.models.enums import EntityType
from conversation_service.models.exceptions import EntityExtractionError, PatternCompilationError
from conversation_service.utils.helpers.text_helpers import get_compiled_pattern

logger = logging.getLogger(__name__)


@dataclass
class EntityPattern:
    """Configuration d'un pattern d'extraction d'entité"""
    entity_type: EntityType
    pattern: str
    confidence_base: float
    default_value: Optional[str] = None
    normalizer: Optional[callable] = None
    validator: Optional[callable] = None


class FinancialPatternMatcher:
    """Matcher de patterns spécialisé entités financières"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration patterns par entité (repris du fichier original)
        self._entity_patterns = {
            EntityType.AMOUNT: [
                EntityPattern(
                    entity_type=EntityType.AMOUNT,
                    pattern=r'\b(\d+(?:[,\.]\d+)?)\s*euros?\b',
                    confidence_base=0.9,
                    normalizer=self._normalize_amount
                ),
                EntityPattern(
                    entity_type=EntityType.AMOUNT,
                    pattern=r'\b(\d+(?:[,\.]\d+)?)\s*€\b',
                    confidence_base=0.95,
                    normalizer=self._normalize_amount
                ),
                EntityPattern(
                    entity_type=EntityType.AMOUNT,
                    pattern=r'\b(\d+(?:[,\.]\d+)?)\s*eur\b',
                    confidence_base=0.85,
                    normalizer=self._normalize_amount
                )
            ],
            
            EntityType.ACCOUNT_TYPE: [
                EntityPattern(
                    entity_type=EntityType.ACCOUNT_TYPE,
                    pattern=r'\b(compte\s+courant|livret\s+a|épargne|livret)\b',
                    confidence_base=0.9,
                    normalizer=self._normalize_account_type
                )
            ],
            
            EntityType.CATEGORY: [
                EntityPattern(
                    entity_type=EntityType.CATEGORY,
                    pattern=r'\b(restaurant|resto|repas|dîner|déjeuner)\b',
                    confidence_base=0.9,
                    default_value="restaurant",
                    normalizer=self._normalize_category
                ),
                EntityPattern(
                    entity_type=EntityType.CATEGORY,
                    pattern=r'\b(courses|alimentation|supermarché|carrefour|leclerc)\b',
                    confidence_base=0.85,
                    default_value="alimentation",
                    normalizer=self._normalize_category
                ),
                EntityPattern(
                    entity_type=EntityType.CATEGORY,
                    pattern=r'\b(transport|essence|carburant|taxi|uber)\b',
                    confidence_base=0.9,
                    default_value="transport",
                    normalizer=self._normalize_category
                ),
                EntityPattern(
                    entity_type=EntityType.CATEGORY,
                    pattern=r'\b(shopping|vêtements|achats|boutique)\b',
                    confidence_base=0.85,
                    default_value="shopping",
                    normalizer=self._normalize_category
                ),
                EntityPattern(
                    entity_type=EntityType.CATEGORY,
                    pattern=r'\b(loisirs|cinéma|sport|vacances)\b',
                    confidence_base=0.85,
                    default_value="loisirs",
                    normalizer=self._normalize_category
                ),
                EntityPattern(
                    entity_type=EntityType.CATEGORY,
                    pattern=r'\b(santé|pharmacie|médecin|dentiste)\b',
                    confidence_base=0.9,
                    default_value="santé",
                    normalizer=self._normalize_category
                )
            ],
            
            EntityType.MONTH: [
                EntityPattern(
                    entity_type=EntityType.MONTH,
                    pattern=r'\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b',
                    confidence_base=0.95,
                    normalizer=self._normalize_month
                )
            ],
            
            EntityType.PERIOD: [
                EntityPattern(
                    entity_type=EntityType.PERIOD,
                    pattern=r'\b(ce\s+mois|mois\s+dernier|cette\s+semaine|semaine\s+dernière)\b',
                    confidence_base=0.9,
                    normalizer=self._normalize_period
                ),
                EntityPattern(
                    entity_type=EntityType.PERIOD,
                    pattern=r'\b(aujourd\'hui|hier|avant-hier)\b',
                    confidence_base=0.95,
                    normalizer=self._normalize_period
                )
            ],
            
            EntityType.RECIPIENT: [
                EntityPattern(
                    entity_type=EntityType.RECIPIENT,
                    pattern=r'\b(à|vers|pour)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                    confidence_base=0.8,
                    normalizer=self._normalize_recipient
                )
            ],
            
            EntityType.CARD_TYPE: [
                EntityPattern(
                    entity_type=EntityType.CARD_TYPE,
                    pattern=r'\b(visa|mastercard|cb|carte\s+bleue)\b',
                    confidence_base=0.95,
                    normalizer=self._normalize_card_type
                )
            ],
            
            EntityType.DATE: [
                EntityPattern(
                    entity_type=EntityType.DATE,
                    pattern=r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
                    confidence_base=0.9,
                    normalizer=self._normalize_date,
                    validator=self._validate_date
                )
            ]
        }
        
        # Compilation des patterns pour performance
        self._compiled_patterns: Dict[EntityType, List[Tuple[Pattern, EntityPattern]]] = {}
        try:
            self._compile_all_patterns()
        except PatternCompilationError as e:
            self.logger.error(f"Erreur compilation patterns: {e}")
            raise EntityExtractionError(
                f"Impossible d'initialiser le matcher: {str(e)}",
                extraction_method="pattern_compilation"
            )
    
    def _compile_all_patterns(self):
        """
        Compile tous les patterns regex pour optimiser performance
        
        Raises:
            PatternCompilationError: Si compilation d'un pattern échoue
        """
        for entity_type, patterns in self._entity_patterns.items():
            compiled_patterns = []
            
            for pattern_config in patterns:
                try:
                    compiled_pattern = get_compiled_pattern(
                        pattern_config.pattern, 
                        re.IGNORECASE
                    )
                    compiled_patterns.append((compiled_pattern, pattern_config))
                except re.error as e:
                    raise PatternCompilationError(
                        pattern=pattern_config.pattern,
                        regex_error=str(e)
                    )
            
            self._compiled_patterns[entity_type] = compiled_patterns
    
    def extract_entities(
        self, 
        text: str, 
        target_entities: Optional[List[EntityType]] = None
    ) -> Dict[str, Any]:
        """
        Extrait toutes les entités d'un texte
        
        Args:
            text: Texte source
            target_entities: Types d'entités à extraire (toutes si None)
            
        Returns:
            Dict avec entités extraites et métadonnées
            
        Raises:
            EntityExtractionError: Si l'extraction échoue complètement
        """
        if not text:
            return {"entities": {}, "metadata": {}}
        
        if not isinstance(text, str):
            raise EntityExtractionError(
                f"Le texte doit être une chaîne, reçu: {type(text)}",
                extraction_method="input_validation"
            )
        
        if target_entities is None:
            target_entities = list(EntityType)
        
        # Validation des types d'entités demandés
        invalid_types = [et for et in target_entities if et not in EntityType]
        if invalid_types:
            raise EntityExtractionError(
                f"Types d'entités invalides: {invalid_types}",
                extraction_method="input_validation"
            )
        
        extracted_entities = {}
        extraction_metadata = {
            "confidence_scores": {},
            "extraction_methods": {},
            "multiple_matches": {},
            "extraction_errors": {}
        }
        
        successful_extractions = 0
        
        for entity_type in target_entities:
            if entity_type not in self._compiled_patterns:
                self.logger.warning(f"Type d'entité non supporté: {entity_type}")
                continue
            
            try:
                entity_results = self._extract_single_entity_type(text, entity_type)
                if entity_results:
                    successful_extractions += 1
                    for entity_key, entity_data in entity_results.items():
                        extracted_entities[entity_key] = entity_data["value"]
                        extraction_metadata["confidence_scores"][entity_key] = entity_data["confidence"]
                        extraction_metadata["extraction_methods"][entity_key] = "pattern_matching"
                        
                        if entity_data.get("multiple_matches", False):
                            extraction_metadata["multiple_matches"][entity_key] = True
                            
            except Exception as e:
                error_msg = f"Erreur extraction {entity_type}: {e}"
                self.logger.warning(error_msg)
                extraction_metadata["extraction_errors"][entity_type.value] = str(e)
                continue
        
        # Si aucune extraction n'a réussi et qu'on a des erreurs, lever une exception
        if successful_extractions == 0 and extraction_metadata["extraction_errors"]:
            raise EntityExtractionError(
                f"Échec complet extraction d'entités: {extraction_metadata['extraction_errors']}",
                extraction_method="pattern_matching",
                context={"text_length": len(text), "target_entities": [et.value for et in target_entities]}
            )
        
        result = {
            "entities": extracted_entities,
            "metadata": extraction_metadata
        }
        
        self.logger.debug(f"Extracted entities from '{text}': {extracted_entities}")
        return result
    
    def _extract_single_entity_type(
        self, 
        text: str, 
        entity_type: EntityType
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extrait toutes les entités d'un type donné
        
        Raises:
            EntityExtractionError: Si l'extraction de ce type spécifique échoue
        """
        results = {}
        patterns = self._compiled_patterns.get(entity_type, [])
        
        if not patterns:
            raise EntityExtractionError(
                f"Aucun pattern configuré pour le type: {entity_type}",
                extraction_method="pattern_matching",
                entity_type=entity_type
            )
        
        extraction_errors = []
        
        for compiled_pattern, pattern_config in patterns:
            try:
                matches = list(compiled_pattern.finditer(text))
                
                for match in matches:
                    try:
                        # Extraction valeur selon type pattern
                        if pattern_config.default_value:
                            raw_value = pattern_config.default_value
                        else:
                            # Prendre premier groupe capturé ou match complet
                            raw_value = match.group(1) if match.groups() else match.group()
                        
                        # Normalisation si spécifiée
                        if pattern_config.normalizer:
                            try:
                                normalized_value = pattern_config.normalizer(raw_value, match)
                            except Exception as e:
                                self.logger.warning(f"Erreur normalisation {entity_type}: {e}")
                                normalized_value = raw_value.strip()
                        else:
                            normalized_value = raw_value.strip()
                        
                        # Validation si spécifiée
                        if pattern_config.validator:
                            try:
                                if not pattern_config.validator(normalized_value):
                                    continue
                            except Exception as e:
                                self.logger.warning(f"Erreur validation {entity_type}: {e}")
                                continue
                        
                        # Calcul confiance basé sur qualité match
                        confidence = self._calculate_match_confidence(
                            match, pattern_config, text
                        )
                        
                        # Clé unique pour cette entité
                        entity_key = entity_type.value
                        
                        # Gestion entités multiples du même type
                        if entity_key in results:
                            # Garder meilleure confiance ou créer liste
                            if confidence > results[entity_key]["confidence"]:
                                results[entity_key] = {
                                    "value": normalized_value,
                                    "confidence": confidence,
                                    "multiple_matches": True
                                }
                        else:
                            results[entity_key] = {
                                "value": normalized_value,
                                "confidence": confidence,
                                "multiple_matches": False
                            }
                            
                    except Exception as e:
                        error_msg = f"Erreur traitement match {entity_type}: {e}"
                        self.logger.warning(error_msg)
                        extraction_errors.append(error_msg)
                        continue
                        
            except Exception as e:
                error_msg = f"Erreur pattern {pattern_config.pattern}: {e}"
                self.logger.warning(error_msg)
                extraction_errors.append(error_msg)
                continue
        
        # Si tous les patterns ont échoué, lever une exception
        if not results and extraction_errors:
            raise EntityExtractionError(
                f"Tous les patterns ont échoué pour {entity_type}: {extraction_errors}",
                extraction_method="pattern_matching",
                entity_type=entity_type,
                context={"text": text[:100], "errors": extraction_errors}
            )
        
        return results
    
    def _calculate_match_confidence(
        self, 
        match: re.Match, 
        pattern_config: EntityPattern, 
        full_text: str
    ) -> float:
        """Calcule confiance d'un match basé sur contexte"""
        base_confidence = pattern_config.confidence_base
        
        # Bonus pour match au début/fin (plus saillant)
        text_length = len(full_text)
        match_position = match.start() / text_length if text_length > 0 else 0.5
        
        if match_position < 0.2 or match_position > 0.8:
            base_confidence *= 1.1
        
        # Bonus pour match complet de mot (pas substring)
        match_text = match.group()
        start_pos = match.start()
        end_pos = match.end()
        
        # Vérifier frontières de mots
        word_boundary_start = (start_pos == 0 or not full_text[start_pos - 1].isalnum())
        word_boundary_end = (end_pos == len(full_text) or not full_text[end_pos].isalnum())
        
        if word_boundary_start and word_boundary_end:
            base_confidence *= 1.05
        
        # Limitation à 1.0
        return min(base_confidence, 1.0)
    
    # Normaliseurs spécialisés par type d'entité
    def _normalize_amount(self, value: str, match: re.Match) -> str:
        """
        Normalise les montants détectés
        
        Raises:
            EntityExtractionError: Si la normalisation échoue
        """
        try:
            # Conversion format français -> format standard
            normalized = value.replace(',', '.')
            float_value = float(normalized)
            
            # Validation montant raisonnable
            if float_value < 0:
                raise EntityExtractionError(
                    f"Montant négatif détecté: {float_value}",
                    extraction_method="amount_normalization"
                )
            if float_value > 1000000:
                self.logger.warning(f"Montant très élevé détecté: {float_value}")
            
            return str(float_value)
        except ValueError as e:
            raise EntityExtractionError(
                f"Impossible de normaliser le montant '{value}': {e}",
                extraction_method="amount_normalization"
            )
    
    def _normalize_account_type(self, value: str, match: re.Match) -> str:
        """Normalise les types de comptes"""
        normalization_map = {
            'compte courant': 'compte_courant',
            'livret a': 'livret_a',
            'épargne': 'epargne',
            'livret': 'livret'
        }
        return normalization_map.get(value.lower(), value)
    
    def _normalize_category(self, value: str, match: re.Match) -> str:
        """Normalise les catégories de dépenses"""
        # Mapping vers catégories standards
        category_map = {
            'resto': 'restaurant',
            'repas': 'restaurant',
            'dîner': 'restaurant',
            'déjeuner': 'restaurant',
            'courses': 'alimentation',
            'supermarché': 'alimentation',
            'carrefour': 'alimentation',
            'leclerc': 'alimentation',
            'essence': 'transport',
            'carburant': 'transport',
            'taxi': 'transport',
            'uber': 'transport',
            'vêtements': 'shopping',
            'achats': 'shopping',
            'boutique': 'shopping',
            'cinéma': 'loisirs',
            'sport': 'loisirs',
            'vacances': 'loisirs',
            'pharmacie': 'santé',
            'médecin': 'santé',
            'dentiste': 'santé'
        }
        return category_map.get(value.lower(), value.lower())
    
    def _normalize_month(self, value: str, match: re.Match) -> str:
        """Normalise les mois français"""
        return value.lower()
    
    def _normalize_period(self, value: str, match: re.Match) -> str:
        """Normalise les périodes temporelles"""
        return value.lower().replace('  ', ' ')
    
    def _normalize_recipient(self, value: str, match: re.Match) -> str:
        """Normalise les destinataires de virement"""
        # Extraire nom propre (groupe 2 du pattern)
        if match.groups() and len(match.groups()) >= 2:
            return match.group(2).strip()
        return value.strip()
    
    def _normalize_card_type(self, value: str, match: re.Match) -> str:
        """Normalise les types de cartes"""
        card_map = {
            'cb': 'carte_bleue',
            'carte bleue': 'carte_bleue',
            'visa': 'visa',
            'mastercard': 'mastercard'
        }
        return card_map.get(value.lower().replace(' ', ' '), value.lower())
    
    def _normalize_date(self, value: str, match: re.Match) -> str:
        """
        Normalise les dates au format standard
        
        Raises:
            EntityExtractionError: Si le format de date est invalide
        """
        try:
            if match.groups() and len(match.groups()) == 3:
                day, month, year = match.groups()
                # Format ISO: YYYY-MM-DD
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                raise EntityExtractionError(
                    f"Format de date invalide: {value}",
                    extraction_method="date_normalization"
                )
        except Exception as e:
            raise EntityExtractionError(
                f"Erreur normalisation date '{value}': {e}",
                extraction_method="date_normalization"
            )
    
    # Validateurs spécialisés
    def _validate_date(self, date_str: str) -> bool:
        """Valide une date extraite"""
        try:
            if '-' in date_str:  # Format ISO normalisé
                year, month, day = date_str.split('-')
            else:  # Format original
                day, month, year = date_str.split('/')
            
            year, month, day = int(year), int(month), int(day)
            
            # Validations basiques
            if not (1 <= month <= 12):
                return False
            if not (1 <= day <= 31):
                return False
            if not (1900 <= year <= 2100):
                return False
            
            # Validation jours par mois simplifiée
            days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if day > days_in_month[month - 1]:
                return False
            
            return True
            
        except (ValueError, IndexError):
            return False
    
    def extract_entity_by_type(
        self, 
        text: str, 
        entity_type: EntityType
    ) -> Optional[Dict[str, Any]]:
        """
        Extrait une entité spécifique d'un type donné
        
        Args:
            text: Texte source
            entity_type: Type d'entité à extraire
            
        Returns:
            Première entité trouvée ou None
            
        Raises:
            EntityExtractionError: Si l'extraction échoue
        """
        try:
            results = self.extract_entities(text, [entity_type])
            entities = results.get("entities", {})
            
            if entity_type.value in entities:
                return {
                    "value": entities[entity_type.value],
                    "confidence": results["metadata"]["confidence_scores"].get(entity_type.value, 0.0)
                }
            
            return None
            
        except EntityExtractionError:
            # Re-raise les erreurs spécifiques
            raise
        except Exception as e:
            raise EntityExtractionError(
                f"Erreur extraction entité {entity_type}: {e}",
                extraction_method="single_entity_extraction",
                entity_type=entity_type
            )
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Retourne la liste des types d'entités supportés"""
        return list(self._entity_patterns.keys())
    
    def validate_entity_value(
        self, 
        entity_type: EntityType, 
        value: str
    ) -> Tuple[bool, str]:
        """
        Valide une valeur d'entité selon son type
        
        Args:
            entity_type: Type d'entité
            value: Valeur à valider
            
        Returns:
            (is_valid, error_message)
        """
        try:
            if entity_type == EntityType.AMOUNT:
                float_val = float(value.replace(',', '.'))
                if float_val < 0:
                    return False, "Montant ne peut pas être négatif"
                if float_val > 1000000:
                    return False, "Montant trop élevé"
                return True, ""
            
            elif entity_type == EntityType.DATE:
                return self._validate_date(value), "Format de date invalide"
            
            elif entity_type == EntityType.MONTH:
                months = [
                    'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                    'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'
                ]
                if value.lower() not in months:
                    return False, "Mois invalide"
                return True, ""
            
            elif entity_type == EntityType.CARD_TYPE:
                valid_cards = ['visa', 'mastercard', 'carte_bleue', 'cb']
                if value.lower() not in valid_cards:
                    return False, "Type de carte invalide"
                return True, ""
            
            else:
                # Validation générique : non vide
                if not value or not value.strip():
                    return False, "Valeur vide"
                return True, ""
                
        except Exception as e:
            return False, f"Erreur validation: {str(e)}"
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Retourne statistiques sur les patterns configurés"""
        stats = {
            "total_entity_types": len(self._entity_patterns),
            "total_patterns": sum(len(patterns) for patterns in self._entity_patterns.values()),
            "patterns_by_type": {},
            "compiled_patterns": len(self._compiled_patterns)
        }
        
        for entity_type, patterns in self._entity_patterns.items():
            stats["patterns_by_type"][entity_type.value] = {
                "count": len(patterns),
                "avg_confidence": sum(p.confidence_base for p in patterns) / len(patterns),
                "has_normalizer": sum(1 for p in patterns if p.normalizer) / len(patterns),
                "has_validator": sum(1 for p in patterns if p.validator) / len(patterns)
            }
        
        return stats


# Instance singleton du pattern matcher
_pattern_matcher_instance = None

def get_pattern_matcher() -> FinancialPatternMatcher:
    """
    Factory function pour récupérer instance PatternMatcher singleton
    
    Raises:
        EntityExtractionError: Si l'initialisation échoue
    """
    global _pattern_matcher_instance
    if _pattern_matcher_instance is None:
        try:
            _pattern_matcher_instance = FinancialPatternMatcher()
        except Exception as e:
            logger.error(f"Erreur initialisation PatternMatcher: {e}")
            raise EntityExtractionError(
                f"Impossible d'initialiser le PatternMatcher: {str(e)}",
                extraction_method="initialization"
            )
    return _pattern_matcher_instance


# Fonctions utilitaires d'extraction rapide
def extract_amounts(text: str) -> List[float]:
    """
    Extraction rapide des montants d'un texte
    
    Raises:
        EntityExtractionError: Si l'extraction échoue
    """
    try:
        matcher = get_pattern_matcher()
        result = matcher.extract_entities(text, [EntityType.AMOUNT])
        
        amounts = []
        entities = result.get("entities", {})
        
        if EntityType.AMOUNT.value in entities:
            try:
                amount_value = float(entities[EntityType.AMOUNT.value])
                amounts.append(amount_value)
            except ValueError as e:
                raise EntityExtractionError(
                    f"Erreur conversion montant: {e}",
                    extraction_method="amount_extraction"
                )
        
        return amounts
        
    except EntityExtractionError:
        raise
    except Exception as e:
        raise EntityExtractionError(
            f"Erreur extraction montants: {e}",
            extraction_method="amount_extraction"
        )


def extract_categories(text: str) -> List[str]:
    """
    Extraction rapide des catégories d'un texte
    
    Raises:
        EntityExtractionError: Si l'extraction échoue
    """
    try:
        matcher = get_pattern_matcher()
        result = matcher.extract_entities(text, [EntityType.CATEGORY])
        
        categories = []
        entities = result.get("entities", {})
        
        if EntityType.CATEGORY.value in entities:
            categories.append(entities[EntityType.CATEGORY.value])
        
        return categories
        
    except EntityExtractionError:
        raise
    except Exception as e:
        raise EntityExtractionError(
            f"Erreur extraction catégories: {e}",
            extraction_method="category_extraction"
        )


def has_financial_entities(text: str) -> bool:
    """
    Vérifie rapidement si texte contient entités financières
    
    Returns:
        bool: True si entités financières détectées
        
    Note: Cette fonction ne lève pas d'exception pour permettre un usage simple
    """
    try:
        financial_entity_types = [
            EntityType.AMOUNT, 
            EntityType.ACCOUNT_TYPE, 
            EntityType.CATEGORY,
            EntityType.CARD_TYPE
        ]
        
        matcher = get_pattern_matcher()
        result = matcher.extract_entities(text, financial_entity_types)
        
        return bool(result.get("entities", {}))
        
    except Exception as e:
        logger.warning(f"Erreur vérification entités financières: {e}")
        return False


# Exports publics
__all__ = [
    "FinancialPatternMatcher",
    "EntityPattern", 
    "get_pattern_matcher",
    "extract_amounts",
    "extract_categories", 
    "has_financial_entities"
]