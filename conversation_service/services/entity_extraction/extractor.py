"""
📝 Service Extraction d'Entités - Orchestrateur Principal

Service principal d'extraction d'entités avec fusion multi-méthodes
et validation cohérence pour domaine financier.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from conversation_service.models.enums import EntityType, IntentType
from conversation_service.models.intent import EntityExtractionRequest, EntityExtractionResponse
from conversation_service.models.exceptions import EntityExtractionError
from .pattern_matcher import get_pattern_matcher, FinancialPatternMatcher
from ..preprocessing.text_cleaner import get_text_cleaner
from conversation_service.config import config

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionResult:
    """Résultat d'extraction d'entité avec métadonnées"""
    entity_type: EntityType
    value: Any
    confidence: float
    extraction_method: str
    raw_text: str
    normalized_value: str
    validation_status: str = "valid"
    context: Optional[str] = None


class EntityExtractor:
    """
    Service principal d'extraction d'entités financières
    
    Orchestre plusieurs méthodes d'extraction et fusionne les résultats
    pour optimiser précision et rappel.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Services sous-jacents
        self.pattern_matcher = get_pattern_matcher()
        self.text_cleaner = get_text_cleaner()
        
        # Configuration extraction par intention
        self.intent_entity_config = {
            IntentType.ACCOUNT_BALANCE: [
                EntityType.ACCOUNT_TYPE, EntityType.MONTH
            ],
            IntentType.SEARCH_BY_CATEGORY: [
                EntityType.CATEGORY, EntityType.PERIOD, EntityType.MONTH, EntityType.AMOUNT
            ],
            IntentType.BUDGET_ANALYSIS: [
                EntityType.AMOUNT, EntityType.PERIOD, EntityType.MONTH, EntityType.CATEGORY
            ],
            IntentType.TRANSFER: [
                EntityType.AMOUNT, EntityType.RECIPIENT, EntityType.ACCOUNT_TYPE
            ],
            IntentType.SEARCH_BY_DATE: [
                EntityType.DATE, EntityType.MONTH, EntityType.PERIOD, EntityType.AMOUNT
            ],
            IntentType.CARD_MANAGEMENT: [
                EntityType.CARD_TYPE, EntityType.AMOUNT
            ]
        }
        
        # Métriques d'extraction
        self._metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "entities_by_type": {},
            "confidence_distribution": {},
            "validation_failures": 0
        }
    
    def extract_entities(
        self, 
        text: str, 
        intent_context: Optional[IntentType] = None,
        target_entities: Optional[List[EntityType]] = None,
        validate_entities: bool = True
    ) -> Dict[str, Any]:
        """
        Extraction principale d'entités avec contexte intentionnel
        
        Args:
            text: Texte source
            intent_context: Contexte d'intention pour cibler extraction
            target_entities: Types d'entités spécifiques à extraire
            validate_entities: Valider cohérence des entités
            
        Returns:
            Dict avec entités extraites et métadonnées
        """
        if not text:
            return {"entities": {}, "metadata": {}}
        
        self._metrics["total_extractions"] += 1
        
        # Préprocessing du texte
        try:
            preprocessed = self.text_cleaner.preprocess_query(text)
            clean_text = preprocessed["normalized_query"]
        except Exception as e:
            self.logger.warning(f"Erreur préprocessing pour extraction: {e}")
            clean_text = text.lower().strip()
        
        # Détermination entités cibles
        if target_entities is None:
            target_entities = self._get_target_entities_for_intent(intent_context)
        
        # Extraction multi-méthodes
        extraction_results = {}
        confidence_scores = {}
        
        # Méthode 1: Pattern matching (principal)
        try:
            pattern_results = self.pattern_matcher.extract_entities(
                clean_text, target_entities
            )
            self._merge_extraction_results(
                extraction_results, 
                confidence_scores,
                pattern_results.get("entities", {}),
                pattern_results.get("metadata", {}).get("confidence_scores", {}),
                "pattern_matching"
            )
        except Exception as e:
            self.logger.warning(f"Erreur pattern matching: {e}")
        
        # Méthode 2: Extraction contextuelle (enrichissement)
        try:
            contextual_results = self._extract_contextual_entities(
                clean_text, intent_context, target_entities
            )
            self._merge_extraction_results(
                extraction_results,
                confidence_scores, 
                contextual_results["entities"],
                contextual_results["confidence_scores"],
                "contextual"
            )
        except Exception as e:
            self.logger.warning(f"Erreur extraction contextuelle: {e}")
        
        # Validation et normalisation
        if validate_entities and extraction_results:
            extraction_results, confidence_scores = self._validate_and_normalize_entities(
                extraction_results, confidence_scores, intent_context
            )
        
        # Génération métadonnées
        metadata = self._generate_extraction_metadata(
            extraction_results, confidence_scores, target_entities, intent_context
        )
        
        # Mise à jour métriques
        self._update_extraction_metrics(extraction_results, confidence_scores)
        
        return {
            "entities": extraction_results,
            "confidence_per_entity": confidence_scores,
            "metadata": metadata
        }
    
    def _get_target_entities_for_intent(
        self, 
        intent: Optional[IntentType]
    ) -> List[EntityType]:
        """Détermine entités cibles selon contexte intentionnel"""
        if intent and intent in self.intent_entity_config:
            return self.intent_entity_config[intent]
        
        # Entités par défaut si pas de contexte
        return [
            EntityType.AMOUNT, EntityType.CATEGORY, EntityType.PERIOD,
            EntityType.MONTH, EntityType.ACCOUNT_TYPE, EntityType.CARD_TYPE
        ]
    
    def _merge_extraction_results(
        self,
        main_results: Dict[str, Any],
        main_confidence: Dict[str, float],
        new_results: Dict[str, Any],
        new_confidence: Dict[str, float],
        method: str
    ):
        """Fusionne résultats de différentes méthodes d'extraction"""
        for entity_key, entity_value in new_results.items():
            if entity_key not in main_results:
                # Nouvelle entité
                main_results[entity_key] = entity_value
                main_confidence[entity_key] = new_confidence.get(entity_key, 0.5)
            else:
                # Entité existante - garder meilleure confiance
                existing_confidence = main_confidence.get(entity_key, 0.0)
                new_conf = new_confidence.get(entity_key, 0.0)
                
                if new_conf > existing_confidence:
                    main_results[entity_key] = entity_value
                    main_confidence[entity_key] = new_conf
    
    def _extract_contextual_entities(
        self,
        text: str,
        intent_context: Optional[IntentType],
        target_entities: List[EntityType]
    ) -> Dict[str, Any]:
        """Extraction contextuelle enrichie selon l'intention"""
        entities = {}
        confidence_scores = {}
        
        # Enrichissement spécifique par intention
        if intent_context == IntentType.SEARCH_BY_CATEGORY:
            # Enrichissement catégories financières
            category_entities = self._extract_financial_categories(text)
            entities.update(category_entities["entities"])
            confidence_scores.update(category_entities["confidence_scores"])
        
        elif intent_context == IntentType.TRANSFER:
            # Enrichissement destinataires et montants
            transfer_entities = self._extract_transfer_entities(text)
            entities.update(transfer_entities["entities"])
            confidence_scores.update(transfer_entities["confidence_scores"])
        
        elif intent_context == IntentType.SEARCH_BY_DATE:
            # Enrichissement dates et périodes
            date_entities = self._extract_temporal_entities(text)
            entities.update(date_entities["entities"])
            confidence_scores.update(date_entities["confidence_scores"])
        
        return {
            "entities": entities,
            "confidence_scores": confidence_scores
        }
    
    def _extract_financial_categories(self, text: str) -> Dict[str, Any]:
        """Extraction enrichie des catégories financières"""
        entities = {}
        confidence_scores = {}
        
        # Mapping étendu catégories
        extended_categories = {
            'resto': 'restaurant', 'restau': 'restaurant', 'dîner': 'restaurant',
            'course': 'alimentation', 'supermarché': 'alimentation', 'épicerie': 'alimentation',
            'essence': 'transport', 'métro': 'transport', 'bus': 'transport',
            'vêtement': 'shopping', 'boutique': 'shopping', 'achat': 'shopping',
            'ciné': 'loisirs', 'cinéma': 'loisirs', 'sortie': 'loisirs',
            'pharma': 'santé', 'médecin': 'santé', 'docteur': 'santé'
        }
        
        text_words = text.lower().split()
        for word in text_words:
            for synonym, category in extended_categories.items():
                if synonym in word:
                    entities["category"] = category
                    confidence_scores["category"] = 0.8
                    break
        
        return {"entities": entities, "confidence_scores": confidence_scores}
    
    def _extract_transfer_entities(self, text: str) -> Dict[str, Any]:
        """Extraction spécialisée entités de virement"""
        entities = {}
        confidence_scores = {}
        
        # Patterns destinataires enrichis
        recipient_patterns = [
            r'\b(?:à|vers|pour)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b([A-Z][a-z]+)\s+(?:doit|recevra|aura)\b'
        ]
        
        for pattern in recipient_patterns:
            import re
            matches = re.findall(pattern, text)
            if matches:
                entities["recipient"] = matches[0] if isinstance(matches[0], str) else matches[0][0]
                confidence_scores["recipient"] = 0.85
                break
        
        return {"entities": entities, "confidence_scores": confidence_scores}
    
    def _extract_temporal_entities(self, text: str) -> Dict[str, Any]:
        """Extraction enrichie entités temporelles"""
        entities = {}
        confidence_scores = {}
        
        # Expressions temporelles étendues
        temporal_expressions = {
            'aujourd\'hui': 'today',
            'hier': 'yesterday', 
            'avant-hier': 'day_before_yesterday',
            'cette semaine': 'this_week',
            'semaine dernière': 'last_week',
            'ce mois': 'this_month',
            'mois dernier': 'last_month',
            'cette année': 'this_year',
            'année dernière': 'last_year'
        }
        
        text_lower = text.lower()
        for expression, normalized in temporal_expressions.items():
            if expression in text_lower:
                entities["period"] = normalized
                confidence_scores["period"] = 0.9
                break
        
        return {"entities": entities, "confidence_scores": confidence_scores}
    
    def _validate_and_normalize_entities(
        self,
        entities: Dict[str, Any],
        confidence_scores: Dict[str, float],
        intent_context: Optional[IntentType]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Valide et normalise les entités extraites"""
        validated_entities = {}
        validated_confidence = {}
        
        for entity_key, entity_value in entities.items():
            try:
                # Détermination type d'entité
                entity_type = self._get_entity_type_from_key(entity_key)
                if not entity_type:
                    continue
                
                # Validation spécialisée
                is_valid, normalized_value = self._validate_entity_value(
                    entity_type, entity_value, intent_context
                )
                
                if is_valid:
                    validated_entities[entity_key] = normalized_value
                    validated_confidence[entity_key] = confidence_scores.get(entity_key, 0.5)
                else:
                    self._metrics["validation_failures"] += 1
                    self.logger.debug(f"Validation failed for {entity_key}: {entity_value}")
                    
            except Exception as e:
                self.logger.warning(f"Erreur validation entité {entity_key}: {e}")
                continue
        
        return validated_entities, validated_confidence
    
    def _get_entity_type_from_key(self, entity_key: str) -> Optional[EntityType]:
        """Convertit clé entité en EntityType"""
        key_mapping = {
            "amount": EntityType.AMOUNT,
            "category": EntityType.CATEGORY,
            "account_type": EntityType.ACCOUNT_TYPE,
            "month": EntityType.MONTH,
            "period": EntityType.PERIOD,
            "recipient": EntityType.RECIPIENT,
            "card_type": EntityType.CARD_TYPE,
            "date": EntityType.DATE,
            "merchant": EntityType.MERCHANT,
            "currency": EntityType.CURRENCY,
            "location": EntityType.LOCATION
        }
        return key_mapping.get(entity_key)
    
    def _validate_entity_value(
        self,
        entity_type: EntityType,
        value: Any,
        intent_context: Optional[IntentType]
    ) -> Tuple[bool, Any]:
        """Validation spécialisée par type d'entité"""
        try:
            if entity_type == EntityType.AMOUNT:
                # Validation montant
                if isinstance(value, str):
                    normalized = value.replace(',', '.')
                    float_val = float(normalized)
                else:
                    float_val = float(value)
                
                if float_val < 0:
                    return False, value
                if float_val > 1000000:  # Seuil arbitraire
                    return False, value
                
                return True, float_val
            
            elif entity_type == EntityType.CATEGORY:
                # Validation catégorie
                valid_categories = {
                    'restaurant', 'alimentation', 'transport', 'shopping', 
                    'loisirs', 'santé', 'logement', 'éducation'
                }
                normalized = str(value).lower()
                if normalized in valid_categories:
                    return True, normalized
                return True, normalized  # Accepter même si pas dans liste prédéfinie
            
            elif entity_type == EntityType.MONTH:
                # Validation mois
                valid_months = [
                    'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                    'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'
                ]
                normalized = str(value).lower()
                return normalized in valid_months, normalized
            
            elif entity_type == EntityType.PERIOD:
                # Les périodes sont généralement valides si détectées
                return True, str(value).lower()
            
            elif entity_type == EntityType.RECIPIENT:
                # Validation destinataire (nom propre basic)
                recipient_str = str(value).strip()
                if len(recipient_str) < 2 or len(recipient_str) > 50:
                    return False, value
                return True, recipient_str
            
            else:
                # Validation générique
                return True, value
                
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Erreur validation {entity_type}: {e}")
            return False, value
    
    def _generate_extraction_metadata(
        self,
        entities: Dict[str, Any],
        confidence_scores: Dict[str, float],
        target_entities: List[EntityType],
        intent_context: Optional[IntentType]
    ) -> Dict[str, Any]:
        """Génère métadonnées complètes d'extraction"""
        return {
            "extraction_summary": {
                "total_entities_found": len(entities),
                "target_entities_count": len(target_entities),
                "extraction_success_rate": len(entities) / len(target_entities) if target_entities else 0.0,
                "average_confidence": sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
            },
            "intent_context": intent_context.value if intent_context else None,
            "entities_by_type": {
                entity_type.value: entity_type.value in [k for k in entities.keys()]
                for entity_type in target_entities
            },
            "high_confidence_entities": [
                k for k, v in confidence_scores.items() if v >= 0.8
            ],
            "extraction_methods_used": ["pattern_matching", "contextual"],
            "validation_applied": True
        }
    
    def _update_extraction_metrics(
        self,
        entities: Dict[str, Any],
        confidence_scores: Dict[str, float]
    ):
        """Met à jour métriques d'extraction"""
        if entities:
            self._metrics["successful_extractions"] += 1
        
        # Distribution par type d'entité
        for entity_key in entities.keys():
            if entity_key not in self._metrics["entities_by_type"]:
                self._metrics["entities_by_type"][entity_key] = 0
            self._metrics["entities_by_type"][entity_key] += 1
        
        # Distribution confiance
        for confidence in confidence_scores.values():
            level = "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low"
            if level not in self._metrics["confidence_distribution"]:
                self._metrics["confidence_distribution"][level] = 0
            self._metrics["confidence_distribution"][level] += 1
    
    def extract_single_entity_type(
        self,
        text: str,
        entity_type: EntityType,
        intent_context: Optional[IntentType] = None
    ) -> Optional[Dict[str, Any]]:
        """Extraction d'un seul type d'entité (optimisé)"""
        result = self.extract_entities(
            text=text,
            intent_context=intent_context,
            target_entities=[entity_type],
            validate_entities=True
        )
        
        entities = result.get("entities", {})
        confidence_scores = result.get("confidence_per_entity", {})
        
        entity_key = entity_type.value
        if entity_key in entities:
            return {
                "value": entities[entity_key],
                "confidence": confidence_scores.get(entity_key, 0.0),
                "entity_type": entity_type,
                "metadata": result.get("metadata", {})
            }
        
        return None
    
    def get_extraction_metrics(self) -> Dict[str, Any]:
        """Retourne métriques détaillées d'extraction"""
        if self._metrics["total_extractions"] == 0:
            return {"total_extractions": 0}
        
        success_rate = (
            self._metrics["successful_extractions"] / 
            self._metrics["total_extractions"]
        )
        
        return {
            "total_extractions": self._metrics["total_extractions"],
            "successful_extractions": self._metrics["successful_extractions"],
            "success_rate": round(success_rate, 3),
            "entities_by_type": self._metrics["entities_by_type"],
            "confidence_distribution": self._metrics["confidence_distribution"],
            "validation_failures": self._metrics["validation_failures"],
            "validation_failure_rate": round(
                self._metrics["validation_failures"] / self._metrics["total_extractions"], 3
            ) if self._metrics["total_extractions"] > 0 else 0.0
        }
    
    def reset_metrics(self):
        """Remet à zéro les métriques"""
        self._metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "entities_by_type": {},
            "confidence_distribution": {},
            "validation_failures": 0
        }


# Instance singleton
_entity_extractor_instance = None

def get_entity_extractor() -> EntityExtractor:
    """Factory function pour récupérer instance EntityExtractor singleton"""
    global _entity_extractor_instance
    if _entity_extractor_instance is None:
        _entity_extractor_instance = EntityExtractor()
    return _entity_extractor_instance


# Fonctions utilitaires d'extraction rapide
def quick_extract_amount(text: str) -> Optional[float]:
    """Extraction rapide d'un montant"""
    extractor = get_entity_extractor()
    result = extractor.extract_single_entity_type(text, EntityType.AMOUNT)
    return result["value"] if result else None


def quick_extract_category(text: str) -> Optional[str]:
    """Extraction rapide d'une catégorie"""
    extractor = get_entity_extractor()
    result = extractor.extract_single_entity_type(text, EntityType.CATEGORY)
    return result["value"] if result else None


# Exports publics
__all__ = [
    "EntityExtractor",
    "EntityExtractionResult",
    "get_entity_extractor",
    "quick_extract_amount",
    "quick_extract_category"
]