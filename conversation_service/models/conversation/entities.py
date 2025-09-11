"""
Modèles Pydantic Phase 2 : Entités Financières
Extraction et validation d'entités financières par multi-agents AutoGen
Compatible Phase 1 avec extensions métier avancées
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from pydantic.types import StrictStr, StrictInt, StrictFloat


class CurrencyCode(str, Enum):
    """Codes devises ISO 4217 supportées"""
    EUR = "EUR"
    USD = "USD" 
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    CAD = "CAD"


class EntityExtractionConfidence(str, Enum):
    """Niveaux de confiance extraction entités"""
    LOW = "low"           # < 0.5
    MEDIUM = "medium"     # 0.5 - 0.8
    HIGH = "high"         # > 0.8


class EntityType(str, Enum):
    """Types d'entités financières supportées"""
    AMOUNT = "amount"
    MERCHANT = "merchant"
    DATE_RANGE = "date_range"
    CATEGORY = "category"
    ACCOUNT = "account"
    LOCATION = "location"
    PAYMENT_METHOD = "payment_method"


class ExtractedAmount(BaseModel):
    """Montant extrait par LLM avec validation métier"""
    
    value: float = Field(..., description="Valeur numérique montant", gt=0)
    currency: CurrencyCode = Field(default=CurrencyCode.EUR, description="Code devise ISO")
    original_text: StrictStr = Field(..., description="Texte original extrait")
    confidence: float = Field(..., description="Score confiance extraction", ge=0.0, le=1.0)
    
    # Métadonnées extraction
    extraction_method: Literal["regex", "llm", "hybrid"] = Field(default="llm")
    validation_status: Literal["valid", "suspicious", "invalid"] = Field(default="valid")
    
    @field_validator('value')
    def validate_amount_range(cls, v):
        """Validation montants business logic"""
        if v <= 0:
            raise ValueError("Montant doit être positif")
        if v > 1000000:  # 1M€ limite business
            raise ValueError("Montant trop élevé (max 1M€)")
        return round(v, 2)  # Arrondi centimes
    
    @field_validator('original_text')
    def validate_original_text(cls, v):
        """Validation texte original"""
        if not v.strip():
            raise ValueError("Texte original requis")
        if len(v) > 100:
            raise ValueError("Texte original trop long (max 100 chars)")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_confidence_consistency(self):
        """Validation cohérence confiance/statut"""
        confidence = self.confidence
        validation_status = self.validation_status
        
        if confidence < 0.3 and validation_status == "valid":
            self.validation_status = "suspicious"
        elif confidence < 0.5 and validation_status == "valid":
            self.validation_status = "suspicious"
            
        return self
    
    def get_confidence_level(self) -> EntityExtractionConfidence:
        """Niveau confiance catégorisé"""
        if self.confidence >= 0.8:
            return EntityExtractionConfidence.HIGH
        elif self.confidence >= 0.5:
            return EntityExtractionConfidence.MEDIUM
        else:
            return EntityExtractionConfidence.LOW
    
    def to_decimal(self) -> Decimal:
        """Conversion Decimal pour calculs précis"""
        return Decimal(str(self.value))


class ExtractedMerchant(BaseModel):
    """Marchand/commerçant extrait"""
    
    name: StrictStr = Field(..., description="Nom marchand normalisé")
    original_text: StrictStr = Field(..., description="Texte original utilisateur")
    confidence: float = Field(..., description="Score confiance", ge=0.0, le=1.0)
    
    # Enrichissement data
    category_hint: Optional[str] = Field(None, description="Catégorie suggérée")
    known_merchant: bool = Field(default=False, description="Marchand connu en base")
    merchant_id: Optional[StrictStr] = Field(None, description="ID marchand si connu")
    
    # Métadonnées
    extraction_source: Literal["user_input", "transaction_data", "llm"] = Field(default="llm")
    normalization_applied: bool = Field(default=False, description="Normalisation nom appliquée")
    
    @field_validator('name')
    def validate_merchant_name(cls, v):
        """Validation nom marchand"""
        if not v.strip():
            raise ValueError("Nom marchand requis")
        if len(v) < 2:
            raise ValueError("Nom marchand trop court (min 2 chars)")
        if len(v) > 80:
            raise ValueError("Nom marchand trop long (max 80 chars)")
        return v.strip().title()  # Normalisation casse
    
    @field_validator('original_text')
    def validate_original_text(cls, v):
        """Validation texte original"""
        if not v.strip():
            raise ValueError("Texte original requis")
        return v.strip()
    
    def get_confidence_level(self) -> EntityExtractionConfidence:
        """Niveau confiance catégorisé"""
        if self.confidence >= 0.8:
            return EntityExtractionConfidence.HIGH
        elif self.confidence >= 0.5:
            return EntityExtractionConfidence.MEDIUM
        else:
            return EntityExtractionConfidence.LOW


class ExtractedDateRange(BaseModel):
    """Plage dates extraite avec validation business"""
    
    start_date: Optional[date] = Field(None, description="Date début")
    end_date: Optional[date] = Field(None, description="Date fin") 
    original_text: StrictStr = Field(..., description="Expression temporelle originale")
    confidence: float = Field(..., description="Score confiance extraction", ge=0.0, le=1.0)
    
    # Types période supportés
    period_type: Literal["specific_date", "date_range", "relative_period", "recurring"] = Field(default="date_range")
    relative_period: Optional[Literal["last_week", "last_month", "last_year", "today", "yesterday"]] = None
    
    # Métadonnées
    extraction_method: Literal["regex", "dateparser", "llm"] = Field(default="llm")
    timezone_hint: StrictStr = Field(default="Europe/Paris")
    
    @field_validator('start_date', 'end_date')
    def validate_date_business_rules(cls, v):
        """Validation règles business dates"""
        if v is None:
            return v
        
        # Pas de dates futures > 1 an
        max_future_date = date.today() + timedelta(days=365)
        if v > max_future_date:
            raise ValueError("Date trop loin dans le futur")
        
        # Pas de dates < 2010 (limite historique)
        min_date = date(2010, 1, 1)
        if v < min_date:
            raise ValueError("Date trop ancienne (min 2010)")
            
        return v
    
    @model_validator(mode='after')
    def validate_date_range_consistency(self):
        """Validation cohérence plage dates"""
        start = self.start_date
        end = self.end_date
        
        if start and end:
            if start > end:
                raise ValueError("Date début doit être <= date fin")
            
            # Max 5 ans de plage
            if (end - start).days > 365 * 5:
                raise ValueError("Plage dates trop large (max 5 ans)")
                
        return self
    
    @field_validator('original_text')
    def validate_original_text(cls, v):
        """Validation texte original"""
        if not v.strip():
            raise ValueError("Texte expression temporelle requis")
        return v.strip()
    
    def get_confidence_level(self) -> EntityExtractionConfidence:
        """Niveau confiance catégorisé"""
        if self.confidence >= 0.8:
            return EntityExtractionConfidence.HIGH
        elif self.confidence >= 0.5:
            return EntityExtractionConfidence.MEDIUM
        else:
            return EntityExtractionConfidence.LOW
    
    def to_dict_range(self) -> Dict[str, Optional[str]]:
        """Export range pour API"""
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None
        }


class ExtractedCategory(BaseModel):
    """Catégorie financière extraite"""
    
    name: StrictStr = Field(..., description="Nom catégorie")
    category_id: Optional[StrictStr] = Field(None, description="ID catégorie si mappée")
    confidence: float = Field(..., description="Score confiance", ge=0.0, le=1.0)
    original_text: StrictStr = Field(..., description="Texte original utilisateur")
    
    # Hiérarchie catégories
    parent_category: Optional[str] = Field(None, description="Catégorie parent")
    subcategory: Optional[str] = Field(None, description="Sous-catégorie")
    
    # Métadonnées
    extraction_method: Literal["keyword_match", "llm_classification", "merchant_mapping"] = Field(default="llm_classification")
    is_custom_category: bool = Field(default=False, description="Catégorie utilisateur")
    
    @field_validator('name')
    def validate_category_name(cls, v):
        """Validation nom catégorie"""
        if not v.strip():
            raise ValueError("Nom catégorie requis")
        if len(v) < 2:
            raise ValueError("Nom catégorie trop court")
        if len(v) > 50:
            raise ValueError("Nom catégorie trop long (max 50 chars)")
        return v.strip().title()
    
    def get_confidence_level(self) -> EntityExtractionConfidence:
        """Niveau confiance catégorisé"""
        if self.confidence >= 0.8:
            return EntityExtractionConfidence.HIGH
        elif self.confidence >= 0.5:
            return EntityExtractionConfidence.MEDIUM
        else:
            return EntityExtractionConfidence.LOW


class ComprehensiveEntityExtraction(BaseModel):
    """Résultat extraction complète d'entités multi-agents"""
    
    # Identifiants
    extraction_id: StrictStr = Field(default_factory=lambda: str(uuid4()))
    user_message: StrictStr = Field(..., description="Message utilisateur original")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Entités extraites
    amounts: List[ExtractedAmount] = Field(default_factory=list, description="Montants extraits")
    merchants: List[ExtractedMerchant] = Field(default_factory=list, description="Marchands extraits")
    date_ranges: List[ExtractedDateRange] = Field(default_factory=list, description="Dates extraites")
    categories: List[ExtractedCategory] = Field(default_factory=list, description="Catégories extraites")
    
    # Métadonnées globales
    overall_confidence: float = Field(default=0.0, description="Confiance globale", ge=0.0, le=1.0)
    extraction_method: Literal["single_agent", "multi_agent_autogen"] = Field(default="multi_agent_autogen")
    processing_time_ms: int = Field(default=0, description="Temps extraction ms")
    
    # Flags qualité
    entities_found: bool = Field(default=False, description="Au moins une entité trouvée")
    high_confidence_entities: int = Field(default=0, description="Nombre entités confiance haute")
    validation_errors: List[str] = Field(default_factory=list, description="Erreurs validation")
    
    @model_validator(mode='after')
    def calculate_derived_fields(self):
        """Calcul champs dérivés"""
        amounts = self.amounts
        merchants = self.merchants
        date_ranges = self.date_ranges
        categories = self.categories
        
        all_entities = amounts + merchants + date_ranges + categories
        
        # Au moins une entité ?
        self.entities_found = len(all_entities) > 0
        
        # Confiance haute count
        high_conf_count = sum(1 for entity in all_entities 
                            if hasattr(entity, 'confidence') and entity.confidence >= 0.8)
        self.high_confidence_entities = high_conf_count
        
        # Confiance globale moyenne
        if all_entities:
            confidences = [entity.confidence for entity in all_entities 
                         if hasattr(entity, 'confidence')]
            self.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return self
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[BaseModel]:
        """Récupération entités par type"""
        if entity_type == EntityType.AMOUNT:
            return self.amounts
        elif entity_type == EntityType.MERCHANT:
            return self.merchants
        elif entity_type == EntityType.DATE_RANGE:
            return self.date_ranges
        elif entity_type == EntityType.CATEGORY:
            return self.categories
        else:
            return []
    
    def get_high_confidence_entities(self, min_confidence: float = 0.8) -> Dict[str, List[BaseModel]]:
        """Entités avec confiance élevée"""
        return {
            "amounts": [a for a in self.amounts if a.confidence >= min_confidence],
            "merchants": [m for m in self.merchants if m.confidence >= min_confidence],
            "date_ranges": [d for d in self.date_ranges if d.confidence >= min_confidence],
            "categories": [c for c in self.categories if c.confidence >= min_confidence]
        }
    
    def to_legacy_entities_dict(self) -> Dict[str, Any]:
        """Conversion format entités Phase 1 (compatibilité)"""
        return {
            "amounts": [{"value": a.value, "currency": a.currency} for a in self.amounts],
            "merchants": [m.name for m in self.merchants],
            "dates": [d.to_dict_range() for d in self.date_ranges],
            "categories": [c.name for c in self.categories],
            "entities_count": len(self.amounts) + len(self.merchants) + len(self.date_ranges) + len(self.categories)
        }


# Factory Functions pour création rapide

def create_amount_from_text(text: str, value: float, confidence: float = 0.8) -> ExtractedAmount:
    """Factory création montant depuis texte"""
    return ExtractedAmount(
        value=value,
        original_text=text,
        confidence=confidence,
        extraction_method="llm"
    )


def create_merchant_from_text(text: str, confidence: float = 0.8) -> ExtractedMerchant:
    """Factory création marchand depuis texte"""
    return ExtractedMerchant(
        name=text,
        original_text=text,
        confidence=confidence,
        extraction_source="user_input"
    )


def create_date_range_from_text(text: str, start_date: Optional[date] = None, 
                               end_date: Optional[date] = None, confidence: float = 0.8) -> ExtractedDateRange:
    """Factory création plage dates"""
    return ExtractedDateRange(
        start_date=start_date,
        end_date=end_date,
        original_text=text,
        confidence=confidence,
        extraction_method="llm"
    )


def create_category_from_text(text: str, confidence: float = 0.8) -> ExtractedCategory:
    """Factory création catégorie"""
    return ExtractedCategory(
        name=text,
        original_text=text,
        confidence=confidence,
        extraction_method="llm_classification"
    )


# Backward compatibility imports
from .intent_entity_models import Intent, Entity

__all__ = [
    # Main classes
    "CurrencyCode", "EntityExtractionConfidence", "EntityType",
    "ExtractedAmount", "ExtractedMerchant", "ExtractedDateRange", 
    "ExtractedCategory", "ComprehensiveEntityExtraction",
    # Factory functions
    "create_amount_from_text", "create_merchant_from_text", 
    "create_date_range_from_text", "create_category_from_text",
    # Backward compatibility
    "Intent", "Entity"
]