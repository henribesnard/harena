"""
Modèles de filtres pour le Search Service.

Ces modèles définissent les structures de données pour tous les filtres
supportés par le service de recherche, avec validation spécialisée pour
le domaine financier et transformation automatique vers Elasticsearch.

ARCHITECTURE:
- Filtres typés par domaine financier
- Validation stricte des valeurs et plages
- Transformation automatique vers Elasticsearch
- Support des filtres complexes et combinaisons
- Sécurité et isolation par utilisateur

CONFIGURATION CENTRALISÉE:
- Validation via config_service
- Limites configurables
- Champs et valeurs validées
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union, Literal, Type
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, NonNegativeInt, NonNegativeFloat

# Configuration centralisée
from config_service.config import settings

# ==================== ENUMS ET CONSTANTES ====================

class FilterType(str, Enum):
    """Types de filtres supportés."""
    USER = "user"
    CATEGORY = "category"
    MERCHANT = "merchant"
    AMOUNT = "amount"
    DATE = "date"
    TEXT = "text"
    TRANSACTION_TYPE = "transaction_type"
    OPERATION_TYPE = "operation_type"
    CURRENCY = "currency"
    CUSTOM = "custom"

class FilterOperator(str, Enum):
    """Opérateurs de filtrage."""
    EQ = "eq"           # Égalité exacte
    NE = "ne"           # Différent de
    GT = "gt"           # Supérieur à
    GTE = "gte"         # Supérieur ou égal
    LT = "lt"           # Inférieur à
    LTE = "lte"         # Inférieur ou égal
    IN = "in"           # Dans la liste
    NOT_IN = "not_in"   # Pas dans la liste
    BETWEEN = "between" # Entre deux valeurs
    CONTAINS = "contains" # Contient (texte)
    STARTS_WITH = "starts_with" # Commence par
    ENDS_WITH = "ends_with" # Finit par
    REGEX = "regex"     # Expression régulière
    EXISTS = "exists"   # Champ existe
    NOT_EXISTS = "not_exists" # Champ n'existe pas

class FilterLogic(str, Enum):
    """Logique de combinaison des filtres."""
    AND = "and"
    OR = "or"
    NOT = "not"

# Constantes financières
TRANSACTION_TYPES = ["debit", "credit"]
CURRENCY_CODES = ["EUR", "USD", "GBP", "CHF", "CAD", "JPY"]
FINANCIAL_CATEGORIES = [
    "alimentation", "restaurant", "transport", "carburant", "santé", "pharmacie",
    "shopping", "loisirs", "sport", "culture", "voyage", "hébergement",
    "services", "banque", "assurance", "impôts", "charges", "loyer",
    "énergie", "télécom", "internet", "abonnements", "éducation", "enfants",
    "animaux", "jardinage", "bricolage", "beauté", "coiffeur", "vêtements",
    "électronique", "informatique", "mobilier", "décoration", "cadeaux",
    "dons", "épargne", "investissement", "remboursement", "virement", "autre"
]

# Champs financiers valides pour filtrage
VALID_FILTER_FIELDS = {
    "user_id": {"type": "integer", "required": True},
    "account_id": {"type": "integer", "required": False},
    "transaction_id": {"type": "string", "required": False},
    "amount": {"type": "float", "required": False},
    "amount_abs": {"type": "float", "required": False},
    "currency_code": {"type": "string", "required": False, "values": CURRENCY_CODES},
    "transaction_type": {"type": "string", "required": False, "values": TRANSACTION_TYPES},
    "operation_type": {"type": "string", "required": False},
    "date": {"type": "date", "required": False},
    "month_year": {"type": "string", "required": False},
    "weekday": {"type": "string", "required": False},
    "primary_description": {"type": "string", "required": False},
    "merchant_name": {"type": "string", "required": False},
    "category_name": {"type": "string", "required": False, "values": FINANCIAL_CATEGORIES}
}

# Opérateurs valides par type de champ
FIELD_VALID_OPERATORS = {
    "integer": [FilterOperator.EQ, FilterOperator.NE, FilterOperator.GT, FilterOperator.GTE, 
                FilterOperator.LT, FilterOperator.LTE, FilterOperator.IN, FilterOperator.NOT_IN, FilterOperator.BETWEEN],
    "float": [FilterOperator.EQ, FilterOperator.NE, FilterOperator.GT, FilterOperator.GTE, 
              FilterOperator.LT, FilterOperator.LTE, FilterOperator.IN, FilterOperator.NOT_IN, FilterOperator.BETWEEN],
    "string": [FilterOperator.EQ, FilterOperator.NE, FilterOperator.IN, FilterOperator.NOT_IN, 
               FilterOperator.CONTAINS, FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH, FilterOperator.REGEX],
    "date": [FilterOperator.EQ, FilterOperator.NE, FilterOperator.GT, FilterOperator.GTE, 
             FilterOperator.LT, FilterOperator.LTE, FilterOperator.BETWEEN],
    "boolean": [FilterOperator.EQ, FilterOperator.NE]
}

# ==================== MODÈLES DE BASE ====================

class BaseFilter(BaseModel, ABC):
    """Classe de base pour tous les filtres."""
    filter_type: FilterType = Field(..., description="Type de filtre")
    field: str = Field(..., description="Champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    boost: Optional[float] = Field(None, ge=0.0, le=10.0, description="Boost de pertinence")
    metadata: Dict[str, Any] = Field(default={}, description="Métadonnées additionnelles")
    
    @abstractmethod
    def validate_value(self) -> bool:
        """Valide la valeur du filtre selon son type."""
        pass
    
    @abstractmethod
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit le filtre en requête Elasticsearch."""
        pass
    
    @validator('field')
    def validate_field(cls, v):
        """Valide que le champ est autorisé."""
        if v not in VALID_FILTER_FIELDS:
            raise ValueError(f"Champ non autorisé: {v}")
        return v
    
    @validator('operator')
    def validate_operator_for_field(cls, v, values):
        """Valide que l'opérateur est compatible avec le type de champ."""
        field = values.get('field')
        if field and field in VALID_FILTER_FIELDS:
            field_type = VALID_FILTER_FIELDS[field]["type"]
            if field_type in FIELD_VALID_OPERATORS:
                if v not in FIELD_VALID_OPERATORS[field_type]:
                    raise ValueError(f"Opérateur {v} non supporté pour le champ {field} de type {field_type}")
        return v
    
    class Config:
        use_enum_values = True

class UserFilter(BaseFilter):
    """Filtre spécialisé pour l'utilisateur (obligatoire pour sécurité)."""
    filter_type: Literal[FilterType.USER] = FilterType.USER
    field: Literal["user_id"] = "user_id"
    operator: Literal[FilterOperator.EQ] = FilterOperator.EQ
    value: PositiveInt = Field(..., description="ID de l'utilisateur")
    
    def validate_value(self) -> bool:
        """Valide l'ID utilisateur."""
        return self.value > 0
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en terme Elasticsearch."""
        return {"term": {"user_id": self.value}}

class CategoryFilter(BaseFilter):
    """Filtre spécialisé pour les catégories financières."""
    filter_type: Literal[FilterType.CATEGORY] = FilterType.CATEGORY
    field: Literal["category_name"] = "category_name"
    operator: FilterOperator = Field(default=FilterOperator.EQ, description="Opérateur")
    value: Union[str, List[str]] = Field(..., description="Catégorie(s) financière(s)")
    case_sensitive: bool = Field(default=False, description="Sensible à la casse")
    
    @validator('value')
    def validate_category_value(cls, v, values):
        """Valide les catégories financières."""
        categories = [v] if isinstance(v, str) else v
        for category in categories:
            if category.lower() not in [c.lower() for c in FINANCIAL_CATEGORIES]:
                raise ValueError(f"Catégorie non reconnue: {category}")
        return v
    
    def validate_value(self) -> bool:
        """Valide la valeur de catégorie."""
        if isinstance(self.value, str):
            return self.value.lower() in [c.lower() for c in FINANCIAL_CATEGORIES]
        return all(cat.lower() in [c.lower() for c in FINANCIAL_CATEGORIES] for cat in self.value)
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        field_name = "category_name.keyword" if not self.case_sensitive else "category_name"
        
        if self.operator == FilterOperator.EQ:
            return {"term": {field_name: self.value}}
        elif self.operator == FilterOperator.IN:
            return {"terms": {field_name: self.value if isinstance(self.value, list) else [self.value]}}
        elif self.operator == FilterOperator.NE:
            return {"bool": {"must_not": {"term": {field_name: self.value}}}}
        elif self.operator == FilterOperator.NOT_IN:
            values = self.value if isinstance(self.value, list) else [self.value]
            return {"bool": {"must_not": {"terms": {field_name: values}}}}
        else:
            raise ValueError(f"Opérateur {self.operator} non supporté pour CategoryFilter")

class MerchantFilter(BaseFilter):
    """Filtre spécialisé pour les marchands."""
    filter_type: Literal[FilterType.MERCHANT] = FilterType.MERCHANT
    field: Literal["merchant_name"] = "merchant_name"
    operator: FilterOperator = Field(default=FilterOperator.EQ, description="Opérateur")
    value: Union[str, List[str]] = Field(..., description="Nom(s) du/des marchand(s)")
    case_sensitive: bool = Field(default=False, description="Sensible à la casse")
    fuzzy_matching: bool = Field(default=False, description="Recherche approximative")
    
    def validate_value(self) -> bool:
        """Valide la valeur du marchand."""
        if isinstance(self.value, str):
            return len(self.value.strip()) > 0
        return all(len(merchant.strip()) > 0 for merchant in self.value)
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        if self.fuzzy_matching and self.operator in [FilterOperator.EQ, FilterOperator.CONTAINS]:
            return {
                "fuzzy": {
                    "merchant_name": {
                        "value": self.value,
                        "fuzziness": "AUTO"
                    }
                }
            }
        
        field_name = "merchant_name.keyword" if not self.case_sensitive else "merchant_name"
        
        if self.operator == FilterOperator.EQ:
            return {"term": {field_name: self.value}}
        elif self.operator == FilterOperator.IN:
            return {"terms": {field_name: self.value if isinstance(self.value, list) else [self.value]}}
        elif self.operator == FilterOperator.CONTAINS:
            return {"wildcard": {field_name: f"*{self.value}*"}}
        elif self.operator == FilterOperator.STARTS_WITH:
            return {"prefix": {field_name: self.value}}
        elif self.operator == FilterOperator.NE:
            return {"bool": {"must_not": {"term": {field_name: self.value}}}}
        else:
            raise ValueError(f"Opérateur {self.operator} non supporté pour MerchantFilter")

class AmountFilter(BaseFilter):
    """Filtre spécialisé pour les montants."""
    filter_type: Literal[FilterType.AMOUNT] = FilterType.AMOUNT
    field: Literal["amount", "amount_abs"] = Field(default="amount_abs", description="Champ montant")
    operator: FilterOperator = Field(..., description="Opérateur de comparaison")
    value: Union[float, List[float]] = Field(..., description="Montant(s) à filtrer")
    currency: Optional[str] = Field(None, description="Code devise pour validation")
    
    @validator('value')
    def validate_amount_value(cls, v):
        """Valide les valeurs de montant."""
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("Le montant ne peut pas être négatif")
            return float(v)
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Liste de montants vide")
            validated = []
            for amount in v:
                if amount < 0:
                    raise ValueError("Les montants ne peuvent pas être négatifs")
                validated.append(float(amount))
            return validated
        else:
            raise ValueError("Valeur de montant invalide")
    
    @validator('currency')
    def validate_currency(cls, v):
        """Valide le code devise."""
        if v and v not in CURRENCY_CODES:
            raise ValueError(f"Code devise non supporté: {v}")
        return v
    
    def validate_value(self) -> bool:
        """Valide la valeur du montant."""
        if isinstance(self.value, (int, float)):
            return self.value >= 0
        return all(amount >= 0 for amount in self.value)
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        if self.operator == FilterOperator.EQ:
            return {"term": {self.field: self.value}}
        elif self.operator == FilterOperator.NE:
            return {"bool": {"must_not": {"term": {self.field: self.value}}}}
        elif self.operator == FilterOperator.GT:
            return {"range": {self.field: {"gt": self.value}}}
        elif self.operator == FilterOperator.GTE:
            return {"range": {self.field: {"gte": self.value}}}
        elif self.operator == FilterOperator.LT:
            return {"range": {self.field: {"lt": self.value}}}
        elif self.operator == FilterOperator.LTE:
            return {"range": {self.field: {"lte": self.value}}}
        elif self.operator == FilterOperator.BETWEEN:
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("BETWEEN nécessite exactement 2 valeurs")
            return {"range": {self.field: {"gte": self.value[0], "lte": self.value[1]}}}
        elif self.operator == FilterOperator.IN:
            values = self.value if isinstance(self.value, list) else [self.value]
            return {"terms": {self.field: values}}
        else:
            raise ValueError(f"Opérateur {self.operator} non supporté pour AmountFilter")

class DateFilter(BaseFilter):
    """Filtre spécialisé pour les dates."""
    filter_type: Literal[FilterType.DATE] = FilterType.DATE
    field: Literal["date", "month_year"] = Field(default="date", description="Champ date")
    operator: FilterOperator = Field(..., description="Opérateur de comparaison")
    value: Union[str, List[str], date, List[date]] = Field(..., description="Date(s) à filtrer")
    format: str = Field(default="yyyy-MM-dd", description="Format de date")
    timezone: Optional[str] = Field(None, description="Fuseau horaire")
    
    @validator('value')
    def validate_date_value(cls, v, values):
        """Valide les valeurs de date."""
        field = values.get('field', 'date')
        
        if field == "month_year":
            # Validation format YYYY-MM
            import re
            month_pattern = r'^\d{4}-\d{2}$'
            if isinstance(v, str):
                if not re.match(month_pattern, v):
                    raise ValueError("Format month_year invalide, attendu: YYYY-MM")
            elif isinstance(v, list):
                for date_str in v:
                    if not re.match(month_pattern, date_str):
                        raise ValueError("Format month_year invalide, attendu: YYYY-MM")
        else:
            # Validation format date standard
            if isinstance(v, str):
                try:
                    datetime.strptime(v, "%Y-%m-%d")
                except ValueError:
                    raise ValueError("Format de date invalide, attendu: YYYY-MM-DD")
            elif isinstance(v, list):
                for date_str in v:
                    try:
                        datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        raise ValueError("Format de date invalide, attendu: YYYY-MM-DD")
        
        return v
    
    def validate_value(self) -> bool:
        """Valide la valeur de date."""
        try:
            if isinstance(self.value, str):
                datetime.strptime(self.value, "%Y-%m-%d")
            elif isinstance(self.value, list):
                for date_str in self.value:
                    datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        range_params = {}
        if self.format:
            range_params["format"] = self.format
        if self.timezone:
            range_params["time_zone"] = self.timezone
        
        if self.operator == FilterOperator.EQ:
            return {"term": {self.field: self.value}}
        elif self.operator == FilterOperator.NE:
            return {"bool": {"must_not": {"term": {self.field: self.value}}}}
        elif self.operator == FilterOperator.GT:
            range_params["gt"] = self.value
            return {"range": {self.field: range_params}}
        elif self.operator == FilterOperator.GTE:
            range_params["gte"] = self.value
            return {"range": {self.field: range_params}}
        elif self.operator == FilterOperator.LT:
            range_params["lt"] = self.value
            return {"range": {self.field: range_params}}
        elif self.operator == FilterOperator.LTE:
            range_params["lte"] = self.value
            return {"range": {self.field: range_params}}
        elif self.operator == FilterOperator.BETWEEN:
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("BETWEEN nécessite exactement 2 valeurs")
            range_params["gte"] = self.value[0]
            range_params["lte"] = self.value[1]
            return {"range": {self.field: range_params}}
        elif self.operator == FilterOperator.IN:
            values = self.value if isinstance(self.value, list) else [self.value]
            return {"terms": {self.field: values}}
        else:
            raise ValueError(f"Opérateur {self.operator} non supporté pour DateFilter")

class TextFilter(BaseFilter):
    """Filtre spécialisé pour la recherche textuelle."""
    filter_type: Literal[FilterType.TEXT] = FilterType.TEXT
    field: str = Field(..., description="Champ textuel")
    operator: FilterOperator = Field(default=FilterOperator.CONTAINS, description="Opérateur textuel")
    value: str = Field(..., min_length=1, max_length=1000, description="Texte à rechercher")
    case_sensitive: bool = Field(default=False, description="Sensible à la casse")
    fuzzy_matching: bool = Field(default=False, description="Recherche approximative")
    analyzer: Optional[str] = Field(None, description="Analyseur Elasticsearch")
    
    @validator('field')
    def validate_text_field(cls, v):
        """Valide que le champ supporte la recherche textuelle."""
        text_fields = ["primary_description", "merchant_name", "searchable_text", "category_name"]
        if v not in text_fields:
            raise ValueError(f"Champ textuel non supporté: {v}")
        return v
    
    def validate_value(self) -> bool:
        """Valide la valeur textuelle."""
        return len(self.value.strip()) > 0
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        if self.fuzzy_matching:
            return {
                "fuzzy": {
                    self.field: {
                        "value": self.value,
                        "fuzziness": "AUTO"
                    }
                }
            }
        
        if self.operator == FilterOperator.EQ:
            return {"term": {f"{self.field}.keyword": self.value}}
        elif self.operator == FilterOperator.CONTAINS:
            return {"wildcard": {f"{self.field}.keyword": f"*{self.value}*"}}
        elif self.operator == FilterOperator.STARTS_WITH:
            return {"prefix": {f"{self.field}.keyword": self.value}}
        elif self.operator == FilterOperator.ENDS_WITH:
            return {"wildcard": {f"{self.field}.keyword": f"*{self.value}"}}
        elif self.operator == FilterOperator.REGEX:
            return {"regexp": {f"{self.field}.keyword": self.value}}
        elif self.operator == FilterOperator.NE:
            return {"bool": {"must_not": {"term": {f"{self.field}.keyword": self.value}}}}
        else:
            # Recherche textuelle avec match
            match_params = {"query": self.value}
            if self.analyzer:
                match_params["analyzer"] = self.analyzer
            return {"match": {self.field: match_params}}

class TransactionTypeFilter(BaseFilter):
    """Filtre spécialisé pour les types de transaction."""
    filter_type: Literal[FilterType.TRANSACTION_TYPE] = FilterType.TRANSACTION_TYPE
    field: Literal["transaction_type"] = "transaction_type"
    operator: FilterOperator = Field(default=FilterOperator.EQ, description="Opérateur")
    value: Union[str, List[str]] = Field(..., description="Type(s) de transaction")
    
    @validator('value')
    def validate_transaction_type(cls, v):
        """Valide les types de transaction."""
        types = [v] if isinstance(v, str) else v
        for trans_type in types:
            if trans_type not in TRANSACTION_TYPES:
                raise ValueError(f"Type de transaction non supporté: {trans_type}")
        return v
    
    def validate_value(self) -> bool:
        """Valide la valeur du type de transaction."""
        if isinstance(self.value, str):
            return self.value in TRANSACTION_TYPES
        return all(t in TRANSACTION_TYPES for t in self.value)
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        if self.operator == FilterOperator.EQ:
            return {"term": {"transaction_type": self.value}}
        elif self.operator == FilterOperator.IN:
            values = self.value if isinstance(self.value, list) else [self.value]
            return {"terms": {"transaction_type": values}}
        elif self.operator == FilterOperator.NE:
            return {"bool": {"must_not": {"term": {"transaction_type": self.value}}}}
        else:
            raise ValueError(f"Opérateur {self.operator} non supporté pour TransactionTypeFilter")

class CustomFilter(BaseFilter):
    """Filtre personnalisé pour cas spéciaux."""
    filter_type: Literal[FilterType.CUSTOM] = FilterType.CUSTOM
    field: str = Field(..., description="Champ personnalisé")
    operator: FilterOperator = Field(..., description="Opérateur")
    value: Any = Field(..., description="Valeur personnalisée")
    elasticsearch_query: Optional[Dict[str, Any]] = Field(None, description="Requête ES personnalisée")
    
    def validate_value(self) -> bool:
        """Valide la valeur personnalisée."""
        return self.value is not None
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        if self.elasticsearch_query:
            return self.elasticsearch_query
        
        # Conversion générique basée sur l'opérateur
        if self.operator == FilterOperator.EQ:
            return {"term": {self.field: self.value}}
        elif self.operator == FilterOperator.IN:
            values = self.value if isinstance(self.value, list) else [self.value]
            return {"terms": {self.field: values}}
        elif self.operator == FilterOperator.EXISTS:
            return {"exists": {"field": self.field}}
        elif self.operator == FilterOperator.NOT_EXISTS:
            return {"bool": {"must_not": {"exists": {"field": self.field}}}}
        else:
            raise ValueError(f"Opérateur {self.operator} non supporté pour CustomFilter")

# ==================== COMBINAISONS DE FILTRES ====================

class FilterCombination(BaseModel):
    """Combinaison de filtres avec logique AND/OR."""
    logic: FilterLogic = Field(default=FilterLogic.AND, description="Logique de combinaison")
    filters: List[BaseFilter] = Field(..., min_items=1, description="Filtres à combiner")
    boost: Optional[float] = Field(None, ge=0.0, le=10.0, description="Boost global")
    
    @validator('filters')
    def validate_filters_count(cls, v):
        """Valide le nombre de filtres."""
        if len(v) > settings.MAX_FILTERS_PER_GROUP:
            raise ValueError(f"Trop de filtres: max {settings.MAX_FILTERS_PER_GROUP}")
        return v
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête bool Elasticsearch."""
        es_filters = [f.to_elasticsearch() for f in self.filters]
        
        if self.logic == FilterLogic.AND:
            bool_query = {"bool": {"filter": es_filters}}
        elif self.logic == FilterLogic.OR:
            bool_query = {"bool": {"should": es_filters, "minimum_should_match": 1}}
        elif self.logic == FilterLogic.NOT:
            bool_query = {"bool": {"must_not": es_filters}}
        else:
            raise ValueError(f"Logique {self.logic} non supportée")
        
        if self.boost:
            bool_query["bool"]["boost"] = self.boost
        
        return bool_query

class FilterGroup(BaseModel):
    """Groupe de filtres avec logique complexe."""
    required: List[BaseFilter] = Field(default=[], description="Filtres obligatoires (AND)")
    optional: List[BaseFilter] = Field(default=[], description="Filtres optionnels (OR)")
    forbidden: List[BaseFilter] = Field(default=[], description="Filtres interdits (NOT)")
    minimum_should_match: Optional[int] = Field(None, description="Nombre minimum de filtres optionnels")
    
    @validator('required', 'optional', 'forbidden')
    def validate_filter_counts(cls, v):
        """Valide le nombre de filtres par groupe."""
        if len(v) > settings.MAX_FILTERS_PER_GROUP:
            raise ValueError(f"Trop de filtres: max {settings.MAX_FILTERS_PER_GROUP}")
        return v
    
    @root_validator
    def validate_user_filter_present(cls, values):
        """Valide qu'un filtre utilisateur est présent."""
        required = values.get('required', [])
        has_user_filter = any(
            isinstance(f, UserFilter) or (hasattr(f, 'field') and f.field == 'user_id')
            for f in required
        )
        
        if not has_user_filter:
            raise ValueError("Filtre user_id obligatoire pour la sécurité")
        
        return values
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête bool Elasticsearch complexe."""
        bool_query = {}
        
        if self.required:
            bool_query["filter"] = [f.to_elasticsearch() for f in self.required]
        
        if self.optional:
            bool_query["should"] = [f.to_elasticsearch() for f in self.optional]
            if self.minimum_should_match:
                bool_query["minimum_should_match"] = self.minimum_should_match
        
        if self.forbidden:
            bool_query["must_not"] = [f.to_elasticsearch() for f in self.forbidden]
        
        return {"bool": bool_query}

# ==================== VALIDATION ET TRANSFORMATION ====================

class FilterValidationError(Exception):
    """Exception pour les erreurs de validation de filtre."""
    pass

class FilterValidator:
    """Validateur pour les filtres de recherche."""
    
    @staticmethod
    def validate_filter(filter_obj: BaseFilter) -> bool:
        """
        Valide un filtre individuellement.
        
        Args:
            filter_obj: Filtre à valider
            
        Returns:
            True si valide
            
        Raises:
            FilterValidationError: Si la validation échoue
        """
        try:
            # Validation Pydantic
            filter_obj.dict()
            
            # Validation spécifique au filtre
            if not filter_obj.validate_value():
                raise FilterValidationError(f"Valeur invalide pour {filter_obj.field}")
            
            # Validation sécurité
            if isinstance(filter_obj, UserFilter):
                if filter_obj.value <= 0:
                    raise FilterValidationError("user_id doit être positif")
            
            return True
            
        except Exception as e:
            raise FilterValidationError(f"Validation échouée: {str(e)}")
    
    @staticmethod
    def validate_filter_group(filter_group: FilterGroup) -> bool:
        """
        Valide un groupe de filtres.
        
        Args:
            filter_group: Groupe de filtres à valider
            
        Returns:
            True si valide
            
        Raises:
            FilterValidationError: Si la validation échoue
        """
        try:
            # Validation Pydantic
            filter_group.dict()
            
            # Validation de chaque filtre
            all_filters = filter_group.required + filter_group.optional + filter_group.forbidden
            for filter_obj in all_filters:
                FilterValidator.validate_filter(filter_obj)
            
            return True
            
        except Exception as e:
            raise FilterValidationError(f"Validation groupe échouée: {str(e)}")

class FilterTransformer:
    """Transformateur pour convertir les filtres en requêtes Elasticsearch."""
    
    @staticmethod
    def transform_filter(filter_obj: BaseFilter) -> Dict[str, Any]:
        """
        Transforme un filtre en requête Elasticsearch.
        
        Args:
            filter_obj: Filtre à transformer
            
        Returns:
            Dictionnaire de requête Elasticsearch
        """
        return filter_obj.to_elasticsearch()
    
    @staticmethod
    def transform_filter_group(filter_group: FilterGroup) -> Dict[str, Any]:
        """
        Transforme un groupe de filtres en requête Elasticsearch.
        
        Args:
            filter_group: Groupe de filtres à transformer
            
        Returns:
            Dictionnaire de requête Elasticsearch
        """
        return filter_group.to_elasticsearch()
    
    @staticmethod
    def optimize_elasticsearch_query(query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimise une requête Elasticsearch générée à partir des filtres.
        
        Args:
            query: Requête Elasticsearch à optimiser
            
        Returns:
            Requête optimisée
        """
        import copy
        optimized = copy.deepcopy(query)
        
        # Optimisation 1: Déplacer les filtres exacts vers filter clause
        if "bool" in optimized:
            bool_query = optimized["bool"]
            
            # Déplacer les term queries vers filter
            if "must" in bool_query:
                must_clauses = bool_query["must"]
                filter_clauses = bool_query.get("filter", [])
                new_must = []
                
                for clause in must_clauses:
                    if isinstance(clause, dict) and "term" in clause:
                        filter_clauses.append(clause)
                    else:
                        new_must.append(clause)
                
                bool_query["must"] = new_must
                if filter_clauses:
                    bool_query["filter"] = filter_clauses
        
        return optimized

# ==================== FACTORY FUNCTIONS ====================

def create_user_filter(user_id: int) -> UserFilter:
    """Factory pour créer un filtre utilisateur."""
    return UserFilter(value=user_id)

def create_category_filter(category: str, operator: FilterOperator = FilterOperator.EQ) -> CategoryFilter:
    """Factory pour créer un filtre de catégorie."""
    return CategoryFilter(operator=operator, value=category)

def create_amount_filter(
    amount: Union[float, List[float]], 
    operator: FilterOperator = FilterOperator.EQ,
    field: str = "amount_abs"
) -> AmountFilter:
    """Factory pour créer un filtre de montant."""
    return AmountFilter(field=field, operator=operator, value=amount)

def create_date_filter(
    date_value: Union[str, List[str]], 
    operator: FilterOperator = FilterOperator.EQ,
    field: str = "date"
) -> DateFilter:
    """Factory pour créer un filtre de date."""
    return DateFilter(field=field, operator=operator, value=date_value)

def create_text_filter(
    text: str, 
    field: str = "primary_description",
    operator: FilterOperator = FilterOperator.CONTAINS
) -> TextFilter:
    """Factory pour créer un filtre textuel."""
    return TextFilter(field=field, operator=operator, value=text)

def create_standard_filter_group(user_id: int) -> FilterGroup:
    """Factory pour créer un groupe de filtres standard avec user_id."""
    return FilterGroup(
        required=[create_user_filter(user_id)]
    )

# ==================== HELPERS ET UTILITAIRES ====================

def filter_to_elasticsearch(filter_obj: BaseFilter) -> Dict[str, Any]:
    """Convertit un filtre en requête Elasticsearch."""
    return FilterTransformer.transform_filter(filter_obj)

def filters_to_elasticsearch(filters: List[BaseFilter]) -> List[Dict[str, Any]]:
    """Convertit une liste de filtres en requêtes Elasticsearch."""
    return [filter_to_elasticsearch(f) for f in filters]

def get_filter_type_for_field(field: str) -> Optional[Type[BaseFilter]]:
    """Retourne le type de filtre approprié pour un champ."""
    field_to_filter = {
        "user_id": UserFilter,
        "category_name": CategoryFilter,
        "merchant_name": MerchantFilter,
        "amount": AmountFilter,
        "amount_abs": AmountFilter,
        "date": DateFilter,
        "month_year": DateFilter,
        "primary_description": TextFilter,
        "searchable_text": TextFilter,
        "transaction_type": TransactionTypeFilter
    }
    return field_to_filter.get(field, CustomFilter)

def validate_filter_field_value(field: str, value: Any) -> bool:
    """Valide qu'une valeur est appropriée pour un champ."""
    if field not in VALID_FILTER_FIELDS:
        return False
    
    field_config = VALID_FILTER_FIELDS[field]
    field_type = field_config["type"]
    
    if field_type == "integer":
        return isinstance(value, int)
    elif field_type == "float":
        return isinstance(value, (int, float))
    elif field_type == "string":
        return isinstance(value, str)
    elif field_type == "date":
        if isinstance(value, str):
            try:
                datetime.strptime(value, "%Y-%m-%d")
                return True
            except ValueError:
                return False
    elif field_type == "boolean":
        return isinstance(value, bool)
    
    return False

# ==================== CONSTANTES ET EXPORTS ====================

# Mapping des opérateurs vers leurs symboles
FILTER_OPERATORS = {
    FilterOperator.EQ: "=",
    FilterOperator.NE: "≠",
    FilterOperator.GT: ">",
    FilterOperator.GTE: "≥",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "≤",
    FilterOperator.IN: "∈",
    FilterOperator.NOT_IN: "∉",
    FilterOperator.BETWEEN: "between",
    FilterOperator.CONTAINS: "contains",
    FilterOperator.STARTS_WITH: "starts_with",
    FilterOperator.ENDS_WITH: "ends_with"
}

__all__ = [
    # Filtres spécialisés
    "BaseFilter",
    "UserFilter",
    "CategoryFilter",
    "MerchantFilter",
    "AmountFilter",
    "DateFilter",
    "TextFilter",
    "TransactionTypeFilter",
    "CustomFilter",
    
    # Combinaisons
    "FilterCombination",
    "FilterGroup",
    
    # Enums
    "FilterType",
    "FilterOperator",
    "FilterLogic",
    
    # Validation et transformation
    "FilterValidationError",
    "FilterValidator",
    "FilterTransformer",
    
    # Factory functions
    "create_user_filter",
    "create_category_filter",
    "create_amount_filter",
    "create_date_filter",
    "create_text_filter",
    "create_standard_filter_group",
    
    # Helpers
    "filter_to_elasticsearch",
    "filters_to_elasticsearch",
    "get_filter_type_for_field",
    "validate_filter_field_value",
    
    # Constantes
    "FILTER_OPERATORS",
    "VALID_FILTER_FIELDS",
    "FINANCIAL_CATEGORIES",
    "TRANSACTION_TYPES",
    "CURRENCY_CODES",
    "FIELD_VALID_OPERATORS"
]