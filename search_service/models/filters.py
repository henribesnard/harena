"""
Modèles de filtres et validation pour le Search Service
Structures spécialisées pour la gestion des filtres Elasticsearch
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from dataclasses import dataclass
import re

from search_service.config import SUPPORTED_FILTER_OPERATORS


class FieldType(str, Enum):
    """Types de champs supportés"""
    TEXT = "text"           # Champs textuels analysés
    KEYWORD = "keyword"     # Champs textuels exacts
    INTEGER = "integer"     # Nombres entiers
    FLOAT = "float"         # Nombres décimaux
    DATE = "date"           # Dates
    BOOLEAN = "boolean"     # Booléens


class FilterPriority(str, Enum):
    """Priorités des filtres pour optimisation"""
    CRITICAL = "critical"   # user_id, sécurité
    HIGH = "high"          # Filtres très sélectifs
    MEDIUM = "medium"      # Filtres modérément sélectifs
    LOW = "low"            # Filtres peu sélectifs


class DatePeriod(str, Enum):
    """Périodes prédéfinies pour les filtres temporels"""
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_QUARTER = "this_quarter"
    LAST_QUARTER = "last_quarter"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"


# === CONFIGURATION DES CHAMPS ===

@dataclass
class FieldConfig:
    """Configuration d'un champ indexé"""
    name: str
    field_type: FieldType
    priority: FilterPriority
    boost: float = 1.0
    is_filterable: bool = True
    is_searchable: bool = False
    supports_range: bool = False
    supports_aggregation: bool = False
    validation_pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    
    def __post_init__(self):
        # Validation automatique selon le type
        if self.field_type in [FieldType.INTEGER, FieldType.FLOAT, FieldType.DATE]:
            self.supports_range = True
        
        if self.field_type == FieldType.TEXT:
            self.is_searchable = True
        
        if self.field_type in [FieldType.KEYWORD, FieldType.INTEGER]:
            self.supports_aggregation = True


# Configuration complète des champs financiers
FIELD_CONFIGURATIONS = {
    # Champs de sécurité
    "user_id": FieldConfig(
        name="user_id",
        field_type=FieldType.INTEGER,
        priority=FilterPriority.CRITICAL,
        boost=1.0,
        validation_pattern=r"^[1-9]\d*$"
    ),
    
    # Champs de recherche textuelle
    "searchable_text": FieldConfig(
        name="searchable_text",
        field_type=FieldType.TEXT,
        priority=FilterPriority.MEDIUM,
        boost=2.0,
        is_searchable=True
    ),
    
    "primary_description": FieldConfig(
        name="primary_description",
        field_type=FieldType.TEXT,
        priority=FilterPriority.MEDIUM,
        boost=1.5,
        is_searchable=True
    ),
    
    "merchant_name": FieldConfig(
        name="merchant_name",
        field_type=FieldType.TEXT,
        priority=FilterPriority.HIGH,
        boost=1.8,
        is_searchable=True
    ),
    
    # Champs de filtrage exact
    "category_name": FieldConfig(
        name="category_name",
        field_type=FieldType.TEXT,
        priority=FilterPriority.HIGH,
        boost=1.2,
        is_searchable=True
    ),
    
    "category_name.keyword": FieldConfig(
        name="category_name.keyword",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.HIGH,
        supports_aggregation=True
    ),
    
    "merchant_name.keyword": FieldConfig(
        name="merchant_name.keyword",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.HIGH,
        supports_aggregation=True
    ),
    
    "transaction_type": FieldConfig(
        name="transaction_type",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.MEDIUM,
        allowed_values=["debit", "credit"]
    ),
    
    "currency_code": FieldConfig(
        name="currency_code",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.LOW,
        allowed_values=["EUR", "USD", "GBP", "CHF", "CAD"]
    ),
    
    "operation_type": FieldConfig(
        name="operation_type",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.MEDIUM,
        supports_aggregation=True
    ),
    
    "operation_type.keyword": FieldConfig(
        name="operation_type.keyword",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.MEDIUM,
        supports_aggregation=True
    ),
    
    # Champs numériques
    "amount": FieldConfig(
        name="amount",
        field_type=FieldType.FLOAT,
        priority=FilterPriority.MEDIUM,
        supports_range=True
    ),
    
    "amount_abs": FieldConfig(
        name="amount_abs",
        field_type=FieldType.FLOAT,
        priority=FilterPriority.MEDIUM,
        supports_range=True,
        supports_aggregation=True
    ),
    
    "account_id": FieldConfig(
        name="account_id",
        field_type=FieldType.INTEGER,
        priority=FilterPriority.MEDIUM,
        validation_pattern=r"^[1-9]\d*$"
    ),
    
    # Champs temporels
    "date": FieldConfig(
        name="date",
        field_type=FieldType.DATE,
        priority=FilterPriority.HIGH,
        supports_range=True,
        validation_pattern=r"^\d{4}-\d{2}-\d{2}$"
    ),
    
    "month_year": FieldConfig(
        name="month_year",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.HIGH,
        supports_aggregation=True,
        validation_pattern=r"^\d{4}-\d{2}$"
    ),
    
    "weekday": FieldConfig(
        name="weekday",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.LOW,
        allowed_values=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ),
    
    # Champs d'identification
    "transaction_id": FieldConfig(
        name="transaction_id",
        field_type=FieldType.KEYWORD,
        priority=FilterPriority.LOW
    )
}


# === MODÈLES DE VALIDATION ===

class FilterValue(BaseModel):
    """Valeur de filtre avec validation selon le type"""
    raw_value: Any = Field(..., description="Valeur brute")
    field_name: str = Field(..., description="Nom du champ")
    
    @field_validator("raw_value")
    @classmethod
    def validate_value_type(cls, v, info):
        """Valide le type de valeur selon le champ"""
        if not info.data:
            return v
            
        field_name = info.data.get("field_name")
        if not field_name or field_name not in FIELD_CONFIGURATIONS:
            return v
        
        field_config = FIELD_CONFIGURATIONS[field_name]
        
        # Validation selon le type de champ
        if field_config.field_type == FieldType.INTEGER:
            if not isinstance(v, int):
                if isinstance(v, str) and v.isdigit():
                    return int(v)
                raise ValueError(f"Valeur doit être un entier pour {field_name}")
        
        elif field_config.field_type == FieldType.FLOAT:
            if not isinstance(v, (int, float)):
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        raise ValueError(f"Valeur doit être un nombre pour {field_name}")
                raise ValueError(f"Valeur doit être un nombre pour {field_name}")
        
        elif field_config.field_type == FieldType.DATE:
            if isinstance(v, str):
                # Validation format date
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                    raise ValueError(f"Date doit être au format YYYY-MM-DD pour {field_name}")
        
        # Validation valeurs autorisées
        if field_config.allowed_values and v not in field_config.allowed_values:
            raise ValueError(f"Valeur {v} non autorisée pour {field_name}. Valeurs possibles: {field_config.allowed_values}")
        
        # Validation pattern
        if field_config.validation_pattern and isinstance(v, str):
            if not re.match(field_config.validation_pattern, v):
                raise ValueError(f"Valeur {v} ne respecte pas le format requis pour {field_name}")
        
        return v
    
    def get_normalized_value(self) -> Any:
        """Retourne la valeur normalisée pour Elasticsearch"""
        field_config = FIELD_CONFIGURATIONS.get(self.field_name)
        if not field_config:
            return self.raw_value
        
        # Normalisation selon le type
        if field_config.field_type == FieldType.DATE:
            if isinstance(self.raw_value, str):
                return self.raw_value  # Format YYYY-MM-DD déjà validé
        
        elif field_config.field_type == FieldType.FLOAT:
            return float(self.raw_value)
        
        elif field_config.field_type == FieldType.INTEGER:
            return int(self.raw_value)
        
        return self.raw_value


class DateRangeFilter(BaseModel):
    """Filtre de plage de dates avec périodes prédéfinies"""
    period: Optional[DatePeriod] = Field(default=None, description="Période prédéfinie")
    start_date: Optional[str] = Field(default=None, description="Date de début (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="Date de fin (YYYY-MM-DD)")
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Valide la cohérence de la plage de dates"""
        if self.period and (self.start_date or self.end_date):
            raise ValueError("Utiliser soit period soit start_date/end_date, pas les deux")
        
        if not self.period and not (self.start_date and self.end_date):
            raise ValueError("period ou start_date/end_date requis")
        
        if self.start_date and self.end_date:
            # Validation format
            for date_str in [self.start_date, self.end_date]:
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                    raise ValueError(f"Date {date_str} doit être au format YYYY-MM-DD")
            
            # Validation ordre
            if self.start_date > self.end_date:
                raise ValueError("start_date doit être antérieure à end_date")
        
        return self
    
    def get_date_range(self) -> Tuple[str, str]:
        """Retourne la plage de dates calculée"""
        if self.start_date and self.end_date:
            return self.start_date, self.end_date
        
        # Calcul basé sur la période
        today = date.today()
        
        if self.period == DatePeriod.TODAY:
            return today.isoformat(), today.isoformat()
        
        elif self.period == DatePeriod.YESTERDAY:
            yesterday = date.today().replace(day=today.day - 1)
            return yesterday.isoformat(), yesterday.isoformat()
        
        elif self.period == DatePeriod.THIS_MONTH:
            start = today.replace(day=1)
            return start.isoformat(), today.isoformat()
        
        elif self.period == DatePeriod.LAST_MONTH:
            first_this_month = today.replace(day=1)
            last_month_end = first_this_month.replace(day=first_this_month.day - 1)
            last_month_start = last_month_end.replace(day=1)
            return last_month_start.isoformat(), last_month_end.isoformat()
        
        elif self.period == DatePeriod.LAST_7_DAYS:
            start = date.today().replace(day=today.day - 7)
            return start.isoformat(), today.isoformat()
        
        elif self.period == DatePeriod.LAST_30_DAYS:
            start = date.today().replace(day=today.day - 30)
            return start.isoformat(), today.isoformat()
        
        elif self.period == DatePeriod.LAST_90_DAYS:
            start = date.today().replace(day=today.day - 90)
            return start.isoformat(), today.isoformat()
        
        # Défaut: aujourd'hui
        return today.isoformat(), today.isoformat()


class AmountRangeFilter(BaseModel):
    """Filtre de plage de montants avec validation"""
    min_amount: Optional[float] = Field(default=None, description="Montant minimum")
    max_amount: Optional[float] = Field(default=None, description="Montant maximum")
    include_negative: bool = Field(default=True, description="Inclure montants négatifs")
    absolute_values: bool = Field(default=False, description="Utiliser valeurs absolues")
    
    @field_validator("min_amount", "max_amount")
    @classmethod
    def validate_amounts(cls, v):
        """Valide les montants"""
        if v is not None:
            if abs(v) > 1000000:  # 1M max
                raise ValueError("Montant ne peut pas dépasser 1,000,000")
            return round(float(v), 2)
        return v
    
    @model_validator(mode='after')
    def validate_range(self):
        """Valide la cohérence de la plage"""
        if self.min_amount is not None and self.max_amount is not None:
            if self.min_amount > self.max_amount:
                raise ValueError("min_amount doit être inférieur à max_amount")
        
        return self
    
    def get_elasticsearch_filter(self, field_name: str = "amount") -> Dict[str, Any]:
        """Génère le filtre Elasticsearch correspondant"""
        if self.absolute_values:
            field_name = "amount_abs"
        
        range_filter = {}
        
        if self.min_amount is not None:
            range_filter["gte"] = self.min_amount
        
        if self.max_amount is not None:
            range_filter["lte"] = self.max_amount
        
        if not range_filter:
            return {}
        
        # Si on exclut les négatifs et qu'on utilise le champ amount
        if not self.include_negative and field_name == "amount":
            # Ajouter un filtre pour exclure les négatifs
            return {
                "bool": {
                    "must": [
                        {"range": {field_name: range_filter}},
                        {"range": {field_name: {"gte": 0}}}
                    ]
                }
            }
        
        return {"range": {field_name: range_filter}}


class ValidatedFilter(BaseModel):
    """Filtre validé et prêt pour Elasticsearch"""
    field: str = Field(..., description="Nom du champ")
    operator: str = Field(..., description="Opérateur de filtrage")
    value: FilterValue = Field(..., description="Valeur validée")
    boost: float = Field(default=1.0, description="Boost pour ce filtre")
    priority: FilterPriority = Field(default=FilterPriority.MEDIUM, description="Priorité du filtre")
    
    @field_validator("field")
    @classmethod
    def validate_field_exists(cls, v):
        """Valide que le champ existe dans la configuration"""
        if v not in FIELD_CONFIGURATIONS:
            raise ValueError(f"Champ {v} non configuré. Champs disponibles: {list(FIELD_CONFIGURATIONS.keys())}")
        return v
    
    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v):
        """Valide l'opérateur"""
        if v not in SUPPORTED_FILTER_OPERATORS:
            raise ValueError(f"Opérateur {v} non supporté. Opérateurs disponibles: {SUPPORTED_FILTER_OPERATORS}")
        return v
    
    @model_validator(mode='after')
    def validate_field_operator_compatibility(self):
        """Valide la compatibilité champ/opérateur"""
        field_config = FIELD_CONFIGURATIONS[self.field]
        
        # Opérateurs de plage seulement pour champs supportant les plages
        range_operators = ["gt", "gte", "lt", "lte", "between"]
        if self.operator in range_operators and not field_config.supports_range:
            raise ValueError(f"Opérateur {self.operator} non supporté pour le champ {self.field}")
        
        # Opérateur match seulement pour champs searchable
        if self.operator == "match" and not field_config.is_searchable:
            raise ValueError(f"Opérateur match non supporté pour le champ {self.field}")
        
        # Mise à jour automatique de la priorité
        self.priority = field_config.priority
        self.boost = field_config.boost
        
        return self
    
    def to_elasticsearch_filter(self) -> Dict[str, Any]:
        """Convertit en filtre Elasticsearch"""
        normalized_value = self.value.get_normalized_value()
        
        if self.operator == "eq":
            return {"term": {self.field: {"value": normalized_value, "boost": self.boost}}}
        
        elif self.operator == "ne":
            return {"bool": {"must_not": {"term": {self.field: normalized_value}}}}
        
        elif self.operator == "in":
            return {"terms": {self.field: normalized_value, "boost": self.boost}}
        
        elif self.operator == "not_in":
            return {"bool": {"must_not": {"terms": {self.field: normalized_value}}}}
        
        elif self.operator in ["gt", "gte", "lt", "lte"]:
            return {"range": {self.field: {self.operator: normalized_value}}}
        
        elif self.operator == "between":
            if not isinstance(normalized_value, list) or len(normalized_value) != 2:
                raise ValueError("between nécessite une liste de 2 valeurs")
            return {"range": {self.field: {"gte": normalized_value[0], "lte": normalized_value[1]}}}
        
        elif self.operator == "exists":
            return {"exists": {"field": self.field}}
        
        elif self.operator == "match":
            return {"match": {self.field: {"query": normalized_value, "boost": self.boost}}}
        
        elif self.operator == "prefix":
            return {"prefix": {self.field: {"value": normalized_value, "boost": self.boost}}}
        
        else:
            raise ValueError(f"Opérateur {self.operator} non implémenté")


class FilterSet(BaseModel):
    """Ensemble de filtres organisés et optimisés"""
    required_filters: List[ValidatedFilter] = Field(default_factory=list, description="Filtres obligatoires (AND)")
    optional_filters: List[ValidatedFilter] = Field(default_factory=list, description="Filtres optionnels (OR)")
    date_range: Optional[DateRangeFilter] = Field(default=None, description="Filtre de plage de dates")
    amount_range: Optional[AmountRangeFilter] = Field(default=None, description="Filtre de plage de montants")
    
    @field_validator("required_filters")
    @classmethod
    def ensure_user_id_filter(cls, v):
        """S'assure qu'un filtre user_id est présent"""
        has_user_id = any(f.field == "user_id" for f in v)
        if not has_user_id:
            raise ValueError("Un filtre user_id est obligatoire pour la sécurité")
        return v
    
    def add_required_filter(self, field: str, operator: str, value: Any):
        """Ajoute un filtre obligatoire"""
        filter_value = FilterValue(raw_value=value, field_name=field)
        validated_filter = ValidatedFilter(
            field=field,
            operator=operator,
            value=filter_value
        )
        self.required_filters.append(validated_filter)
    
    def add_optional_filter(self, field: str, operator: str, value: Any):
        """Ajoute un filtre optionnel"""
        filter_value = FilterValue(raw_value=value, field_name=field)
        validated_filter = ValidatedFilter(
            field=field,
            operator=operator,
            value=filter_value
        )
        self.optional_filters.append(validated_filter)
    
    def get_filter_priorities(self) -> Dict[FilterPriority, int]:
        """Retourne la distribution des priorités"""
        priorities = {}
        all_filters = self.required_filters + self.optional_filters
        
        for filter_obj in all_filters:
            priority = filter_obj.priority
            priorities[priority] = priorities.get(priority, 0) + 1
        
        return priorities
    
    def get_optimization_suggestions(self) -> List[str]:
        """Retourne des suggestions d'optimisation"""
        suggestions = []
        
        # Vérifier l'ordre des filtres critiques
        critical_filters = [f for f in self.required_filters if f.priority == FilterPriority.CRITICAL]
        if critical_filters and self.required_filters[0] not in critical_filters:
            suggestions.append("Placer les filtres critiques (user_id) en premier")
        
        # Vérifier la présence de filtres haute priorité
        high_priority_count = len([f for f in self.required_filters if f.priority == FilterPriority.HIGH])
        if high_priority_count == 0:
            suggestions.append("Ajouter des filtres haute priorité pour améliorer les performances")
        
        # Vérifier les filtres optionnels trop nombreux
        if len(self.optional_filters) > 5:
            suggestions.append("Réduire le nombre de filtres optionnels pour éviter la complexité")
        
        return suggestions
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convertit l'ensemble en requête Elasticsearch"""
        query = {"bool": {}}
        
        # Filtres obligatoires
        if self.required_filters:
            must_filters = [f.to_elasticsearch_filter() for f in self.required_filters]
            query["bool"]["must"] = must_filters
        
        # Filtres optionnels
        if self.optional_filters:
            should_filters = [f.to_elasticsearch_filter() for f in self.optional_filters]
            query["bool"]["should"] = should_filters
            query["bool"]["minimum_should_match"] = 1
        
        # Filtre de dates
        if self.date_range:
            start_date, end_date = self.date_range.get_date_range()
            date_filter = {"range": {"date": {"gte": start_date, "lte": end_date}}}
            
            if "must" not in query["bool"]:
                query["bool"]["must"] = []
            query["bool"]["must"].append(date_filter)
        
        # Filtre de montants
        if self.amount_range:
            amount_filter = self.amount_range.get_elasticsearch_filter()
            if amount_filter:
                if "must" not in query["bool"]:
                    query["bool"]["must"] = []
                query["bool"]["must"].append(amount_filter)
        
        return query


# === FACTORY ET HELPERS ===

class FilterFactory:
    """Factory pour créer des filtres validés"""
    
    @staticmethod
    def create_user_filter(user_id: int) -> ValidatedFilter:
        """Crée un filtre user_id obligatoire"""
        return ValidatedFilter(
            field="user_id",
            operator="eq",
            value=FilterValue(raw_value=user_id, field_name="user_id")
        )
    
    @staticmethod
    def create_category_filter(category: str) -> ValidatedFilter:
        """Crée un filtre par catégorie"""
        return ValidatedFilter(
            field="category_name.keyword",
            operator="eq",
            value=FilterValue(raw_value=category, field_name="category_name.keyword")
        )
    
    @staticmethod
    def create_merchant_filter(merchant: str) -> ValidatedFilter:
        """Crée un filtre par marchand"""
        return ValidatedFilter(
            field="merchant_name.keyword",
            operator="eq",
            value=FilterValue(raw_value=merchant, field_name="merchant_name.keyword")
        )
    
    @staticmethod
    def create_amount_range(min_amount: float = None, max_amount: float = None) -> AmountRangeFilter:
        """Crée un filtre de plage de montants"""
        return AmountRangeFilter(
            min_amount=min_amount,
            max_amount=max_amount
        )
    
    @staticmethod
    def create_date_range(period: DatePeriod = None, start_date: str = None, end_date: str = None) -> DateRangeFilter:
        """Crée un filtre de plage de dates"""
        return DateRangeFilter(
            period=period,
            start_date=start_date,
            end_date=end_date
        )


class FilterValidator:
    """Validateur avancé pour les filtres"""
    
    @staticmethod
    def validate_filter_set(filter_set: FilterSet) -> Tuple[bool, List[str]]:
        """Valide un ensemble de filtres"""
        errors = []
        
        # Vérification sécurité
        has_user_filter = any(f.field == "user_id" for f in filter_set.required_filters)
        if not has_user_filter:
            errors.append("Filtre user_id obligatoire manquant")
        
        # Vérification cohérence
        all_filters = filter_set.required_filters + filter_set.optional_filters
        field_counts = {}
        for filter_obj in all_filters:
            field_counts[filter_obj.field] = field_counts.get(filter_obj.field, 0) + 1
        
        # Avertir des doublons
        for field, count in field_counts.items():
            if count > 1:
                errors.append(f"Champ {field} filtré plusieurs fois (peut causer des conflits)")
        
        # Vérification performance
        if len(all_filters) > 10:
            errors.append("Trop de filtres peuvent impacter les performances")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def estimate_selectivity(filter_set: FilterSet) -> float:
        """Estime la sélectivité des filtres (0.0 = très sélectif, 1.0 = peu sélectif)"""
        selectivity_score = 1.0
        
        for filter_obj in filter_set.required_filters:
            # Facteur de sélectivité selon le type de champ
            field_config = FIELD_CONFIGURATIONS[filter_obj.field]
            
            if field_config.priority == FilterPriority.CRITICAL:
                selectivity_score *= 0.1  # user_id très sélectif
            elif field_config.priority == FilterPriority.HIGH:
                selectivity_score *= 0.3  # category, merchant sélectifs
            elif field_config.priority == FilterPriority.MEDIUM:
                selectivity_score *= 0.6  # amount, date modérément sélectifs
            else:
                selectivity_score *= 0.8  # currency, weekday peu sélectifs
        
        # Facteur pour les filtres de plage
        if filter_set.date_range:
            selectivity_score *= 0.5
        
        if filter_set.amount_range:
            selectivity_score *= 0.7
        
        return min(selectivity_score, 1.0)


# === EXPORTS ===

__all__ = [
    # Enums
    "FieldType",
    "FilterPriority",
    "DatePeriod",
    
    # Configuration
    "FieldConfig",
    "FIELD_CONFIGURATIONS",
    
    # Modèles
    "FilterValue",
    "DateRangeFilter",
    "AmountRangeFilter", 
    "ValidatedFilter",
    "FilterSet",
    
    # Factory et Validators
    "FilterFactory",
    "FilterValidator"
]