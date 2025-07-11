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

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Literal, Type
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator, model_validator
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
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

class DateRange(str, Enum):
    """Plages de dates prédéfinies."""
    TODAY = "today"
    YESTERDAY = "yesterday"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_3_MONTHS = "last_3_months"
    LAST_6_MONTHS = "last_6_months"
    LAST_YEAR = "last_year"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_YEAR = "this_year"
    CUSTOM = "custom"

# ==================== MODÈLES DE BASE ====================

class BaseFilter(BaseModel, ABC):
    """
    Classe de base pour tous les filtres.
    
    Définit l'interface commune et les validations de base
    pour tous les types de filtres.
    """
    filter_type: FilterType = Field(..., description="Type de filtre")
    field: str = Field(..., description="Champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    boost: float = Field(default=1.0, ge=0.1, le=10.0, description="Facteur de boost")
    required: bool = Field(default=True, description="Filtre obligatoire ou optionnel")
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Valide que le champ est autorisé."""
        allowed_fields = getattr(settings, 'ALLOWED_FILTER_FIELDS', [])
        if allowed_fields and v not in allowed_fields:
            raise ValueError(f"Champ de filtre non autorisé: {v}")
        return v
    
    @field_validator('boost')
    @classmethod
    def validate_boost(cls, v):
        """Valide le facteur de boost."""
        if v <= 0 or v > 10:
            raise ValueError("Le boost doit être entre 0.1 et 10")
        return v
    
    @abstractmethod
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convertit le filtre en requête Elasticsearch."""
        pass
    
    @abstractmethod
    def validate_value(self, value: Any) -> bool:
        """Valide la valeur du filtre selon son type."""
        pass

class SimpleFilter(BaseFilter):
    """
    Filtre simple avec une seule valeur.
    
    Supporte les opérateurs d'égalité, comparaison et existence.
    """
    value: Union[str, int, float, bool] = Field(..., description="Valeur du filtre")
    case_sensitive: bool = Field(default=False, description="Sensible à la casse (texte)")
    
    def validate_value(self, value: Any) -> bool:
        """Valide la valeur selon le type de filtre."""
        if self.filter_type == FilterType.USER:
            return isinstance(value, int) and value > 0
        elif self.filter_type == FilterType.AMOUNT:
            return isinstance(value, (int, float)) and value >= 0
        elif self.filter_type == FilterType.TEXT:
            return isinstance(value, str) and len(value.strip()) > 0
        return True
    
    @model_validator(mode='after')
    def validate_simple_filter(self):
        """Valide la cohérence du filtre simple."""
        # Validation de la valeur
        if not self.validate_value(self.value):
            raise ValueError(f"Valeur invalide pour le type {self.filter_type}")
        
        # Validation de l'opérateur selon le type
        numeric_types = {FilterType.AMOUNT, FilterType.USER}
        text_types = {FilterType.TEXT, FilterType.CATEGORY, FilterType.MERCHANT}
        
        if self.filter_type in numeric_types:
            allowed_ops = {FilterOperator.EQ, FilterOperator.NE, FilterOperator.GT, 
                          FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE}
            if self.operator not in allowed_ops:
                raise ValueError(f"Opérateur {self.operator} non supporté pour {self.filter_type}")
        
        if self.filter_type in text_types and self.operator == FilterOperator.BETWEEN:
            raise ValueError("BETWEEN non supporté pour les filtres texte")
        
        return self
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        if self.operator == FilterOperator.EQ:
            return {"term": {self.field: {"value": self.value, "boost": self.boost}}}
        
        elif self.operator == FilterOperator.NE:
            return {"bool": {"must_not": {"term": {self.field: self.value}}}}
        
        elif self.operator in [FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE]:
            range_op = self.operator.value
            return {"range": {self.field: {range_op: self.value, "boost": self.boost}}}
        
        elif self.operator == FilterOperator.EXISTS:
            return {"exists": {"field": self.field}}
        
        elif self.operator == FilterOperator.NOT_EXISTS:
            return {"bool": {"must_not": {"exists": {"field": self.field}}}}
        
        elif self.operator == FilterOperator.CONTAINS:
            if self.case_sensitive:
                return {"wildcard": {self.field: {"value": f"*{self.value}*", "boost": self.boost}}}
            else:
                return {"match": {self.field: {"query": self.value, "boost": self.boost}}}
        
        elif self.operator == FilterOperator.STARTS_WITH:
            return {"prefix": {self.field: {"value": self.value, "boost": self.boost}}}
        
        elif self.operator == FilterOperator.REGEX:
            return {"regexp": {self.field: {"value": self.value, "boost": self.boost}}}
        
        else:
            raise ValueError(f"Opérateur non supporté: {self.operator}")

class ListFilter(BaseFilter):
    """
    Filtre avec liste de valeurs.
    
    Supporte les opérateurs IN et NOT_IN pour filtrer
    sur plusieurs valeurs.
    """
    values: List[Union[str, int, float]] = Field(..., min_length=1, description="Liste de valeurs")
    
    @field_validator('values')
    @classmethod
    def validate_values_list(cls, v):
        """Valide la liste de valeurs."""
        if len(v) > settings.MAX_FILTER_VALUES:
            raise ValueError(f"Trop de valeurs (max {settings.MAX_FILTER_VALUES})")
        
        # Vérifier que toutes les valeurs sont du même type
        if len(set(type(val) for val in v)) > 1:
            raise ValueError("Toutes les valeurs doivent être du même type")
        
        return v
    
    def validate_value(self, value: Any) -> bool:
        """Valide chaque valeur de la liste."""
        return all(
            SimpleFilter(
                filter_type=self.filter_type,
                field=self.field,
                operator=FilterOperator.EQ,
                value=val
            ).validate_value(val) for val in self.values
        )
    
    @model_validator(mode='after')
    def validate_list_filter(self):
        """Valide le filtre de liste."""
        if self.operator not in [FilterOperator.IN, FilterOperator.NOT_IN]:
            raise ValueError("ListFilter supporte uniquement IN et NOT_IN")
        
        if not self.validate_value(self.values):
            raise ValueError("Certaines valeurs de la liste sont invalides")
        
        return self
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        if self.operator == FilterOperator.IN:
            return {"terms": {self.field: self.values, "boost": self.boost}}
        
        elif self.operator == FilterOperator.NOT_IN:
            return {"bool": {"must_not": {"terms": {self.field: self.values}}}}
        
        else:
            raise ValueError(f"Opérateur non supporté pour ListFilter: {self.operator}")

class RangeFilter(BaseFilter):
    """
    Filtre de plage pour valeurs numériques ou dates.
    
    Supporte les plages avec bornes inclusives ou exclusives.
    """
    min_value: Optional[Union[int, float, datetime, date]] = Field(None, description="Valeur minimale")
    max_value: Optional[Union[int, float, datetime, date]] = Field(None, description="Valeur maximale")
    include_min: bool = Field(default=True, description="Inclure la valeur minimale")
    include_max: bool = Field(default=True, description="Inclure la valeur maximale")
    
    def validate_value(self, value: Any) -> bool:
        """Valide les valeurs de la plage."""
        if self.filter_type == FilterType.AMOUNT:
            return all(isinstance(v, (int, float)) and v >= 0 for v in [self.min_value, self.max_value] if v is not None)
        elif self.filter_type == FilterType.DATE:
            return all(isinstance(v, (datetime, date)) for v in [self.min_value, self.max_value] if v is not None)
        return True
    
    @model_validator(mode='after')
    def validate_range_filter(self):
        """Valide le filtre de plage."""
        if self.operator != FilterOperator.BETWEEN:
            raise ValueError("RangeFilter requiert l'opérateur BETWEEN")
        
        if self.min_value is None and self.max_value is None:
            raise ValueError("Au moins une borne doit être définie")
        
        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise ValueError("min_value doit être inférieur à max_value")
        
        if not self.validate_value(None):
            raise ValueError("Valeurs de plage invalides")
        
        return self
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        range_query = {"boost": self.boost}
        
        if self.min_value is not None:
            op = "gte" if self.include_min else "gt"
            range_query[op] = self.min_value
        
        if self.max_value is not None:
            op = "lte" if self.include_max else "lt"
            range_query[op] = self.max_value
        
        return {"range": {self.field: range_query}}

class DateFilter(RangeFilter):
    """
    Filtre de date spécialisé avec plages prédéfinies.
    
    Support des plages courantes (aujourd'hui, cette semaine, etc.)
    et des plages personnalisées.
    """
    date_range: Optional[DateRange] = Field(None, description="Plage de date prédéfinie")
    timezone: str = Field(default="UTC", description="Fuseau horaire")
    
    @model_validator(mode='after')
    def validate_date_filter(self):
        """Valide le filtre de date."""
        self.filter_type = FilterType.DATE
        
        if self.date_range and self.date_range != DateRange.CUSTOM:
            # Calculer les bornes selon la plage prédéfinie
            self._set_predefined_range()
        elif self.date_range == DateRange.CUSTOM:
            if self.min_value is None and self.max_value is None:
                raise ValueError("Bornes requises pour une plage personnalisée")
        
        return self
    
    def _set_predefined_range(self):
        """Définit les bornes selon la plage prédéfinie."""
        now = datetime.now()
        today = now.date()
        
        if self.date_range == DateRange.TODAY:
            self.min_value = datetime.combine(today, datetime.min.time())
            self.max_value = datetime.combine(today, datetime.max.time())
        
        elif self.date_range == DateRange.YESTERDAY:
            yesterday = today - timedelta(days=1)
            self.min_value = datetime.combine(yesterday, datetime.min.time())
            self.max_value = datetime.combine(yesterday, datetime.max.time())
        
        elif self.date_range == DateRange.LAST_7_DAYS:
            self.min_value = datetime.combine(today - timedelta(days=7), datetime.min.time())
            self.max_value = datetime.combine(today, datetime.max.time())
        
        elif self.date_range == DateRange.LAST_30_DAYS:
            self.min_value = datetime.combine(today - timedelta(days=30), datetime.min.time())
            self.max_value = datetime.combine(today, datetime.max.time())
        
        elif self.date_range == DateRange.THIS_MONTH:
            self.min_value = datetime.combine(today.replace(day=1), datetime.min.time())
            self.max_value = datetime.combine(today, datetime.max.time())
        
        # Ajouter d'autres plages selon les besoins

# ==================== GROUPES DE FILTRES ====================

class FilterGroup(BaseModel):
    """
    Groupe de filtres avec logique de combinaison.
    
    Permet de combiner plusieurs filtres avec AND, OR ou NOT.
    """
    logic: FilterLogic = Field(default=FilterLogic.AND, description="Logique de combinaison")
    required: List[Union[SimpleFilter, ListFilter, RangeFilter, DateFilter]] = Field(
        default_factory=list, description="Filtres obligatoires"
    )
    optional: List[Union[SimpleFilter, ListFilter, RangeFilter, DateFilter]] = Field(
        default_factory=list, description="Filtres optionnels"
    )
    exclusions: List[Union[SimpleFilter, ListFilter, RangeFilter, DateFilter]] = Field(
        default_factory=list, description="Filtres d'exclusion"
    )
    
    @model_validator(mode='after')
    def validate_filter_group(self):
        """Valide le groupe de filtres."""
        total_filters = len(self.required) + len(self.optional) + len(self.exclusions)
        
        if total_filters == 0:
            raise ValueError("Un groupe doit contenir au moins un filtre")
        
        max_filters = getattr(settings, 'MAX_FILTERS_PER_GROUP', 50)
        if total_filters > max_filters:
            raise ValueError(f"Trop de filtres dans le groupe (max {max_filters})")
        
        # Validation sécurité: user_id obligatoire
        user_filters = [f for f in self.required if f.field == "user_id"]
        if not user_filters:
            raise ValueError("Filtre user_id obligatoire pour la sécurité")
        
        return self
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convertit le groupe en requête Elasticsearch bool."""
        bool_query = {}
        
        if self.required:
            if self.logic == FilterLogic.AND:
                bool_query["must"] = [f.to_elasticsearch_query() for f in self.required]
            else:  # OR
                bool_query["should"] = [f.to_elasticsearch_query() for f in self.required]
                bool_query["minimum_should_match"] = 1
        
        if self.optional:
            bool_query["should"] = bool_query.get("should", []) + [
                f.to_elasticsearch_query() for f in self.optional
            ]
        
        if self.exclusions:
            bool_query["must_not"] = [f.to_elasticsearch_query() for f in self.exclusions]
        
        return {"bool": bool_query}

# ==================== FACTORY FUNCTIONS ====================

def create_user_filter(user_id: int) -> SimpleFilter:
    """Crée un filtre utilisateur pour la sécurité."""
    return SimpleFilter(
        filter_type=FilterType.USER,
        field="user_id",
        operator=FilterOperator.EQ,
        value=user_id,
        required=True
    )

class UserFilter(SimpleFilter):
    """Filtre spécialisé pour les utilisateurs avec validation de sécurité."""
    filter_type: Literal[FilterType.USER] = FilterType.USER
    field: Literal["user_id"] = "user_id"
    operator: Literal[FilterOperator.EQ] = FilterOperator.EQ
    value: PositiveInt = Field(..., description="ID de l'utilisateur")
    required: Literal[True] = True
    
    def __init__(self, user_id: int, **kwargs):
        """Initialise le filtre utilisateur."""
        super().__init__(
            filter_type=FilterType.USER,
            field="user_id",
            operator=FilterOperator.EQ,
            value=user_id,
            required=True,
            **kwargs
        )
    
    def validate_value(self, value: Any) -> bool:
        """Valide l'ID utilisateur."""
        return isinstance(value, int) and value > 0
    
    @model_validator(mode='after')
    def validate_user_filter(self):
        """Valide le filtre utilisateur."""
        if self.value <= 0:
            raise ValueError("L'ID utilisateur doit être positif")
        
        if self.field != "user_id":
            raise ValueError("Le champ doit être 'user_id'")
        
        if self.operator != FilterOperator.EQ:
            raise ValueError("L'opérateur doit être EQ pour les filtres utilisateur")
        
        return self

def create_amount_range_filter(
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    field: str = "amount"
) -> RangeFilter:
    """Crée un filtre de plage de montant."""
    return RangeFilter(
        filter_type=FilterType.AMOUNT,
        field=field,
        operator=FilterOperator.BETWEEN,
        min_value=min_amount,
        max_value=max_amount,
        include_min=True,
        include_max=True
    )

def create_date_range_filter(
    date_range: DateRange,
    field: str = "transaction_date",
    custom_min: Optional[datetime] = None,
    custom_max: Optional[datetime] = None
) -> DateFilter:
    """Crée un filtre de plage de date."""
    return DateFilter(
        filter_type=FilterType.DATE,
        field=field,
        operator=FilterOperator.BETWEEN,
        date_range=date_range,
        min_value=custom_min if date_range == DateRange.CUSTOM else None,
        max_value=custom_max if date_range == DateRange.CUSTOM else None
    )

def create_category_filter(categories: List[str]) -> ListFilter:
    """Crée un filtre de catégories."""
    return ListFilter(
        filter_type=FilterType.CATEGORY,
        field="category",
        operator=FilterOperator.IN,
        values=categories
    )

def create_merchant_filter(merchant: str, exact_match: bool = True) -> Union[SimpleFilter, SimpleFilter]:
    """Crée un filtre de commerçant."""
    if exact_match:
        return SimpleFilter(
            filter_type=FilterType.MERCHANT,
            field="merchant",
            operator=FilterOperator.EQ,
            value=merchant
        )
    else:
        return SimpleFilter(
            filter_type=FilterType.MERCHANT,
            field="merchant",
            operator=FilterOperator.CONTAINS,
            value=merchant,
            case_sensitive=False
        )

def create_text_search_filter(
    text: str,
    field: str = "description",
    operator: FilterOperator = FilterOperator.CONTAINS
) -> SimpleFilter:
    """Crée un filtre de recherche textuelle."""
    return SimpleFilter(
        filter_type=FilterType.TEXT,
        field=field,
        operator=operator,
        value=text,
        case_sensitive=False
    )

def create_secure_filter_group(
    user_id: int,
    additional_filters: Optional[List[BaseFilter]] = None
) -> FilterGroup:
    """
    Crée un groupe de filtres sécurisé avec user_id obligatoire.
    
    Args:
        user_id: ID de l'utilisateur
        additional_filters: Filtres supplémentaires optionnels
    
    Returns:
        FilterGroup sécurisé
    """
    required_filters = [create_user_filter(user_id)]
    
    if additional_filters:
        # Séparer les filtres selon leur type
        for filter_obj in additional_filters:
            if filter_obj.required:
                required_filters.append(filter_obj)
    
    optional_filters = []
    if additional_filters:
        optional_filters = [f for f in additional_filters if not f.required]
    
    return FilterGroup(
        logic=FilterLogic.AND,
        required=required_filters,
        optional=optional_filters
    )

# ==================== UTILITAIRES DE VALIDATION ====================

class FilterValidator:
    """Validateur spécialisé pour les filtres."""
    
    @staticmethod
    def validate_filter_security(filter_group: FilterGroup) -> bool:
        """
        Valide la sécurité d'un groupe de filtres.
        
        Args:
            filter_group: Groupe à valider
        
        Returns:
            True si sécurisé
        
        Raises:
            ValueError: Si validation échoue
        """
        # Vérifier la présence du filtre user_id
        user_filters = [f for f in filter_group.required if f.field == "user_id"]
        if not user_filters:
            raise ValueError("Filtre user_id obligatoire pour la sécurité")
        
        # Vérifier qu'il n'y a qu'un seul filtre user_id
        if len(user_filters) > 1:
            raise ValueError("Un seul filtre user_id autorisé")
        
        user_filter = user_filters[0]
        if user_filter.operator != FilterOperator.EQ:
            raise ValueError("Filtre user_id doit utiliser l'opérateur EQ")
        
        if not isinstance(user_filter.value, int) or user_filter.value <= 0:
            raise ValueError("user_id doit être un entier positif")
        
        return True
    
    @staticmethod
    def validate_filter_performance(filter_group: FilterGroup) -> List[str]:
        """
        Valide l'impact performance d'un groupe de filtres.
        
        Args:
            filter_group: Groupe à analyser
        
        Returns:
            Liste des avertissements de performance
        """
        warnings = []
        
        # Compter les filtres par type
        total_filters = len(filter_group.required) + len(filter_group.optional) + len(filter_group.exclusions)
        
        if total_filters > 20:
            warnings.append("Beaucoup de filtres peuvent impacter les performances")
        
        # Vérifier les filtres texte avec regex
        for filter_list in [filter_group.required, filter_group.optional]:
            for f in filter_list:
                if isinstance(f, SimpleFilter) and f.operator == FilterOperator.REGEX:
                    warnings.append("Les filtres regex peuvent être lents")
                
                if isinstance(f, SimpleFilter) and f.operator == FilterOperator.CONTAINS:
                    if isinstance(f.value, str) and len(f.value) < 3:
                        warnings.append("Les recherches texte courtes peuvent être lentes")
        
        # Vérifier les plages de dates trop larges
        for filter_list in [filter_group.required, filter_group.optional]:
            for f in filter_list:
                if isinstance(f, DateFilter):
                    if f.min_value and f.max_value:
                        delta = f.max_value - f.min_value
                        if hasattr(delta, 'days') and delta.days > 365:
                            warnings.append("Plage de dates très large peut impacter les performances")
        
        return warnings
    
    @staticmethod
    def optimize_filter_group(filter_group: FilterGroup) -> FilterGroup:
        """
        Optimise un groupe de filtres pour de meilleures performances.
        
        Args:
            filter_group: Groupe à optimiser
        
        Returns:
            Groupe optimisé
        """
        # Réorganiser les filtres par sélectivité (plus sélectifs en premier)
        def get_selectivity_score(f: BaseFilter) -> int:
            """Score de sélectivité (plus bas = plus sélectif)."""
            if f.field == "user_id":
                return 1  # Très sélectif
            elif f.filter_type == FilterType.USER:
                return 1
            elif f.filter_type in [FilterType.CATEGORY, FilterType.MERCHANT]:
                return 2
            elif f.filter_type == FilterType.AMOUNT:
                return 3
            elif f.filter_type == FilterType.DATE:
                return 4
            elif f.filter_type == FilterType.TEXT:
                return 5  # Moins sélectif
            return 6
        
        # Trier les filtres requis par sélectivité
        optimized_required = sorted(filter_group.required, key=get_selectivity_score)
        
        return FilterGroup(
            logic=filter_group.logic,
            required=optimized_required,
            optional=filter_group.optional,
            exclusions=filter_group.exclusions
        )

# ==================== CONVERSIONS ET UTILITAIRES ====================

def filter_group_to_elasticsearch(filter_group: FilterGroup) -> Dict[str, Any]:
    """
    Convertit un groupe de filtres en requête Elasticsearch complète.
    
    Args:
        filter_group: Groupe de filtres
    
    Returns:
        Requête Elasticsearch optimisée
    """
    base_query = filter_group.to_elasticsearch_query()
    
    # Ajouter des optimisations
    if "bool" in base_query:
        # Optimiser l'ordre des clauses bool
        bool_clause = base_query["bool"]
        
        # Les filtres de terme (plus rapides) en premier dans must
        if "must" in bool_clause:
            must_clauses = bool_clause["must"]
            term_clauses = [c for c in must_clauses if "term" in c]
            other_clauses = [c for c in must_clauses if "term" not in c]
            bool_clause["must"] = term_clauses + other_clauses
    
    return base_query

def elasticsearch_to_filter_group(es_query: Dict[str, Any]) -> FilterGroup:
    """
    Convertit une requête Elasticsearch en groupe de filtres (parsing inverse).
    
    Args:
        es_query: Requête Elasticsearch
    
    Returns:
        FilterGroup équivalent
    
    Note:
        Fonction de commodité pour le debugging et la conversion.
        Supporte uniquement les requêtes bool simples.
    """
    if "bool" not in es_query:
        raise ValueError("Seules les requêtes bool sont supportées")
    
    bool_query = es_query["bool"]
    required = []
    optional = []
    exclusions = []
    
    # Parsing basique des clauses term et range
    for must_clause in bool_query.get("must", []):
        if "term" in must_clause:
            field = list(must_clause["term"].keys())[0]
            term_data = must_clause["term"][field]
            value = term_data["value"] if isinstance(term_data, dict) else term_data
            
            required.append(SimpleFilter(
                filter_type=FilterType.CUSTOM,
                field=field,
                operator=FilterOperator.EQ,
                value=value
            ))
    
    # Parsing similaire pour should et must_not...
    # (Implémentation simplifiée)
    
    return FilterGroup(
        logic=FilterLogic.AND,
        required=required,
        optional=optional,
        exclusions=exclusions
    )

# ==================== EXPORTS ====================

__all__ = [
    # Enums
    'FilterType',
    'FilterOperator', 
    'FilterLogic',
    'DateRange',
    
    # Classes de base
    'BaseFilter',
    'SimpleFilter',
    'ListFilter',
    'RangeFilter',
    'DateFilter',
    'FilterGroup',
    'UserFilter',
    
    # Factory functions
    'create_user_filter',
    'create_amount_range_filter',
    'create_date_range_filter',
    'create_category_filter',
    'create_merchant_filter',
    'create_text_search_filter',
    'create_secure_filter_group',
    
    # Validation et utilitaires
    'FilterValidator',
    'filter_group_to_elasticsearch',
    'elasticsearch_to_filter_group'
]