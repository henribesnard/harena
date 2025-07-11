"""
🔧 Modèles Filtres Search Service - Filtrage Avancé
==================================================

Modèles spécialisés pour le filtrage des transactions financières. Ces modèles
étendent les filtres de base avec des fonctionnalités spécifiques au domaine financier.

Responsabilités:
- Filtres spécialisés transactions financières
- Validation domaine financier
- Optimisations performance filtrage
- Filtres composés et complexes
- Mapping vers requêtes Elasticsearch
"""

from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum
import re

from .service_contracts import SearchFilter, FilterOperator


# =============================================================================
# 🎯 ÉNUMÉRATIONS FILTRES FINANCIERS
# =============================================================================

class AmountFilterType(str, Enum):
    """Types de filtres montant."""
    SIGNED = "signed"           # Montant avec signe (-/+)
    ABSOLUTE = "absolute"       # Valeur absolue
    DEBIT_ONLY = "debit_only"   # Seulement débits (négatifs)
    CREDIT_ONLY = "credit_only" # Seulement crédits (positifs)

class DateFilterType(str, Enum):
    """Types de filtres date."""
    EXACT_DATE = "exact_date"
    DATE_RANGE = "date_range"
    RELATIVE_DAYS = "relative_days"
    MONTH_YEAR = "month_year"
    WEEKDAY = "weekday"
    QUARTER = "quarter"
    YEAR = "year"

class TransactionType(str, Enum):
    """Types de transactions."""
    DEBIT = "debit"
    CREDIT = "credit"
    TRANSFER = "transfer"
    REFUND = "refund"
    FEE = "fee"

class OperationType(str, Enum):
    """Types d'opérations."""
    CARD_PAYMENT = "card_payment"
    TRANSFER = "transfer"
    DIRECT_DEBIT = "direct_debit"
    CHECK = "check"
    CASH_WITHDRAWAL = "cash_withdrawal"
    DEPOSIT = "deposit"
    ONLINE_PAYMENT = "online_payment"

class CurrencyCode(str, Enum):
    """Codes devises supportées."""
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    CAD = "CAD"
    JPY = "JPY"


# =============================================================================
# 🔍 FILTRES DE BASE SPÉCIALISÉS
# =============================================================================

class UserIsolationFilter(BaseModel):
    """Filtre isolation utilisateur (obligatoire sécurité)."""
    user_id: int = Field(..., ge=1, description="ID utilisateur")
    
    def to_search_filter(self) -> SearchFilter:
        """Conversion vers SearchFilter standard."""
        return SearchFilter(
            field="user_id",
            operator=FilterOperator.EQ,
            value=self.user_id
        )
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Validation ID utilisateur."""
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v

class AmountFilter(BaseModel):
    """Filtre spécialisé montants financiers."""
    amount_type: AmountFilterType = Field(default=AmountFilterType.ABSOLUTE, description="Type montant")
    min_amount: Optional[Decimal] = Field(None, description="Montant minimum")
    max_amount: Optional[Decimal] = Field(None, description="Montant maximum")
    exact_amount: Optional[Decimal] = Field(None, description="Montant exact")
    tolerance: Optional[Decimal] = Field(None, ge=0, description="Tolérance montant exact")
    currency: Optional[CurrencyCode] = Field(None, description="Code devise")
    
    @validator('max_amount')
    def validate_amount_range(cls, v, values):
        """Validation cohérence plage montants."""
        min_amount = values.get('min_amount')
        if min_amount is not None and v is not None and v < min_amount:
            raise ValueError("max_amount must be greater than min_amount")
        return v
    
    @model_validator(mode='after')
    def validate_amount_specification(self):
        """Validation spécification montant."""
        exact = self.exact_amount
        min_amt = self.min_amount
        max_amt = self.max_amount
        
        # Ne peut pas avoir exact_amount ET plage
        if exact is not None and (min_amt is not None or max_amt is not None):
            raise ValueError("Cannot specify both exact_amount and min/max range")
        
        # Doit avoir au moins une spécification
        if exact is None and min_amt is None and max_amt is None:
            raise ValueError("Must specify at least one amount condition")
        
        return self
    
    def to_search_filters(self) -> List[SearchFilter]:
        """Conversion vers SearchFilters Elasticsearch."""
        filters = []
        
        # Déterminer le champ selon le type
        field_name = "amount" if self.amount_type == AmountFilterType.SIGNED else "amount_abs"
        
        # Filtres selon type transaction
        if self.amount_type == AmountFilterType.DEBIT_ONLY:
            filters.append(SearchFilter(field="amount", operator=FilterOperator.LT, value=0))
        elif self.amount_type == AmountFilterType.CREDIT_ONLY:
            filters.append(SearchFilter(field="amount", operator=FilterOperator.GT, value=0))
        
        # Montant exact avec tolérance
        if self.exact_amount is not None:
            if self.tolerance and self.tolerance > 0:
                min_val = float(self.exact_amount - self.tolerance)
                max_val = float(self.exact_amount + self.tolerance)
                filters.append(SearchFilter(
                    field=field_name,
                    operator=FilterOperator.BETWEEN,
                    value=[min_val, max_val]
                ))
            else:
                filters.append(SearchFilter(
                    field=field_name,
                    operator=FilterOperator.EQ,
                    value=float(self.exact_amount)
                ))
        
        # Plage montants
        if self.min_amount is not None:
            filters.append(SearchFilter(
                field=field_name,
                operator=FilterOperator.GTE,
                value=float(self.min_amount)
            ))
        
        if self.max_amount is not None:
            filters.append(SearchFilter(
                field=field_name,
                operator=FilterOperator.LTE,
                value=float(self.max_amount)
            ))
        
        # Filtre devise
        if self.currency:
            filters.append(SearchFilter(
                field="currency_code",
                operator=FilterOperator.EQ,
                value=self.currency.value
            ))
        
        return filters

class DateFilter(BaseModel):
    """Filtre spécialisé dates."""
    filter_type: DateFilterType = Field(..., description="Type filtre date")
    exact_date: Optional[date] = Field(None, description="Date exacte")
    start_date: Optional[date] = Field(None, description="Date début")
    end_date: Optional[date] = Field(None, description="Date fin")
    relative_days: Optional[int] = Field(None, ge=1, le=3650, description="Jours relatifs (depuis aujourd'hui)")
    month_year: Optional[str] = Field(None, description="Mois-année (YYYY-MM)")
    weekday: Optional[str] = Field(None, description="Jour semaine")
    quarter: Optional[int] = Field(None, ge=1, le=4, description="Trimestre")
    year: Optional[int] = Field(None, ge=2000, le=2100, description="Année")
    
    @validator('month_year')
    def validate_month_year(cls, v):
        """Validation format mois-année."""
        if v and not re.match(r'^\d{4}-\d{2}$', v):
            raise ValueError("month_year must be in YYYY-MM format")
        return v
    
    @validator('weekday')
    def validate_weekday(cls, v):
        """Validation jour semaine."""
        if v:
            valid_weekdays = {
                "monday", "tuesday", "wednesday", "thursday", 
                "friday", "saturday", "sunday"
            }
            if v.lower() not in valid_weekdays:
                raise ValueError(f"weekday must be one of {valid_weekdays}")
        return v.lower() if v else v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validation plage dates."""
        start_date = values.get('start_date')
        if start_date and v and v < start_date:
            raise ValueError("end_date must be after start_date")
        return v
    
    @model_validator(mode='after')
    def validate_date_specification(self):
        """Validation spécification date selon type."""
        filter_type = self.filter_type
        
        if filter_type == DateFilterType.EXACT_DATE:
            if not self.exact_date:
                raise ValueError("exact_date required for EXACT_DATE filter")
        
        elif filter_type == DateFilterType.DATE_RANGE:
            if not self.start_date or not self.end_date:
                raise ValueError("start_date and end_date required for DATE_RANGE filter")
        
        elif filter_type == DateFilterType.RELATIVE_DAYS:
            if not self.relative_days:
                raise ValueError("relative_days required for RELATIVE_DAYS filter")
        
        elif filter_type == DateFilterType.MONTH_YEAR:
            if not self.month_year:
                raise ValueError("month_year required for MONTH_YEAR filter")
        
        elif filter_type == DateFilterType.WEEKDAY:
            if not self.weekday:
                raise ValueError("weekday required for WEEKDAY filter")
        
        elif filter_type == DateFilterType.QUARTER:
            if not self.quarter or not self.year:
                raise ValueError("quarter and year required for QUARTER filter")
        
        elif filter_type == DateFilterType.YEAR:
            if not self.year:
                raise ValueError("year required for YEAR filter")
        
        return self
    
    def to_search_filters(self) -> List[SearchFilter]:
        """Conversion vers SearchFilters."""
        filters = []
        
        if self.filter_type == DateFilterType.EXACT_DATE:
            filters.append(SearchFilter(
                field="date",
                operator=FilterOperator.EQ,
                value=self.exact_date.isoformat()
            ))
        
        elif self.filter_type == DateFilterType.DATE_RANGE:
            filters.append(SearchFilter(
                field="date",
                operator=FilterOperator.BETWEEN,
                value=[self.start_date.isoformat(), self.end_date.isoformat()]
            ))
        
        elif self.filter_type == DateFilterType.RELATIVE_DAYS:
            end_date = date.today()
            start_date = end_date - timedelta(days=self.relative_days)
            filters.append(SearchFilter(
                field="date",
                operator=FilterOperator.BETWEEN,
                value=[start_date.isoformat(), end_date.isoformat()]
            ))
        
        elif self.filter_type == DateFilterType.MONTH_YEAR:
            filters.append(SearchFilter(
                field="month_year",
                operator=FilterOperator.EQ,
                value=self.month_year
            ))
        
        elif self.filter_type == DateFilterType.WEEKDAY:
            filters.append(SearchFilter(
                field="weekday",
                operator=FilterOperator.EQ,
                value=self.weekday.title()
            ))
        
        elif self.filter_type == DateFilterType.QUARTER:
            # Calcul dates trimestre
            quarter_start_month = (self.quarter - 1) * 3 + 1
            start_date = date(self.year, quarter_start_month, 1)
            
            if quarter_start_month + 2 <= 12:
                end_month = quarter_start_month + 2
                end_year = self.year
            else:
                end_month = (quarter_start_month + 2) % 12
                end_year = self.year + 1
            
            # Dernier jour du trimestre
            if end_month in [1, 3, 5, 7, 8, 10, 12]:
                end_day = 31
            elif end_month in [4, 6, 9, 11]:
                end_day = 30
            else:  # Février
                end_day = 29 if end_year % 4 == 0 else 28
            
            end_date = date(end_year, end_month, end_day)
            
            filters.append(SearchFilter(
                field="date",
                operator=FilterOperator.BETWEEN,
                value=[start_date.isoformat(), end_date.isoformat()]
            ))
        
        elif self.filter_type == DateFilterType.YEAR:
            start_date = date(self.year, 1, 1)
            end_date = date(self.year, 12, 31)
            filters.append(SearchFilter(
                field="date",
                operator=FilterOperator.BETWEEN,
                value=[start_date.isoformat(), end_date.isoformat()]
            ))
        
        return filters


# =============================================================================
# 🏪 FILTRES CATÉGORIES ET MARCHANDS
# =============================================================================

class CategoryFilter(BaseModel):
    """Filtre spécialisé catégories."""
    categories: List[str] = Field(..., min_items=1, description="Liste catégories")
    exact_match: bool = Field(default=True, description="Correspondance exacte")
    include_subcategories: bool = Field(default=False, description="Inclure sous-catégories")
    exclude_categories: Optional[List[str]] = Field(None, description="Catégories à exclure")
    
    @validator('categories')
    def validate_categories(cls, v):
        """Validation noms catégories."""
        if not v:
            raise ValueError("At least one category required")
        
        # Normalisation
        normalized = []
        for cat in v:
            if not cat or not cat.strip():
                continue
            normalized.append(cat.strip().lower())
        
        if not normalized:
            raise ValueError("At least one valid category required")
        
        return normalized
    
    @validator('exclude_categories')
    def validate_exclude_categories(cls, v):
        """Validation catégories à exclure."""
        if v:
            return [cat.strip().lower() for cat in v if cat and cat.strip()]
        return v
    
    def to_search_filters(self) -> List[SearchFilter]:
        """Conversion vers SearchFilters."""
        filters = []
        
        # Filtre inclusion catégories
        if len(self.categories) == 1:
            operator = FilterOperator.EQ if self.exact_match else FilterOperator.IN
            filters.append(SearchFilter(
                field="category_name",
                operator=operator,
                value=self.categories[0] if self.exact_match else self.categories
            ))
        else:
            filters.append(SearchFilter(
                field="category_name",
                operator=FilterOperator.IN,
                value=self.categories
            ))
        
        # Filtre exclusion catégories
        if self.exclude_categories:
            filters.append(SearchFilter(
                field="category_name",
                operator=FilterOperator.NOT_IN,
                value=self.exclude_categories
            ))
        
        return filters

class MerchantFilter(BaseModel):
    """Filtre spécialisé marchands."""
    merchants: List[str] = Field(..., min_items=1, description="Liste marchands")
    exact_match: bool = Field(default=False, description="Correspondance exacte")
    case_sensitive: bool = Field(default=False, description="Sensible à la casse")
    exclude_merchants: Optional[List[str]] = Field(None, description="Marchands à exclure")
    merchant_pattern: Optional[str] = Field(None, description="Pattern nom marchand")
    
    @validator('merchants')
    def validate_merchants(cls, v):
        """Validation noms marchands."""
        if not v:
            raise ValueError("At least one merchant required")
        
        normalized = []
        for merchant in v:
            if not merchant or not merchant.strip():
                continue
            normalized.append(merchant.strip())
        
        if not normalized:
            raise ValueError("At least one valid merchant required")
        
        return normalized
    
    @validator('merchant_pattern')
    def validate_merchant_pattern(cls, v):
        """Validation pattern marchand."""
        if v:
            # Validation pattern regex basique
            try:
                re.compile(v)
            except re.error:
                raise ValueError("Invalid merchant pattern regex")
        return v
    
    def to_search_filters(self) -> List[SearchFilter]:
        """Conversion vers SearchFilters."""
        filters = []
        
        # Normalisation selon case sensitivity
        merchants = self.merchants
        if not self.case_sensitive:
            merchants = [m.upper() for m in merchants]
        
        # Filtre inclusion marchands
        if self.exact_match:
            if len(merchants) == 1:
                filters.append(SearchFilter(
                    field="merchant_name.keyword",
                    operator=FilterOperator.EQ,
                    value=merchants[0]
                ))
            else:
                filters.append(SearchFilter(
                    field="merchant_name.keyword",
                    operator=FilterOperator.IN,
                    value=merchants
                ))
        else:
            # Recherche textuelle pour correspondance partielle
            # Sera géré par le moteur de recherche textuelle
            pass
        
        # Filtre exclusion marchands
        if self.exclude_merchants:
            exclude_list = self.exclude_merchants
            if not self.case_sensitive:
                exclude_list = [m.upper() for m in exclude_list]
            
            filters.append(SearchFilter(
                field="merchant_name.keyword",
                operator=FilterOperator.NOT_IN,
                value=exclude_list
            ))
        
        return filters


# =============================================================================
# 🔧 FILTRES COMPOSÉS ET COMPLEXES
# =============================================================================

class TransactionTypeFilter(BaseModel):
    """Filtre types de transactions."""
    transaction_types: List[TransactionType] = Field(..., min_items=1, description="Types transactions")
    operation_types: Optional[List[OperationType]] = Field(None, description="Types opérations")
    exclude_types: Optional[List[TransactionType]] = Field(None, description="Types à exclure")
    
    def to_search_filters(self) -> List[SearchFilter]:
        """Conversion vers SearchFilters."""
        filters = []
        
        # Filtre types transactions
        if len(self.transaction_types) == 1:
            filters.append(SearchFilter(
                field="transaction_type",
                operator=FilterOperator.EQ,
                value=self.transaction_types[0].value
            ))
        else:
            filters.append(SearchFilter(
                field="transaction_type",
                operator=FilterOperator.IN,
                value=[t.value for t in self.transaction_types]
            ))
        
        # Filtre types opérations
        if self.operation_types:
            if len(self.operation_types) == 1:
                filters.append(SearchFilter(
                    field="operation_type",
                    operator=FilterOperator.EQ,
                    value=self.operation_types[0].value
                ))
            else:
                filters.append(SearchFilter(
                    field="operation_type",
                    operator=FilterOperator.IN,
                    value=[t.value for t in self.operation_types]
                ))
        
        # Filtre exclusion types
        if self.exclude_types:
            filters.append(SearchFilter(
                field="transaction_type",
                operator=FilterOperator.NOT_IN,
                value=[t.value for t in self.exclude_types]
            ))
        
        return filters

class CompositeFilter(BaseModel):
    """Filtre composé combinant plusieurs filtres."""
    user_isolation: UserIsolationFilter = Field(..., description="Isolation utilisateur (obligatoire)")
    amount_filter: Optional[AmountFilter] = Field(None, description="Filtre montants")
    date_filter: Optional[DateFilter] = Field(None, description="Filtre dates")
    category_filter: Optional[CategoryFilter] = Field(None, description="Filtre catégories")
    merchant_filter: Optional[MerchantFilter] = Field(None, description="Filtre marchands")
    transaction_type_filter: Optional[TransactionTypeFilter] = Field(None, description="Filtre types transactions")
    custom_filters: List[SearchFilter] = Field(default_factory=list, description="Filtres personnalisés")
    
    def to_search_filters(self) -> Dict[str, List[SearchFilter]]:
        """Conversion vers SearchFilters groupés."""
        filter_groups = {
            "required": [],
            "optional": [],
            "ranges": []
        }
        
        # Filtre utilisateur obligatoire
        filter_groups["required"].append(self.user_isolation.to_search_filter())
        
        # Filtres montants
        if self.amount_filter:
            amount_filters = self.amount_filter.to_search_filters()
            filter_groups["required"].extend(amount_filters)
        
        # Filtres dates
        if self.date_filter:
            date_filters = self.date_filter.to_search_filters()
            # Les filtres de plage vont dans ranges, les autres dans required
            for f in date_filters:
                if f.operator == FilterOperator.BETWEEN:
                    filter_groups["ranges"].append(f)
                else:
                    filter_groups["required"].append(f)
        
        # Filtres catégories
        if self.category_filter:
            category_filters = self.category_filter.to_search_filters()
            filter_groups["required"].extend(category_filters)
        
        # Filtres marchands
        if self.merchant_filter:
            merchant_filters = self.merchant_filter.to_search_filters()
            filter_groups["required"].extend(merchant_filters)
        
        # Filtres types transactions
        if self.transaction_type_filter:
            type_filters = self.transaction_type_filter.to_search_filters()
            filter_groups["required"].extend(type_filters)
        
        # Filtres personnalisés
        filter_groups["optional"].extend(self.custom_filters)
        
        return filter_groups


# =============================================================================
# 🎯 FILTRES SPÉCIALISÉS FINANCIERS
# =============================================================================

class RecurringTransactionFilter(BaseModel):
    """Filtre transactions récurrentes."""
    min_frequency_days: int = Field(default=30, ge=1, description="Fréquence minimum (jours)")
    max_frequency_days: int = Field(default=90, ge=1, description="Fréquence maximum (jours)")
    min_occurrences: int = Field(default=3, ge=2, description="Occurrences minimum")
    amount_tolerance_percent: float = Field(default=5.0, ge=0, le=50, description="Tolérance montant (%)")
    same_merchant: bool = Field(default=True, description="Même marchand requis")
    same_category: bool = Field(default=True, description="Même catégorie requise")
    
    @validator('max_frequency_days')
    def validate_frequency_range(cls, v, values):
        """Validation plage fréquence."""
        min_freq = values.get('min_frequency_days')
        if min_freq and v < min_freq:
            raise ValueError("max_frequency_days must be >= min_frequency_days")
        return v

class SuspiciousActivityFilter(BaseModel):
    """Filtre activités suspectes."""
    unusual_amount_threshold: float = Field(default=3.0, ge=1.0, description="Seuil montant inhabituel (écarts-types)")
    unusual_merchant: bool = Field(default=True, description="Marchand inhabituel")
    unusual_time: bool = Field(default=True, description="Horaire inhabituel")
    unusual_location: bool = Field(default=False, description="Localisation inhabituelle")
    high_frequency_threshold: int = Field(default=10, ge=1, description="Seuil haute fréquence (transactions/jour)")
    amount_spike_threshold: float = Field(default=5.0, ge=1.0, description="Seuil pic montant")

class BudgetFilter(BaseModel):
    """Filtre analyse budget."""
    budget_period: str = Field(..., description="Période budget")
    categories_included: Optional[List[str]] = Field(None, description="Catégories incluses")
    categories_excluded: Optional[List[str]] = Field(None, description="Catégories exclues")
    budget_limits: Dict[str, float] = Field(default_factory=dict, description="Limites budget par catégorie")
    variance_threshold: float = Field(default=10.0, ge=0, description="Seuil variance (%)")
    
    @validator('budget_period')
    def validate_budget_period(cls, v):
        """Validation période budget."""
        valid_periods = {
            "current_month", "current_quarter", "current_year",
            "last_month", "last_quarter", "last_year"
        }
        if v not in valid_periods:
            raise ValueError(f"budget_period must be one of {valid_periods}")
        return v


# =============================================================================
# 🛠️ BUILDER ET FACTORY FILTRES
# =============================================================================

class FilterBuilder:
    """Builder pour construire filtres complexes."""
    
    def __init__(self, user_id: int):
        """Initialisation avec user_id obligatoire."""
        self.user_isolation = UserIsolationFilter(user_id=user_id)
        self._amount_filter = None
        self._date_filter = None
        self._category_filter = None
        self._merchant_filter = None
        self._transaction_type_filter = None
        self._custom_filters = []
    
    def with_amount_range(self, min_amount: float = None, max_amount: float = None, 
                         amount_type: AmountFilterType = AmountFilterType.ABSOLUTE) -> 'FilterBuilder':
        """Ajouter filtre plage montants."""
        self._amount_filter = AmountFilter(
            amount_type=amount_type,
            min_amount=Decimal(str(min_amount)) if min_amount else None,
            max_amount=Decimal(str(max_amount)) if max_amount else None
        )
        return self
    
    def with_exact_amount(self, amount: float, tolerance: float = None, 
                         amount_type: AmountFilterType = AmountFilterType.ABSOLUTE) -> 'FilterBuilder':
        """Ajouter filtre montant exact."""
        self._amount_filter = AmountFilter(
            amount_type=amount_type,
            exact_amount=Decimal(str(amount)),
            tolerance=Decimal(str(tolerance)) if tolerance else None
        )
        return self
    
    def with_date_range(self, start_date: date, end_date: date) -> 'FilterBuilder':
        """Ajouter filtre plage dates."""
        self._date_filter = DateFilter(
            filter_type=DateFilterType.DATE_RANGE,
            start_date=start_date,
            end_date=end_date
        )
        return self
    
    def with_last_days(self, days: int) -> 'FilterBuilder':
        """Ajouter filtre derniers N jours."""
        self._date_filter = DateFilter(
            filter_type=DateFilterType.RELATIVE_DAYS,
            relative_days=days
        )
        return self
    
    def with_month_year(self, month_year: str) -> 'FilterBuilder':
        """Ajouter filtre mois-année."""
        self._date_filter = DateFilter(
            filter_type=DateFilterType.MONTH_YEAR,
            month_year=month_year
        )
        return self
    
    def with_categories(self, categories: List[str], exact_match: bool = True) -> 'FilterBuilder':
        """Ajouter filtre catégories."""
        self._category_filter = CategoryFilter(
            categories=categories,
            exact_match=exact_match
        )
        return self
    
    def with_merchants(self, merchants: List[str], exact_match: bool = False) -> 'FilterBuilder':
        """Ajouter filtre marchands."""
        self._merchant_filter = MerchantFilter(
            merchants=merchants,
            exact_match=exact_match
        )
        return self
    
    def with_transaction_types(self, transaction_types: List[TransactionType]) -> 'FilterBuilder':
        """Ajouter filtre types transactions."""
        self._transaction_type_filter = TransactionTypeFilter(
            transaction_types=transaction_types
        )
        return self
    
    def with_custom_filter(self, field: str, operator: FilterOperator, value: Any) -> 'FilterBuilder':
        """Ajouter filtre personnalisé."""
        custom_filter = SearchFilter(field=field, operator=operator, value=value)
        self._custom_filters.append(custom_filter)
        return self
    
    def build(self) -> CompositeFilter:
        """Construire filtre composite."""
        return CompositeFilter(
            user_isolation=self.user_isolation,
            amount_filter=self._amount_filter,
            date_filter=self._date_filter,
            category_filter=self._category_filter,
            merchant_filter=self._merchant_filter,
            transaction_type_filter=self._transaction_type_filter,
            custom_filters=self._custom_filters
        )

class FilterFactory:
    """Factory pour créer filtres prédéfinis."""
    
    @staticmethod
    def create_category_filter(user_id: int, category: str) -> CompositeFilter:
        """Créer filtre simple par catégorie."""
        return FilterBuilder(user_id).with_categories([category]).build()
    
    @staticmethod
    def create_merchant_filter(user_id: int, merchant: str) -> CompositeFilter:
        """Créer filtre simple par marchand."""
        return FilterBuilder(user_id).with_merchants([merchant]).build()
    
    @staticmethod
    def create_amount_range_filter(user_id: int, min_amount: float, max_amount: float) -> CompositeFilter:
        """Créer filtre plage montants."""
        return FilterBuilder(user_id).with_amount_range(min_amount, max_amount).build()
    
    @staticmethod
    def create_last_month_filter(user_id: int) -> CompositeFilter:
        """Créer filtre mois dernier."""
        today = date.today()
        if today.month == 1:
            last_month = f"{today.year - 1}-12"
        else:
            last_month = f"{today.year}-{today.month - 1:02d}"
        
        return FilterBuilder(user_id).with_month_year(last_month).build()
    
    @staticmethod
    def create_high_amount_filter(user_id: int, threshold: float = 100.0) -> CompositeFilter:
        """Créer filtre montants élevés."""
        return FilterBuilder(user_id).with_amount_range(min_amount=threshold).build()
    
    @staticmethod
    def create_debit_only_filter(user_id: int) -> CompositeFilter:
        """Créer filtre débits uniquement."""
        return FilterBuilder(user_id).with_transaction_types([TransactionType.DEBIT]).build()


# =============================================================================
# 📋 EXPORTS
# =============================================================================

__all__ = [
    # Énumérations
    "AmountFilterType", "DateFilterType", "TransactionType", "OperationType", "CurrencyCode",
    # Filtres de base
    "UserIsolationFilter", "AmountFilter", "DateFilter", "CategoryFilter", "MerchantFilter",
    # Filtres composés
    "TransactionTypeFilter", "CompositeFilter",
    # Filtres spécialisés
    "RecurringTransactionFilter", "SuspiciousActivityFilter", "BudgetFilter",
    # Builder et Factory
    "FilterBuilder", "FilterFactory",
]