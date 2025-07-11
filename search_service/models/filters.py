"""
Modèles de filtres complexes pour la recherche.

Ce module définit les filtres avancés utilisés pour affiner
les recherches de transactions financières.
"""
from typing import Optional, List, Union, Any
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
from search_service.models.search_types import FilterOperator


class AmountFilter(BaseModel):
    """Filtre pour les montants de transaction."""
    operator: FilterOperator
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    currency: Optional[str] = Field(default="EUR", description="Code devise ISO")
    
    @validator("value")
    def validate_single_value(cls, v, values):
        """Valide que value est fournie pour les opérateurs simples."""
        operator = values.get("operator")
        if operator in [FilterOperator.EQ, FilterOperator.GT, FilterOperator.GTE, 
                       FilterOperator.LT, FilterOperator.LTE] and v is None:
            raise ValueError(f"value required for operator {operator}")
        return v
    
    @validator("max_value")
    def validate_range_values(cls, v, values):
        """Valide que min_value et max_value sont fournis pour BETWEEN."""
        operator = values.get("operator")
        min_val = values.get("min_value")
        
        if operator == FilterOperator.BETWEEN:
            if min_val is None or v is None:
                raise ValueError("min_value and max_value required for BETWEEN operator")
            if min_val >= v:
                raise ValueError("min_value must be less than max_value")
        return v


class DateFilter(BaseModel):
    """Filtre pour les dates de transaction."""
    operator: FilterOperator
    value: Optional[Union[datetime, date, str]] = None
    start_date: Optional[Union[datetime, date, str]] = None
    end_date: Optional[Union[datetime, date, str]] = None
    
    @validator("value", "start_date", "end_date", pre=True)
    def parse_date_strings(cls, v):
        """Parse les chaînes de dates en objets datetime."""
        if isinstance(v, str):
            try:
                # Essayer plusieurs formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"]:
                    try:
                        return datetime.strptime(v, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Invalid date format: {v}")
            except ValueError:
                raise ValueError(f"Unable to parse date: {v}")
        return v
    
    @validator("end_date")
    def validate_date_range(cls, v, values):
        """Valide la cohérence des plages de dates."""
        operator = values.get("operator")
        start_date = values.get("start_date")
        
        if operator == FilterOperator.BETWEEN:
            if start_date is None or v is None:
                raise ValueError("start_date and end_date required for BETWEEN operator")
            if start_date >= v:
                raise ValueError("start_date must be before end_date")
        return v


class CategoryFilter(BaseModel):
    """Filtre pour les catégories de transaction."""
    operator: FilterOperator = FilterOperator.IN
    category_ids: Optional[List[int]] = Field(default=None, description="IDs de catégories")
    operation_types: Optional[List[str]] = Field(default=None, description="Types d'opération")
    exclude_categories: Optional[List[int]] = Field(default=None, description="Catégories à exclure")
    
    @validator("category_ids", "operation_types")
    def validate_lists_not_empty(cls, v):
        """Valide que les listes ne sont pas vides si fournies."""
        if v is not None and len(v) == 0:
            raise ValueError("Filter lists cannot be empty")
        return v


class MerchantFilter(BaseModel):
    """Filtre pour les marchands."""
    operator: FilterOperator = FilterOperator.IN
    merchant_names: Optional[List[str]] = Field(default=None, description="Noms de marchands exacts")
    merchant_contains: Optional[str] = Field(default=None, description="Texte contenu dans le marchand")
    exclude_merchants: Optional[List[str]] = Field(default=None, description="Marchands à exclure")


class TextFilter(BaseModel):
    """Filtre pour la recherche textuelle avancée."""
    required_terms: Optional[List[str]] = Field(default=None, description="Termes obligatoires")
    excluded_terms: Optional[List[str]] = Field(default=None, description="Termes à exclure")
    phrase_match: Optional[str] = Field(default=None, description="Correspondance de phrase exacte")
    fuzzy_search: bool = Field(default=True, description="Recherche floue activée")
    minimum_should_match: Optional[str] = Field(default="2<75%", description="Minimum de termes à correspondre")


class GeographicFilter(BaseModel):
    """Filtre géographique (pour les transactions avec géolocalisation)."""
    country_code: Optional[str] = Field(default=None, description="Code pays ISO")
    city: Optional[str] = Field(default=None, description="Ville")
    postal_code: Optional[str] = Field(default=None, description="Code postal")
    radius_km: Optional[float] = Field(default=None, description="Rayon en kilomètres")
    latitude: Optional[float] = Field(default=None, description="Latitude centre")
    longitude: Optional[float] = Field(default=None, description="Longitude centre")
    
    @validator("radius_km")
    def validate_radius_with_coords(cls, v, values):
        """Valide que les coordonnées sont fournies avec le rayon."""
        if v is not None:
            lat = values.get("latitude")
            lon = values.get("longitude")
            if lat is None or lon is None:
                raise ValueError("latitude and longitude required when radius_km is specified")
        return v


class AdvancedFilters(BaseModel):
    """Regroupement de tous les filtres avancés."""
    amount: Optional[AmountFilter] = None
    date: Optional[DateFilter] = None
    category: Optional[CategoryFilter] = None
    merchant: Optional[MerchantFilter] = None
    text: Optional[TextFilter] = None
    geographic: Optional[GeographicFilter] = None
    
    # Filtres simples supplémentaires
    transaction_types: Optional[List[str]] = Field(default=None, description="Types de transaction")
    account_ids: Optional[List[int]] = Field(default=None, description="IDs de comptes")
    has_attachments: Optional[bool] = Field(default=None, description="A des pièces jointes")
    is_recurring: Optional[bool] = Field(default=None, description="Transaction récurrente")
    
    def has_filters(self) -> bool:
        """Vérifie si au moins un filtre est défini."""
        return any([
            self.amount is not None,
            self.date is not None,
            self.category is not None,
            self.merchant is not None,
            self.text is not None,
            self.geographic is not None,
            self.transaction_types is not None,
            self.account_ids is not None,
            self.has_attachments is not None,
            self.is_recurring is not None
        ])
    
    def to_elasticsearch_query(self) -> dict:
        """Convertit les filtres en requête Elasticsearch."""
        must_clauses = []
        must_not_clauses = []
        
        # Filtre de montant
        if self.amount:
            amount_clause = self._build_amount_clause(self.amount)
            if amount_clause:
                must_clauses.append(amount_clause)
        
        # Filtre de date
        if self.date:
            date_clause = self._build_date_clause(self.date)
            if date_clause:
                must_clauses.append(date_clause)
        
        # Filtre de catégorie
        if self.category:
            category_clauses = self._build_category_clauses(self.category)
            must_clauses.extend(category_clauses.get("must", []))
            must_not_clauses.extend(category_clauses.get("must_not", []))
        
        # Filtre de marchand
        if self.merchant:
            merchant_clauses = self._build_merchant_clauses(self.merchant)
            must_clauses.extend(merchant_clauses.get("must", []))
            must_not_clauses.extend(merchant_clauses.get("must_not", []))
        
        # Filtres simples
        if self.transaction_types:
            must_clauses.append({"terms": {"transaction_type": self.transaction_types}})
        
        if self.account_ids:
            must_clauses.append({"terms": {"account_id": self.account_ids}})
        
        if self.has_attachments is not None:
            must_clauses.append({"term": {"has_attachments": self.has_attachments}})
        
        if self.is_recurring is not None:
            must_clauses.append({"term": {"is_recurring": self.is_recurring}})
        
        # Construire la requête finale
        query = {"bool": {}}
        if must_clauses:
            query["bool"]["must"] = must_clauses
        if must_not_clauses:
            query["bool"]["must_not"] = must_not_clauses
        
        return query
    
    def _build_amount_clause(self, amount_filter: AmountFilter) -> dict:
        """Construit une clause pour le filtre de montant."""
        if amount_filter.operator == FilterOperator.EQ:
            return {"term": {"amount": amount_filter.value}}
        elif amount_filter.operator == FilterOperator.GT:
            return {"range": {"amount": {"gt": amount_filter.value}}}
        elif amount_filter.operator == FilterOperator.GTE:
            return {"range": {"amount": {"gte": amount_filter.value}}}
        elif amount_filter.operator == FilterOperator.LT:
            return {"range": {"amount": {"lt": amount_filter.value}}}
        elif amount_filter.operator == FilterOperator.LTE:
            return {"range": {"amount": {"lte": amount_filter.value}}}
        elif amount_filter.operator == FilterOperator.BETWEEN:
            return {"range": {"amount": {"gte": amount_filter.min_value, "lte": amount_filter.max_value}}}
        return {}
    
    def _build_date_clause(self, date_filter: DateFilter) -> dict:
        """Construit une clause pour le filtre de date."""
        if date_filter.operator == FilterOperator.EQ:
            date_str = date_filter.value.strftime("%Y-%m-%d") if hasattr(date_filter.value, 'strftime') else str(date_filter.value)
            return {"term": {"date": date_str}}
        elif date_filter.operator == FilterOperator.BETWEEN:
            start_str = date_filter.start_date.strftime("%Y-%m-%d") if hasattr(date_filter.start_date, 'strftime') else str(date_filter.start_date)
            end_str = date_filter.end_date.strftime("%Y-%m-%d") if hasattr(date_filter.end_date, 'strftime') else str(date_filter.end_date)
            return {"range": {"date": {"gte": start_str, "lte": end_str}}}
        return {}
    
    def _build_category_clauses(self, category_filter: CategoryFilter) -> dict:
        """Construit les clauses pour le filtre de catégorie."""
        clauses = {"must": [], "must_not": []}
        
        if category_filter.category_ids:
            clauses["must"].append({"terms": {"category_id": category_filter.category_ids}})
        
        if category_filter.operation_types:
            clauses["must"].append({"terms": {"operation_type": category_filter.operation_types}})
        
        if category_filter.exclude_categories:
            clauses["must_not"].append({"terms": {"category_id": category_filter.exclude_categories}})
        
        return clauses
    
    def _build_merchant_clauses(self, merchant_filter: MerchantFilter) -> dict:
        """Construit les clauses pour le filtre de marchand."""
        clauses = {"must": [], "must_not": []}
        
        if merchant_filter.merchant_names:
            clauses["must"].append({"terms": {"merchant_name.keyword": merchant_filter.merchant_names}})
        
        if merchant_filter.merchant_contains:
            clauses["must"].append({"wildcard": {"merchant_name": f"*{merchant_filter.merchant_contains}*"}})
        
        if merchant_filter.exclude_merchants:
            clauses["must_not"].append({"terms": {"merchant_name.keyword": merchant_filter.exclude_merchants}})
        
        return clauses