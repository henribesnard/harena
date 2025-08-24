"""Schémas de requêtes pour le service de recherche."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class SearchRequest(BaseModel):
    """Modèle de requête unifié pour le search service"""
    
    # Obligatoire
    user_id: int = Field(..., description="ID utilisateur (obligatoire pour sécurité)", gt=0)
    query: str = Field(default="", description="Requête textuelle libre")
    
    # Optionnel - Filtres
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Filtres additionnels (term, range, terms)"
    )
    
    # Optionnel - Paramètres de pagination
    page: int = Field(default=1, description="Index de page (1 = première)", ge=1)
    page_size: int = Field(
        default=100,
        description="Nombre de résultats par page",
        ge=1,
        le=1000,
    )
    offset: int = Field(
        default=0,
        description="Décalage calculé pour pagination",
        ge=0,
    )
    
    # Optionnel - Métadonnées
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Métadonnées pour debug et contexte"
    )

    aggregations: Optional[Dict[str, Any]] = Field(
        default=None, description="Requête d'agrégation optionnelle"
    )

    highlight: Optional[Dict[str, Any]] = Field(
        default=None, description="Paramètres de surlignage optionnels"
    )

    aggregation_only: bool = Field(
        default=False,
        description="Si vrai, seuls les résultats d'agrégations sont renvoyés",
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_limit(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Supporte l'ancien champ ``limit`` comme alias de ``page_size``."""
        if "page_size" not in values and "limit" in values:
            values["page_size"] = values.pop("limit")
        return values

    @model_validator(mode="after")
    def _compute_offset(self) -> "SearchRequest":
        """Calcule l'offset à partir de la page si non fourni."""
        if "offset" not in self.model_fields_set:
            self.offset = (self.page - 1) * self.page_size
        return self

    @property
    def limit(self) -> int:
        """Alias historique pour ``page_size``."""
        return self.page_size

    @limit.setter
    def limit(self, value: int) -> None:
        self.page_size = value

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validation et nettoyage de la requête"""
        if v is None:
            return ""
        return v.strip()
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validation basique des filtres"""
        if v is None:
            return {}
        
        # Validation des types de filtres supportés (basé sur votre schéma)
        allowed_fields = {
            'category_name', 'merchant_name', 'operation_type',
            'currency_code', 'transaction_type', 'month_year',
            'weekday', 'amount', 'amount_abs', 'date', 'account_id',
            'primary_description', 'searchable_text',
            'account_name', 'account_type', 'account_balance', 'account_currency'
        }
        
        for field in v.keys():
            if field not in allowed_fields:
                # On log un warning mais on ne bloque pas pour la flexibilité
                pass
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 34,
                "query": "restaurant italien",
                "filters": {
                    "category_name": "restaurant",
                    "amount": {"gte": -100, "lte": 0},
                    "date": {"gte": "2024-01-01", "lte": "2024-01-31"}
                },
                "page": 1,
                "page_size": 10,
                "metadata": {"debug": True},
                "aggregations": {
                    "by_category": {"terms": {"field": "category_name"}}
                },
                "highlight": {"fields": {"primary_description": {}}},
                "aggregation_only": False
            }
        }
    )
