from __future__ import annotations

from decimal import Decimal
from datetime import date as Date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class FlexibleFinancialTransaction(BaseModel):
    """Représente une transaction financière extraite d'une entrée LLM."""

    user_id: int = Field(
        ..., description="Identifiant utilisateur, ex: 'pour l'utilisateur 42'", gt=0
    )
    transaction_id: Optional[int] = Field(
        default=None, description="Identifiant de transaction, ex: 'tx 991'"
    )
    amount: Decimal = Field(
        ..., description="Montant de la transaction, ex: '23,45 EUR'"
    )
    currency_code: str = Field(
        default="EUR", description="Devise utilisée, ex: 'EUR'"
    )
    date: Date = Field(
        ..., description="Date de la transaction, ex: '2024-01-30'"
    )
    description: Optional[str] = Field(
        default=None, description="Description libre, ex: 'achat café'"
    )


class DynamicSpendingAnalysis(BaseModel):
    """Analyse dynamique des dépenses pour un utilisateur."""

    user_id: int = Field(
        ..., description="Identifiant utilisateur, ex: 'utilisateur 42'", gt=0
    )
    period_start: Optional[Date] = Field(
        default=None, description="Début de période, ex: '2024-01-01'"
    )
    period_end: Optional[Date] = Field(
        default=None, description="Fin de période, ex: '2024-01-31'"
    )
    total_spent: Decimal = Field(
        default=Decimal("0"), description="Total des dépenses, ex: '152.33'"
    )
    category_totals: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Dépenses par catégorie, ex: {'restaurant': '45.00'}",
    )
    transactions: List[FlexibleFinancialTransaction] = Field(
        default_factory=list,
        description="Transactions analysées, ex: [<FlexibleFinancialTransaction>]",
    )


class FlexibleSearchCriteria(BaseModel):
    """Critères de recherche flexibles interprétés depuis du texte libre."""

    user_id: int = Field(
        ..., description="Identifiant utilisateur, ex: 'utilisateur 42'", gt=0
    )
    query: str = Field(
        default="",
        description="Requête textuelle, ex: 'dépenses restaurants janvier'",
    )
    min_amount: Optional[Decimal] = Field(
        default=None, description="Montant minimum, ex: '-50'"
    )
    max_amount: Optional[Decimal] = Field(
        default=None, description="Montant maximum, ex: '0'"
    )
    start_date: Optional[Date] = Field(
        default=None, description="Date de début, ex: '2024-01-01'"
    )
    end_date: Optional[Date] = Field(
        default=None, description="Date de fin, ex: '2024-01-31'"
    )
    categories: Optional[List[str]] = Field(
        default=None, description="Catégories ciblées, ex: ['restaurant', 'courses']"
    )
    merchant: Optional[str] = Field(
        default=None, description="Nom du commerçant, ex: 'Amazon'"
    )

    @model_validator(mode="after")
    def check_dates(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class LLMExtractedInsights(BaseModel):
    """Contient les informations extraites par le LLM."""

    criteria: FlexibleSearchCriteria = Field(
        ..., description="Critères utilisés pour l'analyse, ex: {...}"
    )
    analysis: Optional[DynamicSpendingAnalysis] = Field(
        default=None, description="Résultat d'analyse, ex: {...}"
    )
    notes: Optional[str] = Field(
        default=None, description="Commentaire libre, ex: 'Tu dépenses beaucoup en cafés'"
    )


__all__ = [
    "FlexibleFinancialTransaction",
    "DynamicSpendingAnalysis",
    "FlexibleSearchCriteria",
    "LLMExtractedInsights",
]
