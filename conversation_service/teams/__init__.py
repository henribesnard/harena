"""Package des équipes pour le service de conversation.

Ce module fournit un point d'entrée unique pour accéder aux différentes
équipes disponibles. Actuellement, seule `FinancialAnalysisTeam` est
exposée publiquement.
"""

from __future__ import annotations

try:
    from .financial_analysis_team import FinancialAnalysisTeam  # pragma: no cover
except ModuleNotFoundError:  # pragma: no cover - impl optionnelle absente
    FinancialAnalysisTeam = None  # type: ignore[assignment]

__all__ = ["FinancialAnalysisTeam"]
