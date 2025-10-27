"""
Routes API pour le profilage budgétaire
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, field_validator
import logging

from budget_profiling_service.api.dependencies import get_db, get_current_user_id
from budget_profiling_service.services.transaction_service import TransactionService
from budget_profiling_service.services.fixed_charge_detector import FixedChargeDetector
from budget_profiling_service.services.budget_profiler import BudgetProfiler
from budget_profiling_service.services.user_preferences_service import UserPreferencesService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/budget", tags=["Budget Profiling"])


# ===== MODÈLES PYDANTIC =====

class ProfileResponse(BaseModel):
    """Réponse avec le profil budgétaire"""
    # Métriques de base
    user_segment: str
    behavioral_pattern: str
    avg_monthly_income: float
    avg_monthly_expenses: float
    avg_monthly_savings: float
    savings_rate: float
    fixed_charges_total: float
    semi_fixed_charges_total: float
    variable_charges_total: float
    remaining_to_live: float
    profile_completeness: float
    last_analyzed_at: str

    # Nouvelles métriques avancées
    financial_health_score: Optional[float] = None
    debt_to_income_ratio: Optional[float] = None
    expense_volatility: Optional[float] = None
    spending_trend: Optional[str] = None
    spending_trend_pct: Optional[float] = None
    prev_period_income: Optional[float] = None
    prev_period_expenses: Optional[float] = None
    income_change_pct: Optional[float] = None
    expense_change_pct: Optional[float] = None
    risk_level: Optional[str] = None
    active_alerts: Optional[List[Dict[str, Any]]] = None
    projected_annual_savings: Optional[float] = None
    months_of_expenses_saved: Optional[float] = None
    segment_details: Optional[Dict[str, Any]] = None
    behavioral_patterns: Optional[Dict[str, Any]] = None


class FixedChargeResponse(BaseModel):
    """Réponse avec une charge fixe"""
    id: int
    merchant_name: str
    category: str
    avg_amount: float
    recurrence_day: int
    recurrence_confidence: float
    validated_by_user: bool
    transaction_count: int


class MonthlyAggregateResponse(BaseModel):
    """Agrégat mensuel"""
    month: str
    total_income: float
    total_expenses: float
    net_cashflow: float
    transaction_count: int


class AnalyzeProfileRequest(BaseModel):
    """Requête pour analyser le profil avec validation renforcée"""
    months_analysis: Optional[int] = Field(
        default=None,
        description="Nombre de mois à analyser (None = toutes les transactions disponibles)"
    )

    @field_validator('months_analysis')
    @classmethod
    def validate_months(cls, v: Optional[int]) -> Optional[int]:
        """
        Valide le nombre de mois demandé

        Règles:
        - None est accepté (analyse complète)
        - Minimum: 1 mois
        - Maximum: 60 mois (5 ans)
        """
        if v is None:
            return v

        if v < 1:
            raise ValueError("months_analysis doit être au moins 1")

        if v > 60:
            raise ValueError("months_analysis ne peut pas dépasser 60 mois (5 ans)")

        return v


# ===== ENDPOINTS =====

@router.get("/profile", response_model=ProfileResponse)
def get_budget_profile(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Récupère le profil budgétaire de l'utilisateur connecté
    """
    try:
        profiler = BudgetProfiler(db)
        profile = profiler.get_user_profile(user_id)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profil budgétaire non trouvé. Lancez une analyse d'abord."
            )

        return ProfileResponse(
            # Métriques de base
            user_segment=profile.user_segment or "indéterminé",
            behavioral_pattern=profile.behavioral_pattern or "indéterminé",
            avg_monthly_income=float(profile.avg_monthly_income or 0),
            avg_monthly_expenses=float(profile.avg_monthly_expenses or 0),
            avg_monthly_savings=float(profile.avg_monthly_savings or 0),
            savings_rate=float(profile.savings_rate or 0),
            fixed_charges_total=float(profile.fixed_charges_total or 0),
            semi_fixed_charges_total=float(profile.semi_fixed_charges_total or 0),
            variable_charges_total=float(profile.variable_charges_total or 0),
            remaining_to_live=float(profile.remaining_to_live or 0),
            profile_completeness=float(profile.profile_completeness or 0),
            last_analyzed_at=profile.last_analyzed_at.isoformat() if profile.last_analyzed_at else "",
            # Nouvelles métriques avancées
            financial_health_score=float(profile.financial_health_score) if profile.financial_health_score else None,
            debt_to_income_ratio=float(profile.debt_to_income_ratio) if profile.debt_to_income_ratio else None,
            expense_volatility=float(profile.expense_volatility) if profile.expense_volatility else None,
            spending_trend=profile.spending_trend,
            spending_trend_pct=float(profile.spending_trend_pct) if profile.spending_trend_pct else None,
            prev_period_income=float(profile.prev_period_income) if profile.prev_period_income else None,
            prev_period_expenses=float(profile.prev_period_expenses) if profile.prev_period_expenses else None,
            income_change_pct=float(profile.income_change_pct) if profile.income_change_pct else None,
            expense_change_pct=float(profile.expense_change_pct) if profile.expense_change_pct else None,
            risk_level=profile.risk_level,
            active_alerts=profile.active_alerts,
            projected_annual_savings=float(profile.projected_annual_savings) if profile.projected_annual_savings else None,
            months_of_expenses_saved=float(profile.months_of_expenses_saved) if profile.months_of_expenses_saved else None,
            segment_details=profile.segment_details,
            behavioral_patterns=profile.behavioral_patterns
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération profil: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération du profil"
        )


@router.post("/profile/analyze", response_model=ProfileResponse)
def analyze_budget_profile(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Analyse et calcule le profil budgétaire de l'utilisateur

    Les paramètres d'analyse sont automatiquement récupérés depuis les préférences utilisateur.
    Pour modifier les paramètres, utilisez l'endpoint PUT /api/v1/budget/settings
    """
    try:
        logger.info(f"Analyse profil pour user {user_id}")

        # 1. Détecter les charges fixes
        detector = FixedChargeDetector(db)
        detected_charges = detector.detect_fixed_charges(user_id)

        # Sauvegarder les charges détectées
        detector.save_detected_charges(user_id, detected_charges)

        # 2. Calculer le profil budgétaire
        profiler = BudgetProfiler(db)
        profile_data = profiler.calculate_user_profile(user_id)

        # 3. Sauvegarder le profil
        profile = profiler.save_profile(user_id, profile_data)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur lors de la sauvegarde du profil"
            )

        return ProfileResponse(
            # Métriques de base
            user_segment=profile.user_segment or "indéterminé",
            behavioral_pattern=profile.behavioral_pattern or "indéterminé",
            avg_monthly_income=float(profile.avg_monthly_income or 0),
            avg_monthly_expenses=float(profile.avg_monthly_expenses or 0),
            avg_monthly_savings=float(profile.avg_monthly_savings or 0),
            savings_rate=float(profile.savings_rate or 0),
            fixed_charges_total=float(profile.fixed_charges_total or 0),
            semi_fixed_charges_total=float(profile.semi_fixed_charges_total or 0),
            variable_charges_total=float(profile.variable_charges_total or 0),
            remaining_to_live=float(profile.remaining_to_live or 0),
            profile_completeness=float(profile.profile_completeness or 0),
            last_analyzed_at=profile.last_analyzed_at.isoformat() if profile.last_analyzed_at else "",
            # Nouvelles métriques avancées
            financial_health_score=float(profile.financial_health_score) if profile.financial_health_score else None,
            debt_to_income_ratio=float(profile.debt_to_income_ratio) if profile.debt_to_income_ratio else None,
            expense_volatility=float(profile.expense_volatility) if profile.expense_volatility else None,
            spending_trend=profile.spending_trend,
            spending_trend_pct=float(profile.spending_trend_pct) if profile.spending_trend_pct else None,
            prev_period_income=float(profile.prev_period_income) if profile.prev_period_income else None,
            prev_period_expenses=float(profile.prev_period_expenses) if profile.prev_period_expenses else None,
            income_change_pct=float(profile.income_change_pct) if profile.income_change_pct else None,
            expense_change_pct=float(profile.expense_change_pct) if profile.expense_change_pct else None,
            risk_level=profile.risk_level,
            active_alerts=profile.active_alerts,
            projected_annual_savings=float(profile.projected_annual_savings) if profile.projected_annual_savings else None,
            months_of_expenses_saved=float(profile.months_of_expenses_saved) if profile.months_of_expenses_saved else None,
            segment_details=profile.segment_details,
            behavioral_patterns=profile.behavioral_patterns
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur analyse profil: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'analyse du profil"
        )


@router.get("/fixed-charges", response_model=List[FixedChargeResponse])
def get_fixed_charges(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Récupère les charges fixes détectées
    """
    try:
        detector = FixedChargeDetector(db)
        charges = detector.get_user_fixed_charges(user_id, active_only=True)

        return [
            FixedChargeResponse(
                id=charge.id,
                merchant_name=charge.merchant_name or "",
                category=charge.category or "",
                avg_amount=float(charge.avg_amount or 0),
                recurrence_day=charge.recurrence_day or 0,
                recurrence_confidence=float(charge.recurrence_confidence or 0),
                validated_by_user=charge.validated_by_user or False,
                transaction_count=charge.transaction_count or 0
            )
            for charge in charges
        ]

    except Exception as e:
        logger.error(f"Erreur récupération charges fixes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des charges fixes"
        )


@router.get("/monthly-aggregates", response_model=List[MonthlyAggregateResponse])
def get_monthly_aggregates(
    months: int = 3,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Récupère les agrégats mensuels (revenus, dépenses)
    """
    try:
        transaction_service = TransactionService(db)
        aggregates = transaction_service.get_monthly_aggregates(user_id, months)

        return [
            MonthlyAggregateResponse(**agg)
            for agg in aggregates
        ]

    except Exception as e:
        logger.error(f"Erreur récupération agrégats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des agrégats"
        )


@router.get("/category-breakdown")
def get_category_breakdown(
    months: int = 3,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
) -> Dict[str, float]:
    """
    Récupère la répartition des dépenses par catégorie
    """
    try:
        transaction_service = TransactionService(db)
        breakdown = transaction_service.get_category_breakdown(user_id, months)

        return breakdown

    except Exception as e:
        logger.error(f"Erreur récupération breakdown: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération du breakdown"
        )


# ===== ENDPOINTS GESTION DES PARAMÈTRES =====

@router.get("/settings")
def get_budget_settings(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Récupère les paramètres budgétaires de l'utilisateur

    Retourne tous les paramètres de profiling avec les valeurs par défaut si non personnalisés.
    """
    try:
        preferences_service = UserPreferencesService(db)
        settings = preferences_service.get_budget_settings(user_id)

        return {
            "settings": settings,
            "message": "Paramètres récupérés avec succès"
        }

    except Exception as e:
        logger.error(f"Erreur récupération paramètres: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des paramètres"
        )


@router.put("/settings")
def update_budget_settings(
    new_settings: Dict[str, Any],
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Met à jour les paramètres budgétaires de l'utilisateur

    La mise à jour est partielle : seuls les paramètres fournis sont modifiés,
    les autres conservent leur valeur actuelle.

    Body exemple:
    ```json
    {
        "months_analysis": 18,
        "fixed_charge_detection": {
            "min_occurrences": 6,
            "max_amount_variance_pct": 25.0
        }
    }
    ```
    """
    try:
        preferences_service = UserPreferencesService(db)
        success, updated_settings, errors = preferences_service.update_budget_settings(
            user_id,
            new_settings,
            partial=True
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Validation des paramètres échouée", "errors": errors}
            )

        return {
            "settings": updated_settings,
            "message": "Paramètres mis à jour avec succès"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur mise à jour paramètres: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la mise à jour des paramètres"
        )


@router.post("/settings/reset")
def reset_budget_settings(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Réinitialise les paramètres budgétaires aux valeurs par défaut

    Cette action est irréversible.
    """
    try:
        preferences_service = UserPreferencesService(db)
        success = preferences_service.reset_to_defaults(user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur lors de la réinitialisation des paramètres"
            )

        # Récupérer les paramètres réinitialisés
        settings = preferences_service.get_budget_settings(user_id)

        return {
            "settings": settings,
            "message": "Paramètres réinitialisés aux valeurs par défaut"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur réinitialisation paramètres: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la réinitialisation des paramètres"
        )
