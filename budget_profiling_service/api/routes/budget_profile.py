"""
Routes API pour le profilage budgétaire
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging

from budget_profiling_service.api.dependencies import get_db, get_current_user_id
from budget_profiling_service.services.transaction_service import TransactionService
from budget_profiling_service.services.fixed_charge_detector import FixedChargeDetector
from budget_profiling_service.services.budget_profiler import BudgetProfiler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/budget", tags=["Budget Profiling"])


# ===== MODÈLES PYDANTIC =====

class ProfileResponse(BaseModel):
    """Réponse avec le profil budgétaire"""
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
    """Requête pour analyser le profil"""
    months_analysis: Optional[int] = Field(
        default=None,
        ge=1,
        description="Nombre de mois à analyser (None = toutes les transactions disponibles)"
    )


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
            last_analyzed_at=profile.last_analyzed_at.isoformat() if profile.last_analyzed_at else ""
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
    request: AnalyzeProfileRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Analyse et calcule le profil budgétaire de l'utilisateur
    """
    try:
        months = request.months_analysis
        log_msg = f"TOUTES les transactions" if months is None else f"{months} mois"
        logger.info(f"Analyse profil pour user {user_id} sur {log_msg}")

        # 1. Détecter les charges fixes
        detector = FixedChargeDetector(db)
        # Pour la détection, doubler la période si spécifiée, sinon None (tout)
        detection_months = (months * 2) if months is not None else None
        detected_charges = detector.detect_fixed_charges(
            user_id,
            months_back=detection_months
        )

        # Sauvegarder les charges détectées
        detector.save_detected_charges(user_id, detected_charges)

        # 2. Calculer le profil budgétaire
        profiler = BudgetProfiler(db)
        profile_data = profiler.calculate_user_profile(
            user_id,
            months_analysis=months
        )

        # 3. Sauvegarder le profil
        profile = profiler.save_profile(user_id, profile_data)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur lors de la sauvegarde du profil"
            )

        return ProfileResponse(
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
            last_analyzed_at=profile.last_analyzed_at.isoformat() if profile.last_analyzed_at else ""
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
