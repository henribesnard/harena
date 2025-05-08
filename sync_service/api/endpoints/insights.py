"""
Endpoints pour les insights financiers.

Ce module expose les endpoints pour accéder aux insights financiers
fournis par Bridge API et stockés dans la base de données.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

from user_service.db.session import get_db
from user_service.api.deps import get_current_active_user
from user_service.models.user import User
from sync_service.models.sync import BridgeInsight
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_insights(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère les insights financiers pour l'utilisateur.
    
    Returns:
        Dict: Insights financiers
    """
    # Récupérer les insights de l'utilisateur
    insights = db.query(BridgeInsight).filter(
        BridgeInsight.user_id == current_user.id
    ).first()
    
    if not insights:
        return {
            "status": "warning",
            "message": "No insights available",
            "insights": None
        }
    
    return {
        "status": "success",
        "insights": {
            "global_kpis": insights.global_kpis,
            "monthly_kpis": insights.monthly_kpis,
            "oldest_existing_transaction": insights.oldest_existing_transaction.isoformat() if insights.oldest_existing_transaction else None,
            "fully_analyzed_month": insights.fully_analyzed_month,
            "fully_analyzed_day": insights.fully_analyzed_day,
            "updated_at": insights.updated_at.isoformat()
        }
    }

@router.post("/refresh")
async def refresh_insights(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Rafraîchit les insights financiers pour l'utilisateur.
    
    Returns:
        Dict: Résultat du rafraîchissement
    """
    try:
        # Récupérer le token Bridge
        from user_service.services.bridge import get_bridge_token
        token_data = await get_bridge_token(db, current_user.id)
        access_token = token_data["access_token"]
        
        # Récupérer les insights depuis Bridge API
        from user_service.services.bridge import get_bridge_insights
        insights = await get_bridge_insights(db, current_user.id)
        
        if not insights:
            return {
                "status": "warning",
                "message": "No insights available from Bridge API"
            }
        
        # Stocker les insights
        from sync_service.sync_manager.insight_handler import store_bridge_insights
        result = await store_bridge_insights(db, current_user.id, insights)
        
        return {
            "status": result.get("status", "error"),
            "message": "Insights refreshed successfully" if result.get("status") == "success" else "Failed to refresh insights",
            "insights_updated": result.get("insights_updated", False)
        }
    except Exception as e:
        logger.error(f"Error refreshing insights: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh insights: {str(e)}"
        )