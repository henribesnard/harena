"""
Modèles de réponse API épurée pour l'endpoint public conversation/{user_id}
Version nettoyée qui ne retourne que les informations essentielles à l'utilisateur final
"""
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

from conversation_service.models.responses.conversation_responses import (
    ResponseContent, ResponseQuality, Insight, Suggestion, StructuredData
)


def create_enhanced_structured_data(
    search_results, 
    entities_result=None, 
    intent_type=None
) -> StructuredData:
    """Crée des données structurées enrichies pour l'utilisateur final"""
    
    if not search_results or not search_results.hits:
        return StructuredData()
    
    # Calcul des métriques réelles
    total_amount = sum(hit.source.get("amount", 0) for hit in search_results.hits)
    transaction_count = len(search_results.hits)
    average_amount = total_amount / transaction_count if transaction_count > 0 else 0
    
    # Détermination de la période à partir des entités
    period_start = None
    period_end = None
    period_description = None
    
    if entities_result and entities_result.get("entities", {}).get("dates"):
        dates = entities_result["entities"]["dates"]
        if dates:
            date_info = dates[0]  # Premier élément de date
            if date_info.get("type") == "period" and date_info.get("value"):
                period_value = date_info["value"]
                if len(period_value) == 7:  # Format YYYY-MM
                    year, month = period_value.split("-")
                    period_start = f"{period_value}-01"
                    # Dernier jour du mois
                    import calendar
                    last_day = calendar.monthrange(int(year), int(month))[1]
                    period_end = f"{period_value}-{last_day:02d}"
                    period_description = f"{date_info.get('text', period_value)}"
    
    # Type d'analyse basé sur l'intent
    analysis_type = None
    if intent_type:
        intent_name = str(intent_type).replace("HarenaIntentType.", "").lower()
        if "merchant" in intent_name:
            analysis_type = "par marchand"
        elif "category" in intent_name:
            analysis_type = "par catégorie"
        elif "operation_type" in intent_name:
            analysis_type = "par type d'opération"
        elif "amount" in intent_name:
            analysis_type = "par montant"
        else:
            analysis_type = "transactions"
    
    return StructuredData(
        total_amount=round(total_amount, 2),
        currency="EUR",
        transaction_count=transaction_count,
        average_amount=round(average_amount, 2),
        period=period_description,
        period_start=period_start,
        period_end=period_end,
        analysis_type=analysis_type
    )


class CleanConversationResponse(BaseModel):
    """Réponse API épurée pour l'utilisateur final"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "request_id": "phase5_1756754833163_34",
                "processing_time_ms": 42394,
                "response": {
                    "message": "**Vos rentrées d'argent pour mai 2025**\n\nSur la période du 1er au 31 mai 2025, vous avez enregistré **6 746,43€** de rentrées d'argent.\n\n**Détail des transactions créditrices :**\n- Salaire : 2 302,20€ (le 30/05)\n- Virement : 1 880€ (le 30/05)\n- Prestations : 994,14€ (le 16/05)\n- Autres virements : 1 570,09€\n\n**Analyse :**\nVos revenus sont diversifiés avec un salaire principal complété par différents virements. Le mois de mai présente une belle stabilité financière.\n\nPour une analyse plus détaillée ou pour consulter d'autres mois, n'hésitez pas à me le demander.",
                    "structured_data": {
                        "total_amount": 6746.43,
                        "currency": "EUR",
                        "transaction_count": 13,
                        "average_amount": 518.99,
                        "analysis_type": "rentrées d'argent"
                    },
                    "insights": [
                        {
                            "type": "spending",
                            "title": "Montant moyen élevé",
                            "description": "Vos transactions ont un montant moyen de 518,99€",
                            "severity": "info",
                            "confidence": 0.7
                        }
                    ],
                    "suggestions": [],
                    "next_actions": [
                        "Voir le détail des transactions",
                        "Filtrer par montant",
                        "Grouper par période"
                    ]
                },
                "quality": {
                    "relevance_score": 0.8,
                    "completeness": "partial",
                    "actionability": "none",
                    "tone": "professional_friendly"
                }
            }
        }
    )
    
    # Identifiants essentiels
    request_id: Optional[str] = None
    processing_time_ms: int
    
    # Réponse principale (contenu pour l'utilisateur)
    response: ResponseContent
    
    # Qualité de la réponse
    quality: Optional[ResponseQuality] = None
    
    # Statut simple
    status: str = "success"
    
    # Métriques utilisateur (optionnelles, masquées par défaut)
    user_metrics: Optional[Dict[str, Any]] = None


class CleanErrorResponse(BaseModel):
    """Réponse d'erreur épurée"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    request_id: Optional[str] = None
    status: str = "error"
    error: str
    processing_time_ms: int
    timestamp: datetime
    
    # Suggestions d'actions pour l'utilisateur
    suggestions: List[str] = []