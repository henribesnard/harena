"""
Modèles Pydantic V2 pour les requêtes conversation
"""
from pydantic import BaseModel, field_validator
from typing import Optional

class ConversationRequest(BaseModel):
    """Requête POST /conversation/{user_id} - Phase 1"""
    
    message: str
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Le message ne peut pas être vide")
        
        if len(v.strip()) > 1000:
            raise ValueError("Le message ne peut pas dépasser 1000 caractères")
        
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Combien j'ai dépensé chez Amazon ce mois ?"
            }
        }