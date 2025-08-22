from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict, EmailStr


class UserServiceProfile(BaseModel):
    """Profil utilisateur retourné par le user service."""

    user_id: int = Field(
        ..., description="Identifiant unique de l'utilisateur", gt=0
    )
    email: Optional[EmailStr] = Field(
        None, description="Adresse e-mail de l'utilisateur"
    )
    first_name: Optional[str] = Field(None, description="Prénom")
    last_name: Optional[str] = Field(None, description="Nom")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Métadonnées flexibles"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 123,
                "email": "user@example.com",
                "first_name": "Alice",
                "metadata": {"locale": "fr-FR"},
            }
        }
    )
