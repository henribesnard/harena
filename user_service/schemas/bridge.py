"""Schémas pour l'API Bridge Connect."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ConnectSessionRequest(BaseModel):
    """Modèle de requête pour initialiser une session Bridge Connect."""

    callback_url: Optional[str] = Field(
        default=None, description="URL de rappel après la connexion"
    )
    country_code: Optional[str] = Field(
        default="FR", description="Code pays ISO à deux lettres"
    )
    account_types: Optional[str] = Field(
        default="payment", description="Types de comptes demandés"
    )
    context: Optional[str] = Field(
        default=None, description="Contexte opaque renvoyé après la session"
    )
    provider_id: Optional[int] = Field(
        default=None, description="Identifiant du fournisseur Bridge"
    )
    item_id: Optional[int] = Field(
        default=None, description="Identifiant de l'item existant"
    )

    @field_validator("context")
    @classmethod
    def context_length(cls, v: Optional[str]) -> Optional[str]:
        """Valide la longueur du champ `context`."""
        if v and len(v) > 100:
            raise ValueError("context must be 100 characters or less")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "callback_url": "https://votre-app.com/callback",
                "country_code": "FR",
                "account_types": "payment",
                "context": "user-session-123",
                "provider_id": 12,
                "item_id": 34,
            }
        }
    )


class ConnectSessionResponse(BaseModel):
    """Réponse contenant l'URL Bridge Connect à utiliser."""

    connect_url: str = Field(..., description="URL de redirection Bridge Connect")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connect_url": "https://connect.bridgeapi.io/session/abc123"
            }
        }
    )

