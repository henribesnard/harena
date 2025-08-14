from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class ConnectSessionRequest(BaseModel):
    callback_url: Optional[str] = None
    country_code: Optional[str] = "FR"
    account_types: Optional[str] = "payment"
    context: Optional[str] = None
    provider_id: Optional[int] = None
    item_id: Optional[int] = None

    @field_validator('context')
    @classmethod
    def context_length(cls, v: Optional[str]):
        if v and len(v) > 100:
            raise ValueError('context must be 100 characters or less')
        return v

    class Config:
        schema_extra = {
            "example": {
                "callback_url": "https://votre-app.com/callback",
                "country_code": "FR",
                "account_types": "payment",
                "context": "user-session-123"
            }
        }


class ConnectSessionResponse(BaseModel):
    connect_url: str
