from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class ConnectSessionRequest(BaseModel):
    callback_url: Optional[str] = None
    country_code: Optional[str] = "FR"
    account_types: Optional[str] = "payment"
    context: Optional[str] = None
    provider_id: Optional[int] = None
    item_id: Optional[int] = None

    class Config:
        schema_extra = {
    @field_validator('context')
    def context_length(cls, v):
    @classmethod
    def context_length(cls, v: Optional[str]):
        if v and len(v) > 100:
            raise ValueError('context must be 100 characters or less')
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "callback_url": "https://votre-app.com/callback",
                "country_code": "FR",
                "account_types": "payment",
                "context": "user-session-123"
            }
        }
    )


class ConnectSessionResponse(BaseModel):
    connect_url: str
