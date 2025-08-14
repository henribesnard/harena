from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    FieldValidationInfo,
    ConfigDict,
    field_validator,
)
from typing import Optional, Dict, Any, List
from datetime import datetime


# Schémas de base
class UserBase(BaseModel):
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None


# Création d'utilisateur
class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    confirm_password: str

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info: FieldValidationInfo):
        if info.data.get("password") and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


# Mise à jour d'utilisateur
class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    password: Optional[str] = None


# Préférences utilisateur
class UserPreferenceBase(BaseModel):
    notification_settings: Dict[str, Any] = {}
    display_preferences: Dict[str, Any] = {}
    budget_settings: Dict[str, Any] = {}


class UserPreferenceCreate(UserPreferenceBase):
    pass


class UserPreferenceUpdate(UserPreferenceBase):
    pass


class UserPreferenceInDB(UserPreferenceBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Bridge connections
class BridgeConnectionBase(BaseModel):
    external_user_id: str


class BridgeConnectionCreate(BridgeConnectionBase):
    pass


class BridgeConnectionInDB(BridgeConnectionBase):
    id: int
    user_id: int
    bridge_user_uuid: str
    last_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# User in DB
class UserInDBBase(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserInDB(UserInDBBase):
    password_hash: str


class User(UserInDBBase):
    preferences: Optional[UserPreferenceInDB] = None
    bridge_connections: List[BridgeConnectionInDB] = []
    permissions: List[str] = []


# Token
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[int] = None
    permissions: List[str] = []


# Bridge API response schemas
class BridgeUserResponse(BaseModel):
    uuid: str
    external_user_id: str


class BridgeTokenResponse(BaseModel):
    access_token: str
    expires_at: datetime
    user: BridgeUserResponse
