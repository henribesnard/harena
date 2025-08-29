"""Entity models used in the conversation service."""
from __future__ import annotations

from datetime import date as Date, datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict, field_validator


class ExtractedAmount(BaseModel):
    """Represents a monetary amount mentioned in a conversation."""

    value: float = Field(..., description="Montant mentionné")
    currency: str = Field(..., description="Code devise ISO 4217")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 3:
            raise ValueError("currency must be a 3-letter ISO code")
        return v

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class ExtractedMerchant(BaseModel):
    """Represents a merchant extracted from a conversation."""

    name: str = Field(..., description="Nom du commerçant")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("merchant name cannot be empty")
        return v

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class ExtractedDate(BaseModel):
    """Represents a date extracted from a conversation."""

    date: Date = Field(..., description="Date de la transaction")

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> date:
        if isinstance(v, str):
            return Date.fromisoformat(v)
        return v

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class CategoryEntity(BaseModel):
    """Represents a transaction category extracted from a conversation."""

    name: str

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class TransactionTypeEntity(BaseModel):
    """Represents a transaction type such as credit or debit."""

    transaction_type: str

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class EntityExtractionResult(BaseModel):
    """Aggregates all entities extracted from a conversation."""

    amounts: List[ExtractedAmount] = Field(default_factory=list)
    merchants: List[ExtractedMerchant] = Field(default_factory=list)
    dates: List[ExtractedDate] = Field(default_factory=list)
    categories: List[CategoryEntity] = Field(default_factory=list)
    transaction_types: List[TransactionTypeEntity] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    team_context: Dict[str, Any] = Field(default_factory=dict)
    global_confidence: float = Field(0.0, ge=0.0, le=1.0)

    @classmethod
    def from_llm_response(cls, data: Dict[str, Any]) -> "EntityExtractionResult":
        """Build an extraction result from a raw LLM response."""

        entities_data = data.get("entities", [])
        if not isinstance(entities_data, list):
            raise ValueError("entities must be a list")

        result = cls(
            extraction_metadata=data.get("extraction_metadata", {}),
            team_context=data.get("team_context", {}),
            global_confidence=float(data.get("global_confidence", 0.0)),
        )

        for entity in entities_data:
            etype = entity.get("type")
            value = entity.get("value")
            if etype == "amount":
                result.amounts.append(
                    ExtractedAmount(
                        value=float(value),
                        currency=entity.get("currency", ""),
                    )
                )
            elif etype == "merchant":
                result.merchants.append(ExtractedMerchant(name=str(value)))
            elif etype == "date":
                result.dates.append(ExtractedDate(date=value))
            elif etype == "category":
                result.categories.append(CategoryEntity(name=str(value)))
            elif etype == "transaction_type":
                result.transaction_types.append(
                    TransactionTypeEntity(transaction_type=str(value))
                )

        return result

    @classmethod
    def create_fallback_result(
        cls, error: str, team_context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Create a standardized fallback extraction result."""

        return {
            "extraction_success": False,
            "entities": cls(extraction_metadata={"error": error}),
            "team_context": team_context or {},
        }

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True,
        extra="forbid",
    )


__all__ = [
    "ExtractedAmount",
    "ExtractedMerchant",
    "ExtractedDate",
    "CategoryEntity",
    "TransactionTypeEntity",
    "EntityExtractionResult",
]

