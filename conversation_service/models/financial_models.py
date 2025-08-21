from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import re

from .enums import EntityType


class FinancialEntity(BaseModel):
    """Financial entity with comprehensive validation and normalization."""

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "entity_type": "AMOUNT",
                "raw_value": "500 euros",
                "normalized_value": "500.00",
                "confidence": 0.95,
                "start_position": 15,
                "end_position": 25,
                "context": "spending analysis"
            }
        }
    )

    entity_type: EntityType = Field(
        ...,
        description="Type of financial entity according to Harena taxonomy",
    )
    raw_value: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Original extracted value from user input",
    )
    normalized_value: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Normalized value following business rules",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0-1.0)",
    )

    # Optional metadata
    start_position: Optional[int] = Field(
        None,
        ge=0,
        description="Start position in source text",
    )
    end_position: Optional[int] = Field(
        None,
        ge=0,
        description="End position in source text",
    )
    alternative_values: Optional[List[str]] = Field(
        None,
        max_length=5,
        description="Alternative extraction candidates",
    )
    context: Optional[str] = Field(
        None,
        max_length=200,
        description="Extraction context for disambiguation",
    )

    @field_validator('normalized_value')
    @classmethod
    def validate_normalized_value(cls, v: str, info) -> str:
        entity_type = info.data.get('entity_type')

        if entity_type == EntityType.AMOUNT:
            try:
                clean_amount = re.sub(r'[^\d.,\-+]', '', v)
                clean_amount = clean_amount.replace(',', '.')
                amount_float = float(clean_amount)
                if abs(amount_float) > 10000000:
                    raise ValueError(f"Amount exceeds maximum limit: {v}")
                return f"{amount_float:.2f}"
            except ValueError as e:
                raise ValueError(f"Invalid amount format '{v}': {str(e)}")

        elif entity_type == EntityType.CURRENCY:
            normalized = v.upper().strip()
            if len(normalized) > 5:
                raise ValueError(f"Currency code too long: {v}")
            return normalized

        elif entity_type == EntityType.PERCENTAGE:
            try:
                numeric_part = re.sub(r'[^\d.,\-]', '', v)
                numeric_part = numeric_part.replace(',', '.')
                pct_value = float(numeric_part)
                if abs(pct_value) > 1000:
                    raise ValueError(f"Percentage value too extreme: {v}")
                return f"{pct_value:.2f}%"
            except ValueError:
                raise ValueError(f"Invalid percentage format: {v}")

        elif entity_type == EntityType.CATEGORY:
            normalized = v.lower().strip()
            if len(normalized) < 2 or len(normalized) > 50:
                raise ValueError(f"Category name invalid length: {v}")
            return normalized

        elif entity_type == EntityType.MERCHANT:
            if len(v.strip()) < 1:
                raise ValueError("Merchant name cannot be empty")
            clean_name = re.sub(r'[<>{}"]', '', v.strip())
            if len(clean_name) > 200:
                clean_name = clean_name[:200]
            return clean_name

        elif entity_type == EntityType.DATE_RANGE:
            normalized = v.lower().strip()
            if len(normalized) > 100:
                raise ValueError(f"Date range description too long: {v}")
            return normalized

        cleaned = v.strip()
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        return cleaned

    @model_validator(mode='after')
    def validate_positions(self):
        if self.start_position is not None and self.end_position is not None:
            if self.start_position >= self.end_position:
                raise ValueError("start_position must be less than end_position")
        return self

    def is_action_related(self) -> bool:
        action_entities = {
            EntityType.BENEFICIARY,
            EntityType.AUTHENTICATION,
            EntityType.COMMUNICATION,
        }
        return self.entity_type in action_entities

    def to_search_filter(self) -> Dict[str, Any]:
        if self.entity_type == EntityType.AMOUNT:
            amount_value = float(self.normalized_value)
            return {
                "range": {
                    "amount_abs": {
                        "gte": abs(amount_value) * 0.9,
                        "lte": abs(amount_value) * 1.1,
                    }
                }
            }
        elif self.entity_type == EntityType.MERCHANT:
            return {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "merchant_name": {
                                    "query": self.normalized_value,
                                    "fuzziness": "AUTO",
                                    "boost": 2.0,
                                }
                            }
                        },
                        {
                            "term": {
                                "merchant_name.keyword": self.normalized_value
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        elif self.entity_type == EntityType.CATEGORY:
            return {
                "term": {
                    "category_name.keyword": self.normalized_value
                }
            }
        elif self.entity_type == EntityType.DATE_RANGE:
            return {
                "exists": {
                    "field": "date",
                }
            }
        elif self.entity_type == EntityType.CURRENCY:
            return {
                "term": {
                    "currency_code": self.normalized_value
                }
            }
        return {
            "match": {
                "searchable_text": {
                    "query": self.normalized_value,
                    "fuzziness": "AUTO",
                }
            }
        }


__all__ = ["FinancialEntity"]
