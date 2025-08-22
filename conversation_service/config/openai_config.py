"""OpenAI model configuration for the conversation service.

The configuration describes available models, their costs, rate limits
and fallbacks.  It is intentionally static and can be imported by other
modules to perform budgeting or model selection.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Detailed configuration for a single OpenAI model."""

    name: str = Field(..., description="Model identifier as used by the API")
    prompt_cost_per_million: float = Field(
        ..., ge=0, description="USD cost for one million prompt tokens"
    )
    completion_cost_per_million: float = Field(
        ..., ge=0, description="USD cost for one million completion tokens"
    )
    rpm_limit: int = Field(..., ge=0, description="Requests per minute limit")
    tpm_limit: int = Field(..., ge=0, description="Tokens per minute limit")
    fallback: Optional[str] = Field(
        None, description="Fallback model name if this model is unavailable"
    )


class BudgetConfig(BaseModel):
    """Soft and hard budget limits for OpenAI usage."""

    soft_limit_usd: float = Field(10.0, ge=0, description="Warning threshold")
    hard_limit_usd: float = Field(50.0, ge=0, description="Maximum allowed cost")


class OpenAIConfig(BaseModel):
    """Aggregated configuration for OpenAI usage."""

    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    budget: BudgetConfig = BudgetConfig()

    def get_model(self, name: str) -> ModelConfig:
        """Return the configuration for *name* or raise ``KeyError``."""

        return self.models[name]

    def get_fallback(self, name: str) -> Optional[ModelConfig]:
        """Return the fallback model for *name* if defined."""

        model = self.get_model(name)
        if model.fallback:
            return self.models.get(model.fallback)
        return None


# Default configuration with a couple of common models.
DEFAULT_OPENAI_CONFIG = OpenAIConfig(
    models={
        "gpt-4o-mini": ModelConfig(
            name="gpt-4o-mini",
            prompt_cost_per_million=0.15,
            completion_cost_per_million=0.60,
            rpm_limit=10000,
            tpm_limit=200000,
        ),
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            prompt_cost_per_million=5.00,
            completion_cost_per_million=15.00,
            rpm_limit=5000,
            tpm_limit=300000,
            fallback="gpt-4o-mini",
        ),
    },
)
