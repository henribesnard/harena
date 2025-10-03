"""
Pydantic Models pour les métriques
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MetricType(str, Enum):
    """Types de métriques"""
    MOM = "mom"
    YOY = "yoy"
    SAVINGS_RATE = "savings_rate"
    EXPENSE_RATIO = "expense_ratio"
    BURN_RATE = "burn_rate"
    BALANCE_FORECAST = "balance_forecast"
    RECURRING = "recurring_expenses"

class TrendDirection(str, Enum):
    """Direction de la tendance"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"

# === Trends ===

class MoMMetric(BaseModel):
    """Month-over-Month metric"""
    current_month: str
    previous_month: str
    current_amount: float
    previous_amount: float
    change_amount: float
    change_percent: float
    trend: TrendDirection

class YoYMetric(BaseModel):
    """Year-over-Year metric"""
    current_year: int
    previous_year: int
    current_amount: float
    previous_amount: float
    change_amount: float
    change_percent: float
    trend: TrendDirection

# === Health ===

class SavingsRateMetric(BaseModel):
    """Taux d'épargne"""
    period_start: str
    period_end: str
    total_income: float
    total_expenses: float
    net_savings: float
    savings_rate: float  # En pourcentage
    health_status: str  # "excellent" | "good" | "fair" | "poor"
    recommendation: Optional[str] = None

class ExpenseRatioMetric(BaseModel):
    """Ratios de dépenses (50/30/20)"""
    period_start: str
    period_end: str
    total_expenses: float
    essentials: float  # 50%
    essentials_percent: float
    lifestyle: float  # 30%
    lifestyle_percent: float
    savings: float  # 20%
    savings_percent: float
    is_balanced: bool
    recommendations: List[str] = []

class BurnRateMetric(BaseModel):
    """Burn rate et runway"""
    period_start: str
    period_end: str
    current_balance: float
    monthly_burn_rate: float
    runway_days: Optional[int] = None
    runway_months: Optional[float] = None
    risk_level: str  # "low" | "medium" | "high" | "critical"
    alert: Optional[str] = None

# === Forecasts ===

class BalancePrediction(BaseModel):
    """Prédiction de solde pour une date"""
    date: str
    balance: float
    balance_lower: float
    balance_upper: float

class BalanceForecastMetric(BaseModel):
    """Prévision de solde avec Prophet"""
    forecast_type: str  # "prophet" | "linear" | "average"
    periods: int
    current_balance: float
    predictions: List[BalancePrediction]
    trend: TrendDirection
    confidence: str  # "high" | "medium" | "low"

class ExpensePrediction(BaseModel):
    """Prédiction de dépense pour une date"""
    date: str
    amount: float
    amount_lower: float
    amount_upper: float

class ExpenseForecastMetric(BaseModel):
    """Prévision des dépenses"""
    forecast_type: str
    periods: int
    predictions: List[ExpensePrediction]
    total_forecast: float
    confidence: str

# === Patterns ===

class RecurringExpense(BaseModel):
    """Dépense récurrente détectée"""
    merchant: str
    category: Optional[str] = None
    frequency: str  # "weekly" | "monthly" | "yearly"
    average_amount: float
    last_occurrence: str
    next_expected: str
    confidence: float
    occurrences: int

class RecurringExpensesMetric(BaseModel):
    """Liste des dépenses récurrentes"""
    period_start: str
    period_end: str
    recurring_expenses: List[RecurringExpense]
    total_monthly_recurring: float
    recurring_percent_of_expenses: float

# === Generic Response ===

class MetricResponse(BaseModel):
    """Réponse générique pour une métrique"""
    user_id: int
    metric_type: MetricType
    computed_at: datetime
    data: Dict[str, Any]
    cached: bool = False
    cache_ttl: Optional[int] = None

class MetricsListResponse(BaseModel):
    """Réponse pour plusieurs métriques"""
    user_id: int
    metrics: List[MetricResponse]
    computed_at: datetime
