from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class GoalProgress(BaseModel):
  label: str
  value: float
  target: float


class CashFlowPoint(BaseModel):
  month: str = Field(description="Short month label, e.g. Oct 24")
  income: float
  expenses: float
  net: float


class CategorySlice(BaseModel):
  category: str
  amount: float


class DashboardOverview(BaseModel):
  kpis: Dict[str, float]
  risk_score: Dict[str, Any]
  goal_progress: List[GoalProgress]


class DashboardCashFlowSeries(BaseModel):
  series: List[CashFlowPoint]


class DashboardCategoryBreakdown(BaseModel):
  categories: List[CategorySlice]


class DashboardInsightsBundle(BaseModel):
  overview: DashboardOverview
  cash_flow: DashboardCashFlowSeries
  category_breakdown: DashboardCategoryBreakdown
