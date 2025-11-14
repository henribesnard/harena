"""
Service layer responsible for building REAL analytics returned by the dashboard endpoints.
If the user has no data yet, everything falls back to 0 instead of fictive numbers.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from db_service.session import SessionLocal
from db_service.models.sync import SyncAccount, SyncItem, RawTransaction, Category

from dashboard_service.schemas.dashboard import (
  DashboardOverview,
  DashboardCashFlowSeries,
  DashboardCategoryBreakdown,
  DashboardInsightsBundle,
  GoalProgress,
  CashFlowPoint,
  CategorySlice,
)


def _to_float(value: float | None) -> float:
  return round(float(value or 0.0), 2)


def _risk_level(score: float) -> str:
  if score >= 80:
    return "excellent"
  if score >= 60:
    return "good"
  if score >= 40:
    return "balanced"
  if score > 0:
    return "fragile"
  return "incomplet"


def _goal_entry(label: str, value: float, target: float) -> GoalProgress:
  adjusted_target = target if target > 0 else 1.0
  safe_value = max(value, 0.0)
  return GoalProgress(label=label, value=_to_float(safe_value), target=_to_float(adjusted_target))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _sum_total_assets(db: Session, user_id: int) -> float:
  total = (
    db.query(func.coalesce(func.sum(SyncAccount.balance), 0.0))
    .join(SyncItem, SyncAccount.item_id == SyncItem.id)
    .filter(SyncItem.user_id == user_id)
    .scalar()
  )
  return _to_float(total)


def _recent_flows(db: Session, user_id: int, days: int = 30) -> Tuple[float, float]:
  window_start = datetime.utcnow() - timedelta(days=days)

  income = (
    db.query(func.coalesce(func.sum(RawTransaction.amount), 0.0))
    .filter(
      RawTransaction.user_id == user_id,
      RawTransaction.date >= window_start,
      RawTransaction.amount > 0,
    )
    .scalar()
  )

  expenses = (
    db.query(func.coalesce(func.sum(RawTransaction.amount), 0.0))
    .filter(
      RawTransaction.user_id == user_id,
      RawTransaction.date >= window_start,
      RawTransaction.amount < 0,
    )
    .scalar()
  )

  return _to_float(income), abs(_to_float(expenses))


def _build_cash_flow_points(db: Session, user_id: int, months: int = 6) -> List[CashFlowPoint]:
  period = func.date_trunc("month", RawTransaction.date).label("period")
  income_sum = func.sum(case((RawTransaction.amount > 0, RawTransaction.amount), else_=0.0)).label("income")
  expense_sum = func.sum(case((RawTransaction.amount < 0, RawTransaction.amount), else_=0.0)).label("expenses")

  rows = (
    db.query(period, income_sum, expense_sum)
    .filter(RawTransaction.user_id == user_id)
    .group_by(period)
    .order_by(period.desc())
    .limit(months)
    .all()
  )

  series: List[CashFlowPoint] = []
  for row in reversed(rows):
    if not row.period:
      continue
    label = row.period.strftime("%b %y")
    income = _to_float(row.income)
    expenses = abs(_to_float(row.expenses))
    net = _to_float(income - expenses)
    series.append(CashFlowPoint(month=label, income=income, expenses=expenses, net=net))

  return series


def _build_category_breakdown(db: Session, user_id: int, limit: int = 6) -> List[CategorySlice]:
  ninety_days_ago = datetime.utcnow() - timedelta(days=90)
  rows = (
    db.query(
      func.coalesce(Category.category_name, "Autres").label("label"),
      func.coalesce(func.sum(func.abs(RawTransaction.amount)), 0.0).label("amount"),
    )
    .outerjoin(Category, RawTransaction.category_id == Category.category_id)
    .filter(
      RawTransaction.user_id == user_id,
      RawTransaction.amount < 0,
      RawTransaction.date >= ninety_days_ago,
    )
    .group_by("label")
    .order_by(func.sum(func.abs(RawTransaction.amount)).desc())
    .limit(limit)
    .all()
  )

  return [
    CategorySlice(category=row.label or "Autres", amount=_to_float(row.amount))
    for row in rows
    if row.amount
  ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dashboard_overview(user_id: int) -> DashboardOverview:
  db = SessionLocal()
  try:
    cash_flow = _build_cash_flow_points(db, user_id, months=6)
    income, expenses = _recent_flows(db, user_id)
    total_assets = _sum_total_assets(db, user_id)

    monthly_savings = _to_float(income - expenses)
    burn_rate = _to_float(expenses / income) if income > 0 else 0.0
    coverage_ratio = _to_float(income / expenses) if expenses > 0 else 0.0
    savings_rate = monthly_savings / income if income > 0 else 0.0

    score = min(100.0, max(0.0, (coverage_ratio * 40) + (savings_rate * 60)))
    risk_score = {"score": round(score), "level": _risk_level(score)}

    positive_months = sum(1 for point in cash_flow if point.net >= 0)
    goal_progress = [
      _goal_entry("Fonds d'urgence (3 mois)", total_assets, expenses * 3),
      _goal_entry("Epargne mensuelle", monthly_savings, income * 0.2),
      _goal_entry("Mois positifs", float(positive_months), float(len(cash_flow) or 1)),
    ]

    return DashboardOverview(
      kpis={
        "total_assets": total_assets,
        "monthly_savings": monthly_savings,
        "burn_rate": burn_rate,
        "coverage_ratio": coverage_ratio,
      },
      risk_score=risk_score,
      goal_progress=goal_progress,
    )
  finally:
    db.close()


def get_cash_flow_series(user_id: int) -> DashboardCashFlowSeries:
  db = SessionLocal()
  try:
    return DashboardCashFlowSeries(series=_build_cash_flow_points(db, user_id, months=6))
  finally:
    db.close()


def get_category_breakdown(user_id: int) -> DashboardCategoryBreakdown:
  db = SessionLocal()
  try:
    return DashboardCategoryBreakdown(categories=_build_category_breakdown(db, user_id))
  finally:
    db.close()


def get_dashboard_bundle(user_id: int) -> DashboardInsightsBundle:
  db = SessionLocal()
  try:
    cash_flow = _build_cash_flow_points(db, user_id, months=6)
    income, expenses = _recent_flows(db, user_id)
    total_assets = _sum_total_assets(db, user_id)

    monthly_savings = _to_float(income - expenses)
    burn_rate = _to_float(expenses / income) if income > 0 else 0.0
    coverage_ratio = _to_float(income / expenses) if expenses > 0 else 0.0
    savings_rate = monthly_savings / income if income > 0 else 0.0
    score = min(100.0, max(0.0, (coverage_ratio * 40) + (savings_rate * 60)))

    positive_months = sum(1 for point in cash_flow if point.net >= 0)
    goal_progress = [
      _goal_entry("Fonds d'urgence (3 mois)", total_assets, expenses * 3),
      _goal_entry("Epargne mensuelle", monthly_savings, income * 0.2),
      _goal_entry("Mois positifs", float(positive_months), float(len(cash_flow) or 1)),
    ]

    overview = DashboardOverview(
      kpis={
        "total_assets": total_assets,
        "monthly_savings": monthly_savings,
        "burn_rate": burn_rate,
        "coverage_ratio": coverage_ratio,
      },
      risk_score={"score": round(score), "level": _risk_level(score)},
      goal_progress=goal_progress,
    )

    categories = _build_category_breakdown(db, user_id)

    return DashboardInsightsBundle(
      overview=overview,
      cash_flow=DashboardCashFlowSeries(series=cash_flow),
      category_breakdown=DashboardCategoryBreakdown(categories=categories),
    )
  finally:
    db.close()
