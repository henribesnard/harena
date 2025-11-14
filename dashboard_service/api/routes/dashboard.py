from fastapi import APIRouter, Request, HTTPException

from dashboard_service.services.aggregations import (
  get_dashboard_overview,
  get_cash_flow_series,
  get_category_breakdown,
  get_dashboard_bundle,
)
from dashboard_service.schemas.dashboard import (
  DashboardOverview,
  DashboardCashFlowSeries,
  DashboardCategoryBreakdown,
  DashboardInsightsBundle,
)


router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def _require_user_id(request: Request) -> int:
  user = getattr(request.state, "user", None)
  user_id = user.get("id") if isinstance(user, dict) else None
  if not user_id:
    raise HTTPException(status_code=401, detail="Utilisateur non authentifié")
  return int(user_id)


@router.get("/overview", response_model=DashboardOverview)
async def read_overview(request: Request):
  user_id = _require_user_id(request)
  return get_dashboard_overview(user_id)


@router.get("/cash-flow", response_model=DashboardCashFlowSeries)
async def read_cash_flow(request: Request):
  user_id = _require_user_id(request)
  return get_cash_flow_series(user_id)


@router.get("/category-breakdown", response_model=DashboardCategoryBreakdown)
async def read_category_breakdown(request: Request):
  user_id = _require_user_id(request)
  return get_category_breakdown(user_id)


@router.get("/all", response_model=DashboardInsightsBundle)
async def read_dashboard_bundle(request: Request):
  user_id = _require_user_id(request)
  return get_dashboard_bundle(user_id)
