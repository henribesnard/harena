"""
Visualization Schemas - Data Models for Chart Specifications

Supports Chart.js format for frontend rendering.
Defines KPI cards, line charts, bar charts, pie charts, etc.

Sprint 1.3 - T3.1
"""

from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class VisualizationType(str, Enum):
    """Types de visualisations supportés"""
    KPI_CARD = "kpi_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    DOUGHNUT_CHART = "doughnut_chart"
    AREA_CHART = "area_chart"


class KPICard(BaseModel):
    """
    KPI Card specification for displaying key metrics

    Examples:
    - Total spending
    - Transaction count
    - Average amount
    - Comparison vs previous period
    """
    type: Literal["kpi_card"] = "kpi_card"
    title: str = Field(..., description="KPI title (e.g., 'Total Dépenses')")
    value: float = Field(..., description="KPI numeric value")
    unit: str = Field(default="", description="Unit of measurement (e.g., '€', '%')")
    change_percent: Optional[float] = Field(default=None, description="Percentage change vs comparison period")
    change_direction: Optional[Literal["up", "down", "stable"]] = Field(default=None, description="Direction of change")
    comparison_text: Optional[str] = Field(default=None, description="Comparison period text (e.g., 'vs mois précédent')")
    icon: Optional[str] = Field(default=None, description="Icon name for display")
    color: Optional[str] = Field(default=None, description="Color theme (red, green, blue, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "kpi_card",
                "title": "Total Dépenses",
                "value": 1250.50,
                "unit": "€",
                "change_percent": 5.2,
                "change_direction": "up",
                "comparison_text": "vs mois précédent",
                "icon": "credit-card",
                "color": "red"
            }
        }


class ChartDataset(BaseModel):
    """
    Dataset for Chart.js charts

    Represents one series of data in a chart
    """
    label: str = Field(..., description="Dataset label")
    data: List[float] = Field(..., description="Data points")
    backgroundColor: Optional[Union[List[str], str]] = Field(default=None, description="Background color(s)")
    borderColor: Optional[str] = Field(default=None, description="Border color")
    borderWidth: Optional[int] = Field(default=2, description="Border width")
    fill: Optional[bool] = Field(default=False, description="Fill area under line (for area charts)")
    tension: Optional[float] = Field(default=0, description="Line tension/curve (0=straight, 0.4=curved)")

    class Config:
        json_schema_extra = {
            "example": {
                "label": "Dépenses",
                "data": [1000, 1200, 1100, 1300],
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "#36A2EB",
                "borderWidth": 2,
                "fill": True,
                "tension": 0.4
            }
        }


class ChartData(BaseModel):
    """
    Data structure for Chart.js

    Contains labels (x-axis) and datasets (series)
    """
    labels: List[str] = Field(..., description="X-axis labels")
    datasets: List[ChartDataset] = Field(..., description="Chart datasets")

    class Config:
        json_schema_extra = {
            "example": {
                "labels": ["Jan", "Feb", "Mar", "Apr"],
                "datasets": [
                    {
                        "label": "Dépenses",
                        "data": [1000, 1200, 1100, 1300],
                        "borderColor": "#36A2EB"
                    }
                ]
            }
        }


class ChartOptions(BaseModel):
    """
    Options for Chart.js configuration

    Defines chart behavior, styling, and interactivity
    """
    responsive: bool = Field(default=True, description="Chart responsive to container")
    maintainAspectRatio: bool = Field(default=False, description="Maintain aspect ratio")
    plugins: Dict[str, Any] = Field(default_factory=dict, description="Plugin configurations (legend, tooltip, etc.)")
    scales: Optional[Dict[str, Any]] = Field(default=None, description="Axis scales configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {"position": "top"},
                    "tooltip": {"enabled": True}
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "ticks": {
                            "callback": "function(value) { return value + '€'; }"
                        }
                    }
                }
            }
        }


class ChartVisualization(BaseModel):
    """
    Complete Chart.js visualization specification

    Ready to be consumed by frontend Chart.js library
    """
    type: VisualizationType = Field(..., description="Type of chart")
    title: str = Field(..., description="Chart title")
    description: Optional[str] = Field(default=None, description="Chart description")
    data: ChartData = Field(..., description="Chart data (labels + datasets)")
    options: ChartOptions = Field(default_factory=ChartOptions, description="Chart options")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "line_chart",
                "title": "Évolution des Dépenses",
                "description": "Tendance sur les 6 derniers mois",
                "data": {
                    "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                    "datasets": [
                        {
                            "label": "Dépenses",
                            "data": [1000, 1100, 1050, 1200, 1150, 1300],
                            "borderColor": "#36A2EB",
                            "fill": True
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"legend": {"display": False}}
                }
            }
        }


class VisualizationResponse(BaseModel):
    """
    Complete response with multiple visualizations

    Container for all visualizations generated for a request
    """
    visualizations: List[Union[KPICard, ChartVisualization]] = Field(..., description="List of visualizations")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    intent_group: str = Field(..., description="Intent group that triggered generation")
    intent_subtype: str = Field(..., description="Intent subtype")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "visualizations": [
                    {
                        "type": "kpi_card",
                        "title": "Total Dépenses",
                        "value": 1250.50,
                        "unit": "€"
                    },
                    {
                        "type": "pie_chart",
                        "title": "Répartition par Catégorie",
                        "data": {
                            "labels": ["Restaurant", "Transport", "Shopping"],
                            "datasets": [{"label": "Montant", "data": [500, 400, 350]}]
                        }
                    }
                ],
                "generated_at": "2025-10-12T15:30:00",
                "intent_group": "transaction_search",
                "intent_subtype": "by_category"
            }
        }


__all__ = [
    "VisualizationType",
    "KPICard",
    "ChartDataset",
    "ChartData",
    "ChartOptions",
    "ChartVisualization",
    "VisualizationResponse"
]
