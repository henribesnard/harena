"""
Logic Agents Module

This module contains logic-based agents that perform analytical computations
without LLM calls (pure Python logic).

Agents:
- AnalyticsAgent: Temporal comparisons, trends, anomaly detection, pivot analysis
"""

from .analytics_agent import AnalyticsAgent

__all__ = ["AnalyticsAgent"]
