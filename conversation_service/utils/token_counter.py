"""
Compteur de tokens pour les appels API DeepSeek.

Ce module suit l'utilisation des tokens et calcule les coûts
pour le monitoring et la facturation.
"""
import logging
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Structure pour l'usage de tokens."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    timestamp: str = ""
    model: str = ""
    user_id: Optional[int] = None
    operation: str = ""

class TokenCounter:
    """Compteur et tracker pour l'usage des tokens DeepSeek."""
    
    def __init__(self):
        self.usage_history: List[TokenUsage] = []
        self.user_usage: Dict[int, List[TokenUsage]] = defaultdict(list)
        self.daily_usage: Dict[str, TokenUsage] = {}
        self.monthly_usage: Dict[str, TokenUsage] = {}
        
        # Coûts par modèle (prix par 1K tokens)
        self.model_costs = {
            "deepseek-chat": {
                "input": 0.00014,   # $0.14 per 1M tokens
                "output": 0.00028   # $0.28 per 1M tokens
            },
            "deepseek-reasoner": {
                "input": 0.00055,   # $0.55 per 1M tokens
                "output": 0.00220   # $2.20 per 1M tokens
            }
        }
        
        self.is_enabled = True
        logger.info("TokenCounter initialisé")
    
    def initialize(self):
        """Initialise le compteur de tokens."""
        logger.info("TokenCounter initialized")
        
    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "deepseek-chat",
        user_id: Optional[int] = None,
        operation: str = "chat"
    ) -> TokenUsage:
        """
        Enregistre l'usage de tokens pour un appel API.
        
        Args:
            input_tokens: Nombre de tokens d'entrée
            output_tokens: Nombre de tokens de sortie
            model: Modèle utilisé
            user_id: ID de l'utilisateur
            operation: Type d'opération
            
        Returns:
            TokenUsage: Enregistrement de l'usage
        """
        if not self.is_enabled:
            return TokenUsage()
        
        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens, model)
        
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            timestamp=datetime.now().isoformat(),
            model=model,
            user_id=user_id,
            operation=operation
        )
        
        # Stocker l'usage
        self.usage_history.append(usage)
        
        if user_id:
            self.user_usage[user_id].append(usage)
        
        # Mise à jour des statistiques quotidiennes et mensuelles
        self._update_period_stats(usage)
        
        logger.debug(f"Token usage recorded: {total_tokens} tokens, ${cost:.6f}, user: {user_id}")
        return usage
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calcule le coût d'un appel API."""
        if model not in self.model_costs:
            logger.warning(f"Unknown model {model}, using deepseek-chat costs")
            model = "deepseek-chat"
        
        costs = self.model_costs[model]
        
        # Coût par 1000 tokens
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def _update_period_stats(self, usage: TokenUsage):
        """Met à jour les statistiques par période."""
        now = datetime.now()
        day_key = now.strftime("%Y-%m-%d")
        month_key = now.strftime("%Y-%m")
        
        # Statistiques quotidiennes
        if day_key in self.daily_usage:
            daily = self.daily_usage[day_key]
            daily.input_tokens += usage.input_tokens
            daily.output_tokens += usage.output_tokens
            daily.total_tokens += usage.total_tokens
            daily.cost += usage.cost
        else:
            self.daily_usage[day_key] = TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cost=usage.cost,
                timestamp=day_key
            )
        
        # Statistiques mensuelles
        if month_key in self.monthly_usage:
            monthly = self.monthly_usage[month_key]
            monthly.input_tokens += usage.input_tokens
            monthly.output_tokens += usage.output_tokens
            monthly.total_tokens += usage.total_tokens
            monthly.cost += usage.cost
        else:
            self.monthly_usage[month_key] = TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cost=usage.cost,
                timestamp=month_key
            )
    
    def get_user_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Récupère les statistiques d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days: Nombre de jours à inclure
            
        Returns:
            Dict: Statistiques de l'utilisateur
        """
        if user_id not in self.user_usage:
            return {
                "user_id": user_id,
                "total_tokens": 0,
                "total_cost": 0.0,
                "calls_count": 0,
                "period_days": days
            }
        
        # Filtrer par période
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_usage = [
            usage for usage in self.user_usage[user_id]
            if datetime.fromisoformat(usage.timestamp) >= cutoff_date
        ]
        
        if not recent_usage:
            return {
                "user_id": user_id,
                "total_tokens": 0,
                "total_cost": 0.0,
                "calls_count": 0,
                "period_days": days
            }
        
        total_tokens = sum(usage.total_tokens for usage in recent_usage)
        total_cost = sum(usage.cost for usage in recent_usage)
        calls_count = len(recent_usage)
        
        # Statistiques par modèle
        by_model = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
        for usage in recent_usage:
            by_model[usage.model]["tokens"] += usage.total_tokens
            by_model[usage.model]["cost"] += usage.cost
            by_model[usage.model]["calls"] += 1
        
        # Statistiques par opération
        by_operation = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
        for usage in recent_usage:
            by_operation[usage.operation]["tokens"] += usage.total_tokens
            by_operation[usage.operation]["cost"] += usage.cost
            by_operation[usage.operation]["calls"] += 1
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "calls_count": calls_count,
            "avg_tokens_per_call": total_tokens / calls_count if calls_count > 0 else 0,
            "avg_cost_per_call": total_cost / calls_count if calls_count > 0 else 0,
            "by_model": dict(by_model),
            "by_operation": dict(by_operation),
            "latest_call": recent_usage[-1].timestamp if recent_usage else None
        }
    
    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Récupère les statistiques globales.
        
        Args:
            days: Nombre de jours à inclure
            
        Returns:
            Dict: Statistiques globales
        """
        if not self.usage_history:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "calls_count": 0,
                "unique_users": 0,
                "period_days": days
            }
        
        # Filtrer par période
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_usage = [
            usage for usage in self.usage_history
            if datetime.fromisoformat(usage.timestamp) >= cutoff_date
        ]
        
        if not recent_usage:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "calls_count": 0,
                "unique_users": 0,
                "period_days": days
            }
        
        total_tokens = sum(usage.total_tokens for usage in recent_usage)
        total_cost = sum(usage.cost for usage in recent_usage)
        calls_count = len(recent_usage)
        unique_users = len(set(usage.user_id for usage in recent_usage if usage.user_id))
        
        # Statistiques par modèle
        by_model = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
        for usage in recent_usage:
            by_model[usage.model]["tokens"] += usage.total_tokens
            by_model[usage.model]["cost"] += usage.cost
            by_model[usage.model]["calls"] += 1
        
        return {
            "period_days": days,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "calls_count": calls_count,
            "unique_users": unique_users,
            "avg_tokens_per_call": total_tokens / calls_count if calls_count > 0 else 0,
            "avg_cost_per_call": total_cost / calls_count if calls_count > 0 else 0,
            "by_model": dict(by_model),
            "cost_breakdown": {
                model: costs for model, costs in by_model.items()
            }
        }
    
    def get_daily_stats(self, days: int = 7) -> Dict[str, Any]:
        """Récupère les statistiques quotidiennes."""
        daily_stats = {}
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            day_key = date.strftime("%Y-%m-%d")
            
            if day_key in self.daily_usage:
                usage = self.daily_usage[day_key]
                daily_stats[day_key] = {
                    "tokens": usage.total_tokens,
                    "cost": usage.cost,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens
                }
            else:
                daily_stats[day_key] = {
                    "tokens": 0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
        
        return daily_stats
    
    def get_monthly_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques mensuelles."""
        return {
            month: {
                "tokens": usage.total_tokens,
                "cost": usage.cost,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens
            }
            for month, usage in self.monthly_usage.items()
        }
    
    def get_top_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les utilisateurs avec le plus d'usage.
        
        Args:
            limit: Nombre d'utilisateurs à retourner
            
        Returns:
            List: Liste des top utilisateurs
        """
        if not self.user_usage:
            return []
        
        # Calculer l'usage total par utilisateur
        user_totals = {}
        for user_id, usages in self.user_usage.items():
            total_tokens = sum(usage.total_tokens for usage in usages)
            total_cost = sum(usage.cost for usage in usages)
            user_totals[user_id] = {
                "user_id": user_id,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "calls_count": len(usages)
            }
        
        # Trier par usage total de tokens
        sorted_users = sorted(
            user_totals.values(),
            key=lambda x: x["total_tokens"],
            reverse=True
        )
        
        return sorted_users[:limit]
    
    def export_usage_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Exporte les données d'usage pour analyse.
        
        Args:
            days: Nombre de jours à exporter
            
        Returns:
            List: Liste des enregistrements d'usage
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_usage = [
            usage for usage in self.usage_history
            if datetime.fromisoformat(usage.timestamp) >= cutoff_date
        ]
        
        return [asdict(usage) for usage in recent_usage]
    
    def reset_stats(self):
        """Remet à zéro toutes les statistiques."""
        self.usage_history.clear()
        self.user_usage.clear()
        self.daily_usage.clear()
        self.monthly_usage.clear()
        logger.info("Token counter stats reset")
    
    def enable(self):
        """Active le compteur de tokens."""
        self.is_enabled = True
        logger.info("Token counting enabled")
    
    def disable(self):
        """Désactive le compteur de tokens."""
        self.is_enabled = False
        logger.info("Token counting disabled")

# Instance globale
token_counter = TokenCounter()