"""
Compteur de tokens pour le suivi des coûts.

Ce module gère le comptage et le suivi des tokens utilisés
avec l'API DeepSeek pour estimer les coûts.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TokenCounter:
    """Compteur et gestionnaire des tokens utilisés."""
    
    def __init__(self):
        self.daily_usage = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
        self.user_usage = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
        self.model_usage = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
        
        # Historique des sessions récentes (pour les stats)
        self.recent_sessions = deque(maxlen=1000)
        
        # Coûts par modèle (prix par 1K tokens)
        self.token_costs = {
            "deepseek-chat": {
                "input": 0.00014,   # $0.14 per 1M input tokens
                "output": 0.00028   # $0.28 per 1M output tokens
            },
            "deepseek-reasoner": {
                "input": 0.00055,   # $0.55 per 1M input tokens  
                "output": 0.00222   # $2.22 per 1M output tokens
            }
        }
        
        self._initialized = False
    
    def initialize(self):
        """Initialise le compteur de tokens."""
        self._initialized = True
        logger.info("TokenCounter initialisé")
    
    def record_usage(
        self,
        user_id: int,
        input_tokens: int,
        output_tokens: int,
        model: str = "deepseek-chat"
    ):
        """
        Enregistre l'utilisation de tokens.
        
        Args:
            user_id: ID de l'utilisateur
            input_tokens: Nombre de tokens d'entrée
            output_tokens: Nombre de tokens de sortie
            model: Modèle utilisé
        """
        try:
            total_tokens = input_tokens + output_tokens
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Mise à jour des compteurs journaliers
            self.daily_usage[today]["input"] += input_tokens
            self.daily_usage[today]["output"] += output_tokens
            self.daily_usage[today]["total"] += total_tokens
            
            # Mise à jour des compteurs par utilisateur
            self.user_usage[user_id]["input"] += input_tokens
            self.user_usage[user_id]["output"] += output_tokens
            self.user_usage[user_id]["total"] += total_tokens
            
            # Mise à jour des compteurs par modèle
            self.model_usage[model]["input"] += input_tokens
            self.model_usage[model]["output"] += output_tokens
            self.model_usage[model]["total"] += total_tokens
            
            # Ajouter à l'historique des sessions
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": self.calculate_cost(input_tokens, output_tokens, model)
            }
            self.recent_sessions.append(session_data)
            
            logger.debug(
                f"Tokens enregistrés: user={user_id}, model={model}, "
                f"input={input_tokens}, output={output_tokens}"
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des tokens: {e}")
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "deepseek-chat"
    ) -> float:
        """
        Calcule le coût estimé pour une utilisation de tokens.
        
        Args:
            input_tokens: Nombre de tokens d'entrée
            output_tokens: Nombre de tokens de sortie
            model: Modèle utilisé
            
        Returns:
            float: Coût estimé en USD
        """
        if model not in self.token_costs:
            logger.warning(f"Modèle {model} non reconnu pour le calcul du coût")
            model = "deepseek-chat"  # Fallback
        
        costs = self.token_costs[model]
        
        # Calcul du coût (prix par 1K tokens)
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def get_user_usage(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère l'utilisation d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques d'utilisation
        """
        usage = self.user_usage.get(user_id, {"input": 0, "output": 0, "total": 0})
        
        # Calculer le coût total estimé pour l'utilisateur
        user_sessions = [s for s in self.recent_sessions if s["user_id"] == user_id]
        total_cost = sum(s["estimated_cost"] for s in user_sessions)
        
        # Calculer les moyennes
        session_count = len(user_sessions)
        avg_tokens_per_session = usage["total"] / session_count if session_count > 0 else 0
        avg_cost_per_session = total_cost / session_count if session_count > 0 else 0
        
        return {
            "user_id": user_id,
            "total_input_tokens": usage["input"],
            "total_output_tokens": usage["output"],
            "total_tokens": usage["total"],
            "total_estimated_cost_usd": total_cost,
            "session_count": session_count,
            "avg_tokens_per_session": avg_tokens_per_session,
            "avg_cost_per_session": avg_cost_per_session
        }
    
    def get_daily_usage(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère l'utilisation quotidienne.
        
        Args:
            date: Date au format YYYY-MM-DD (aujourd'hui par défaut)
            
        Returns:
            Dict: Statistiques quotidiennes
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        usage = self.daily_usage.get(date, {"input": 0, "output": 0, "total": 0})
        
        # Sessions du jour
        day_sessions = [
            s for s in self.recent_sessions 
            if s["timestamp"].startswith(date)
        ]
        
        total_cost = sum(s["estimated_cost"] for s in day_sessions)
        unique_users = len(set(s["user_id"] for s in day_sessions))
        
        return {
            "date": date,
            "total_input_tokens": usage["input"],
            "total_output_tokens": usage["output"],
            "total_tokens": usage["total"],
            "total_estimated_cost_usd": total_cost,
            "session_count": len(day_sessions),
            "unique_users": unique_users
        }
    
    def get_model_usage(self) -> Dict[str, Any]:
        """
        Récupère l'utilisation par modèle.
        
        Returns:
            Dict: Statistiques par modèle
        """
        model_stats = {}
        
        for model, usage in self.model_usage.items():
            # Sessions pour ce modèle
            model_sessions = [s for s in self.recent_sessions if s["model"] == model]
            total_cost = sum(s["estimated_cost"] for s in model_sessions)
            
            model_stats[model] = {
                "total_input_tokens": usage["input"],
                "total_output_tokens": usage["output"],
                "total_tokens": usage["total"],
                "total_estimated_cost_usd": total_cost,
                "session_count": len(model_sessions),
                "cost_per_1k_input": self.token_costs.get(model, {}).get("input", 0),
                "cost_per_1k_output": self.token_costs.get(model, {}).get("output", 0)
            }
        
        return model_stats
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques globales du compteur.
        
        Returns:
            Dict: Statistiques complètes
        """
        # Statistiques générales
        total_sessions = len(self.recent_sessions)
        total_cost = sum(s["estimated_cost"] for s in self.recent_sessions)
        
        # Calculs sur les dernières 24h
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        recent_sessions = [
            s for s in self.recent_sessions 
            if datetime.fromisoformat(s["timestamp"]) > last_24h
        ]
        
        recent_cost = sum(s["estimated_cost"] for s in recent_sessions)
        recent_tokens = sum(s["total_tokens"] for s in recent_sessions)
        
        return {
            "total_sessions": total_sessions,
            "total_estimated_cost_usd": total_cost,
            "total_users": len(self.user_usage),
            "total_models": len(self.model_usage),
            "last_24h": {
                "sessions": len(recent_sessions),
                "tokens": recent_tokens,
                "estimated_cost_usd": recent_cost
            },
            "daily_usage_days": len(self.daily_usage),
            "most_expensive_session": max(
                self.recent_sessions, 
                key=lambda x: x["estimated_cost"], 
                default={}
            ),
            "average_tokens_per_session": (
                sum(s["total_tokens"] for s in self.recent_sessions) / total_sessions
                if total_sessions > 0 else 0
            ),
            "average_cost_per_session": total_cost / total_sessions if total_sessions > 0 else 0
        }
    
    def get_top_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les utilisateurs avec le plus d'utilisation.
        
        Args:
            limit: Nombre d'utilisateurs à retourner
            
        Returns:
            List[Dict]: Top utilisateurs par utilisation
        """
        user_costs = {}
        
        for session in self.recent_sessions:
            user_id = session["user_id"]
            if user_id not in user_costs:
                user_costs[user_id] = 0
            user_costs[user_id] += session["estimated_cost"]
        
        # Trier par coût décroissant
        sorted_users = sorted(
            user_costs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return [
            {
                "user_id": user_id,
                "total_estimated_cost_usd": cost,
                "total_tokens": self.user_usage[user_id]["total"],
                "sessions": len([s for s in self.recent_sessions if s["user_id"] == user_id])
            }
            for user_id, cost in sorted_users
        ]
    
    def estimate_monthly_cost(self) -> float:
        """
        Estime le coût mensuel basé sur l'utilisation récente.
        
        Returns:
            float: Coût mensuel estimé en USD
        """
        # Utiliser les 7 derniers jours pour extrapoler
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
    def estimate_monthly_cost(self) -> float:
        """
        Estime le coût mensuel basé sur l'utilisation récente.
        
        Returns:
            float: Coût mensuel estimé en USD
        """
        # Utiliser les 7 derniers jours pour extrapoler
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        week_sessions = [
            s for s in self.recent_sessions 
            if datetime.fromisoformat(s["timestamp"]) > week_ago
        ]
        
        if not week_sessions:
            return 0.0
        
        # Coût de la semaine
        week_cost = sum(s["estimated_cost"] for s in week_sessions)
        
        # Extrapoler sur un mois (30 jours)
        monthly_estimate = (week_cost / 7) * 30
        
        return monthly_estimate
    
    def reset_user_usage(self, user_id: int):
        """
        Remet à zéro l'utilisation d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
        """
        if user_id in self.user_usage:
            del self.user_usage[user_id]
        
        # Supprimer les sessions de cet utilisateur
        self.recent_sessions = deque(
            [s for s in self.recent_sessions if s["user_id"] != user_id],
            maxlen=1000
        )
        
        logger.info(f"Utilisation remise à zéro pour l'utilisateur {user_id}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Nettoie les anciennes données de comptage.
        
        Args:
            days_to_keep: Nombre de jours de données à conserver
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Nettoyer les sessions anciennes
        filtered_sessions = [
            s for s in self.recent_sessions 
            if datetime.fromisoformat(s["timestamp"]) > cutoff_date
        ]
        self.recent_sessions = deque(filtered_sessions, maxlen=1000)
        
        # Nettoyer les données quotidiennes anciennes
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
        old_dates = [
            date for date in self.daily_usage.keys()
            if date < cutoff_date_str
        ]
        
        for date in old_dates:
            del self.daily_usage[date]
        
        # Recalculer les compteurs utilisateurs et modèles
        self._recalculate_counters()
        
        logger.info(f"Nettoyage effectué: {len(old_dates)} jours supprimés")
    
    def _recalculate_counters(self):
        """Recalcule les compteurs utilisateurs et modèles depuis les sessions."""
        # Réinitialiser les compteurs
        self.user_usage.clear()
        self.model_usage.clear()
        
        # Recalculer depuis les sessions restantes
        for session in self.recent_sessions:
            user_id = session["user_id"]
            model = session["model"]
            input_tokens = session["input_tokens"]
            output_tokens = session["output_tokens"]
            total_tokens = session["total_tokens"]
            
            # Compteurs utilisateur
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {"input": 0, "output": 0, "total": 0}
            
            self.user_usage[user_id]["input"] += input_tokens
            self.user_usage[user_id]["output"] += output_tokens
            self.user_usage[user_id]["total"] += total_tokens
            
            # Compteurs modèle
            if model not in self.model_usage:
                self.model_usage[model] = {"input": 0, "output": 0, "total": 0}
            
            self.model_usage[model]["input"] += input_tokens
            self.model_usage[model]["output"] += output_tokens
            self.model_usage[model]["total"] += total_tokens
    
    def export_usage_data(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Exporte les données d'utilisation pour analyse.
        
        Args:
            user_id: ID de l'utilisateur spécifique (optionnel)
            
        Returns:
            Dict: Données d'utilisation exportées
        """
        if user_id:
            # Exporter pour un utilisateur spécifique
            user_sessions = [s for s in self.recent_sessions if s["user_id"] == user_id]
            
            return {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "sessions": user_sessions,
                "summary": self.get_user_usage(user_id)
            }
        else:
            # Exporter toutes les données
            return {
                "export_timestamp": datetime.now().isoformat(),
                "sessions": list(self.recent_sessions),
                "daily_usage": dict(self.daily_usage),
                "user_usage": dict(self.user_usage),
                "model_usage": dict(self.model_usage),
                "summary": self.get_stats()
            }


# Instance globale
token_counter = TokenCounter()