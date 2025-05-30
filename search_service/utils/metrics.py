"""
Collecteur de métriques pour le service de recherche.

Ce module collecte et agrège les métriques de performance
pour le monitoring et l'optimisation.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio

from search_service.models import SearchType

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collecte et agrège les métriques du service de recherche."""
    
    def __init__(self, window_size: int = 3600):
        """
        Initialise le collecteur.
        
        Args:
            window_size: Taille de la fenêtre temporelle en secondes
        """
        self.window_size = window_size
        
        # Métriques globales
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Métriques par type de recherche
        self.search_counts = defaultdict(int)
        self.search_durations = defaultdict(list)
        self.search_result_counts = defaultdict(list)
        
        # Métriques par utilisateur
        self.user_metrics = defaultdict(lambda: {
            "searches": 0,
            "cache_hits": 0,
            "recent_queries": deque(maxlen=10)
        })
        
        # Métriques temporelles (pour les graphiques)
        self.time_series = {
            "requests": deque(maxlen=window_size),
            "response_times": deque(maxlen=window_size),
            "errors": deque(maxlen=window_size)
        }
        
        # Feedback utilisateur
        self.feedback_stats = {
            "relevant": 0,
            "not_relevant": 0,
            "by_query": defaultdict(lambda: {"relevant": 0, "not_relevant": 0})
        }
        
        self._lock = asyncio.Lock()
    
    def record_request(
        self,
        path: str,
        method: str,
        status_code: int,
        duration: float
    ):
        """
        Enregistre une requête HTTP.
        
        Args:
            path: Chemin de la requête
            method: Méthode HTTP
            status_code: Code de statut
            duration: Durée en secondes
        """
        self.request_count += 1
        
        if status_code >= 400:
            self.error_count += 1
            self.time_series["errors"].append({
                "timestamp": time.time(),
                "path": path,
                "status_code": status_code
            })
        
        self.time_series["requests"].append({
            "timestamp": time.time(),
            "path": path,
            "method": method
        })
        
        self.time_series["response_times"].append({
            "timestamp": time.time(),
            "duration": duration * 1000  # Convertir en ms
        })
    
    def record_search(
        self,
        search_type: SearchType,
        results_count: int,
        duration: float
    ):
        """
        Enregistre une recherche.
        
        Args:
            search_type: Type de recherche
            results_count: Nombre de résultats
            duration: Durée en secondes
        """
        self.search_counts[search_type.value] += 1
        self.search_durations[search_type.value].append(duration)
        self.search_result_counts[search_type.value].append(results_count)
        
        # Limiter la taille des listes
        if len(self.search_durations[search_type.value]) > 1000:
            self.search_durations[search_type.value] = self.search_durations[search_type.value][-1000:]
        if len(self.search_result_counts[search_type.value]) > 1000:
            self.search_result_counts[search_type.value] = self.search_result_counts[search_type.value][-1000:]
    
    def record_cache_hit(self):
        """Enregistre un hit de cache."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Enregistre un miss de cache."""
        self.cache_misses += 1
    
    def record_user_search(
        self,
        user_id: int,
        query: str,
        cache_hit: bool = False
    ):
        """
        Enregistre une recherche utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            cache_hit: Si c'était un hit de cache
        """
        metrics = self.user_metrics[user_id]
        metrics["searches"] += 1
        
        if cache_hit:
            metrics["cache_hits"] += 1
        
        # Ajouter à l'historique des requêtes
        metrics["recent_queries"].append({
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_feedback(
        self,
        user_id: int,
        query: str,
        transaction_id: int,
        relevant: bool
    ):
        """
        Enregistre un feedback utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            transaction_id: ID de la transaction
            relevant: Si le résultat était pertinent
        """
        # Stats globales
        if relevant:
            self.feedback_stats["relevant"] += 1
        else:
            self.feedback_stats["not_relevant"] += 1
        
        # Stats par requête
        query_stats = self.feedback_stats["by_query"][query.lower()]
        if relevant:
            query_stats["relevant"] += 1
        else:
            query_stats["not_relevant"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des métriques.
        
        Returns:
            Dict: Résumé des métriques
        """
        # Calculer les moyennes
        total_searches = sum(self.search_counts.values())
        
        avg_durations = {}
        for search_type, durations in self.search_durations.items():
            if durations:
                avg_durations[search_type] = sum(durations) / len(durations)
        
        avg_results = {}
        for search_type, counts in self.search_result_counts.items():
            if counts:
                avg_results[search_type] = sum(counts) / len(counts)
        
        # Calculer le taux de requêtes
        now = time.time()
        recent_requests = [
            req for req in self.time_series["requests"]
            if now - req["timestamp"] < 60  # Dernière minute
        ]
        requests_per_minute = len(recent_requests)
        
        # Calculer le temps de réponse moyen
        recent_times = [
            rt["duration"] for rt in self.time_series["response_times"]
            if now - rt["timestamp"] < 300  # Dernières 5 minutes
        ]
        avg_response_time = sum(recent_times) / len(recent_times) if recent_times else 0
        
        # Taux de cache
        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "total_searches": total_searches,
            "searches_by_type": dict(self.search_counts),
            "avg_search_duration": avg_durations,
            "avg_result_count": avg_results,
            "requests_per_minute": requests_per_minute,
            "avg_response_time_ms": avg_response_time,
            "cache_hit_rate": cache_hit_rate,
            "feedback_stats": {
                "total": self.feedback_stats["relevant"] + self.feedback_stats["not_relevant"],
                "relevance_rate": self.feedback_stats["relevant"] / (
                    self.feedback_stats["relevant"] + self.feedback_stats["not_relevant"]
                ) if (self.feedback_stats["relevant"] + self.feedback_stats["not_relevant"]) > 0 else 0
            }
        }
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Retourne les statistiques pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques utilisateur
        """
        if user_id not in self.user_metrics:
            return {
                "total_searches": 0,
                "cache_hit_rate": 0,
                "recent_queries": []
            }
        
        metrics = self.user_metrics[user_id]
        cache_hit_rate = (
            metrics["cache_hits"] / metrics["searches"]
            if metrics["searches"] > 0 else 0
        )
        
        # Compter les types de recherche
        search_types = defaultdict(int)
        for query_info in metrics["recent_queries"]:
            # Simplification: on pourrait stocker le type réel
            search_types["hybrid"] += 1
        
        return {
            "total_searches": metrics["searches"],
            "cache_hit_rate": cache_hit_rate,
            "search_types": dict(search_types),
            "recent_queries": list(metrics["recent_queries"])
        }
    
    def get_time_series(
        self,
        metric: str,
        window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Retourne une série temporelle pour un métrique.
        
        Args:
            metric: Nom du métrique
            window_minutes: Fenêtre temporelle en minutes
            
        Returns:
            List: Points de données
        """
        if metric not in self.time_series:
            return []
        
        now = time.time()
        window_seconds = window_minutes * 60
        
        # Filtrer les points dans la fenêtre
        points = [
            point for point in self.time_series[metric]
            if now - point["timestamp"] < window_seconds
        ]
        
        return points
    
    async def cleanup_task(self):
        """
        Tâche de nettoyage périodique des métriques.
        À lancer en arrière-plan.
        """
        while True:
            try:
                # Attendre 1 heure
                await asyncio.sleep(3600)
                
                # Nettoyer les vieilles métriques
                async with self._lock:
                    # Nettoyer les séries temporelles
                    now = time.time()
                    for series in self.time_series.values():
                        # Garder seulement les données de la dernière heure
                        while series and now - series[0]["timestamp"] > self.window_size:
                            series.popleft()
                    
                    # Limiter la taille des métriques par type
                    for search_type in self.search_durations:
                        if len(self.search_durations[search_type]) > 10000:
                            self.search_durations[search_type] = self.search_durations[search_type][-5000:]
                        if len(self.search_result_counts[search_type]) > 10000:
                            self.search_result_counts[search_type] = self.search_result_counts[search_type][-5000:]
                
                logger.info("Nettoyage des métriques terminé")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans la tâche de nettoyage des métriques: {e}")