"""
Gestionnaire de contexte temporaire Phase 5
Gère la mémoire conversationnelle en session pour personnaliser les réponses
"""
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from threading import Lock
import json

logger = logging.getLogger(__name__)


class TemporaryContextManager:
    """Gestionnaire de contexte conversationnel temporaire sans persistance"""
    
    def __init__(self, context_ttl_seconds: int = 3600):
        """
        Initialise le gestionnaire de contexte
        
        Args:
            context_ttl_seconds: TTL du contexte en secondes (défaut: 1 heure)
        """
        self.session_contexts = {}  # user_id -> context_data
        self.context_ttl = context_ttl_seconds
        self._lock = Lock()
        
        # Configuration des éléments de contexte
        self.max_recent_queries = 5
        self.max_recent_intents = 10
        self.max_merchants_history = 20
        self.max_categories_history = 15
    
    def get_user_context(self, user_id: int) -> Dict[str, Any]:
        """Récupère le contexte utilisateur temporaire"""
        
        with self._lock:
            context = self.session_contexts.get(user_id)
            
            if not context:
                return self._create_empty_context()
            
            # Vérification TTL
            if self._is_context_expired(context):
                logger.info(f"Contexte expiré pour utilisateur {user_id}, suppression")
                self.session_contexts.pop(user_id, None)
                return self._create_empty_context()
            
            return context.copy()  # Copie pour éviter les modifications externes
    
    def update_context(
        self, 
        user_id: int, 
        message: str,
        intent: Dict[str, Any],
        entities: Dict[str, Any],
        search_results: Optional[Dict[str, Any]] = None,
        response_generated: Optional[Dict[str, Any]] = None
    ) -> None:
        """Met à jour le contexte utilisateur avec les nouvelles données"""
        
        with self._lock:
            try:
                current_context = self.session_contexts.get(user_id, self._create_empty_context())
                
                # Mise à jour timestamp
                current_context["updated_at"] = datetime.now(timezone.utc).isoformat()
                current_context["interaction_count"] += 1
                
                # Historique des requêtes
                self._update_query_history(current_context, message, intent)
                
                # Historique des entités
                self._update_entity_history(current_context, entities)
                
                # Patterns de comportement
                self._update_behavior_patterns(current_context, intent, entities, search_results)
                
                # Préférences déduites
                self._update_inferred_preferences(current_context, intent, entities, response_generated)
                
                # Métriques d'engagement
                self._update_engagement_metrics(current_context, search_results, response_generated)
                
                # Sauvegarde du contexte mis à jour
                self.session_contexts[user_id] = current_context
                
                logger.debug(f"Contexte mis à jour pour utilisateur {user_id}: {current_context['interaction_count']} interactions")
                
            except Exception as e:
                logger.error(f"Erreur mise à jour contexte pour {user_id}: {str(e)}")
    
    def get_context_summary(self, user_id: int) -> Dict[str, Any]:
        """Récupère un résumé du contexte pour personnalisation"""
        
        context = self.get_user_context(user_id)
        
        return {
            "is_returning_user": context["interaction_count"] > 1,
            "interaction_count": context["interaction_count"],
            "preferred_intents": self._get_top_intents(context),
            "frequent_merchants": self._get_frequent_merchants(context),
            "preferred_categories": self._get_preferred_categories(context),
            "communication_style": context.get("communication_preferences", {}).get("style", "balanced"),
            "detail_level": context.get("communication_preferences", {}).get("detail_level", "medium"),
            "recent_topics": context.get("recent_queries", [])[-3:],  # 3 derniers sujets
            "engagement_level": self._calculate_engagement_level(context)
        }
    
    def cleanup_expired_contexts(self) -> int:
        """Nettoie les contextes expirés et retourne le nombre de contextes supprimés"""
        
        with self._lock:
            expired_users = []
            
            for user_id, context in self.session_contexts.items():
                if self._is_context_expired(context):
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                self.session_contexts.pop(user_id, None)
            
            if expired_users:
                logger.info(f"Nettoyage: {len(expired_users)} contextes expirés supprimés")
            
            return len(expired_users)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les contextes actifs"""
        
        with self._lock:
            active_contexts = len(self.session_contexts)
            
            if active_contexts == 0:
                return {
                    "active_contexts": 0,
                    "total_interactions": 0,
                    "average_interactions_per_user": 0
                }
            
            total_interactions = sum(ctx["interaction_count"] for ctx in self.session_contexts.values())
            
            return {
                "active_contexts": active_contexts,
                "total_interactions": total_interactions,
                "average_interactions_per_user": total_interactions / active_contexts,
                "oldest_context_age_minutes": self._get_oldest_context_age_minutes(),
                "memory_usage_estimate_mb": self._estimate_memory_usage()
            }
    
    def _create_empty_context(self) -> Dict[str, Any]:
        """Crée un contexte vide pour un nouvel utilisateur"""
        
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "interaction_count": 0,
            "recent_queries": [],
            "intent_history": {},
            "entity_history": {
                "merchants": {},
                "categories": {},
                "date_patterns": []
            },
            "behavior_patterns": {
                "most_common_intent": None,
                "average_query_complexity": 0,
                "preferred_time_periods": [],
                "typical_amounts": []
            },
            "communication_preferences": {
                "style": "balanced",  # concise, balanced, detailed
                "detail_level": "medium",  # basic, medium, advanced
                "tone_preference": "professional_friendly"
            },
            "engagement_metrics": {
                "queries_with_results": 0,
                "total_queries": 0,
                "insights_shown": 0,
                "suggestions_provided": 0
            }
        }
    
    def _is_context_expired(self, context: Dict[str, Any]) -> bool:
        """Vérifie si un contexte est expiré"""
        
        try:
            updated_at_str = context.get("updated_at")
            if not updated_at_str:
                return True
            
            updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
            age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
            
            return age_seconds > self.context_ttl
            
        except Exception as e:
            logger.error(f"Erreur vérification expiration contexte: {str(e)}")
            return True
    
    def _update_query_history(self, context: Dict[str, Any], message: str, intent: Dict[str, Any]) -> None:
        """Met à jour l'historique des requêtes"""
        
        # Ajouter la nouvelle requête
        query_entry = {
            "message": message,
            "intent": intent.get("intent_type"),
            "confidence": intent.get("confidence", 0),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        context["recent_queries"].append(query_entry)
        
        # Limiter la taille de l'historique
        if len(context["recent_queries"]) > self.max_recent_queries:
            context["recent_queries"] = context["recent_queries"][-self.max_recent_queries:]
        
        # Mettre à jour statistiques intentions
        intent_type = intent.get("intent_type")
        if intent_type:
            context["intent_history"][intent_type] = context["intent_history"].get(intent_type, 0) + 1
    
    def _update_entity_history(self, context: Dict[str, Any], entities: Dict[str, Any]) -> None:
        """Met à jour l'historique des entités"""
        
        # Marchands
        if entities.get("merchants"):
            merchants = entities["merchants"]
            if isinstance(merchants, list):
                for merchant in merchants:
                    merchant_name = merchant if isinstance(merchant, str) else merchant.get("name")
                    if merchant_name:
                        context["entity_history"]["merchants"][merchant_name] = \
                            context["entity_history"]["merchants"].get(merchant_name, 0) + 1
        
        # Catégories
        if entities.get("categories"):
            categories = entities["categories"]
            if isinstance(categories, list):
                for category in categories:
                    context["entity_history"]["categories"][category] = \
                        context["entity_history"]["categories"].get(category, 0) + 1
        
        # Patterns de dates
        if entities.get("dates"):
            date_info = entities["dates"]
            if isinstance(date_info, dict):
                pattern = self._extract_date_pattern(date_info)
                if pattern:
                    context["entity_history"]["date_patterns"].append(pattern)
                    # Garder seulement les 10 derniers patterns
                    context["entity_history"]["date_patterns"] = \
                        context["entity_history"]["date_patterns"][-10:]
    
    def _update_behavior_patterns(
        self, 
        context: Dict[str, Any], 
        intent: Dict[str, Any], 
        entities: Dict[str, Any],
        search_results: Optional[Dict[str, Any]]
    ) -> None:
        """Met à jour les patterns de comportement détectés"""
        
        # Intent le plus commun
        if context["intent_history"]:
            most_common = max(context["intent_history"], key=context["intent_history"].get)
            context["behavior_patterns"]["most_common_intent"] = most_common
        
        # Complexité moyenne des requêtes
        complexity_score = self._calculate_query_complexity(intent, entities)
        current_avg = context["behavior_patterns"].get("average_query_complexity", 0)
        interaction_count = context["interaction_count"]
        new_avg = (current_avg * (interaction_count - 1) + complexity_score) / interaction_count
        context["behavior_patterns"]["average_query_complexity"] = new_avg
        
        # Montants typiques si disponibles
        if search_results and "total_amount" in search_results:
            amount = search_results["total_amount"]
            context["behavior_patterns"]["typical_amounts"].append(amount)
            # Garder seulement les 20 derniers montants
            context["behavior_patterns"]["typical_amounts"] = \
                context["behavior_patterns"]["typical_amounts"][-20:]
    
    def _update_inferred_preferences(
        self, 
        context: Dict[str, Any], 
        intent: Dict[str, Any], 
        entities: Dict[str, Any],
        response_generated: Optional[Dict[str, Any]]
    ) -> None:
        """Met à jour les préférences déduites de l'utilisateur"""
        
        # Style de communication basé sur la complexité des requêtes
        avg_complexity = context["behavior_patterns"].get("average_query_complexity", 0)
        if avg_complexity > 0.7:
            context["communication_preferences"]["detail_level"] = "advanced"
            context["communication_preferences"]["style"] = "detailed"
        elif avg_complexity < 0.3:
            context["communication_preferences"]["detail_level"] = "basic"
            context["communication_preferences"]["style"] = "concise"
    
    def _update_engagement_metrics(
        self, 
        context: Dict[str, Any], 
        search_results: Optional[Dict[str, Any]],
        response_generated: Optional[Dict[str, Any]]
    ) -> None:
        """Met à jour les métriques d'engagement"""
        
        metrics = context["engagement_metrics"]
        metrics["total_queries"] += 1
        
        if search_results and search_results.get("has_results"):
            metrics["queries_with_results"] += 1
        
        if response_generated:
            if response_generated.get("insights"):
                metrics["insights_shown"] += len(response_generated["insights"])
            if response_generated.get("suggestions"):
                metrics["suggestions_provided"] += len(response_generated["suggestions"])
    
    def _extract_date_pattern(self, date_info: Dict[str, Any]) -> Optional[str]:
        """Extrait un pattern de date lisible"""
        
        try:
            if "original" in date_info:
                original = date_info["original"].lower()
                if "mois" in original:
                    return "monthly_analysis"
                elif "semaine" in original:
                    return "weekly_analysis"
                elif "année" in original:
                    return "yearly_analysis"
                elif "jour" in original or "aujourd'hui" in original:
                    return "daily_analysis"
            
            return "date_range_analysis"
            
        except Exception:
            return None
    
    def _calculate_query_complexity(self, intent: Dict[str, Any], entities: Dict[str, Any]) -> float:
        """Calcule un score de complexité de la requête (0-1)"""
        
        complexity = 0.3  # Base
        
        # Bonus pour la confiance de l'intent
        if intent.get("confidence", 0) > 0.8:
            complexity += 0.2
        
        # Bonus pour le nombre d'entités
        entity_count = 0
        for entity_type, entity_data in entities.items():
            if entity_data:
                if isinstance(entity_data, list):
                    entity_count += len(entity_data)
                else:
                    entity_count += 1
        
        complexity += min(0.3, entity_count * 0.1)
        
        # Bonus pour les dates spécifiques
        if entities.get("dates") and isinstance(entities["dates"], dict):
            if "normalized" in entities["dates"]:
                complexity += 0.2
        
        return min(1.0, complexity)
    
    def _get_top_intents(self, context: Dict[str, Any], limit: int = 3) -> List[str]:
        """Récupère les intentions les plus fréquentes"""
        
        intent_history = context.get("intent_history", {})
        if not intent_history:
            return []
        
        sorted_intents = sorted(intent_history.items(), key=lambda x: x[1], reverse=True)
        return [intent for intent, _ in sorted_intents[:limit]]
    
    def _get_frequent_merchants(self, context: Dict[str, Any], limit: int = 5) -> List[str]:
        """Récupère les marchands les plus fréquents"""
        
        merchants = context.get("entity_history", {}).get("merchants", {})
        if not merchants:
            return []
        
        sorted_merchants = sorted(merchants.items(), key=lambda x: x[1], reverse=True)
        return [merchant for merchant, _ in sorted_merchants[:limit]]
    
    def _get_preferred_categories(self, context: Dict[str, Any], limit: int = 3) -> List[str]:
        """Récupère les catégories préférées"""
        
        categories = context.get("entity_history", {}).get("categories", {})
        if not categories:
            return []
        
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        return [category for category, _ in sorted_categories[:limit]]
    
    def _calculate_engagement_level(self, context: Dict[str, Any]) -> str:
        """Calcule le niveau d'engagement de l'utilisateur"""
        
        metrics = context.get("engagement_metrics", {})
        total_queries = metrics.get("total_queries", 0)
        
        if total_queries == 0:
            return "new"
        
        success_rate = metrics.get("queries_with_results", 0) / total_queries
        interaction_count = context.get("interaction_count", 0)
        
        if interaction_count >= 10 and success_rate > 0.8:
            return "high"
        elif interaction_count >= 5 and success_rate > 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_oldest_context_age_minutes(self) -> float:
        """Récupère l'âge du plus ancien contexte en minutes"""
        
        if not self.session_contexts:
            return 0
        
        oldest_time = None
        for context in self.session_contexts.values():
            try:
                created_at = datetime.fromisoformat(context["created_at"].replace('Z', '+00:00'))
                if oldest_time is None or created_at < oldest_time:
                    oldest_time = created_at
            except Exception:
                continue
        
        if oldest_time:
            age_seconds = (datetime.now(timezone.utc) - oldest_time).total_seconds()
            return age_seconds / 60
        
        return 0
    
    def _estimate_memory_usage(self) -> float:
        """Estime l'usage mémoire en MB (approximatif)"""
        
        try:
            # Estimation très approximative basée sur la sérialisation JSON
            total_size = 0
            for context in self.session_contexts.values():
                json_str = json.dumps(context)
                total_size += len(json_str.encode('utf-8'))
            
            return total_size / (1024 * 1024)  # Conversion en MB
            
        except Exception:
            return 0.0


class PersonalizationEngine:
    """Moteur de personnalisation basé sur le contexte utilisateur"""
    
    def __init__(self, context_manager: TemporaryContextManager):
        self.context_manager = context_manager
    
    def personalize_response_generation(self, user_id: int, base_prompt: str) -> str:
        """Personnalise le prompt de génération selon le contexte utilisateur"""
        
        context_summary = self.context_manager.get_context_summary(user_id)
        
        personalization_addons = []
        
        # Style de communication
        style = context_summary.get("communication_style", "balanced")
        if style == "concise":
            personalization_addons.append("Réponds de manière concise et directe.")
        elif style == "detailed":
            personalization_addons.append("Fournis une réponse détaillée avec des explications approfondies.")
        
        # Niveau de détail
        detail_level = context_summary.get("detail_level", "medium")
        if detail_level == "basic":
            personalization_addons.append("Utilise un langage simple et évite les termes techniques.")
        elif detail_level == "advanced":
            personalization_addons.append("Tu peux utiliser des termes financiers techniques et des analyses avancées.")
        
        # Utilisateur récurrent
        if context_summary.get("is_returning_user"):
            frequent_merchants = context_summary.get("frequent_merchants", [])
            if frequent_merchants:
                merchants_str = ", ".join(frequent_merchants[:3])
                personalization_addons.append(f"L'utilisateur consulte fréquemment: {merchants_str}.")
        
        # Préférences d'intention
        preferred_intents = context_summary.get("preferred_intents", [])
        if preferred_intents:
            intent_str = ", ".join(preferred_intents[:2])
            personalization_addons.append(f"L'utilisateur s'intéresse souvent à: {intent_str}.")
        
        if personalization_addons:
            personalization_text = " ".join(personalization_addons)
            return f"{base_prompt}\n\nPersonnalisation: {personalization_text}"
        
        return base_prompt
    
    def get_personalization_context(self, user_id: int) -> Dict[str, Any]:
        """Récupère le contexte de personnalisation pour les templates"""
        
        return self.context_manager.get_context_summary(user_id)