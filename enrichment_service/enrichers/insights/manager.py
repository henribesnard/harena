"""
Gestionnaire d'insights financiers.

Ce module contient la logique de gestion des insights : filtrage, 
classement, stockage et récupération.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
from uuid import uuid4

from enrichment_service.core.logging import get_contextual_logger
from enrichment_service.core.config import enrichment_settings
from enrichment_service.enrichers.insights.data_models import (
    FinancialInsight, InsightType, Priority, InsightAnalytics, 
    InsightRecommendation
)

logger = logging.getLogger(__name__)

class InsightManager:
    """
    Gestionnaire central des insights financiers.
    
    Cette classe gère le cycle de vie complet des insights :
    filtrage, classement, stockage, récupération et analytics.
    """
    
    def __init__(self, embedding_service, qdrant_service):
        """
        Initialise le gestionnaire d'insights.
        
        Args:
            embedding_service: Service d'embedding
            qdrant_service: Service Qdrant
        """
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        
        # Configuration
        self.confidence_threshold = enrichment_settings.insight_confidence_threshold
        self.max_insights_per_user = enrichment_settings.max_insights_per_user
    
    async def filter_and_rank_insights(self, insights: List[FinancialInsight]) -> List[FinancialInsight]:
        """
        Filtre et classe les insights par importance.
        
        Args:
            insights: Liste des insights générés
            
        Returns:
            List[FinancialInsight]: Insights filtrés et classés
        """
        # Filtrer par score de confiance
        filtered_insights = [
            insight for insight in insights 
            if insight.metrics.confidence_score >= self.confidence_threshold
        ]
        
        # Éviter les doublons (même type d'insight)
        seen_types = set()
        unique_insights = []
        
        for insight in filtered_insights:
            # Créer une clé unique basée sur le type et l'utilisateur
            insight_key = f"{insight.insight_type.value}_{insight.user_id}"
            
            if insight_key not in seen_types:
                unique_insights.append(insight)
                seen_types.add(insight_key)
        
        # Classer par priorité puis par score de confiance
        unique_insights.sort(
            key=lambda x: (x.priority.value, x.metrics.confidence_score), 
            reverse=True
        )
        
        # Limiter le nombre d'insights
        return unique_insights[:self.max_insights_per_user]
    
    async def store_insights(self, insights: List[FinancialInsight]) -> Dict[str, Any]:
        """
        Stocke une liste d'insights dans Qdrant.
        
        Args:
            insights: Liste des insights à stocker
            
        Returns:
            Dict: Résultat du stockage
        """
        stored_count = 0
        errors = []
        
        for insight in insights:
            try:
                await self._store_single_insight(insight)
                stored_count += 1
            except Exception as e:
                error_msg = f"Erreur stockage insight {insight.insight_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return {
            "stored_count": stored_count,
            "total_insights": len(insights),
            "errors": errors,
            "success_rate": (stored_count / len(insights)) * 100 if insights else 0
        }
    
    async def _store_single_insight(self, insight: FinancialInsight):
        """
        Stocke un insight unique dans Qdrant.
        
        Args:
            insight: Insight à stocker
        """
        # Générer l'embedding de l'insight
        embedding_text = f"{insight.title} {insight.description} {insight.narrative}"
        
        vector = await self.embedding_service.generate_embedding(embedding_text)
        
        # Stocker dans Qdrant
        await self.qdrant_service.upsert_point(
            collection_name="financial_insights",
            point_id=insight.insight_id,
            vector=vector,
            payload=insight.to_dict()
        )
    
    async def get_user_insights(
        self, 
        user_id: int, 
        insight_type: Optional[InsightType] = None, 
        limit: int = 10
    ) -> List[FinancialInsight]:
        """
        Récupère les insights d'un utilisateur depuis Qdrant.
        
        Args:
            user_id: ID de l'utilisateur
            insight_type: Type d'insight à filtrer (optionnel)
            limit: Nombre maximum d'insights
            
        Returns:
            List[FinancialInsight]: Liste des insights
        """
        filter_conditions = {"user_id": user_id}
        if insight_type:
            filter_conditions["insight_type"] = insight_type.value
        
        try:
            results = await self.qdrant_service.search_points(
                collection_name="financial_insights",
                filter_conditions=filter_conditions,
                limit=limit
            )
            
            insights = []
            for result in results:
                try:
                    insight = FinancialInsight.from_dict(result["payload"])
                    insights.append(insight)
                except Exception as e:
                    logger.warning(f"Erreur lors de la désérialisation de l'insight: {e}")
            
            # Trier par priorité et date de création
            insights.sort(
                key=lambda x: (x.priority.value, x.created_at), 
                reverse=True
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des insights: {e}")
            return []
    
    async def search_insights(
        self, 
        user_id: int, 
        query: str, 
        limit: int = 10
    ) -> List[FinancialInsight]:
        """
        Recherche des insights par contenu textuel.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            List[FinancialInsight]: Insights correspondants
        """
        try:
            # Générer l'embedding de la requête
            query_vector = await self.embedding_service.generate_embedding(query)
            
            # Rechercher dans Qdrant
            results = await self.qdrant_service.search_similar(
                collection_name="financial_insights",
                query_vector=query_vector,
                filter_conditions={"user_id": user_id},
                limit=limit
            )
            
            insights = []
            for result in results:
                try:
                    insight = FinancialInsight.from_dict(result["payload"])
                    # Ajouter le score de similarité
                    insight.metrics.confidence_score = result.get("score", 0)
                    insights.append(insight)
                except Exception as e:
                    logger.warning(f"Erreur lors de la désérialisation: {e}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche d'insights: {e}")
            return []
    
    async def mark_insight_as_read(self, insight_id: str, user_id: int) -> bool:
        """
        Marque un insight comme lu par l'utilisateur.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si marqué avec succès
        """
        try:
            # Récupérer l'insight
            results = await self.qdrant_service.search_points(
                collection_name="financial_insights",
                filter_conditions={"id": insight_id, "user_id": user_id},
                limit=1
            )
            
            if not results:
                return False
            
            insight_data = results[0]["payload"]
            
            # Mettre à jour le statut
            insight_data["read_at"] = datetime.now().isoformat()
            insight_data["status"] = "read"
            
            # Regénérer l'embedding
            embedding_text = f"{insight_data['title']} {insight_data['description']}"
            vector = await self.embedding_service.generate_embedding(embedding_text)
            
            # Mettre à jour dans Qdrant
            await self.qdrant_service.upsert_point(
                collection_name="financial_insights",
                point_id=insight_id,
                vector=vector,
                payload=insight_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du marquage comme lu: {e}")
            return False
    
    async def dismiss_insight(self, insight_id: str, user_id: int) -> bool:
    async def dismiss_insight(self, insight_id: str, user_id: int) -> bool:
        """
        Supprime/rejette un insight spécifique.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        try:
            return await self.qdrant_service.delete_points(
                collection_name="financial_insights",
                filter_conditions={"id": insight_id, "user_id": user_id}
            )
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'insight: {e}")
            return False
    
    async def delete_user_insights(self, user_id: int) -> bool:
        """
        Supprime tous les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        try:
            return await self.qdrant_service.delete_points(
                collection_name="financial_insights",
                filter_conditions={"user_id": user_id}
            )
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des insights: {e}")
            return False
    
    async def get_insight_analytics(self, user_id: int, days: int = 30) -> InsightAnalytics:
        """
        Génère des analytics sur les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days: Période d'analyse en jours
            
        Returns:
            InsightAnalytics: Analytics des insights
        """
        insights = await self.get_user_insights(user_id, limit=100)
        
        if not insights:
            return InsightAnalytics(period_days=days)
        
        # Filtrer par période
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_insights = [
            insight for insight in insights
            if insight.created_at >= cutoff_date
        ]
        
        # Calculer les métriques
        total_generated = len(recent_insights)
        read_insights = len([i for i in recent_insights if i.read_at])
        dismissed_insights = len([i for i in recent_insights if i.status == "dismissed"])
        
        engagement_rate = (read_insights / total_generated * 100) if total_generated > 0 else 0
        
        # Analyser les types
        type_frequency = defaultdict(int)
        for insight in recent_insights:
            type_frequency[insight.insight_type.value] += 1
        
        # Analyser les priorités
        priority_distribution = defaultdict(int)
        for insight in recent_insights:
            priority_distribution[insight.priority.value] += 1
        
        # Analyser les domaines financiers
        scope_distribution = defaultdict(int)
        for insight in recent_insights:
            scope_distribution[insight.financial_scope.value] += 1
        
        # Calculer l'impact potentiel
        total_potential_impact = 0
        for insight in recent_insights:
            if insight.metrics.potential_impact:
                total_potential_impact += insight.metrics.potential_impact
        
        # Calculer la confiance moyenne
        average_confidence = 0
        if recent_insights:
            total_confidence = sum(i.metrics.confidence_score for i in recent_insights)
            average_confidence = total_confidence / len(recent_insights)
        
        # Trouver le type le plus commun
        most_common_type = None
        if type_frequency:
            most_common_type = max(type_frequency.items(), key=lambda x: x[1])[0]
        
        return InsightAnalytics(
            total_insights=total_generated,
            insights_by_type=dict(type_frequency),
            insights_by_priority=dict(priority_distribution),
            insights_by_scope=dict(scope_distribution),
            engagement_rate=round(engagement_rate, 1),
            average_confidence=round(average_confidence, 2),
            total_potential_impact=total_potential_impact,
            most_common_type=most_common_type,
            period_days=days
        )
    
    async def generate_personalized_recommendations(self, user_id: int) -> List[InsightRecommendation]:
        """
        Génère des recommandations personnalisées basées sur tous les insights.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[InsightRecommendation]: Recommandations personnalisées
        """
        insights = await self.get_user_insights(user_id, limit=50)
        
        if not insights:
            return []
        
        recommendations = []
        
        # Analyser les patterns dans les insights
        spending_insights = [
            i for i in insights 
            if i.financial_scope == FinancialScope.SPENDING
        ]
        saving_insights = [
            i for i in insights 
            if i.financial_scope == FinancialScope.SAVING
        ]
        
        # Recommandation globale sur les dépenses
        if len(spending_insights) >= 3:
            total_potential_savings = sum(
                i.metrics.numerical_value or 0 for i in spending_insights 
                if i.insight_type in [InsightType.SAVINGS_OPPORTUNITY, InsightType.SUBSCRIPTION_ANALYSIS]
            )
            
            if total_potential_savings > 50:
                recommendations.append(InsightRecommendation(
                    recommendation_type="spending_optimization",
                    priority="high",
                    title="Optimisation des dépenses",
                    description=f"Économies potentielles identifiées: ~{total_potential_savings:.0f}€/mois",
                    actions=[
                        "Réviser vos abonnements et services",
                        "Optimiser vos catégories de dépenses variables",
                        "Mettre en place un budget strict"
                    ],
                    estimated_impact=f"{total_potential_savings:.0f}€/mois",
                    affected_insights=[i.insight_id for i in spending_insights[:3]]
                ))
        
        # Recommandation sur l'épargne
        if len(saving_insights) >= 2:
            low_savings_insights = [
                i for i in saving_insights 
                if i.insight_type == InsightType.LOW_SAVINGS_RATE
            ]
            
            if low_savings_insights:
                recommendations.append(InsightRecommendation(
                    recommendation_type="savings_improvement",
                    priority="high",
                    title="Amélioration de l'épargne",
                    description="Votre taux d'épargne peut être optimisé",
                    actions=[
                        "Automatiser un virement d'épargne mensuel",
                        "Fixer un objectif d'épargne de 10-20% des revenus",
                        "Explorer des placements pour faire fructifier votre épargne"
                    ],
                    estimated_impact="Amélioration de la sécurité financière",
                    affected_insights=[i.insight_id for i in low_savings_insights]
                ))
        
        # Recommandation sur la planification budgétaire
        budget_insights = [
            i for i in insights 
            if "budget" in i.tags
        ]
        
        if len(budget_insights) >= 2:
            recommendations.append(InsightRecommendation(
                recommendation_type="budget_planning",
                priority="medium",
                title="Planification budgétaire",
                description="Votre gestion budgétaire pourrait être structurée",
                actions=[
                    "Créer un budget mensuel détaillé",
                    "Suivre vos dépenses par catégorie",
                    "Planifier vos gros achats à l'avance"
                ],
                estimated_impact="Meilleure maîtrise des finances",
                affected_insights=[i.insight_id for i in budget_insights[:2]]
            ))
        
        # Recommandation personnalisée basée sur les insights les plus fréquents
        insight_types = [i.insight_type for i in insights]
        if insight_types:
            from collections import Counter
            most_common_type = Counter(insight_types).most_common(1)[0][0]
            type_count = insight_types.count(most_common_type)
            
            if type_count >= 3:
                type_recommendations = {
                    InsightType.SPENDING_INCREASE: InsightRecommendation(
                        recommendation_type="personalized",
                        priority="medium",
                        title="Maîtrise des dépenses croissantes",
                        description="Vos dépenses ont tendance à augmenter",
                        actions=[
                            "Identifier les causes d'augmentation", 
                            "Mettre en place des alertes de budget"
                        ],
                        estimated_impact="Amélioration ciblée"
                    ),
                    InsightType.CATEGORY_CONCENTRATION: InsightRecommendation(
                        recommendation_type="personalized",
                        priority="medium",
                        title="Diversification des dépenses",
                        description="Vos dépenses se concentrent sur peu de catégories",
                        actions=[
                            "Analyser si cette concentration est justifiée", 
                            "Chercher des alternatives moins coûteuses"
                        ],
                        estimated_impact="Amélioration ciblée"
                    ),
                    InsightType.LARGE_TRANSACTION: InsightRecommendation(
                        recommendation_type="personalized",
                        priority="medium",
                        title="Gestion des gros achats",
                        description="Vous effectuez régulièrement de gros achats",
                        actions=[
                            "Planifier les gros achats", 
                            "Constituer une réserve pour les achats importants"
                        ],
                        estimated_impact="Amélioration ciblée"
                    )
                }
                
                if most_common_type in type_recommendations:
                    recommendation = type_recommendations[most_common_type]
                    recommendation.affected_insights = [
                        i.insight_id for i in insights 
                        if i.insight_type == most_common_type
                    ][:3]
                    recommendations.append(recommendation)
        
        return recommendations[:5]  # Limiter à 5 recommandations
    
    async def get_insights_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Génère un résumé des insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Résumé des insights
        """
        insights = await self.get_user_insights(user_id, limit=50)
        
        if not insights:
            return {
                "total_insights": 0,
                "by_priority": {},
                "by_type": {},
                "by_scope": {},
                "key_recommendations": [],
                "summary_text": "Aucun insight disponible pour le moment."
            }
        
        # Analyser les insights
        by_priority = defaultdict(int)
        by_type = defaultdict(int)
        by_scope = defaultdict(int)
        
        high_priority_insights = []
        
        for insight in insights:
            priority = insight.priority.value
            insight_type = insight.insight_type.value
            financial_scope = insight.financial_scope.value
            
            by_priority[priority] += 1
            by_type[insight_type] += 1
            by_scope[financial_scope] += 1
            
            if insight.priority.value >= 4:
                high_priority_insights.append(insight)
        
        # Extraire les recommandations clés
        key_recommendations = []
        for insight in high_priority_insights[:5]:  # Top 5 insights prioritaires
            for action in insight.suggested_actions[:2]:  # 2 premières actions
                key_recommendations.append(action.action_text)
        
        # Générer le texte de résumé
        total_insights = len(insights)
        critical_count = by_priority.get(5, 0)
        high_count = by_priority.get(4, 0)
        
        summary_parts = [f"Vous avez {total_insights} insights actifs."]
        
        if critical_count > 0:
            summary_parts.append(f"{critical_count} insight(s) critique(s) nécessitent votre attention immédiate.")
        
        if high_count > 0:
            summary_parts.append(f"{high_count} insight(s) importante(s) à considérer.")
        
        # Domaines principaux
        if by_scope:
            top_scope = max(by_scope.items(), key=lambda x: x[1])[0]
            summary_parts.append(f"Focus principal sur: {top_scope}.")
        
        summary_text = " ".join(summary_parts)
        
        return {
            "total_insights": total_insights,
            "by_priority": dict(by_priority),
            "by_type": dict(by_type),
            "by_scope": dict(by_scope),
            "key_recommendations": key_recommendations[:8],  # Limiter à 8 recommandations
            "high_priority_insights": len(high_priority_insights),
            "summary_text": summary_text
        }
    
    async def cleanup_expired_insights(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Nettoie les insights expirés.
        
        Args:
            user_id: ID de l'utilisateur (optionnel, sinon tous)
            
        Returns:
            Dict: Résultat du nettoyage
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="insight_cleanup"
        )
        
        try:
            filter_conditions = {}
            if user_id:
                filter_conditions["user_id"] = user_id
            
            # Récupérer tous les insights
            results = await self.qdrant_service.search_points(
                collection_name="financial_insights",
                filter_conditions=filter_conditions,
                limit=1000  # Limite élevée pour le nettoyage
            )
            
            now = datetime.now()
            expired_insights = []
            
            for result in results:
                insight_data = result["payload"]
                expires_at = insight_data.get("expires_at")
                
                if expires_at:
                    expiry_date = datetime.fromisoformat(expires_at)
                    if expiry_date < now:
                        expired_insights.append(insight_data["id"])
            
            # Supprimer les insights expirés
            deleted_count = 0
            for insight_id in expired_insights:
                try:
                    await self.qdrant_service.delete_points(
                        collection_name="financial_insights",
                        filter_conditions={"id": insight_id}
                    )
                    deleted_count += 1
                except Exception as e:
                    ctx_logger.warning(f"Erreur suppression insight {insight_id}: {e}")
            
            result = {
                "total_checked": len(results),
                "expired_found": len(expired_insights),
                "deleted_count": deleted_count,
                "cleanup_date": now.isoformat()
            }
            
            ctx_logger.info(f"Nettoyage terminé: {deleted_count}/{len(expired_insights)} insights expirés supprimés")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur lors du nettoyage des insights: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
    
    async def bulk_update_insights_status(
        self, 
        user_id: int, 
        insight_ids: List[str], 
        new_status: str
    ) -> Dict[str, Any]:
        """
        Met à jour le statut de plusieurs insights en une fois.
        
        Args:
            user_id: ID de l'utilisateur
            insight_ids: Liste des IDs d'insights
            new_status: Nouveau statut
            
        Returns:
            Dict: Résultat de la mise à jour
        """
        updated_count = 0
        errors = []
        
        for insight_id in insight_ids:
            try:
                # Récupérer l'insight
                results = await self.qdrant_service.search_points(
                    collection_name="financial_insights",
                    filter_conditions={"id": insight_id, "user_id": user_id},
                    limit=1
                )
                
                if results:
                    insight_data = results[0]["payload"]
                    insight_data["status"] = new_status
                    
                    if new_status == "read":
                        insight_data["read_at"] = datetime.now().isoformat()
                    
                    # Regénérer l'embedding
                    embedding_text = f"{insight_data['title']} {insight_data['description']}"
                    vector = await self.embedding_service.generate_embedding(embedding_text)
                    
                    # Mettre à jour
                    await self.qdrant_service.upsert_point(
                        collection_name="financial_insights",
                        point_id=insight_id,
                        vector=vector,
                        payload=insight_data
                    )
                    
                    updated_count += 1
                
            except Exception as e:
                error_msg = f"Erreur mise à jour insight {insight_id}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return {
            "requested_updates": len(insight_ids),
            "successful_updates": updated_count,
            "errors": errors,
            "success_rate": (updated_count / len(insight_ids)) * 100 if insight_ids else 0
        }