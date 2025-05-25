async def update_insights_for_transaction(self, transaction: RawTransaction):
        """
        Met à jour les insights affectés par une nouvelle transaction.
        
        Args:
            transaction: Nouvelle transaction
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=transaction.user_id,
            transaction_id=transaction.id,
            enrichment_type="insight_update"
        )
        
        try:
            # Pour l'instant, on régénère tous les insights de l'utilisateur
            # Une approche plus sophistiquée pourrait mettre à jour sélectivement
            await self.refresh_user_insights(transaction.user_id)
            ctx_logger.debug("Insights mis à jour suite à nouvelle transaction")
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors de la mise à jour des insights: {e}", exc_info=True)
    
    async def get_insight_by_id(self, insight_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupère un insight spécifique par son ID.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            Optional[Dict]: Insight trouvé ou None
        """
        try:
            insights = await self.manager.get_user_insights(user_id, limit=100)
            
            for insight in insights:
                if insight.insight_id == insight_id:
                    return insight.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'insight {insight_id}: {e}")
            return None
    
    async def search_insights(self, user_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche des insights par contenu textuel.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            List[Dict]: Insights correspondants
        """
        insights = await self.manager.search_insights(user_id, query, limit)
        return [insight.to_dict() for insight in insights]
    
    async def get_insights_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les statistiques globales des insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques des insights
        """
        insights = await self.manager.get_user_insights(user_id, limit=100)
        
        if not insights:
            return {
                "total_insights": 0,
                "insights_by_month": {},
                "average_priority": 0,
                "most_actionable_type": None
            }
        
        # Analyser par mois de création
        insights_by_month = defaultdict(int)
        total_priority = 0
        type_action_count = defaultdict(int)
        
        for insight in insights:
            # Par mois
            month_key = insight.created_at.strftime("%Y-%m")
            insights_by_month[month_key] += 1
            
            # Priorité moyenne"""
Générateur d'insights financiers refactorisé.

Ce module coordonne la génération d'insights en utilisant des analyseurs
spécialisés et un gestionnaire centralisé.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from uuid import uuid4

from enrichment_service.core.logging import get_contextual_logger, log_performance
from enrichment_service.core.exceptions import InsightGenerationError
from enrichment_service.core.config import enrichment_settings
from enrichment_service.db.models import RawTransaction

# Import des modules refactorisés
from enrichment_service.enrichers.insights.data_models import (
    FinancialInsight, InsightType, TimeScope, FinancialScope, Priority
)
from enrichment_service.enrichers.insights.analyzers import (
    SpendingAnalyzer, SavingsAnalyzer, TrendAnalyzer, 
    AnomalyAnalyzer, OpportunityAnalyzer
)
from enrichment_service.enrichers.insights.manager import InsightManager

logger = logging.getLogger(__name__)

class InsightGenerator:
    """
    Générateur d'insights financiers refactorisé.
    
    Cette classe coordonne la génération d'insights en utilisant des
    analyseurs spécialisés et délègue la gestion au InsightManager.
    """
    
    def __init__(self, db: Session, embedding_service, qdrant_service):
        """
        Initialise le générateur d'insights.
        
        Args:
            db: Session de base de données
            embedding_service: Service d'embedding
            qdrant_service: Service Qdrant
        """
        self.db = db
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        
        # Initialiser les analyseurs spécialisés
        self.spending_analyzer = SpendingAnalyzer(db)
        self.savings_analyzer = SavingsAnalyzer(db)
        self.trend_analyzer = TrendAnalyzer(db)
        self.anomaly_analyzer = AnomalyAnalyzer(db)
        self.opportunity_analyzer = OpportunityAnalyzer(db)
        
        # Gestionnaire centralisé
        self.manager = InsightManager(embedding_service, qdrant_service)
    
    @log_performance
    async def generate_user_insights(self, user_id: int) -> List[FinancialInsight]:
        """
        Génère tous les insights pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[FinancialInsight]: Liste des insights générés
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="insight_generation"
        )
        
        ctx_logger.info(f"Génération d'insights pour l'utilisateur {user_id}")
        
        try:
            # Récupérer les données nécessaires
            recent_transactions = await self._get_recent_transactions(user_id, days=30)
            historical_transactions = await self._get_recent_transactions(user_id, days=90)
            
            if not recent_transactions:
                ctx_logger.info("Pas de transactions récentes, aucun insight généré")
                return []
            
            insights = []
            
            # Utiliser les analyseurs spécialisés
            ctx_logger.debug("Analyse des patterns de dépenses")
            insights.extend(await self.spending_analyzer.analyze_spending_patterns(
                user_id, recent_transactions, historical_transactions
            ))
            
            ctx_logger.debug("Analyse des patterns d'épargne")
            insights.extend(await self.savings_analyzer.analyze_savings_patterns(
                user_id, recent_transactions, historical_transactions
            ))
            
            ctx_logger.debug("Analyse des tendances financières")
            insights.extend(await self.trend_analyzer.analyze_financial_trends(
                user_id, recent_transactions, historical_transactions
            ))
            
            ctx_logger.debug("Détection d'anomalies")
            insights.extend(await self.anomaly_analyzer.analyze_anomalies(
                user_id, recent_transactions
            ))
            
            ctx_logger.debug("Analyse des opportunités d'optimisation")
            insights.extend(await self.opportunity_analyzer.analyze_optimization_opportunities(
                user_id, recent_transactions, historical_transactions
            ))
            
            # Filtrer et classer via le gestionnaire
            filtered_insights = await self.manager.filter_and_rank_insights(insights)
            
            # Stocker dans Qdrant
            storage_result = await self.manager.store_insights(filtered_insights)
            
            ctx_logger.info(f"Génération terminée: {len(filtered_insights)} insights créés, {storage_result['stored_count']} stockés")
            
            return filtered_insights
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération d'insights: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise InsightGenerationError(error_msg, "general", {"user_id": user_id})
    
    async def _get_recent_transactions(self, user_id: int, days: int) -> List[RawTransaction]:
        """
        Récupère les transactions récentes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days: Nombre de jours à récupérer
            
        Returns:
            List[RawTransaction]: Liste des transactions
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        transactions = self.db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.date >= cutoff_date,
            RawTransaction.deleted.is_(False),
            RawTransaction.amount != 0
        ).order_by(RawTransaction.date.desc()).all()
        
        return transactions
    
    # Méthodes de délégation vers le gestionnaire
    
    async def get_user_insights(
        self, 
        user_id: int, 
        insight_type: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Récupère les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            insight_type: Type d'insight à filtrer (optionnel)
            limit: Nombre maximum d'insights
            
        Returns:
            List[Dict]: Liste des insights (format dict pour compatibilité)
        """
        insight_type_enum = None
        if insight_type:
            try:
                insight_type_enum = InsightType(insight_type)
            except ValueError:
                logger.warning(f"Type d'insight invalide: {insight_type}")
        
        insights = await self.manager.get_user_insights(user_id, insight_type_enum, limit)
        
        # Convertir en format dict pour compatibilité avec l'ancienne API
        return [insight.to_dict() for insight in insights]
    
    async def delete_user_insights(self, user_id: int) -> bool:
        """
        Supprime tous les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        return await self.manager.delete_user_insights(user_id)
    
    async def refresh_user_insights(self, user_id: int) -> List[FinancialInsight]:
        """
        Rafraîchit tous les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[FinancialInsight]: Nouveaux insights générés
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="insight_refresh"
        )
        
        try:
            # Supprimer les anciens insights
            await self.manager.delete_user_insights(user_id)
            ctx_logger.info("Anciens insights supprimés")
            
            # Générer de nouveaux insights
            new_insights = await self.generate_user_insights(user_id)
            ctx_logger.info(f"Génération de {len(new_insights)} nouveaux insights")
            
            return new_insights
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors du rafraîchissement des insights: {e}")
            raise
    
    async def generate_insight_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Génère un résumé des insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Résumé des insights
        """
        return await self.manager.get_insights_summary(user_id)
    
    async def mark_insight_as_read(self, insight_id: str, user_id: int) -> bool:
        """
        Marque un insight comme lu par l'utilisateur.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si marqué avec succès
        """
        return await self.manager.mark_insight_as_read(insight_id, user_id)
    
    async def dismiss_insight(self, insight_id: str, user_id: int) -> bool:
        """
        Supprime/rejette un insight spécifique.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        return await self.manager.dismiss_insight(insight_id, user_id)
    
    async def get_insight_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Génère des analytics sur les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days: Période d'analyse en jours
            
        Returns:
            Dict: Analytics des insights
        """
        analytics = await self.manager.get_insight_analytics(user_id, days)
        
        # Convertir en dict pour compatibilité
        return {
            "period_days": analytics.period_days,
            "total_insights_generated": analytics.total_insights,
            "insights_read": sum(1 for insight in await self.manager.get_user_insights(user_id, limit=100) if insight.read_at),
            "insights_dismissed": sum(1 for insight in await self.manager.get_user_insights(user_id, limit=100) if insight.status == "dismissed"),
            "engagement_rate": analytics.engagement_rate,
            "type_frequency": analytics.insights_by_type,
            "priority_distribution": analytics.insights_by_priority,
            "scope_distribution": analytics.insights_by_scope,
            "total_potential_impact": analytics.total_potential_impact,
            "average_confidence": analytics.average_confidence,
            "most_common_type": analytics.most_common_type or "none"
        }
    
    async def generate_personalized_recommendations(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Génère des recommandations personnalisées basées sur tous les insights.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[Dict]: Recommandations personnalisées
        """
        recommendations = await self.manager.generate_personalized_recommendations(user_id)
        
        # Convertir en format dict pour compatibilité
        return [
            {
                "type": rec.recommendation_type,
                "priority": rec.priority,
                "title": rec.title,
                "description": rec.description,
                "actions": rec.actions,
                "estimated_impact": rec.estimated_impact,
                "affected_insights": rec.affected_insights
            }
            for rec in recommendations
        ]
    
    async def update_insights_for_transaction(self, transaction: RawTransaction):
        """
        Met à jour les insights affectés par une nouvelle transaction.
        
        Args:
            transaction: Nouvelle