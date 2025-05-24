"""
Générateur de résumés financiers périodiques (refactorisé).

Ce module génère des résumés automatiques des activités financières
par période (mensuel, trimestriel, annuel) avec analyses et insights.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from uuid import uuid4
from calendar import monthrange

from enrichment_service.core.logging import get_contextual_logger, log_performance
from enrichment_service.core.exceptions import SummaryGenerationError
from enrichment_service.core.config import enrichment_settings
from enrichment_service.db.models import RawTransaction
from enrichment_service.enrichers.summary.data_models import FinancialSummary
from enrichment_service.enrichers.summary.analyzer import FinancialAnalyzer
from enrichment_service.enrichers.summary.narrator import SummaryNarrator
from enrichment_service.enrichers.summary.comparator import SummaryComparator

logger = logging.getLogger(__name__)

class SummaryGenerator:
    """
    Générateur de résumés financiers périodiques.
    
    Cette classe analyse les transactions sur différentes périodes
    et génère des résumés complets avec insights et comparaisons.
    """
    
    def __init__(self, db: Session, embedding_service, qdrant_service):
        """
        Initialise le générateur de résumés.
        
        Args:
            db: Session de base de données
            embedding_service: Service d'embedding
            qdrant_service: Service Qdrant
        """
        self.db = db
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        
        # Composants spécialisés
        self.analyzer = FinancialAnalyzer(db)
        self.narrator = SummaryNarrator()
        self.comparator = SummaryComparator(self)
    
    @log_performance
    async def generate_monthly_summary(self, user_id: int, year: int, month: int) -> FinancialSummary:
        """
        Génère un résumé mensuel pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            year: Année
            month: Mois (1-12)
            
        Returns:
            FinancialSummary: Résumé mensuel généré
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="monthly_summary"
        )
        
        # Calculer les dates de début et fin du mois
        period_start = datetime(year, month, 1)
        _, last_day = monthrange(year, month)
        period_end = datetime(year, month, last_day, 23, 59, 59)
        
        period_name = f"{period_start.strftime('%B %Y')}"
        is_complete = period_end < datetime.now()
        
        ctx_logger.info(f"Génération du résumé mensuel pour {period_name}")
        
        try:
            # Récupérer les transactions du mois
            transactions = await self._get_period_transactions(user_id, period_start, period_end)
            
            if not transactions:
                ctx_logger.info("Aucune transaction trouvée pour cette période")
                return self._create_empty_summary(user_id, "monthly", period_name, period_start, period_end, is_complete)
            
            # Créer le résumé de base
            summary = FinancialSummary(
                summary_id=str(uuid4()),
                user_id=user_id,
                period_type="monthly",
                period_name=period_name,
                period_start=period_start,
                period_end=period_end,
                is_complete=is_complete
            )
            
            # Analyser les transactions
            summary = self.analyzer.analyze_transactions(summary, transactions)
            
            # Ajouter les comparaisons mensuelles
            await self.comparator.add_monthly_comparisons(summary, user_id, year, month)
            
            # Générer le narratif
            self.narrator.generate_monthly_narrative(summary)
            
            # Stocker dans Qdrant
            await self._store_summary_in_qdrant(summary)
            
            ctx_logger.info(f"Résumé mensuel généré: {summary.transaction_count} transactions, {summary.net_flow:.2f}€ net")
            
            return summary
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération du résumé mensuel: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise SummaryGenerationError(error_msg, "monthly", {"user_id": user_id, "period": period_name})
    
    @log_performance
    async def generate_quarterly_summary(self, user_id: int, year: int, quarter: int) -> FinancialSummary:
        """
        Génère un résumé trimestriel pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            year: Année
            quarter: Trimestre (1-4)
            
        Returns:
            FinancialSummary: Résumé trimestriel généré
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="quarterly_summary"
        )
        
        # Calculer les dates du trimestre
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        
        period_start = datetime(year, start_month, 1)
        _, last_day = monthrange(year, end_month)
        period_end = datetime(year, end_month, last_day, 23, 59, 59)
        
        period_name = f"Q{quarter} {year}"
        is_complete = period_end < datetime.now()
        
        ctx_logger.info(f"Génération du résumé trimestriel pour {period_name}")
        
        try:
            # Récupérer les transactions du trimestre
            transactions = await self._get_period_transactions(user_id, period_start, period_end)
            
            if not transactions:
                return self._create_empty_summary(user_id, "quarterly", period_name, period_start, period_end, is_complete)
            
            # Créer le résumé de base
            summary = FinancialSummary(
                summary_id=str(uuid4()),
                user_id=user_id,
                period_type="quarterly",
                period_name=period_name,
                period_start=period_start,
                period_end=period_end,
                is_complete=is_complete
            )
            
            # Analyser les transactions
            summary = self.analyzer.analyze_transactions(summary, transactions)
            
            # Ajouter les comparaisons trimestrielles
            await self.comparator.add_quarterly_comparisons(summary, user_id, year, quarter)
            
            # Générer le narratif
            self.narrator.generate_quarterly_narrative(summary)
            
            # Stocker dans Qdrant
            await self._store_summary_in_qdrant(summary)
            
            ctx_logger.info(f"Résumé trimestriel généré: {summary.transaction_count} transactions")
            
            return summary
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération du résumé trimestriel: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise SummaryGenerationError(error_msg, "quarterly", {"user_id": user_id, "period": period_name})
    
    @log_performance
    async def generate_yearly_summary(self, user_id: int, year: int) -> FinancialSummary:
        """
        Génère un résumé annuel pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            year: Année
            
        Returns:
            FinancialSummary: Résumé annuel généré
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="yearly_summary"
        )
        
        period_start = datetime(year, 1, 1)
        period_end = datetime(year, 12, 31, 23, 59, 59)
        period_name = str(year)
        is_complete = period_end < datetime.now()
        
        ctx_logger.info(f"Génération du résumé annuel pour {year}")
        
        try:
            # Récupérer les transactions de l'année
            transactions = await self._get_period_transactions(user_id, period_start, period_end)
            
            if not transactions:
                return self._create_empty_summary(user_id, "yearly", period_name, period_start, period_end, is_complete)
            
            # Créer le résumé de base
            summary = FinancialSummary(
                summary_id=str(uuid4()),
                user_id=user_id,
                period_type="yearly",
                period_name=period_name,
                period_start=period_start,
                period_end=period_end,
                is_complete=is_complete
            )
            
            # Analyser les transactions
            summary = self.analyzer.analyze_transactions(summary, transactions)
            
            # Ajouter les comparaisons annuelles
            await self.comparator.add_yearly_comparisons(summary, user_id, year)
            
            # Générer le narratif
            self.narrator.generate_yearly_narrative(summary)
            
            # Stocker dans Qdrant
            await self._store_summary_in_qdrant(summary)
            
            ctx_logger.info(f"Résumé annuel généré: {summary.transaction_count} transactions")
            
            return summary
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération du résumé annuel: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise SummaryGenerationError(error_msg, "yearly", {"user_id": user_id, "period": period_name})
    
    async def _get_period_transactions(self, user_id: int, start_date: datetime, end_date: datetime) -> List[RawTransaction]:
        """
        Récupère les transactions pour une période donnée.
        
        Args:
            user_id: ID de l'utilisateur
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            List[RawTransaction]: Liste des transactions
        """
        transactions = self.db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.date >= start_date,
            RawTransaction.date <= end_date,
            RawTransaction.deleted.is_(False),
            RawTransaction.amount != 0
        ).order_by(RawTransaction.date.asc()).all()
        
        return transactions
    
    def _create_empty_summary(
        self, 
        user_id: int, 
        period_type: str, 
        period_name: str, 
        period_start: datetime, 
        period_end: datetime, 
        is_complete: bool
    ) -> FinancialSummary:
        """
        Crée un résumé vide pour une période sans transactions.
        
        Args:
            user_id: ID de l'utilisateur
            period_type: Type de période
            period_name: Nom de la période
            period_start: Date de début
            period_end: Date de fin
            is_complete: Si la période est complète
            
        Returns:
            FinancialSummary: Résumé vide
        """
        summary = FinancialSummary(
            summary_id=str(uuid4()),
            user_id=user_id,
            period_type=period_type,
            period_name=period_name,
            period_start=period_start,
            period_end=period_end,
            is_complete=is_complete
        )
        
        summary.narrative_highlights = ["Aucune activité financière pour cette période"]
        summary.summary_text = f"Aucune transaction enregistrée pour {period_name}"
        summary.tags = [period_type, "inactif"]
        
        return summary
    
    async def _store_summary_in_qdrant(self, summary: FinancialSummary):
        """
        Stocke un résumé dans Qdrant.
        
        Args:
            summary: Résumé à stocker
        """
        # Générer l'embedding du résumé
        embedding_text = summary.summary_text
        if summary.narrative_highlights:
            embedding_text += " " + " ".join(summary.narrative_highlights)
        
        vector = await self.embedding_service.generate_embedding(embedding_text)
        
        # Construire le payload
        payload = {
            "id": summary.summary_id,
            "user_id": summary.user_id,
            "period_type": summary.period_type,
            "period_name": summary.period_name,
            "period_start": summary.period_start.isoformat(),
            "period_end": summary.period_end.isoformat(),
            "is_complete": summary.is_complete,
            
            # Métriques globales
            "total_income": summary.total_income,
            "total_expenses": summary.total_expenses,
            "net_flow": summary.net_flow,
            "savings_rate": summary.savings_rate,
            
            # Répartitions
            "income_breakdown": summary.income_breakdown,
            "expense_breakdown": summary.expense_breakdown,
            "top_categories": [
                {
                    "name": cat.category_name,
                    "amount": cat.total_amount,
                    "percentage": cat.percentage
                } for cat in summary.top_categories[:5]
            ],
            "top_merchants": [
                {
                    "name": merchant.merchant_name,
                    "amount": merchant.total_amount,
                    "percentage": merchant.percentage
                } for merchant in summary.top_merchants[:5]
            ],
            "recurring_spending": summary.recurring_spending,
            "recurring_percentage": summary.recurring_percentage,
            
            # Comparaisons
            "vs_previous_period": summary.vs_previous_period,
            "vs_average": summary.vs_average,
            "significant_changes": summary.significant_changes,
            "anomalies": summary.anomalies,
            
            # Narratif
            "narrative_highlights": summary.narrative_highlights,
            "financial_health_indicators": summary.financial_health_indicators,
            "summary_text": summary.summary_text,
            "tags": summary.tags
        }
        
        # Stocker dans Qdrant
        await self.qdrant_service.upsert_point(
            collection_name="financial_summaries",
            point_id=summary.summary_id,
            vector=vector,
            payload=payload
        )
    
    # Méthodes utilitaires
    async def update_summaries_for_transaction(self, transaction: RawTransaction):
        """
        Met à jour les résumés affectés par une nouvelle transaction.
        
        Args:
            transaction: Nouvelle transaction
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=transaction.user_id,
            transaction_id=transaction.id,
            enrichment_type="summary_update"
        )
        
        try:
            # Déterminer les périodes à mettre à jour
            tx_date = transaction.date
            
            # Résumé mensuel
            await self.generate_monthly_summary(transaction.user_id, tx_date.year, tx_date.month)
            ctx_logger.debug(f"Résumé mensuel mis à jour pour {tx_date.year}-{tx_date.month}")
            
            # Résumé trimestriel
            quarter = ((tx_date.month - 1) // 3) + 1
            await self.generate_quarterly_summary(transaction.user_id, tx_date.year, quarter)
            ctx_logger.debug(f"Résumé trimestriel mis à jour pour Q{quarter} {tx_date.year}")
            
            # Résumé annuel
            await self.generate_yearly_summary(transaction.user_id, tx_date.year)
            ctx_logger.debug(f"Résumé annuel mis à jour pour {tx_date.year}")
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors de la mise à jour des résumés: {e}", exc_info=True)
    
    async def get_user_summaries(self, user_id: int, period_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les résumés d'un utilisateur depuis Qdrant.
        
        Args:
            user_id: ID de l'utilisateur
            period_type: Type de période à filtrer (optionnel)
            limit: Nombre maximum de résumés
            
        Returns:
            List[Dict]: Liste des résumés
        """
        filter_conditions = {"user_id": user_id}
        if period_type:
            filter_conditions["period_type"] = period_type
        
        try:
            results = await self.qdrant_service.search_points(
                collection_name="financial_summaries",
                filter_conditions=filter_conditions,
                limit=limit
            )
            
            summaries = []
            for result in results:
                summaries.append(result["payload"])
            
            return summaries
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des résumés: {e}")
            return []
    
    async def delete_user_summaries(self, user_id: int) -> bool:
        """
        Supprime tous les résumés d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        try:
            return await self.qdrant_service.delete_points(
                collection_name="financial_summaries",
                filter_conditions={"user_id": user_id}
            )
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des résumés: {e}")
            return False