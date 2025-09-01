"""
Service de génération d'insights automatiques Phase 5
Analyse les données financières pour produire des insights pertinents et actionnables
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from conversation_service.models.responses.conversation_responses import Insight
from conversation_service.models.contracts.search_service import SearchResponse

logger = logging.getLogger(__name__)


class InsightGenerator:
    """Générateur d'insights automatiques basé sur l'analyse des données financières"""
    
    def __init__(self):
        # Configuration des seuils pour la détection d'insights
        self.thresholds = {
            "spending_trends": {
                "significant_increase": 0.20,  # > 20% augmentation
                "significant_decrease": -0.15,  # > 15% diminution
                "unusual_spike": 2.0,  # 2x moyenne habituelle
            },
            "transaction_patterns": {
                "high_frequency": 15,  # > 15 transactions
                "low_frequency": 3,  # < 3 transactions
                "high_average": 100.0,  # > 100€ moyenne
                "low_average": 5.0,  # < 5€ moyenne
            },
            "merchant_patterns": {
                "loyalty_threshold": 5,  # > 5 transactions chez même marchand
                "diversity_threshold": 10,  # > 10 marchands différents
            },
            "category_patterns": {
                "dominance_threshold": 0.30,  # > 30% des dépenses
                "emergence_threshold": 0.10,  # > 10% nouvelle catégorie
            }
        }
    
    def generate_insights(
        self, 
        search_results: Optional[SearchResponse],
        intent: Dict[str, Any],
        entities: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ) -> List[Insight]:
        """Génère une liste d'insights basés sur les données disponibles"""
        
        insights = []
        
        try:
            # Insights de base sur les résultats
            basic_insights = self._generate_basic_insights(analysis_data)
            insights.extend(basic_insights)
            
            # Insights sur les patterns de dépenses
            if analysis_data.get("has_results"):
                spending_insights = self._generate_spending_insights(analysis_data)
                insights.extend(spending_insights)
                
                # Insights sur les patterns de transactions
                transaction_insights = self._generate_transaction_insights(analysis_data)
                insights.extend(transaction_insights)
                
                # Insights spécifiques par intention
                intent_insights = self._generate_intent_specific_insights(intent, analysis_data)
                insights.extend(intent_insights)
                
                # Insights sur la diversification
                diversity_insights = self._generate_diversity_insights(analysis_data)
                insights.extend(diversity_insights)
            
            # Limiter le nombre d'insights pour éviter la surcharge
            insights = self._prioritize_insights(insights)
            
            logger.info(f"Généré {len(insights)} insights automatiques")
            
        except Exception as e:
            logger.error(f"Erreur génération insights: {str(e)}")
            # Insight de fallback
            insights.append(Insight(
                type="pattern",
                title="Analyse en cours",
                description="L'analyse de vos données est en cours de traitement",
                severity="info",
                confidence=0.5
            ))
        
        return insights
    
    def _generate_basic_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Génère des insights de base sur les résultats"""
        
        insights = []
        
        if not analysis_data.get("has_results"):
            insights.append(Insight(
                type="pattern",
                title="Aucune donnée trouvée",
                description="Aucune transaction correspondante pour cette recherche",
                severity="neutral",
                confidence=0.95
            ))
            return insights
        
        total_hits = analysis_data.get("total_hits", 0)
        
        # Insight sur le volume de données
        if total_hits == 1:
            insights.append(Insight(
                type="pattern",
                title="Transaction unique",
                description="Une seule transaction correspond à vos critères",
                severity="info",
                confidence=0.9
            ))
        elif total_hits < self.thresholds["transaction_patterns"]["low_frequency"]:
            insights.append(Insight(
                type="pattern",
                title="Peu d'activité",
                description=f"Seulement {total_hits} transactions trouvées",
                severity="neutral",
                confidence=0.8
            ))
        elif total_hits > self.thresholds["transaction_patterns"]["high_frequency"]:
            insights.append(Insight(
                type="pattern",
                title="Activité élevée",
                description=f"{total_hits} transactions identifiées - activité soutenue",
                severity="info",
                confidence=0.8
            ))
        
        return insights
    
    def _generate_spending_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Génère des insights sur les patterns de dépenses"""
        
        insights = []
        
        total_amount = analysis_data.get("total_amount")
        average_amount = analysis_data.get("average_amount")
        
        if not total_amount:
            return insights
        
        # Insight sur les montants élevés
        if total_amount > 1000:
            insights.append(Insight(
                type="spending",
                title="Dépenses importantes",
                description=f"Montant total de {total_amount:.2f}€ sur cette période",
                severity="warning",
                confidence=0.85
            ))
        elif total_amount > 500:
            insights.append(Insight(
                type="spending",
                title="Dépenses modérées",
                description=f"{total_amount:.2f}€ dépensés - montant raisonnable",
                severity="info",
                confidence=0.8
            ))
        
        # Insight sur les montants moyens
        if average_amount:
            if average_amount > self.thresholds["transaction_patterns"]["high_average"]:
                insights.append(Insight(
                    type="spending",
                    title="Transactions de montant élevé",
                    description=f"Montant moyen de {average_amount:.2f}€ par transaction",
                    severity="info",
                    confidence=0.75
                ))
            elif average_amount < self.thresholds["transaction_patterns"]["low_average"]:
                insights.append(Insight(
                    type="spending",
                    title="Petites transactions",
                    description=f"Montant moyen faible de {average_amount:.2f}€",
                    severity="neutral",
                    confidence=0.7
                ))
        
        return insights
    
    def _generate_transaction_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Génère des insights sur les patterns de transactions"""
        
        insights = []
        
        transaction_count = analysis_data.get("transaction_count")
        min_amount = analysis_data.get("min_amount")
        max_amount = analysis_data.get("max_amount")
        
        # Insight sur la dispersion des montants
        if min_amount and max_amount and min_amount > 0:
            ratio = max_amount / min_amount
            if ratio > 10:  # Grande dispersion
                insights.append(Insight(
                    type="pattern",
                    title="Large éventail de montants",
                    description=f"Montants variant de {min_amount:.2f}€ à {max_amount:.2f}€",
                    severity="info",
                    confidence=0.7
                ))
        
        # Insight sur la régularité
        if transaction_count and analysis_data.get("average_amount"):
            total_amount = analysis_data.get("total_amount", 0)
            expected_total = transaction_count * analysis_data["average_amount"]
            if abs(total_amount - expected_total) < 0.01:  # Montants très réguliers
                insights.append(Insight(
                    type="pattern",
                    title="Montants réguliers",
                    description="Vos transactions ont des montants très similaires",
                    severity="neutral",
                    confidence=0.6
                ))
        
        return insights
    
    def _generate_intent_specific_insights(self, intent: Dict[str, Any], analysis_data: Dict[str, Any]) -> List[Insight]:
        """Génère des insights spécifiques selon l'intention"""
        
        insights = []
        intent_type = intent.get("intent_type")
        
        if intent_type == "SEARCH_BY_MERCHANT":
            insights.extend(self._generate_merchant_insights(analysis_data))
        elif intent_type == "SPENDING_ANALYSIS":
            insights.extend(self._generate_spending_analysis_insights(analysis_data))
        elif intent_type == "CATEGORY_ANALYSIS":
            insights.extend(self._generate_category_insights(analysis_data))
        elif intent_type == "BALANCE_INQUIRY":
            insights.extend(self._generate_balance_insights(analysis_data))
        
        return insights
    
    def _generate_merchant_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Insights spécifiques pour l'analyse par marchand"""
        
        insights = []
        primary_entity = analysis_data.get("primary_entity")
        transaction_count = analysis_data.get("transaction_count", 0)
        
        if not primary_entity:
            return insights
        
        # Fidélité au marchand
        if transaction_count >= self.thresholds["merchant_patterns"]["loyalty_threshold"]:
            insights.append(Insight(
                type="pattern",
                title="Fidélité élevée",
                description=f"Vous êtes un client fidèle de {primary_entity} avec {transaction_count} transactions",
                severity="positive",
                confidence=0.8
            ))
        
        # Premier achat
        if transaction_count == 1:
            insights.append(Insight(
                type="pattern",
                title="Nouveau marchand",
                description=f"Premier achat identifié chez {primary_entity}",
                severity="info",
                confidence=0.9
            ))
        
        return insights
    
    def _generate_spending_analysis_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Insights pour l'analyse générale des dépenses"""
        
        insights = []
        
        # Analyse de la répartition si données disponibles
        if "merchants_breakdown" in analysis_data:
            merchants_data = analysis_data["merchants_breakdown"]
            if len(merchants_data) > 1:
                # Marchand dominant
                sorted_merchants = sorted(merchants_data, key=lambda x: x["amount"], reverse=True)
                top_merchant = sorted_merchants[0]
                total_spent = sum(m["amount"] for m in merchants_data)
                
                if total_spent > 0:
                    dominance = top_merchant["amount"] / total_spent
                    if dominance > 0.5:
                        insights.append(Insight(
                            type="pattern",
                            title="Marchand dominant",
                            description=f"{top_merchant['name']} représente {dominance*100:.0f}% de vos dépenses",
                            severity="info",
                            confidence=0.8
                        ))
        
        return insights
    
    def _generate_category_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Insights pour l'analyse par catégorie"""
        
        insights = []
        
        if "categories_breakdown" in analysis_data:
            categories = analysis_data["categories_breakdown"]
            
            # Catégorie dominante
            if len(categories) > 0:
                top_category = max(categories, key=lambda x: x.get("amount", 0))
                percentage = top_category.get("percentage", 0)
                
                if percentage > self.thresholds["category_patterns"]["dominance_threshold"] * 100:
                    insights.append(Insight(
                        type="category",
                        title="Catégorie dominante",
                        description=f"La catégorie '{top_category['name']}' représente {percentage:.0f}% de vos dépenses",
                        severity="info",
                        confidence=0.8
                    ))
        
        return insights
    
    def _generate_diversity_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Insights sur la diversification des dépenses"""
        
        insights = []
        unique_merchants = analysis_data.get("unique_merchants", 0)
        
        if unique_merchants >= self.thresholds["merchant_patterns"]["diversity_threshold"]:
            insights.append(Insight(
                type="pattern",
                title="Forte diversification",
                description=f"Vos dépenses sont réparties sur {unique_merchants} marchands différents",
                severity="positive",
                confidence=0.7
            ))
        elif unique_merchants == 1:
            merchant_name = analysis_data.get("primary_entity", "un seul marchand")
            insights.append(Insight(
                type="pattern",
                title="Concentration sur un marchand",
                description=f"Toutes vos transactions sont chez {merchant_name}",
                severity="neutral",
                confidence=0.9
            ))
        
        return insights
    
    def _generate_balance_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Insights pour les demandes de solde"""
        
        insights = []
        
        # Ces insights seraient générés si nous avions accès aux données de solde
        # Pour l'instant, nous retournons des insights génériques
        insights.append(Insight(
            type="liquidity",
            title="Information de solde",
            description="Votre solde reflète votre situation financière actuelle",
            severity="info",
            confidence=0.5
        ))
        
        return insights
    
    def _prioritize_insights(self, insights: List[Insight]) -> List[Insight]:
        """Priorise et limite le nombre d'insights pour éviter la surcharge"""
        
        if len(insights) <= 5:
            return insights
        
        # Tri par pertinence : confidence * severity_weight
        severity_weights = {
            "alert": 1.0,
            "warning": 0.8,
            "positive": 0.7,
            "info": 0.6,
            "neutral": 0.4
        }
        
        def insight_score(insight: Insight) -> float:
            severity_weight = severity_weights.get(insight.severity, 0.5)
            return insight.confidence * severity_weight
        
        # Trier par score décroissant et prendre les 5 meilleurs
        prioritized = sorted(insights, key=insight_score, reverse=True)
        return prioritized[:5]


class AdvancedInsightGenerator(InsightGenerator):
    """Version avancée avec comparaisons temporelles et détection de tendances"""
    
    def __init__(self):
        super().__init__()
        # Configurations avancées pour les comparaisons temporelles
        self.temporal_thresholds = {
            "trend_significance": 0.15,  # 15% de variation
            "seasonality_detection": 0.25,  # 25% variation saisonnière
            "volatility_threshold": 0.40  # 40% coefficient de variation
        }
    
    def generate_temporal_insights(
        self, 
        current_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> List[Insight]:
        """Génère des insights basés sur les comparaisons temporelles"""
        
        insights = []
        
        if not historical_data:
            return insights
        
        try:
            # Comparaison des montants
            current_amount = current_data.get("total_amount", 0)
            previous_amount = historical_data.get("total_amount", 0)
            
            if previous_amount > 0:
                change_ratio = (current_amount - previous_amount) / previous_amount
                
                if abs(change_ratio) >= self.temporal_thresholds["trend_significance"]:
                    trend_type = "augmentation" if change_ratio > 0 else "diminution"
                    percentage_change = abs(change_ratio) * 100
                    
                    severity = "warning" if change_ratio > 0.3 else "info"
                    if change_ratio < -0.2:  # Diminution importante
                        severity = "positive"
                    
                    insights.append(Insight(
                        type="trend",
                        title=f"Tendance : {trend_type}",
                        description=f"{trend_type.capitalize()} de {percentage_change:.0f}% par rapport à la période précédente",
                        severity=severity,
                        confidence=0.85
                    ))
            
            # Comparaison de la fréquence
            current_count = current_data.get("transaction_count", 0)
            previous_count = historical_data.get("transaction_count", 0)
            
            if previous_count > 0:
                frequency_change = (current_count - previous_count) / previous_count
                
                if abs(frequency_change) >= 0.2:  # 20% de changement
                    change_desc = "augmenté" if frequency_change > 0 else "diminué"
                    insights.append(Insight(
                        type="pattern",
                        title="Évolution de la fréquence",
                        description=f"Le nombre de transactions a {change_desc} par rapport à la période précédente",
                        severity="info",
                        confidence=0.75
                    ))
        
        except Exception as e:
            logger.error(f"Erreur génération insights temporels: {str(e)}")
        
        return insights
    
    def detect_spending_patterns(self, time_series_data: List[Dict[str, Any]]) -> List[Insight]:
        """Détecte des patterns dans les données temporelles"""
        
        insights = []
        
        if len(time_series_data) < 3:
            return insights
        
        try:
            amounts = [data.get("total_amount", 0) for data in time_series_data]
            
            # Détection de tendance
            if self._is_increasing_trend(amounts):
                insights.append(Insight(
                    type="trend",
                    title="Tendance croissante",
                    description="Vos dépenses suivent une tendance croissante",
                    severity="warning",
                    confidence=0.8
                ))
            elif self._is_decreasing_trend(amounts):
                insights.append(Insight(
                    type="trend",
                    title="Tendance décroissante",
                    description="Vos dépenses diminuent progressivement",
                    severity="positive",
                    confidence=0.8
                ))
            
            # Détection de volatilité
            if self._is_volatile(amounts):
                insights.append(Insight(
                    type="pattern",
                    title="Dépenses irrégulières",
                    description="Vos dépenses varient beaucoup d'une période à l'autre",
                    severity="info",
                    confidence=0.7
                ))
        
        except Exception as e:
            logger.error(f"Erreur détection patterns: {str(e)}")
        
        return insights
    
    def _is_increasing_trend(self, values: List[float]) -> bool:
        """Détecte une tendance croissante"""
        if len(values) < 3:
            return False
        
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        return increases >= len(values) * 0.6  # 60% d'augmentations
    
    def _is_decreasing_trend(self, values: List[float]) -> bool:
        """Détecte une tendance décroissante"""
        if len(values) < 3:
            return False
        
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        return decreases >= len(values) * 0.6  # 60% de diminutions
    
    def _is_volatile(self, values: List[float]) -> bool:
        """Détecte la volatilité des dépenses"""
        if len(values) < 2:
            return False
        
        mean_value = sum(values) / len(values)
        if mean_value == 0:
            return False
        
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
        coefficient_variation = (variance ** 0.5) / mean_value
        
        return coefficient_variation > self.temporal_thresholds["volatility_threshold"]