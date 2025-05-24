"""
Profileur de comptes bancaires.

Ce module analyse les comptes bancaires pour créer des profils enrichis
avec patterns d'utilisation, tendances et recommandations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from uuid import uuid4

from enrichment_service.core.logging import get_contextual_logger, log_performance
from enrichment_service.core.exceptions import AccountProfilingError
from enrichment_service.core.config import enrichment_settings
from enrichment_service.db.models import SyncAccount, RawTransaction, SyncItem

logger = logging.getLogger(__name__)

@dataclass
class CashFlowPattern:
    """Pattern de flux de trésorerie."""
    pattern_type: str  # weekly, monthly, quarterly
    typical_inflow: float
    typical_outflow: float
    net_flow: float
    regularity_score: float
    trend: str  # increasing, stable, decreasing

@dataclass
class UsagePattern:
    """Pattern d'utilisation du compte."""
    primary_purpose: str
    usage_frequency: str  # high, medium, low
    transaction_types: Dict[str, int]
    peak_activity_days: List[str]
    peak_activity_hours: List[int]
    seasonal_patterns: Dict[str, float]

@dataclass
class AccountProfile:
    """Profil complet d'un compte bancaire."""
    profile_id: str
    user_id: int
    bridge_account_id: int
    account_name: str
    account_type: str
    institution_name: str
    
    # Soldes et balances
    current_balance: float = 0.0
    available_balance: float = 0.0
    currency: str = "EUR"
    last_update: Optional[datetime] = None
    
    # Flux typiques
    typical_monthly_inflow: float = 0.0
    typical_monthly_outflow: float = 0.0
    main_income_sources: Dict[str, float] = field(default_factory=dict)
    main_expense_categories: Dict[str, float] = field(default_factory=dict)
    
    # Tendances historiques
    historical_balances: Dict[str, float] = field(default_factory=dict)
    lowest_balance_30d: float = 0.0
    highest_balance_30d: float = 0.0
    balance_trend: str = "stable"
    
    # Patterns d'utilisation
    cash_flow_patterns: List[CashFlowPattern] = field(default_factory=list)
    usage_patterns: Optional[UsagePattern] = None
    
    # Analyse comportementale
    primary_purpose: str = "unknown"
    importance_score: float = 0.0
    health_indicators: Dict[str, Any] = field(default_factory=dict)
    cash_flow_stability: float = 0.0
    
    # Métriques de risque
    overdraft_risk: float = 0.0
    liquidity_score: float = 0.0
    spending_predictability: float = 0.0
    
    # Comparaisons et insights
    vs_user_average: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Métadonnées
    account_summary: str = ""
    iban: str = ""
    account_metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class AccountProfiler:
    """
    Profileur avancé de comptes bancaires.
    
    Cette classe analyse les comptes bancaires et leurs transactions
    pour créer des profils comportementaux détaillés.
    """
    
    def __init__(self, db: Session, embedding_service, qdrant_service):
        """
        Initialise le profileur de comptes.
        
        Args:
            db: Session de base de données
            embedding_service: Service d'embedding
            qdrant_service: Service Qdrant
        """
        self.db = db
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        
        # Paramètres d'analyse
        self.analysis_window_days = 90  # Fenêtre d'analyse par défaut
        self.min_transactions_for_profile = 10
        
        # Seuils pour la classification
        self.high_activity_threshold = 50  # transactions/mois
        self.low_balance_threshold = 100.0  # euros
        self.overdraft_risk_threshold = 0.7
    
    @log_performance
    async def profile_account(self, account: SyncAccount) -> AccountProfile:
        """
        Crée un profil complet pour un compte bancaire.
        
        Args:
            account: Compte à profiler
            
        Returns:
            AccountProfile: Profil généré
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=self._get_user_id_from_account(account),
            account_id=account.bridge_account_id,
            enrichment_type="account_profiling"
        )
        
        ctx_logger.info(f"Début du profilage du compte {account.account_name}")
        
        try:
            # Récupérer les informations de base
            user_id = self._get_user_id_from_account(account)
            institution_name = self._get_institution_name(account)
            
            # Créer le profil de base
            profile = AccountProfile(
                profile_id=str(uuid4()),
                user_id=user_id,
                bridge_account_id=account.bridge_account_id,
                account_name=account.account_name or "Compte sans nom",
                account_type=account.account_type or "unknown",
                institution_name=institution_name,
                current_balance=account.balance or 0.0,
                available_balance=account.balance or 0.0,
                currency=account.currency_code or "EUR",
                last_update=datetime.now()
            )
            
            # Récupérer et analyser les transactions
            transactions = await self._get_account_transactions(account, self.analysis_window_days)
            
            if len(transactions) < self.min_transactions_for_profile:
                ctx_logger.info(f"Pas assez de transactions ({len(transactions)}) pour un profil détaillé")
                return await self._create_basic_profile(profile, transactions)
            
            # Analyses détaillées
            await self._analyze_cash_flows(profile, transactions)
            await self._analyze_usage_patterns(profile, transactions)
            await self._analyze_balance_trends(profile, account, transactions)
            await self._calculate_health_indicators(profile, transactions)
            await self._generate_recommendations(profile, transactions)
            await self._detect_account_alerts(profile, transactions)
            
            # Comparaisons avec les autres comptes de l'utilisateur
            await self._add_user_comparisons(profile)
            
            # Générer le résumé narratif
            await self._generate_account_summary(profile)
            
            # Stocker dans Qdrant
            await self._store_profile_in_qdrant(profile)
            
            ctx_logger.info(f"Profil de compte généré avec succès (score d'importance: {profile.importance_score:.2f})")
            
            return profile
            
        except Exception as e:
            error_msg = f"Erreur lors du profilage du compte {account.bridge_account_id}: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise AccountProfilingError(error_msg, account.bridge_account_id)
    
    def _get_user_id_from_account(self, account: SyncAccount) -> int:
        """
        Récupère l'ID utilisateur depuis un compte via l'item.
        
        Args:
            account: Compte bancaire
            
        Returns:
            int: ID de l'utilisateur
        """
        item = self.db.query(SyncItem).filter(SyncItem.id == account.item_id).first()
        return item.user_id if item else 0
    
    def _get_institution_name(self, account: SyncAccount) -> str:
        """
        Détermine le nom de l'institution bancaire.
        
        Args:
            account: Compte bancaire
            
        Returns:
            str: Nom de l'institution
        """
        # Pour l'instant, extraction basique depuis le nom du compte
        # TODO: Améliorer avec une base de données des institutions
        account_name = account.account_name or ""
        
        # Patterns courants
        institution_patterns = {
            "credit agricole": "Crédit Agricole",
            "bnp": "BNP Paribas",
            "societe generale": "Société Générale",
            "lcl": "LCL",
            "caisse epargne": "Caisse d'Épargne",
            "banque populaire": "Banque Populaire",
            "credit mutuel": "Crédit Mutuel",
            "ing": "ING",
            "boursorama": "Boursorama",
            "revolut": "Revolut",
            "n26": "N26"
        }
        
        account_lower = account_name.lower()
        for pattern, institution in institution_patterns.items():
            if pattern in account_lower:
                return institution
        
        return "Institution inconnue"
    
    async def _get_account_transactions(self, account: SyncAccount, days: int) -> List[RawTransaction]:
        """
        Récupère les transactions d'un compte pour la période d'analyse.
        
        Args:
            account: Compte bancaire
            days: Nombre de jours à analyser
            
        Returns:
            List[RawTransaction]: Liste des transactions
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        transactions = self.db.query(RawTransaction).filter(
            RawTransaction.account_id == account.id,
            RawTransaction.date >= cutoff_date,
            RawTransaction.deleted.is_(False),
            RawTransaction.amount != 0
        ).order_by(RawTransaction.date.asc()).all()
        
        return transactions
    
    async def _create_basic_profile(self, profile: AccountProfile, transactions: List[RawTransaction]) -> AccountProfile:
        """
        Crée un profil basique avec peu de transactions.
        
        Args:
            profile: Profil de base
            transactions: Liste des transactions
            
        Returns:
            AccountProfile: Profil basique complété
        """
        if transactions:
            total_amount = sum(abs(t.amount) for t in transactions)
            avg_amount = total_amount / len(transactions)
            
            profile.account_summary = f"Compte avec activité limitée ({len(transactions)} transactions, montant moyen: {avg_amount:.2f}€)"
            profile.primary_purpose = "low_activity"
            profile.importance_score = 0.3
            profile.tags = ["activité_faible", "données_limitées"]
        else:
            profile.account_summary = "Compte inactif sans transactions récentes"
            profile.primary_purpose = "inactive"
            profile.importance_score = 0.1
            profile.tags = ["inactif"]
        
        return profile
    
    async def _analyze_cash_flows(self, profile: AccountProfile, transactions: List[RawTransaction]):
        """
        Analyse les flux de trésorerie du compte.
        
        Args:
            profile: Profil à enrichir
            transactions: Liste des transactions
        """
        if not transactions:
            return
        
        # Séparer les entrées et sorties
        inflows = [t for t in transactions if t.amount > 0]
        outflows = [t for t in transactions if t.amount < 0]
        
        # Calculs mensuels
        total_days = (transactions[-1].date - transactions[0].date).days or 1
        months_analyzed = max(1, total_days / 30.0)
        
        total_inflow = sum(t.amount for t in inflows)
        total_outflow = abs(sum(t.amount for t in outflows))
        
        profile.typical_monthly_inflow = total_inflow / months_analyzed
        profile.typical_monthly_outflow = total_outflow / months_analyzed
        
        # Analyser les sources de revenus principales
        income_sources = defaultdict(float)
        for transaction in inflows:
            source = self._categorize_income_source(transaction)
            income_sources[source] += transaction.amount
        
        profile.main_income_sources = dict(sorted(
            income_sources.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
        
        # Analyser les catégories de dépenses principales
        expense_categories = defaultdict(float)
        for transaction in outflows:
            category = self._get_expense_category(transaction)
            expense_categories[category] += abs(transaction.amount)
        
        profile.main_expense_categories = dict(sorted(
            expense_categories.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
        
        # Détecter les patterns de flux
        await self._detect_cash_flow_patterns(profile, transactions)
    
    def _categorize_income_source(self, transaction: RawTransaction) -> str:
        """
        Catégorise une transaction de revenu par source.
        
        Args:
            transaction: Transaction de revenu
            
        Returns:
            str: Catégorie de source de revenu
        """
        description = (transaction.clean_description or transaction.provider_description or "").lower()
        
        # Patterns de reconnaissance
        if any(keyword in description for keyword in ['salaire', 'paie', 'salary', 'wages']):
            return "Salaire"
        elif any(keyword in description for keyword in ['virement', 'transfer', 'vir']):
            return "Virement"
        elif any(keyword in description for keyword in ['remboursement', 'refund']):
            return "Remboursement"
        elif any(keyword in description for keyword in ['allocation', 'aide', 'caf']):
            return "Aides sociales"
        elif any(keyword in description for keyword in ['dividende', 'interet', 'dividend']):
            return "Revenus financiers"
        elif any(keyword in description for keyword in ['freelance', 'consultant', 'honoraire']):
            return "Revenus indépendants"
        else:
            return "Autres revenus"
    
    def _get_expense_category(self, transaction: RawTransaction) -> str:
        """
        Détermine la catégorie d'une dépense.
        
        Args:
            transaction: Transaction de dépense
            
        Returns:
            str: Catégorie de dépense
        """
        description = (transaction.clean_description or transaction.provider_description or "").lower()
        
        # Categories basées sur les mots-clés
        if any(keyword in description for keyword in ['supermarche', 'carrefour', 'leclerc', 'alimentation']):
            return "Alimentation"
        elif any(keyword in description for keyword in ['essence', 'station', 'carburant']):
            return "Transport"
        elif any(keyword in description for keyword in ['restaurant', 'cafe', 'mcdonalds']):
            return "Restaurants"
        elif any(keyword in description for keyword in ['loyer', 'rent']):
            return "Logement"
        elif any(keyword in description for keyword in ['edf', 'gdf', 'orange', 'sfr']):
            return "Factures"
        elif any(keyword in description for keyword in ['assurance', 'mutuelle']):
            return "Assurances"
        elif any(keyword in description for keyword in ['retrait', 'dab', 'atm']):
            return "Retraits"
        elif any(keyword in description for keyword in ['virement', 'transfer']):
            return "Virements"
        else:
            return "Autres dépenses"
    
    async def _detect_cash_flow_patterns(self, profile: AccountProfile, transactions: List[RawTransaction]):
        """
        Détecte les patterns de flux de trésorerie.
        
        Args:
            profile: Profil à enrichir
            transactions: Liste des transactions
        """
        # Analyser les patterns hebdomadaires
        weekly_pattern = await self._analyze_weekly_pattern(transactions)
        if weekly_pattern:
            profile.cash_flow_patterns.append(weekly_pattern)
        
        # Analyser les patterns mensuels
        monthly_pattern = await self._analyze_monthly_pattern(transactions)
        if monthly_pattern:
            profile.cash_flow_patterns.append(monthly_pattern)
        
        # Calculer la stabilité des flux
        profile.cash_flow_stability = await self._calculate_cash_flow_stability(transactions)
    
    async def _analyze_weekly_pattern(self, transactions: List[RawTransaction]) -> Optional[CashFlowPattern]:
        """
        Analyse le pattern hebdomadaire des flux.
        
        Args:
            transactions: Liste des transactions
            
        Returns:
            Optional[CashFlowPattern]: Pattern hebdomadaire ou None
        """
        if len(transactions) < 14:  # Au moins 2 semaines
            return None
        
        # Grouper par semaine
        weekly_data = defaultdict(lambda: {"inflow": 0.0, "outflow": 0.0})
        
        for transaction in transactions:
            week_key = transaction.date.strftime("%Y-W%U")
            if transaction.amount > 0:
                weekly_data[week_key]["inflow"] += transaction.amount
            else:
                weekly_data[week_key]["outflow"] += abs(transaction.amount)
        
        if len(weekly_data) < 2:
            return None
        
        # Calculer les moyennes
        inflows = [data["inflow"] for data in weekly_data.values()]
        outflows = [data["outflow"] for data in weekly_data.values()]
        
        avg_inflow = sum(inflows) / len(inflows)
        avg_outflow = sum(outflows) / len(outflows)
        
        # Calculer la régularité (coefficient de variation inverse)
        inflow_std = (sum((x - avg_inflow) ** 2 for x in inflows) / len(inflows)) ** 0.5
        outflow_std = (sum((x - avg_outflow) ** 2 for x in outflows) / len(outflows)) ** 0.5
        
        inflow_regularity = 1 - (inflow_std / avg_inflow) if avg_inflow > 0 else 0
        outflow_regularity = 1 - (outflow_std / avg_outflow) if avg_outflow > 0 else 0
        
        regularity_score = (inflow_regularity + outflow_regularity) / 2
        
        # Détecter la tendance
        trend = self._detect_trend([data["inflow"] - data["outflow"] for data in weekly_data.values()])
        
        return CashFlowPattern(
            pattern_type="weekly",
            typical_inflow=avg_inflow,
            typical_outflow=avg_outflow,
            net_flow=avg_inflow - avg_outflow,
            regularity_score=max(0, min(1, regularity_score)),
            trend=trend
        )
    
    async def _analyze_monthly_pattern(self, transactions: List[RawTransaction]) -> Optional[CashFlowPattern]:
        """
        Analyse le pattern mensuel des flux.
        
        Args:
            transactions: Liste des transactions
            
        Returns:
            Optional[CashFlowPattern]: Pattern mensuel ou None
        """
        if len(transactions) < 30:  # Au moins un mois
            return None
        
        # Grouper par mois
        monthly_data = defaultdict(lambda: {"inflow": 0.0, "outflow": 0.0})
        
        for transaction in transactions:
            month_key = transaction.date.strftime("%Y-%m")
            if transaction.amount > 0:
                monthly_data[month_key]["inflow"] += transaction.amount
            else:
                monthly_data[month_key]["outflow"] += abs(transaction.amount)
        
        if len(monthly_data) < 2:
            return None
        
        # Calculer les moyennes
        inflows = [data["inflow"] for data in monthly_data.values()]
        outflows = [data["outflow"] for data in monthly_data.values()]
        
        avg_inflow = sum(inflows) / len(inflows)
        avg_outflow = sum(outflows) / len(outflows)
        
        # Calculer la régularité
        inflow_std = (sum((x - avg_inflow) ** 2 for x in inflows) / len(inflows)) ** 0.5
        outflow_std = (sum((x - avg_outflow) ** 2 for x in outflows) / len(outflows)) ** 0.5
        
        inflow_regularity = 1 - (inflow_std / avg_inflow) if avg_inflow > 0 else 0
        outflow_regularity = 1 - (outflow_std / avg_outflow) if avg_outflow > 0 else 0
        
        regularity_score = (inflow_regularity + outflow_regularity) / 2
        
        # Détecter la tendance
        trend = self._detect_trend([data["inflow"] - data["outflow"] for data in monthly_data.values()])
        
        return CashFlowPattern(
            pattern_type="monthly",
            typical_inflow=avg_inflow,
            typical_outflow=avg_outflow,
            net_flow=avg_inflow - avg_outflow,
            regularity_score=max(0, min(1, regularity_score)),
            trend=trend
        )
    
    def _detect_trend(self, values: List[float]) -> str:
        """
        Détecte la tendance dans une série de valeurs.
        
        Args:
            values: Liste de valeurs
            
        Returns:
            str: Tendance détectée
        """
        if len(values) < 2:
            return "stable"
        
        # Régression linéaire simple
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Seuils pour déterminer la tendance
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    async def _calculate_cash_flow_stability(self, transactions: List[RawTransaction]) -> float:
        """
        Calcule un score de stabilité des flux de trésorerie.
        
        Args:
            transactions: Liste des transactions
            
        Returns:
            float: Score de stabilité (0-1)
        """
        if len(transactions) < 7:
            return 0.0
        
        # Grouper par jour
        daily_flows = defaultdict(float)
        for transaction in transactions:
            day_key = transaction.date.strftime("%Y-%m-%d")
            daily_flows[day_key] += transaction.amount
        
        flows = list(daily_flows.values())
        
        # Calculer le coefficient de variation
        if not flows:
            return 0.0
        
        mean_flow = sum(flows) / len(flows)
        if mean_flow == 0:
            return 0.0
        
        variance = sum((flow - mean_flow) ** 2 for flow in flows) / len(flows)
        std_dev = variance ** 0.5
        
        coefficient_of_variation = std_dev / abs(mean_flow)
        
        # Convertir en score de stabilité (inverse du coefficient de variation)
        stability_score = 1 / (1 + coefficient_of_variation)
        
        return min(1.0, max(0.0, stability_score))
    
    async def _analyze_usage_patterns(self, profile: AccountProfile, transactions: List[RawTransaction]):
        """
        Analyse les patterns d'utilisation du compte.
        
        Args:
            profile: Profil à enrichir
            transactions: Liste des transactions
        """
        if not transactions:
            return
        
        # Fréquence d'utilisation
        total_days = (transactions[-1].date - transactions[0].date).days or 1
        transactions_per_day = len(transactions) / total_days
        
        if transactions_per_day >= 2:
            usage_frequency = "high"
        elif transactions_per_day >= 0.5:
            usage_frequency = "medium"
        else:
            usage_frequency = "low"
        
        # Types de transactions
        transaction_types = defaultdict(int)
        for transaction in transactions:
            if transaction.amount > 0:
                transaction_types["credit"] += 1
            else:
                transaction_types["debit"] += 1
        
        # Jours de pic d'activité
        daily_activity = defaultdict(int)
        for transaction in transactions:
            day_name = transaction.date.strftime("%A")
            daily_activity[day_name] += 1
        
        peak_days = sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_activity_days = [day for day, _ in peak_days]
        
        # Heures de pic d'activité (si disponible)
        hourly_activity = defaultdict(int)
        for transaction in transactions:
            if hasattr(transaction.date, 'hour'):
                hourly_activity[transaction.date.hour] += 1
        
        peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_activity_hours = [hour for hour, _ in peak_hours]
        
        # Déterminer l'usage principal
        primary_purpose = self._determine_primary_purpose(profile, transactions)
        
        profile.usage_patterns = UsagePattern(
            primary_purpose=primary_purpose,
            usage_frequency=usage_frequency,
            transaction_types=dict(transaction_types),
            peak_activity_days=peak_activity_days,
            peak_activity_hours=peak_activity_hours,
            seasonal_patterns={}  # TODO: Implémenter l'analyse saisonnière
        )
        
        profile.primary_purpose = primary_purpose
    
    def _determine_primary_purpose(self, profile: AccountProfile, transactions: List[RawTransaction]) -> str:
        """
        Détermine l'usage principal du compte.
        
        Args:
            profile: Profil du compte
            transactions: Liste des transactions
            
        Returns:
            str: Usage principal déterminé
        """
        # Analyser les ratios et patterns
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_expenses = abs(sum(t.amount for t in transactions if t.amount < 0))
        
        income_ratio = total_income / (total_income + total_expenses) if (total_income + total_expenses) > 0 else 0
        
        # Analyser les sources de revenus
        salary_indicators = 0
        savings_indicators = 0
        business_indicators = 0
        
        for transaction in transactions:
            if transaction.amount > 0:
                description = (transaction.clean_description or transaction.provider_description or "").lower()
                if any(keyword in description for keyword in ['salaire', 'paie', 'salary']):
                    salary_indicators += 1
                elif any(keyword in description for keyword in ['epargne', 'placement', 'saving']):
                    savings_indicators += 1
                elif any(keyword in description for keyword in ['freelance', 'entreprise', 'business']):
                    business_indicators += 1
        
        # Classification basée sur les indicateurs
        if salary_indicators > 0 and income_ratio > 0.3:
            return "primary_account"  # Compte principal avec salaire
        elif total_expenses > total_income * 2:
            return "spending_account"  # Compte principalement pour les dépenses
        elif savings_indicators > 0 or income_ratio > 0.8:
            return "savings_account"  # Compte d'épargne
        elif business_indicators > 0:
            return "business_account"  # Compte professionnel
        elif len(transactions) < 20:
            return "occasional_use"  # Usage occasionnel
        else:
            return "secondary_account"  # Compte secondaire
    
    async def _analyze_balance_trends(self, profile: AccountProfile, account: SyncAccount, transactions: List[RawTransaction]):
        """
        Analyse les tendances de solde du compte.
        
        Args:
            profile: Profil à enrichir
            account: Compte bancaire
            transactions: Liste des transactions
        """
        # Calculer l'historique des soldes (simulation)
        current_balance = account.balance or 0.0
        balance_history = []
        running_balance = current_balance
        
        # Reconstruire l'historique en partant du solde actuel
        for transaction in reversed(transactions[-30:]):  # 30 dernières transactions
            balance_history.append({
                "date": transaction.date.isoformat(),
                "balance": running_balance
            })
            running_balance -= transaction.amount
        
        balance_history.reverse()
        
        # Stocker l'historique
        profile.historical_balances = {
            item["date"]: item["balance"] for item in balance_history[-10:]  # 10 derniers points
        }
        
        # Calculer les extremums sur 30 jours
        if balance_history:
            balances = [item["balance"] for item in balance_history]
            profile.lowest_balance_30d = min(balances)
            profile.highest_balance_30d = max(balances)
            
            # Détecter la tendance
            profile.balance_trend = self._detect_trend(balances)
        else:
            profile.lowest_balance_30d = current_balance
            profile.highest_balance_30d = current_balance
            profile.balance_trend = "stable"
    
    async def _calculate_health_indicators(self, profile: AccountProfile, transactions: List[RawTransaction]):
        """
        Calcule les indicateurs de santé du compte.
        
        Args:
            profile: Profil à enrichir
            transactions: Liste des transactions
        """
        indicators = {}
        
        # Score de liquidité
        current_balance = profile.current_balance
        monthly_expenses = profile.typical_monthly_outflow
        
        if monthly_expenses > 0:
            months_coverage = current_balance / monthly_expenses
            if months_coverage >= 3:
                indicators["liquidity_status"] = "excellent"
                profile.liquidity_score = 1.0
            elif months_coverage >= 1:
                indicators["liquidity_status"] = "good"
                profile.liquidity_score = 0.7
            elif months_coverage >= 0.5:
                indicators["liquidity_status"] = "fair"
                profile.liquidity_score = 0.5
            else:
                indicators["liquidity_status"] = "low"
                profile.liquidity_score = 0.2
        else:
            indicators["liquidity_status"] = "unknown"
            profile.liquidity_score = 0.5
        
        # Risque de découvert
        lowest_balance = profile.lowest_balance_30d
        if lowest_balance < 0:
            profile.overdraft_risk = 1.0
            indicators["overdraft_risk"] = "high"
        elif lowest_balance < self.low_balance_threshold:
            profile.overdraft_risk = 0.7
            indicators["overdraft_risk"] = "medium"
        else:
            profile.overdraft_risk = 0.1
            indicators["overdraft_risk"] = "low"
        
        # Prévisibilité des dépenses
        if profile.cash_flow_stability > 0.7:
            profile.spending_predictability = 1.0
            indicators["spending_predictability"] = "high"
        elif profile.cash_flow_stability > 0.4:
            profile.spending_predictability = 0.6
            indicators["spending_predictability"] = "medium"
        else:
            profile.spending_predictability = 0.3
            indicators["spending_predictability"] = "low"
        
        # Score d'importance du compte
        factors = [
            profile.typical_monthly_inflow / 1000,  # Normaliser sur 1000€
            profile.typical_monthly_outflow / 1000,
            len(transactions) / 100,  # Normaliser sur 100 transactions
            profile.liquidity_score,
            profile.cash_flow_stability
        ]
        
        profile.importance_score = min(1.0, sum(factors) / len(factors))
        
        # Santé globale du compte
        health_score = (profile.liquidity_score + (1 - profile.overdraft_risk) + profile.spending_predictability) / 3
        
        if health_score >= 0.8:
            indicators["overall_health"] = "excellent"
        elif health_score >= 0.6:
            indicators["overall_health"] = "good"
        elif health_score >= 0.4:
            indicators["overall_health"] = "fair"
        else:
            indicators["overall_health"] = "concerning"
        
        profile.health_indicators = indicators
    
    async def _generate_recommendations(self, profile: AccountProfile, transactions: List[RawTransaction]):
        """
        Génère des recommandations pour le compte.
        
        Args:
            profile: Profil à enrichir
            transactions: Liste des transactions
        """
        recommendations = []
        
        # Recommandations basées sur la liquidité
        if profile.liquidity_score < 0.5:
            recommendations.append("Considérez augmenter le solde de ce compte pour améliorer votre liquidité")
        
        # Recommandations basées sur le risque de découvert
        if profile.overdraft_risk > 0.5:
            recommendations.append("Surveillez ce compte de près pour éviter les découverts")
        
        # Recommandations basées sur l'usage
        if profile.primary_purpose == "spending_account" and profile.current_balance > profile.typical_monthly_outflow * 2:
            recommendations.append("Ce compte a un solde élevé pour un compte de dépenses, considérez transférer vers l'épargne")
        
        # Recommandations basées sur la stabilité
        if profile.cash_flow_stability < 0.4:
            recommendations.append("Les flux de ce compte sont irréguliers, considérez une meilleure planification budgétaire")
        
        # Recommandations basées sur l'activité
        if profile.usage_patterns and profile.usage_patterns.usage_frequency == "low":
            recommendations.append("Ce compte est peu utilisé, vérifiez s'il est toujours nécessaire")
        
        profile.recommendations = recommendations
    
    async def _detect_account_alerts(self, profile: AccountProfile, transactions: List[RawTransaction]):
        """
        Détecte les alertes pour le compte.
        
        Args:
            profile: Profil à enrichir
            transactions: Liste des transactions
        """
        alerts = []
        
        # Alerte solde faible
        if profile.current_balance < self.low_balance_threshold:
            alerts.append({
                "type": "low_balance",
                "severity": "warning",
                "message": f"Solde faible: {profile.current_balance:.2f}€",
                "recommendation": "Surveiller les prochaines dépenses"
            })
        
        # Alerte découvert potentiel
        if profile.overdraft_risk > self.overdraft_risk_threshold:
            alerts.append({
                "type": "overdraft_risk",
                "severity": "high",
                "message": "Risque élevé de découvert détecté",
                "recommendation": "Réduire les dépenses ou approvisionner le compte"
            })
        
        # Alerte activité inhabituelle
        recent_transactions = [t for t in transactions if (datetime.now() - t.date).days <= 7]
        if len(recent_transactions) == 0 and profile.usage_patterns and profile.usage_patterns.usage_frequency != "low":
            alerts.append({
                "type": "unusual_activity",
                "severity": "info",
                "message": "Aucune activité récente sur ce compte actif",
                "recommendation": "Vérifier si le compte fonctionne normalement"
            })
        
        # Alerte grosse transaction
        if recent_transactions:
            avg_amount = sum(abs(t.amount) for t in transactions) / len(transactions)
            large_transactions = [t for t in recent_transactions if abs(t.amount) > avg_amount * 3]
            
            for transaction in large_transactions:
                alerts.append({
                    "type": "large_transaction",
                    "severity": "info",
                    "message": f"Transaction importante: {abs(transaction.amount):.2f}€",
                    "recommendation": "Vérifier si cette transaction est attendue",
                    "transaction_date": transaction.date.isoformat()
                })
        
        profile.alerts = alerts
    
    async def _add_user_comparisons(self, profile: AccountProfile):
        """
        Ajoute les comparaisons avec les autres comptes de l'utilisateur.
        
        Args:
            profile: Profil à enrichir
        """
        # Récupérer les autres comptes de l'utilisateur
        user_accounts = self.db.query(SyncAccount).join(SyncItem).filter(
            SyncItem.user_id == profile.user_id,
            SyncAccount.id != profile.bridge_account_id
        ).all()
        
        if not user_accounts:
            return
        
        # Calculer les moyennes utilisateur (simplification)
        total_balances = sum(acc.balance or 0 for acc in user_accounts) + profile.current_balance
        avg_balance = total_balances / (len(user_accounts) + 1)
        
        profile.vs_user_average = {
            "balance_vs_avg": profile.current_balance - avg_balance,
            "is_primary_account": profile.importance_score > 0.7,
            "accounts_count": len(user_accounts) + 1
        }
    
    async def _generate_account_summary(self, profile: AccountProfile):
        """
        Génère le résumé narratif du compte.
        
        Args:
            profile: Profil à enrichir
        """
        summary_parts = []
        
        # Description de base
        summary_parts.append(f"Compte {profile.account_type} chez {profile.institution_name}")
        
        # Usage principal
        purpose_descriptions = {
            "primary_account": "compte principal",
            "spending_account": "compte de dépenses",
            "savings_account": "compte d'épargne", 
            "business_account": "compte professionnel",
            "secondary_account": "compte secondaire",
            "occasional_use": "usage occasionnel"
        }
        
        purpose_desc = purpose_descriptions.get(profile.primary_purpose, "usage indéterminé")
        summary_parts.append(f"utilisé comme {purpose_desc}")
        
        # Activité
        if profile.usage_patterns:
            freq_desc = {
                "high": "très actif",
                "medium": "moyennement actif", 
                "low": "peu actif"
            }
            activity_desc = freq_desc.get(profile.usage_patterns.usage_frequency, "activité inconnue")
            summary_parts.append(f"avec une activité {activity_desc}")
        
        # Flux mensuels
        if profile.typical_monthly_inflow > 0 or profile.typical_monthly_outflow > 0:
            summary_parts.append(
                f"(~{profile.typical_monthly_inflow:.0f}€/mois entrants, "
                f"{profile.typical_monthly_outflow:.0f}€/mois sortants)"
            )
        
        # Santé financière
        health_status = profile.health_indicators.get("overall_health", "inconnue")
        health_descriptions = {
            "excellent": "en excellente santé",
            "good": "en bonne santé",
            "fair": "santé correcte",
            "concerning": "nécessite attention"
        }
        
        health_desc = health_descriptions.get(health_status, "santé inconnue")
        summary_parts.append(f"Compte {health_desc}")
        
        profile.account_summary = ". ".join(summary_parts) + "."
        
        profile.account_summary = ". ".join(summary_parts) + "."
        
        # Générer les tags
        tags = [profile.account_type, profile.primary_purpose]
        
        if profile.usage_patterns:
            tags.append(f"activité_{profile.usage_patterns.usage_frequency}")
        
        health_status = profile.health_indicators.get("overall_health")
        if health_status:
            tags.append(f"santé_{health_status}")
        
        if profile.importance_score > 0.7:
            tags.append("compte_important")
        elif profile.importance_score < 0.3:
            tags.append("compte_secondaire")
        
        if profile.overdraft_risk > 0.5:
            tags.append("risque_découvert")
        
        if profile.liquidity_score > 0.8:
            tags.append("bonne_liquidité")
        elif profile.liquidity_score < 0.3:
            tags.append("liquidité_faible")
        
        profile.tags = tags
    
    async def _store_profile_in_qdrant(self, profile: AccountProfile):
        """
        Stocke le profil de compte dans Qdrant.
        
        Args:
            profile: Profil à stocker
        """
        # Générer l'embedding du profil
        embedding_text = profile.account_summary
        
        # Ajouter des informations contextuelles
        context_parts = [
            f"Compte {profile.account_type} {profile.institution_name}",
            f"Usage: {profile.primary_purpose}",
            f"Solde: {profile.current_balance:.2f}€"
        ]
        
        if profile.main_income_sources:
            top_income = max(profile.main_income_sources.items(), key=lambda x: x[1])
            context_parts.append(f"Principal revenu: {top_income[0]}")
        
        if profile.main_expense_categories:
            top_expense = max(profile.main_expense_categories.items(), key=lambda x: x[1])
            context_parts.append(f"Principale dépense: {top_expense[0]}")
        
        full_embedding_text = embedding_text + " | " + " | ".join(context_parts)
        
        vector = await self.embedding_service.generate_embedding(full_embedding_text)
        
        # Construire le payload
        payload = {
            "id": profile.profile_id,
            "user_id": profile.user_id,
            "bridge_account_id": profile.bridge_account_id,
            "account_name": profile.account_name,
            "account_type": profile.account_type,
            "institution_name": profile.institution_name,
            
            # Soldes
            "current_balance": profile.current_balance,
            "available_balance": profile.available_balance,
            "currency": profile.currency,
            "last_update": profile.last_update.isoformat() if profile.last_update else None,
            
            # Flux typiques
            "typical_monthly_inflow": profile.typical_monthly_inflow,
            "typical_monthly_outflow": profile.typical_monthly_outflow,
            "main_income_sources": profile.main_income_sources,
            "main_expense_categories": profile.main_expense_categories,
            
            # Tendances
            "historical_balances": profile.historical_balances,
            "lowest_balance_30d": profile.lowest_balance_30d,
            "highest_balance_30d": profile.highest_balance_30d,
            "balance_trend": profile.balance_trend,
            
            # Analyse
            "primary_purpose": profile.primary_purpose,
            "importance_score": profile.importance_score,
            "health_indicators": profile.health_indicators,
            "cash_flow_stability": profile.cash_flow_stability,
            
            # Métriques de risque
            "overdraft_risk": profile.overdraft_risk,
            "liquidity_score": profile.liquidity_score,
            "spending_predictability": profile.spending_predictability,
            
            # Patterns
            "cash_flow_patterns": [
                {
                    "type": pattern.pattern_type,
                    "inflow": pattern.typical_inflow,
                    "outflow": pattern.typical_outflow,
                    "regularity": pattern.regularity_score,
                    "trend": pattern.trend
                } for pattern in profile.cash_flow_patterns
            ],
            
            "usage_patterns": {
                "primary_purpose": profile.usage_patterns.primary_purpose,
                "usage_frequency": profile.usage_patterns.usage_frequency,
                "transaction_types": profile.usage_patterns.transaction_types,
                "peak_activity_days": profile.usage_patterns.peak_activity_days,
                "peak_activity_hours": profile.usage_patterns.peak_activity_hours
            } if profile.usage_patterns else None,
            
            # Recommandations et alertes
            "recommendations": profile.recommendations,
            "alerts": profile.alerts,
            "vs_user_average": profile.vs_user_average,
            
            # Métadonnées
            "account_summary": profile.account_summary,
            "iban": profile.iban,
            "account_metadata": profile.account_metadata,
            "tags": profile.tags
        }
        
        # Stocker dans Qdrant
        await self.qdrant_service.upsert_point(
            collection_name="enriched_accounts",
            point_id=profile.profile_id,
            vector=vector,
            payload=payload
        )
    
    async def profile_user_accounts(self, user_id: int) -> List[AccountProfile]:
        """
        Profile tous les comptes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[AccountProfile]: Liste des profils générés
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="user_accounts_profiling"
        )
        
        ctx_logger.info(f"Début du profilage de tous les comptes de l'utilisateur {user_id}")
        
        try:
            # Récupérer tous les comptes de l'utilisateur
            accounts = self.db.query(SyncAccount).join(SyncItem).filter(
                SyncItem.user_id == user_id
            ).all()
            
            if not accounts:
                ctx_logger.info("Aucun compte trouvé pour cet utilisateur")
                return []
            
            profiles = []
            for account in accounts:
                try:
                    profile = await self.profile_account(account)
                    profiles.append(profile)
                except Exception as e:
                    ctx_logger.error(f"Erreur lors du profilage du compte {account.bridge_account_id}: {e}")
            
            ctx_logger.info(f"Profilage terminé: {len(profiles)}/{len(accounts)} comptes traités")
            
            return profiles
            
        except Exception as e:
            error_msg = f"Erreur lors du profilage des comptes utilisateur: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise AccountProfilingError(error_msg, user_id)
    
    async def update_account_profile(self, account: SyncAccount) -> AccountProfile:
        """
        Met à jour le profil d'un compte existant.
        
        Args:
            account: Compte à mettre à jour
            
        Returns:
            AccountProfile: Profil mis à jour
        """
        # Rechercher le profil existant dans Qdrant
        try:
            existing_profiles = await self.qdrant_service.search_points(
                collection_name="enriched_accounts",
                filter_conditions={"bridge_account_id": account.bridge_account_id}
            )
            
            if existing_profiles:
                # Supprimer l'ancien profil
                await self.qdrant_service.delete_points(
                    collection_name="enriched_accounts",
                    filter_conditions={"bridge_account_id": account.bridge_account_id}
                )
        except Exception as e:
            logger.warning(f"Impossible de supprimer l'ancien profil: {e}")
        
        # Créer un nouveau profil
        return await self.profile_account(account)
    
    async def get_user_account_profiles(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Récupère les profils de comptes d'un utilisateur depuis Qdrant.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[Dict]: Liste des profils de comptes
        """
        try:
            results = await self.qdrant_service.search_points(
                collection_name="enriched_accounts",
                filter_conditions={"user_id": user_id},
                limit=50
            )
            
            profiles = []
            for result in results:
                profiles.append(result["payload"])
            
            # Trier par score d'importance
            profiles.sort(key=lambda p: p.get("importance_score", 0), reverse=True)
            
            return profiles
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des profils de comptes: {e}")
            return []
    
    async def delete_user_account_profiles(self, user_id: int) -> bool:
        """
        Supprime tous les profils de comptes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        try:
            return await self.qdrant_service.delete_points(
                collection_name="enriched_accounts",
                filter_conditions={"user_id": user_id}
            )
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des profils de comptes: {e}")
            return False
    
    async def get_account_recommendations(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Génère des recommandations globales basées sur tous les comptes de l'utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[Dict]: Liste des recommandations
        """
        profiles = await self.get_user_account_profiles(user_id)
        
        if not profiles:
            return []
        
        recommendations = []
        
        # Analyser la répartition des comptes
        total_balance = sum(p.get("current_balance", 0) for p in profiles)
        primary_accounts = [p for p in profiles if p.get("primary_purpose") == "primary_account"]
        savings_accounts = [p for p in profiles if p.get("primary_purpose") == "savings_account"]
        
        # Recommandations sur la structure des comptes
        if len(primary_accounts) > 2:
            recommendations.append({
                "type": "account_structure",
                "priority": "medium",
                "message": "Vous avez plusieurs comptes principaux, considérez simplifier votre structure",
                "action": "Centraliser vos revenus sur un compte principal unique"
            })
        
        if not savings_accounts and total_balance > 1000:
            recommendations.append({
                "type": "savings_optimization",
                "priority": "high",
                "message": "Aucun compte d'épargne identifié malgré un patrimoine suffisant",
                "action": "Ouvrir un compte d'épargne pour optimiser vos finances"
            })
        
        # Recommandations sur les risques
        risky_accounts = [p for p in profiles if p.get("overdraft_risk", 0) > 0.5]
        if risky_accounts:
            recommendations.append({
                "type": "risk_management",
                "priority": "high",
                "message": f"{len(risky_accounts)} compte(s) à risque de découvert",
                "action": "Surveiller de près ces comptes et rééquilibrer si nécessaire",
                "affected_accounts": [p.get("account_name") for p in risky_accounts]
            })
        
        # Recommandations sur l'optimisation
        low_activity_accounts = [
            p for p in profiles 
            if p.get("usage_patterns", {}).get("usage_frequency") == "low"
            and p.get("current_balance", 0) > 500
        ]
        
        if low_activity_accounts:
            recommendations.append({
                "type": "optimization",
                "priority": "medium",
                "message": f"{len(low_activity_accounts)} compte(s) peu utilisé(s) avec solde élevé",
                "action": "Considérez fermer ou réaffecter ces comptes",
                "affected_accounts": [p.get("account_name") for p in low_activity_accounts]
            })
        
        return recommendations
    
    async def generate_account_comparison(self, account_id1: int, account_id2: int) -> Dict[str, Any]:
        """
        Compare deux comptes et génère un rapport de comparaison.
        
        Args:
            account_id1: ID du premier compte
            account_id2: ID du deuxième compte
            
        Returns:
            Dict: Rapport de comparaison
        """
        try:
            # Récupérer les profils des deux comptes
            profile1_results = await self.qdrant_service.search_points(
                collection_name="enriched_accounts",
                filter_conditions={"bridge_account_id": account_id1},
                limit=1
            )
            
            profile2_results = await self.qdrant_service.search_points(
                collection_name="enriched_accounts",
                filter_conditions={"bridge_account_id": account_id2},
                limit=1
            )
            
            if not profile1_results or not profile2_results:
                return {"error": "Un ou plusieurs profils de comptes non trouvés"}
            
            profile1 = profile1_results[0]["payload"]
            profile2 = profile2_results[0]["payload"]
            
            # Générer la comparaison
            comparison = {
                "accounts": {
                    "account1": {
                        "name": profile1.get("account_name"),
                        "type": profile1.get("account_type"),
                        "institution": profile1.get("institution_name")
                    },
                    "account2": {
                        "name": profile2.get("account_name"),
                        "type": profile2.get("account_type"),
                        "institution": profile2.get("institution_name")
                    }
                },
                "balance_comparison": {
                    "account1_balance": profile1.get("current_balance", 0),
                    "account2_balance": profile2.get("current_balance", 0),
                    "difference": profile1.get("current_balance", 0) - profile2.get("current_balance", 0)
                },
                "activity_comparison": {
                    "account1_inflow": profile1.get("typical_monthly_inflow", 0),
                    "account2_inflow": profile2.get("typical_monthly_inflow", 0),
                    "account1_outflow": profile1.get("typical_monthly_outflow", 0),
                    "account2_outflow": profile2.get("typical_monthly_outflow", 0)
                },
                "health_comparison": {
                    "account1_health": profile1.get("health_indicators", {}).get("overall_health"),
                    "account2_health": profile2.get("health_indicators", {}).get("overall_health"),
                    "account1_importance": profile1.get("importance_score", 0),
                    "account2_importance": profile2.get("importance_score", 0)
                },
                "usage_comparison": {
                    "account1_purpose": profile1.get("primary_purpose"),
                    "account2_purpose": profile2.get("primary_purpose"),
                    "account1_frequency": profile1.get("usage_patterns", {}).get("usage_frequency"),
                    "account2_frequency": profile2.get("usage_patterns", {}).get("usage_frequency")
                },
                "recommendations": []
            }
            
            # Générer des recommandations basées sur la comparaison
            if comparison["balance_comparison"]["difference"] > 1000:
                comparison["recommendations"].append(
                    f"Le compte {profile1.get('account_name')} a un solde significativement plus élevé. "
                    "Considérez rééquilibrer si approprié."
                )
            
            # Recommandations sur l'usage
            purpose1 = profile1.get("primary_purpose")
            purpose2 = profile2.get("primary_purpose")
            
            if purpose1 == purpose2 and purpose1 != "secondary_account":
                comparison["recommendations"].append(
                    f"Les deux comptes ont le même usage ({purpose1}). "
                    "Considérez spécialiser leurs rôles pour optimiser votre gestion."
                )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des comptes: {e}")
            return {"error": f"Erreur lors de la comparaison: {str(e)}"}
    
    async def detect_duplicate_accounts(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Détecte les comptes potentiellement en doublon pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[Dict]: Liste des doublons potentiels
        """
        profiles = await self.get_user_account_profiles(user_id)
        
        if len(profiles) < 2:
            return []
        
        duplicates = []
        
        for i, profile1 in enumerate(profiles):
            for profile2 in profiles[i+1:]:
                # Vérifier la similarité
                similarity_score = 0
                
                # Même institution
                if profile1.get("institution_name") == profile2.get("institution_name"):
                    similarity_score += 0.3
                
                # Même type de compte
                if profile1.get("account_type") == profile2.get("account_type"):
                    similarity_score += 0.2
                
                # Usage similaire
                if profile1.get("primary_purpose") == profile2.get("primary_purpose"):
                    similarity_score += 0.3
                
                # Soldes similaires
                balance1 = profile1.get("current_balance", 0)
                balance2 = profile2.get("current_balance", 0)
                if abs(balance1 - balance2) < max(balance1, balance2) * 0.1:  # Différence < 10%
                    similarity_score += 0.2
                
                # Si score de similarité élevé, considérer comme doublon potentiel
                if similarity_score >= 0.6:
                    duplicates.append({
                        "account1": {
                            "id": profile1.get("bridge_account_id"),
                            "name": profile1.get("account_name"),
                            "balance": profile1.get("current_balance", 0)
                        },
                        "account2": {
                            "id": profile2.get("bridge_account_id"),
                            "name": profile2.get("account_name"),
                            "balance": profile2.get("current_balance", 0)
                        },
                        "similarity_score": similarity_score,
                        "recommendation": "Vérifiez si ces comptes font doublon et considérez en fermer un"
                    })
        
        return duplicates