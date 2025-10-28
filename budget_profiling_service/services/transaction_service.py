"""
Service de récupération et préparation des transactions pour analyse
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, func, case, extract
from sqlalchemy.orm import Session
import logging

from db_service.models.sync import RawTransaction, Category

logger = logging.getLogger(__name__)

# Cache partagé au niveau du module pour les catégories
_CATEGORY_CACHE: Dict[int, str] = {}
_CATEGORY_CACHE_TIMESTAMP: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=1)


class TransactionService:
    """
    Service de récupération et formatage des transactions
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self._preload_categories()

    def get_user_transactions(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        months_back: Optional[int] = None,
        account_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les transactions d'un utilisateur sur une période

        Args:
            user_id: ID utilisateur
            start_date: Date début (optionnel)
            end_date: Date fin (optionnel)
            months_back: Nombre de mois en arrière (None = toutes les transactions)
            account_ids: Liste de bridge_account_ids à filtrer (optionnel)

        Returns:
            Liste de transactions formatées pour analyse
        """
        try:
            if not end_date:
                end_date = datetime.now()

            if not start_date:
                if months_back is None:
                    # Récupérer TOUTES les transactions - mettre une date très ancienne
                    start_date = datetime(2000, 1, 1)
                else:
                    start_date = datetime.now() - timedelta(days=30 * months_back)

            logger.info(f"Récupération transactions user {user_id} de {start_date} à {end_date}")

            # Construire les conditions WHERE
            where_conditions = [
                RawTransaction.user_id == user_id,
                RawTransaction.date >= start_date,
                RawTransaction.date <= end_date,
                RawTransaction.deleted == False
            ]

            # Ajouter filtre par comptes si spécifié
            if account_ids:
                # Récupérer les IDs internes depuis les bridge_account_ids
                from db_service.models.sync import SyncAccount, SyncItem
                stmt_accounts = (
                    select(SyncAccount.id)
                    .join(SyncItem)
                    .where(SyncItem.user_id == user_id)
                    .where(SyncAccount.bridge_account_id.in_(account_ids))
                )
                internal_ids_result = self.db.execute(stmt_accounts)
                internal_ids = [row[0] for row in internal_ids_result]

                if internal_ids:
                    where_conditions.append(RawTransaction.account_id.in_(internal_ids))
                    logger.info(f"Filtrage sur {len(internal_ids)} comptes (bridge_ids: {account_ids})")
                else:
                    logger.warning(f"Aucun compte trouvé pour bridge_account_ids: {account_ids}")

            query = select(RawTransaction).where(
                and_(*where_conditions)
            ).order_by(RawTransaction.date.desc())

            result = self.db.execute(query)
            transactions = result.scalars().all()

            logger.info(f"Trouvé {len(transactions)} transactions")

            # Formater pour analyse
            formatted = []
            for tx in transactions:
                formatted.append({
                    'id': tx.bridge_transaction_id,
                    'user_id': tx.user_id,
                    'merchant_name': tx.merchant_name or tx.clean_description,
                    'amount': float(tx.amount) if tx.amount else 0.0,
                    'date': tx.date,
                    'category': self._get_category_name(tx.category_id),
                    'operation_type': tx.operation_type,
                    'is_debit': tx.amount < 0 if tx.amount else False,
                    'is_credit': tx.amount > 0 if tx.amount else False
                })

            return formatted

        except Exception as e:
            logger.error(f"Erreur récupération transactions: {e}", exc_info=True)
            return []

    def _preload_categories(self):
        """
        Précharge toutes les catégories au démarrage (cache partagé)
        """
        global _CATEGORY_CACHE, _CATEGORY_CACHE_TIMESTAMP

        # Vérifier si le cache est déjà chargé et valide
        if _CATEGORY_CACHE and _CATEGORY_CACHE_TIMESTAMP:
            if datetime.now() - _CATEGORY_CACHE_TIMESTAMP < _CACHE_TTL:
                return  # Cache valide, pas besoin de recharger

        try:
            result = self.db.execute(select(Category))
            categories = result.scalars().all()

            # Vider et recharger le cache
            _CATEGORY_CACHE.clear()
            for cat in categories:
                _CATEGORY_CACHE[cat.category_id] = cat.category_name

            _CATEGORY_CACHE_TIMESTAMP = datetime.now()
            logger.info(f"Préchargé {len(_CATEGORY_CACHE)} catégories dans le cache partagé")

        except Exception as e:
            logger.error(f"Erreur préchargement catégories: {e}")

    def _get_category_name(self, category_id: Optional[int]) -> str:
        """
        Récupère nom catégorie depuis category_id avec cache partagé et TTL
        """
        global _CATEGORY_CACHE, _CATEGORY_CACHE_TIMESTAMP

        if not category_id:
            return 'uncategorized'

        # Invalider cache si trop ancien
        if _CATEGORY_CACHE_TIMESTAMP and datetime.now() - _CATEGORY_CACHE_TIMESTAMP > _CACHE_TTL:
            logger.info("Cache catégories expiré, rechargement...")
            self._preload_categories()

        # Vérifier cache partagé
        if category_id in _CATEGORY_CACHE:
            return _CATEGORY_CACHE[category_id]

        # Si pas dans cache, récupérer depuis DB et ajouter au cache
        try:
            result = self.db.execute(
                select(Category).where(Category.category_id == category_id)
            )
            category = result.scalar_one_or_none()

            if category:
                _CATEGORY_CACHE[category_id] = category.category_name
                return category.category_name
            else:
                return 'uncategorized'

        except Exception as e:
            logger.error(f"Erreur récupération catégorie {category_id}: {e}")
            return 'uncategorized'

    def get_monthly_aggregates(
        self,
        user_id: int,
        months: Optional[int] = None,
        account_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Agrégations mensuelles (revenus, dépenses) - OPTIMISÉ DB-SIDE

        Args:
            user_id: ID utilisateur
            months: Nombre de mois (None = toutes les transactions)
            account_ids: Liste d'IDs de comptes à inclure (None = tous les comptes)

        Returns:
            [
                {
                    'month': '2025-01',
                    'total_income': 3000.0,
                    'total_expenses': 2400.0,
                    'net_cashflow': 600.0,
                    'transaction_count': 145
                },
                ...
            ]
        """
        try:
            # Calculer date de début
            if months is None:
                start_date = datetime(2000, 1, 1)
            else:
                start_date = datetime.now() - timedelta(days=30 * months)

            end_date = datetime.now()

            accounts_log = f" sur {len(account_ids)} comptes" if account_ids else ""
            logger.info(f"Agrégats mensuels user {user_id}{accounts_log} (DB-side) de {start_date} à {end_date}")

            # Construire conditions WHERE
            where_conditions = [
                RawTransaction.user_id == user_id,
                RawTransaction.date >= start_date,
                RawTransaction.date <= end_date,
                RawTransaction.deleted == False
            ]

            # Ajouter filtre par comptes si spécifié
            if account_ids:
                # Récupérer les IDs internes depuis les bridge_account_ids
                from db_service.models.sync import SyncAccount, SyncItem
                stmt_accounts = (
                    select(SyncAccount.id)
                    .join(SyncItem)
                    .where(SyncItem.user_id == user_id)
                    .where(SyncAccount.bridge_account_id.in_(account_ids))
                )
                internal_ids_result = self.db.execute(stmt_accounts)
                internal_ids = [row[0] for row in internal_ids_result]

                if internal_ids:
                    where_conditions.append(RawTransaction.account_id.in_(internal_ids))
                    logger.debug(f"Filtrage sur {len(internal_ids)} comptes internes (depuis {len(account_ids)} bridge IDs)")
                else:
                    logger.warning(f"Aucun compte trouvé pour les bridge_account_ids fournis")
                    return []  # Aucune transaction si aucun compte trouvé

            # Agrégation DB-side avec GROUP BY sur année/mois
            query = (
                select(
                    # Colonnes groupées
                    extract('year', RawTransaction.date).label('year'),
                    extract('month', RawTransaction.date).label('month'),
                    # Agrégations
                    func.sum(
                        case(
                            (RawTransaction.amount > 0, RawTransaction.amount),
                            else_=0
                        )
                    ).label('total_income'),
                    func.sum(
                        case(
                            (RawTransaction.amount < 0, func.abs(RawTransaction.amount)),
                            else_=0
                        )
                    ).label('total_expenses'),
                    func.count(RawTransaction.id).label('transaction_count')
                )
                .where(and_(*where_conditions))
                .group_by(
                    extract('year', RawTransaction.date),
                    extract('month', RawTransaction.date)
                )
                .order_by('year', 'month')
            )

            result_set = self.db.execute(query)

            # Formater résultats
            results = []
            for row in result_set:
                month_str = f"{int(row.year)}-{int(row.month):02d}"
                total_income = float(row.total_income or 0)
                total_expenses = float(row.total_expenses or 0)

                results.append({
                    'month': month_str,
                    'total_income': round(total_income, 2),
                    'total_expenses': round(total_expenses, 2),
                    'net_cashflow': round(total_income - total_expenses, 2),
                    'transaction_count': int(row.transaction_count or 0)
                })

            logger.info(f"Agrégats calculés: {len(results)} mois")
            return results

        except Exception as e:
            logger.error(f"Erreur calcul agrégats mensuels: {e}", exc_info=True)
            return []

    def get_category_breakdown(
        self,
        user_id: int,
        months: Optional[int] = None,
        account_ids: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Répartition des dépenses par catégorie (MOYENNES MENSUELLES) - OPTIMISÉ DB-SIDE

        Args:
            user_id: ID utilisateur
            months: Nombre de mois (None = toutes les transactions)
            account_ids: Liste d'IDs de comptes à inclure (None = tous les comptes)

        Returns:
            {
                'alimentation': 450.0,  # Moyenne mensuelle
                'transport': 120.0,
                'loisirs': 230.0,
                ...
            }
        """
        try:
            # Calculer date de début
            if months is None:
                start_date = datetime(2000, 1, 1)
            else:
                start_date = datetime.now() - timedelta(days=30 * months)

            end_date = datetime.now()

            accounts_log = f" sur {len(account_ids)} comptes" if account_ids else ""
            logger.info(f"Breakdown catégories user {user_id}{accounts_log} (DB-side) de {start_date} à {end_date}")

            # Construire conditions WHERE communes
            where_conditions = [
                RawTransaction.user_id == user_id,
                RawTransaction.date >= start_date,
                RawTransaction.date <= end_date,
                RawTransaction.deleted == False
            ]

            # Ajouter filtre par comptes si spécifié
            if account_ids:
                from db_service.models.sync import SyncAccount, SyncItem
                stmt_accounts = (
                    select(SyncAccount.id)
                    .join(SyncItem)
                    .where(SyncItem.user_id == user_id)
                    .where(SyncAccount.bridge_account_id.in_(account_ids))
                )
                internal_ids_result = self.db.execute(stmt_accounts)
                internal_ids = [row[0] for row in internal_ids_result]

                if internal_ids:
                    where_conditions.append(RawTransaction.account_id.in_(internal_ids))
                else:
                    logger.warning(f"Aucun compte trouvé pour les bridge_account_ids fournis")
                    return {}

            # 1. Compter le nombre de mois distincts (pour calculer moyennes)
            months_query = (
                select(
                    func.count(
                        func.distinct(
                            func.concat(
                                extract('year', RawTransaction.date),
                                '-',
                                extract('month', RawTransaction.date)
                            )
                        )
                    ).label('nb_months')
                )
                .where(and_(*where_conditions))
            )

            nb_months_result = self.db.execute(months_query)
            nb_months = nb_months_result.scalar() or 0

            if nb_months == 0:
                logger.warning(f"Aucun mois trouvé pour user {user_id}")
                return {}

            # 2. Agrégation par catégorie (seulement débits)
            category_where = where_conditions + [RawTransaction.amount < 0]  # Seulement débits
            category_query = (
                select(
                    Category.category_name,
                    func.sum(func.abs(RawTransaction.amount)).label('total_amount')
                )
                .join(Category, RawTransaction.category_id == Category.category_id, isouter=True)
                .where(and_(*category_where))
                .group_by(Category.category_name)
            )

            result_set = self.db.execute(category_query)

            # 3. Calculer moyennes mensuelles
            category_averages = {}
            for row in result_set:
                category_name = row.category_name or 'uncategorized'
                total_amount = float(row.total_amount or 0)
                avg_amount = total_amount / nb_months

                category_averages[category_name] = round(avg_amount, 2)

            logger.info(f"Breakdown calculé: {len(category_averages)} catégories sur {nb_months} mois")
            return category_averages

        except Exception as e:
            logger.error(f"Erreur calcul breakdown catégories: {e}", exc_info=True)
            return {}

    def get_merchant_transactions(
        self,
        user_id: int,
        merchant_name: str,
        months_back: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Récupère toutes les transactions pour un marchand spécifique
        Utile pour la détection de charges fixes
        """
        try:
            transactions = self.get_user_transactions(
                user_id,
                months_back=months_back
            )

            # Filtrer par merchant_name
            merchant_txs = [
                tx for tx in transactions
                if tx['merchant_name'] and merchant_name.lower() in tx['merchant_name'].lower()
            ]

            return merchant_txs

        except Exception as e:
            logger.error(f"Erreur récupération transactions marchand: {e}", exc_info=True)
            return []
