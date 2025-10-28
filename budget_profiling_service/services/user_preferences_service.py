"""
Service de gestion des préférences utilisateur pour le Budget Profiling Service
"""
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import select
import logging

from db_service.models.user import UserPreference
from db_service.models.sync import SyncAccount, SyncItem
from db_service.config.default_preferences import (
    get_default_budget_settings,
    merge_with_defaults,
    validate_budget_settings
)

logger = logging.getLogger(__name__)


class UserPreferencesService:
    """
    Service de gestion des préférences utilisateur pour le profiling budgétaire
    """

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_or_create_preferences(self, user_id: int) -> UserPreference:
        """
        Récupère les préférences d'un utilisateur ou les crée avec les valeurs par défaut

        Args:
            user_id: ID de l'utilisateur

        Returns:
            UserPreference: Instance de préférences utilisateur
        """
        try:
            # Rechercher les préférences existantes
            stmt = select(UserPreference).where(UserPreference.user_id == user_id)
            result = self.db.execute(stmt)
            preferences = result.scalar_one_or_none()

            if preferences:
                logger.info(f"Préférences trouvées pour user {user_id}")
                return preferences

            # Créer nouvelles préférences avec valeurs par défaut
            logger.info(f"Création des préférences par défaut pour user {user_id}")
            preferences = UserPreference(
                user_id=user_id,
                notification_settings={},
                display_preferences={},
                budget_settings=get_default_budget_settings()
            )

            self.db.add(preferences)
            self.db.commit()
            self.db.refresh(preferences)

            logger.info(f"Préférences créées avec succès pour user {user_id}")
            return preferences

        except Exception as e:
            logger.error(f"Erreur lors de la récupération/création des préférences pour user {user_id}: {e}", exc_info=True)
            self.db.rollback()
            raise

    def get_budget_settings(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les paramètres budgétaires d'un utilisateur (avec fusion des valeurs par défaut)

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Dict contenant tous les paramètres budgétaires (complets avec valeurs par défaut)
        """
        try:
            preferences = self.get_or_create_preferences(user_id)
            user_budget_settings = preferences.budget_settings or {}

            # Fusionner avec les valeurs par défaut pour garantir complétude
            complete_settings = merge_with_defaults(user_budget_settings)

            logger.debug(f"Budget settings récupérés pour user {user_id}")
            return complete_settings

        except Exception as e:
            logger.error(f"Erreur récupération budget_settings pour user {user_id}: {e}", exc_info=True)
            # En cas d'erreur, retourner les valeurs par défaut
            return get_default_budget_settings()

    def update_budget_settings(
        self,
        user_id: int,
        new_settings: Dict[str, Any],
        partial: bool = True
    ) -> tuple[bool, Optional[Dict[str, Any]], Optional[list[str]]]:
        """
        Met à jour les paramètres budgétaires d'un utilisateur

        Args:
            user_id: ID de l'utilisateur
            new_settings: Nouveaux paramètres (peut être partiel)
            partial: Si True, fusionne avec les paramètres existants. Si False, remplace complètement

        Returns:
            (success, updated_settings, errors)
        """
        try:
            # Valider les nouveaux paramètres
            is_valid, errors = validate_budget_settings(new_settings)
            if not is_valid:
                logger.warning(f"Validation échouée pour user {user_id}: {errors}")
                return False, None, errors

            # Récupérer ou créer les préférences
            preferences = self.get_or_create_preferences(user_id)

            if partial:
                # Fusion partielle : conserver les valeurs existantes et mettre à jour
                current_settings = preferences.budget_settings or {}
                updated_settings = merge_with_defaults(current_settings)

                # Appliquer les nouvelles valeurs (fusion profonde)
                def deep_update(base: dict, updates: dict):
                    for key, value in updates.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            deep_update(base[key], value)
                        else:
                            base[key] = value

                deep_update(updated_settings, new_settings)
            else:
                # Remplacement complet (avec fusion des défauts pour garantir complétude)
                updated_settings = merge_with_defaults(new_settings)

            # Enregistrer
            preferences.budget_settings = updated_settings
            self.db.commit()
            self.db.refresh(preferences)

            logger.info(f"Budget settings mis à jour pour user {user_id}")
            return True, updated_settings, None

        except Exception as e:
            logger.error(f"Erreur mise à jour budget_settings pour user {user_id}: {e}", exc_info=True)
            self.db.rollback()
            return False, None, [str(e)]

    def reset_to_defaults(self, user_id: int) -> bool:
        """
        Réinitialise les paramètres budgétaires aux valeurs par défaut

        Args:
            user_id: ID de l'utilisateur

        Returns:
            True si succès, False sinon
        """
        try:
            preferences = self.get_or_create_preferences(user_id)
            preferences.budget_settings = get_default_budget_settings()

            self.db.commit()
            self.db.refresh(preferences)

            logger.info(f"Budget settings réinitialisés pour user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Erreur réinitialisation budget_settings pour user {user_id}: {e}", exc_info=True)
            self.db.rollback()
            return False

    def get_parameter(self, user_id: int, parameter_path: str, default: Any = None) -> Any:
        """
        Récupère un paramètre spécifique en utilisant un chemin pointé

        Args:
            user_id: ID de l'utilisateur
            parameter_path: Chemin du paramètre (ex: "fixed_charge_detection.min_occurrences")
            default: Valeur par défaut si le paramètre n'existe pas

        Returns:
            Valeur du paramètre ou default

        Examples:
            >>> service.get_parameter(1, "months_analysis")
            12
            >>> service.get_parameter(1, "fixed_charge_detection.min_occurrences")
            5
        """
        try:
            settings = self.get_budget_settings(user_id)

            # Navigation dans le dictionnaire avec le chemin pointé
            keys = parameter_path.split(".")
            value = settings

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default

            return value

        except Exception as e:
            logger.error(f"Erreur récupération paramètre {parameter_path} pour user {user_id}: {e}")
            return default

    def get_filtered_account_ids(self, user_id: int) -> List[int]:
        """
        Récupère les IDs des comptes à inclure dans les calculs budgétaires
        en fonction des préférences de l'utilisateur

        IMPORTANT: Seuls les comptes "checking" et "card" sont éligibles pour les calculs
        budgétaires (whitelist). Les autres types (savings, loan, investment, other)
        sont toujours exclus automatiquement.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Liste des bridge_account_ids à inclure dans les calculs (checking + card uniquement)

        Examples:
            Mode "all": Tous les comptes checking + card
            Mode "exclude_types": Exclure card (garder seulement checking)
            Mode "include_specific": Seulement les comptes listés (parmi checking + card)
        """
        try:
            # Récupérer les paramètres de sélection
            settings = self.get_budget_settings(user_id)
            account_selection = settings.get("account_selection", {})
            mode = account_selection.get("mode", "all")
            additional_excluded_types = account_selection.get("excluded_types", [])
            included_accounts = account_selection.get("included_accounts", [])

            # Récupérer tous les comptes de l'utilisateur
            stmt = (
                select(SyncAccount)
                .join(SyncItem)
                .where(SyncItem.user_id == user_id)
            )
            result = self.db.execute(stmt)
            all_accounts = result.scalars().all()

            logger.debug(f"User {user_id}: {len(all_accounts)} comptes actifs trouvés")

            # ÉTAPE 1: Filtrer sur types éligibles (WHITELIST: checking + card uniquement)
            ELIGIBLE_TYPES = ["checking", "card"]
            eligible_accounts = [
                acc for acc in all_accounts
                if acc.account_type in ELIGIBLE_TYPES
            ]
            ineligible_count = len(all_accounts) - len(eligible_accounts)
            if ineligible_count > 0:
                logger.info(
                    f"User {user_id}: {ineligible_count} compte(s) non-éligibles exclus "
                    f"(seuls checking/card sont inclus dans les calculs)"
                )

            # ÉTAPE 2: Appliquer les préférences utilisateur sur les comptes éligibles
            if mode == "all":
                # Tous les comptes éligibles (checking + card)
                filtered = eligible_accounts
                logger.info(f"User {user_id}: Mode 'all' - {len(filtered)} comptes checking/card inclus")

            elif mode == "exclude_types":
                # Exclure certains types éligibles (ex: exclure card, garder seulement checking)
                filtered = [
                    acc for acc in eligible_accounts
                    if acc.account_type not in additional_excluded_types
                ]
                additional_excluded = len(eligible_accounts) - len(filtered)
                logger.info(
                    f"User {user_id}: Mode 'exclude_types' - {len(filtered)} comptes inclus, "
                    f"{additional_excluded} exclus en plus (types: {additional_excluded_types})"
                )

            elif mode == "include_specific":
                # Seulement les comptes spécifiques (parmi les éligibles)
                filtered = [
                    acc for acc in eligible_accounts
                    if acc.bridge_account_id in included_accounts
                ]
                logger.info(
                    f"User {user_id}: Mode 'include_specific' - {len(filtered)} comptes inclus "
                    f"sur {len(included_accounts)} demandés (parmi checking/card)"
                )

            else:
                # Fallback: tous les éligibles
                logger.warning(f"User {user_id}: Mode inconnu '{mode}', fallback sur tous (checking/card)")
                filtered = eligible_accounts

            # Retourner les bridge_account_ids
            account_ids = [acc.bridge_account_id for acc in filtered if acc.bridge_account_id]
            logger.info(f"User {user_id}: {len(account_ids)} comptes finaux pour calculs budgétaires")
            return account_ids

        except Exception as e:
            logger.error(f"Erreur filtrage comptes pour user {user_id}: {e}", exc_info=True)
            # En cas d'erreur, retourner liste vide (safer que tous les comptes)
            return []

    def get_filtered_accounts_details(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les détails complets des comptes utilisés dans les calculs budgétaires

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Dict avec informations sur les comptes utilisés:
            {
                'total_accounts': int,  # Nombre total de comptes
                'eligible_accounts': int,  # Comptes éligibles (checking + card)
                'used_accounts': int,  # Comptes effectivement utilisés
                'accounts': [  # Liste des comptes utilisés
                    {
                        'bridge_account_id': int,
                        'account_name': str,
                        'account_type': str,
                        'balance': float,
                        'currency': str
                    }
                ]
            }
        """
        try:
            # Récupérer les paramètres de sélection
            settings = self.get_budget_settings(user_id)
            account_selection = settings.get("account_selection", {})
            mode = account_selection.get("mode", "all")
            additional_excluded_types = account_selection.get("excluded_types", [])
            included_accounts = account_selection.get("included_accounts", [])

            # Récupérer tous les comptes de l'utilisateur
            stmt = (
                select(SyncAccount)
                .join(SyncItem)
                .where(SyncItem.user_id == user_id)
            )
            result = self.db.execute(stmt)
            all_accounts = result.scalars().all()

            # ÉTAPE 1: Filtrer sur types éligibles (WHITELIST)
            ELIGIBLE_TYPES = ["checking", "card"]
            eligible_accounts = [
                acc for acc in all_accounts
                if acc.account_type in ELIGIBLE_TYPES
            ]

            # ÉTAPE 2: Appliquer les préférences utilisateur
            if mode == "all":
                filtered = eligible_accounts
            elif mode == "exclude_types":
                filtered = [
                    acc for acc in eligible_accounts
                    if acc.account_type not in additional_excluded_types
                ]
            elif mode == "include_specific":
                filtered = [
                    acc for acc in eligible_accounts
                    if acc.bridge_account_id in included_accounts
                ]
            else:
                filtered = eligible_accounts

            # Formater les détails des comptes utilisés
            accounts_details = [
                {
                    'bridge_account_id': acc.bridge_account_id,
                    'account_name': acc.account_name or f"Compte {acc.bridge_account_id}",
                    'account_type': acc.account_type,
                    'balance': float(acc.balance) if acc.balance is not None else None,
                    'currency': acc.currency_code
                }
                for acc in filtered
                if acc.bridge_account_id
            ]

            return {
                'total_accounts': len(all_accounts),
                'eligible_accounts': len(eligible_accounts),
                'used_accounts': len(accounts_details),
                'selection_mode': mode,
                'accounts': accounts_details
            }

        except Exception as e:
            logger.error(f"Erreur récupération détails comptes pour user {user_id}: {e}", exc_info=True)
            return {
                'total_accounts': 0,
                'eligible_accounts': 0,
                'used_accounts': 0,
                'selection_mode': 'all',
                'accounts': []
            }
