"""
Service de gestion des préférences utilisateur pour le Budget Profiling Service
"""
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select
import logging

from db_service.models.user import UserPreference
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
