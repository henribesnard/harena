"""
Module de healthcheck pour la base de données.

Fournit des fonctions pour vérifier la santé de la connexion PostgreSQL.
"""
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging

from db_service.session import engine

logger = logging.getLogger(__name__)


def check_database_health() -> tuple[bool, str]:
    """
    Vérifie la santé de la connexion à la base de données.

    Returns:
        tuple[bool, str]: (is_healthy, message)
            - is_healthy: True si la DB est accessible, False sinon
            - message: Message descriptif de l'état
    """
    try:
        # Essayer d'exécuter une simple requête SELECT 1
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()

        return True, "Database connection successful"

    except SQLAlchemyError as e:
        error_msg = f"Database connection failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error checking database health: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def check_database_pool_health() -> dict:
    """
    Vérifie l'état du pool de connexions à la base de données.

    Returns:
        dict: Informations sur l'état du pool
    """
    try:
        pool = engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in_connections": pool.checkedin(),
            "checked_out_connections": pool.checkedout(),
            "overflow": pool.overflow(),
            "max_overflow": engine.pool._max_overflow,
        }
    except Exception as e:
        logger.error(f"Error checking pool health: {str(e)}")
        return {"error": str(e)}
