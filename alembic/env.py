"""
Configuration d'Alembic pour les migrations de base de données.

Ce module configure l'environnement Alembic pour gérer les migrations
de schéma de base de données pour les services user_service et sync_service
de la plateforme Harena.
"""

import logging
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, MetaData
from sqlalchemy import pool
from alembic import context

# Configurer le logger
logger = logging.getLogger("alembic")

# Ajouter le répertoire racine au chemin Python pour permettre les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# -----------------------------------------------------------------------------
# Import des modèles des services
# -----------------------------------------------------------------------------

# Import des modèles user_service
from user_service.models.base import Base as UserBase
from user_service.models.user import User, BridgeConnection, UserPreference

# Import de la configuration
from config_service.config import settings

# Import des modèles sync_service
try:
    from sync_service.models.base import Base as SyncBase
    from sync_service.models.sync import (
        WebhookEvent, SyncItem, SyncAccount, LoanDetail, 
        RawTransaction, BridgeCategory, RawStock,
        AccountInformation, BridgeInsight, SyncTask, SyncStat
    )
    sync_models_imported = True
    logger.info("Modèles sync_service importés avec succès")
except ImportError as e:
    sync_models_imported = False
    logger.warning(f"Impossible d'importer les modèles sync_service: {e}")

# -----------------------------------------------------------------------------
# Configuration Alembic
# -----------------------------------------------------------------------------

# Récupération de la configuration Alembic
config = context.config

# Configurer la journalisation Alembic
fileConfig(config.config_file_name)

# -----------------------------------------------------------------------------
# Configuration de la connexion à la base de données
# -----------------------------------------------------------------------------

# Gestion de l'URL de base de données fournie par Heroku ou l'environnement
database_url = os.environ.get("DATABASE_URL")

if database_url:
    # Heroku utilise "postgres://" mais SQLAlchemy 1.4+ requiert "postgresql://"
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        logger.info("URL de base de données convertie de postgres:// à postgresql://")
    
    # Utiliser DATABASE_URL de l'environnement au lieu de la configuration locale
    config.set_main_option("sqlalchemy.url", database_url)
    logger.info("Utilisation de l'URL de base de données depuis l'environnement")
else:
    # Utiliser la configuration locale depuis settings
    config.set_main_option("sqlalchemy.url", str(settings.SQLALCHEMY_DATABASE_URI))
    logger.info("Utilisation de l'URL de base de données depuis la configuration")

# -----------------------------------------------------------------------------
# Construction des métadonnées combinées pour tous les services
# -----------------------------------------------------------------------------

combined_metadata = MetaData()

# Ajouter les métadonnées de user_service
logger.info("Ajout des métadonnées de user_service")
for table in UserBase.metadata.tables.values():
    table.tometadata(combined_metadata)

# Ajouter les métadonnées de sync_service si disponibles
if sync_models_imported:
    logger.info("Ajout des métadonnées de sync_service")
    for table in SyncBase.metadata.tables.values():
        if table.name not in combined_metadata.tables:
            table.tometadata(combined_metadata)

# Définir les métadonnées cibles pour les migrations
target_metadata = combined_metadata
logger.info(f"Métadonnées combinées créées avec {len(target_metadata.tables)} tables")

# -----------------------------------------------------------------------------
# Fonctions d'exécution de migration
# -----------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Exécute les migrations en mode 'offline'.
    
    En mode offline, les commandes sont écrites dans un script SQL
    sans connexion active à la base de données.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        logger.info("Exécution des migrations en mode offline")
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Exécute les migrations en mode 'online'.
    
    En mode online, une connexion active à la base de données est établie
    et les migrations sont exécutées directement.
    """
    # Configuration de l'engine
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Connexion et exécution des migrations
    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            # Options supplémentaires pour les migrations
            compare_type=True,       # Comparer les types lors de la génération des migrations
            compare_server_default=True,  # Comparer les valeurs par défaut
            include_schemas=True,    # Inclure les schémas si existants
            render_as_batch=True,    # Rendre les migrations en mode batch pour SQLite
        )

        with context.begin_transaction():
            logger.info("Exécution des migrations en mode online")
            context.run_migrations()


# -----------------------------------------------------------------------------
# Point d'entrée principal
# -----------------------------------------------------------------------------

# Exécuter la fonction appropriée selon le mode
if context.is_offline_mode():
    logger.info("Mode offline détecté")
    run_migrations_offline()
else:
    logger.info("Mode online détecté")
    run_migrations_online()

logger.info("Processus de migration terminé")