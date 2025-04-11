# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Ajout des chemins d'importation pour les modèles
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import des modèles et configuration
from user_service.models.base import Base
from user_service.models.user import User, BridgeConnection, UserPreference
from user_service.core.config import settings

# Importer les modèles sync_service si disponibles
try:
    from sync_service.models.sync import WebhookEvent, SyncItem, SyncAccount
    sync_models_imported = True
except ImportError:
    sync_models_imported = False
    print("Avertissement: Les modèles sync_service n'ont pas pu être importés.")

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
fileConfig(config.config_file_name)

# Définir l'URL de la base de données depuis les settings
config.set_main_option("sqlalchemy.url", str(settings.SQLALCHEMY_DATABASE_URI))

# Ajout des métadonnées à travers lesquelles nous voulons faire des migrations
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()