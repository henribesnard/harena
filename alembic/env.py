from logging.config import fileConfig
from sqlalchemy import engine_from_config, MetaData
from sqlalchemy import pool
from alembic import context

# Ajout des chemins d'importation pour les modèles
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import des modèles et configuration
from user_service.models.base import Base as UserBase
from user_service.models.user import User, BridgeConnection, UserPreference
from user_service.core.config import settings

# Importer les modèles sync_service si disponibles
try:
    from sync_service.models.sync import WebhookEvent, SyncItem, SyncAccount
    sync_models_imported = True
except ImportError:
    sync_models_imported = False
    print("Avertissement: Les modèles sync_service n'ont pas pu être importés.")

# Importer les modèles conversation_service si disponibles
try:
    from conversation_service.db.models import Base as ConversationBase
    from conversation_service.db.models import Conversation, Message, ConversationContext
    conversation_models_imported = True
except ImportError:
    conversation_models_imported = False
    print("Avertissement: Les modèles conversation_service n'ont pas pu être importés.")

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
fileConfig(config.config_file_name)

# Gestion de DATABASE_URL fourni par Heroku
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # Heroku utilise "postgres://" mais SQLAlchemy 1.4+ requiert "postgresql://"
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    # Utiliser DATABASE_URL de Heroku au lieu de la configuration locale
    config.set_main_option("sqlalchemy.url", database_url)
else:
    # Utiliser la configuration locale
    config.set_main_option("sqlalchemy.url", str(settings.SQLALCHEMY_DATABASE_URI))

# Combiner les métadonnées des différents modèles
combined_metadata = MetaData()

# Ajouter les métadonnées de user_service
for table in UserBase.metadata.tables.values():
    table.tometadata(combined_metadata)

# Ajouter les métadonnées de sync_service si disponibles
if sync_models_imported:
    # Ajout direct des tables car nous n'avons pas importé Base explicitement
    for table in SyncItem.__table__.metadata.tables.values():
        if table.name not in combined_metadata.tables:
            table.tometadata(combined_metadata)

# Ajouter les métadonnées de conversation_service si disponibles
if conversation_models_imported:
    for table in ConversationBase.metadata.tables.values():
        if table.name not in combined_metadata.tables:
            table.tometadata(combined_metadata)

# Utiliser les métadonnées combinées pour les migrations
target_metadata = combined_metadata


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