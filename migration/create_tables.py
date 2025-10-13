#!/usr/bin/env python
"""
Créer toutes les tables directement depuis les models SQLAlchemy
Sans passer par les migrations Alembic qui ont des problèmes avec une base vide
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔧 Importation des modules...")

from config.database import Base, engine
from sqlalchemy import text

# Import de tous les models pour qu'ils soient enregistrés dans Base.metadata
print("📦 Chargement des models...")
from user_service.models import User
from conversation_service.models import (
    ConversationHistory,
    ConversationMetrics,
    ConversationTurn
)
from sync_service.models import (
    BankAccount,
    BankConnection,
    RawTransaction,
    Transaction,
    Category,
    CategoryGroup
)
from enrichment_service.models import EnrichedTransaction
from metric_service.models import (
    MetricDefinition,
    MetricValue,
    MetricAlert
)

print(f"✅ {len(Base.metadata.tables)} tables trouvées dans les models")
print("\nTables à créer:")
for table_name in sorted(Base.metadata.tables.keys()):
    print(f"  - {table_name}")

print("\n" + "="*50)
print("CRÉATION DES TABLES DANS RDS")
print("="*50)

try:
    # Créer toutes les tables
    print("\n🏗️  Création des tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables créées avec succès!")

    # Vérifier les tables créées
    print("\n🔍 Vérification...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """))
        tables = [row[0] for row in result]

        print(f"\n✅ {len(tables)} tables créées dans RDS:")
        for table in tables:
            print(f"  ✓ {table}")

    # Marquer Alembic comme à jour (sur la dernière migration)
    print("\n📋 Marquage Alembic à la dernière version...")
    with engine.connect() as conn:
        # Créer la table alembic_version si elle n'existe pas
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS alembic_version (
                version_num VARCHAR(32) NOT NULL,
                CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
            )
        """))

        # Marquer comme étant à la dernière migration
        conn.execute(text("DELETE FROM alembic_version"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('065c1780d509')"))
        conn.commit()

    print("✅ Alembic marqué à la version: 065c1780d509 (head)")

    print("\n" + "="*50)
    print("✅ SCHÉMA RDS CRÉÉ AVEC SUCCÈS!")
    print("="*50)
    print("\nProchaîne étape: importer les données depuis Heroku")

except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
