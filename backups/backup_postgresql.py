"""
Script de backup PostgreSQL pour Harena
Sauvegarde complète de la base de données avant déploiement AWS
"""
import os
import sys
from datetime import datetime
import subprocess

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import create_engine, text
import pandas as pd

# Configuration
DATABASE_URL = 'postgresql://harena_admin:HaReNa2024SecureDbPassword123@63.35.52.216:5432/harena'
BACKUP_DIR = os.path.join(os.path.dirname(__file__))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

def backup_table_to_csv(engine, table_name, backup_dir):
    """Sauvegarde une table en CSV"""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)

        output_file = os.path.join(backup_dir, f"{table_name}_{TIMESTAMP}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"OK {table_name}: {len(df)} lignes -> {output_file}")
        return True
    except Exception as e:
        print(f"ERROR {table_name}: {str(e)}")
        return False

def backup_postgresql():
    """Backup complet PostgreSQL"""
    print("="*80)
    print(f"BACKUP POSTGRESQL - {TIMESTAMP}")
    print("="*80)
    print()

    # Connexion
    print("Connexion à PostgreSQL...")
    engine = create_engine(DATABASE_URL)

    # Liste des tables
    tables_query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """

    with engine.connect() as conn:
        result = conn.execute(text(tables_query))
        tables = [row[0] for row in result]

    print(f"Tables trouvées: {len(tables)}\n")

    # Backup de chaque table
    success_count = 0
    for table in tables:
        if backup_table_to_csv(engine, table, BACKUP_DIR):
            success_count += 1

    print()
    print("="*80)
    print(f"RÉSULTAT: {success_count}/{len(tables)} tables sauvegardées")
    print(f"Répertoire: {BACKUP_DIR}")
    print("="*80)

    # Créer un fichier de métadonnées
    metadata_file = os.path.join(BACKUP_DIR, f"backup_metadata_{TIMESTAMP}.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"Backup PostgreSQL Harena\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Database: harena\n")
        f.write(f"Host: 63.35.52.216:5432\n")
        f.write(f"Tables: {len(tables)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"\nTables sauvegardées:\n")
        for table in tables:
            f.write(f"  - {table}\n")

    print(f"\nOK Metadonnees: {metadata_file}")

    return success_count == len(tables)

if __name__ == "__main__":
    try:
        success = backup_postgresql()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR FATALE: {str(e)}")
        sys.exit(1)
