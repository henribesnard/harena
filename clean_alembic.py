#!/usr/bin/env python3
"""
Script de nettoyage complet Alembic pour résoudre les problèmes de révisions.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Execute une commande avec gestion d'erreur."""
    print(f"🔧 {description}")
    print(f"💻 {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"✅ {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur: {e}")
        if e.stdout:
            print(f"📤 Sortie: {e.stdout}")
        if e.stderr:
            print(f"📥 Erreur: {e.stderr}")
        return False

def main():
    print("🚨 === NETTOYAGE COMPLET ALEMBIC ===")
    print("⚠️  Ceci va supprimer l'historique des migrations et repartir de zéro")
    
    response = input("\n❓ Continuer avec le nettoyage complet? (oui/non): ").lower().strip()
    if response not in ['oui', 'o', 'yes', 'y']:
        print("❌ Nettoyage annulé")
        sys.exit(0)
    
    # 1. Sauvegarde
    print("\n📦 Sauvegarde recommandée...")
    print("   python scripts/db_backup.py --output backups/before_cleanup.sql")
    
    # 2. Supprimer TOUTES les migrations conversation
    print("\n🗑️  Suppression de toutes les migrations conversation...")
    migrations_dir = Path("alembic/versions")
    
    conversation_migrations = [
        "48f2e3ebd17c_enrich_conversation_turns.py",
        "d40dee0675ee_enrich_conversation_models_with_nlp_.py", 
        "12d193f2a3ae_enhance_conversation_for_ai.py",
        "44775ef78f2d_modif_conversation_service_pour_.py",
        "89cf29c22f13_merge_conversation_nlp_and_other_changes.py",
        "41b5fbb88ff5_*.py",  # Migration problématique
        "053199a23dec_*.py",  # Head problématique
    ]
    
    for pattern in conversation_migrations:
        # Utiliser glob pour trouver les fichiers correspondants
        for migration_file in migrations_dir.glob(pattern):
            print(f"🗑️  Suppression de {migration_file.name}")
            migration_file.unlink()
    
    # 3. Forcer Alembic à une révision stable connue
    print("\n🔧 Force reset vers révision stable...")
    
    # Essayer de trouver une révision stable
    stable_revisions = [
        "93f0d886307b",  # Migration de base
        # Ajoutez d'autres révisions stables si nécessaire
    ]
    
    for revision in stable_revisions:
        print(f"🎯 Tentative de stamp vers {revision}...")
        if run_command([
            "alembic", "stamp", revision
        ], f"Force stamp vers {revision}", check=False):
            print(f"✅ Stamp réussi vers {revision}")
            break
    else:
        print("⚠️  Aucune révision stable trouvée, stamp vers head")
        run_command(["alembic", "stamp", "head"], "Force stamp head", check=False)
    
    # 4. Créer la migration simple
    print("\n📝 Création de la migration simple...")
    
    migration_content = '''"""create simple conversations tables

Revision ID: simple_conv_clean
Revises: 93f0d886307b
Create Date: 2025-08-23 23:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'simple_conv_clean'
down_revision: Union[str, None] = '93f0d886307b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    print("🚀 Création tables conversation simples...")
    
    op.create_table(
        'conversations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.String(255), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('total_turns', sa.Integer(), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('conversation_id')
    )
    
    op.create_table(
        'conversation_turns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('turn_number', sa.Integer(), nullable=False),
        sa.Column('user_message', sa.Text(), nullable=False),
        sa.Column('assistant_response', sa.Text(), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE')
    )
    
    print("✅ Tables créées!")

def downgrade() -> None:
    op.drop_table('conversation_turns')
    op.drop_table('conversations')
'''
    
    clean_migration_file = migrations_dir / "simple_conv_clean_create_conversations.py"
    with open(clean_migration_file, 'w', encoding='utf-8') as f:
        f.write(migration_content)
    
    print(f"✅ Migration propre créée: {clean_migration_file}")
    
    # 5. Vérifier l'état
    print("\n📍 Vérification état...")
    run_command(["alembic", "current"], "État actuel", check=False)
    run_command(["alembic", "history"], "Historique", check=False)
    
    print("\n🎯 === NETTOYAGE TERMINÉ ===")
    print("Prochaines étapes:")
    print("1. alembic upgrade head")
    print("2. Vérifier que les tables sont créées")

if __name__ == "__main__":
    main()