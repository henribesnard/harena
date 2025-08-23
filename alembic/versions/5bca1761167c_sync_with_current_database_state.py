"""sync with current database state - FIXED VERSION

Revision ID: 5bca1761167c
Revises:
Create Date: 2025-08-23 23:30:00.000000

Version corrigée pour gérer les données existantes de manière sécurisée.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5bca1761167c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    print("🔧 Migration sécurisée des tables conversation...")

    conn = op.get_bind()
    insp = sa.inspect(conn)

    # =========================================================================
    # ÉTAPE 1: Supprimer conversation_summaries (si elle existe)
    # NB: Inutile de supprimer ses index à part; DROP TABLE CASCADE les supprime.
    # =========================================================================
    print("🗑️  Suppression table conversation_summaries (si existe)...")
    op.execute("DROP TABLE IF EXISTS conversation_summaries CASCADE")

    # =========================================================================
    # ÉTAPE 2: conversation_turns
    # =========================================================================
    print("🔄 Modification sécurisée de conversation_turns...")

    # 2.a Ajouter la colonne data (nullable), si absente
    ct_cols = {c["name"] for c in insp.get_columns("conversation_turns")}
    if "data" not in ct_cols:
        with op.batch_alter_table("conversation_turns") as batch_op:
            print("   📝 Ajout colonne data (nullable)...")
            batch_op.add_column(sa.Column("data", sa.JSON(), nullable=True))

    # 2.b Peupler et mettre NOT NULL + default
    print("   🔄 Peuplement des données existantes...")
    op.execute(sa.text("UPDATE conversation_turns SET data = '{}' WHERE data IS NULL"))

    print("   🔒 Application contrainte NOT NULL...")
    with op.batch_alter_table("conversation_turns") as batch_op:
        batch_op.alter_column("data", nullable=False, server_default='{}')

    # 2.c Supprimer proprement les colonnes obsolètes (uniquement si elles existent)
    columns_to_remove_ct = [
        'turn_id', 'error_message', 'intent_result', 'search_execution_time_ms',
        'processing_time_ms', 'search_query_used', 'agent_chain', 'completion_tokens',
        'total_tokens', 'prompt_tokens', 'intent_classification', 'turn_metadata',
        'intent', 'confidence_score', 'intent_confidence', 'total_tokens_used',
        'error_occurred', 'search_results_count', 'entities', 'entities_extracted'
    ]
    ct_cols = {c["name"] for c in insp.get_columns("conversation_turns")}
    cols_present_ct = [c for c in columns_to_remove_ct if c in ct_cols]
    if cols_present_ct:
        print("   🗑️  Suppression colonnes obsolètes...")
        with op.batch_alter_table("conversation_turns") as batch_op:
            for col in cols_present_ct:
                batch_op.drop_column(col)
                print(f"      ✅ {col}")

    # 2.d Supprimer l'index obsolète HORS batch, de manière sûre
    print("   🧹 Nettoyage index obsolètes...")
    # Variante 1 (fonctionne partout) :
    op.execute("DROP INDEX IF EXISTS ix_conversation_turns_turn_id")
    # Variante 2 (si votre Alembic >= 1.13): 
    # op.drop_index('ix_conversation_turns_turn_id', table_name='conversation_turns', if_exists=True)

    # =========================================================================
    # ÉTAPE 3: conversations
    # =========================================================================
    print("🔄 Modification sécurisée de conversations...")

    conv_cols = {c["name"] for c in insp.get_columns("conversations")}
    if "data" not in conv_cols:
        with op.batch_alter_table("conversations") as batch_op:
            print("   📝 Ajout colonne data (nullable)...")
            batch_op.add_column(sa.Column("data", sa.JSON(), nullable=True))

    print("   🔄 Peuplement des données existantes...")
    op.execute(sa.text("UPDATE conversations SET data = '{}' WHERE data IS NULL"))

    print("   🔒 Application contrainte NOT NULL et nettoyage...")
    with op.batch_alter_table("conversations") as batch_op:
        batch_op.alter_column("data", nullable=False, server_default='{}')

    columns_to_remove_conv = [
        'intents', 'language', 'conversation_metadata', 'domain',
        'completion_tokens', 'max_turns', 'prompt_tokens', 'entities',
        'session_metadata', 'user_preferences', 'total_tokens'
    ]
    conv_cols = {c["name"] for c in insp.get_columns("conversations")}
    cols_present_conv = [c for c in columns_to_remove_conv if c in conv_cols]
    if cols_present_conv:
        print("   🗑️  Suppression colonnes obsolètes...")
        with op.batch_alter_table("conversations") as batch_op:
            for col in cols_present_conv:
                batch_op.drop_column(col)
                print(f"      ✅ {col}")

    print("✅ Migration sécurisée terminée!")
    print("🎯 Tables conversation simplifiées avec préservation des données")


def downgrade() -> None:
    print("⏪ Restauration de l'ancien schéma (partielle)...")
    # Attention: downgrade non exhaustif, on retire juste data
    with op.batch_alter_table("conversations") as batch_op:
        if "data" in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("conversations")}:
            batch_op.drop_column("data")
    with op.batch_alter_table("conversation_turns") as batch_op:
        if "data" in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("conversation_turns")}:
            batch_op.drop_column("data")

    op.execute("CREATE TABLE IF NOT EXISTS conversation_summaries (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL)")
    print("✅ Downgrade terminé (simplifié)")
