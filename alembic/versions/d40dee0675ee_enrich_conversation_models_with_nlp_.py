"""enrich conversation models with nlp metadata

Revision ID: d40dee0675ee
Revises: 12d193f2a3ae
Create Date: 2025-08-22 13:02:16.023232

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd40dee0675ee'
down_revision: Union[str, None] = '12d193f2a3ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema with safe migration for existing data."""
    
    # Étape 1 : Ajouter les nouvelles colonnes comme NULLABLE
    with op.batch_alter_table('conversation_turns', schema=None) as batch_op:
        batch_op.add_column(sa.Column('intent_classification', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('entities_extracted', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('intent_confidence', sa.DECIMAL(precision=5, scale=4), nullable=True))
        batch_op.add_column(sa.Column('total_tokens_used', sa.Integer(), nullable=True))
    
    # Étape 2 : Mettre des valeurs par défaut pour les enregistrements existants
    connection = op.get_bind()
    
    # Vérifier s'il y a des données existantes
    result = connection.execute(sa.text("SELECT COUNT(*) FROM conversation_turns"))
    count = result.scalar()
    
    if count > 0:
        print(f"🔄 Mise à jour de {count} enregistrement(s) existant(s) avec des valeurs par défaut...")
        
        # Mettre à jour les enregistrements existants
        connection.execute(
            sa.text("""
            UPDATE conversation_turns 
            SET 
                intent_classification = '{}',
                entities_extracted = '[]',
                intent_confidence = 0.0,
                total_tokens_used = 0
            WHERE intent_confidence IS NULL
            """)
        )
        
        print("✅ Mise à jour des données existantes terminée")
    else:
        print("ℹ️  Aucune donnée existante à migrer")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('conversation_turns', schema=None) as batch_op:
        batch_op.drop_column('total_tokens_used')
        batch_op.drop_column('intent_confidence')
        batch_op.drop_column('entities_extracted')
        batch_op.drop_column('intent_classification')