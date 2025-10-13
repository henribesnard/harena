"""add_foreign_key_raw_transactions_to_categories

Revision ID: fd37009410f7
Revises: c97f5f6b7b37
Create Date: 2025-09-05 16:03:05.174092

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fd37009410f7'
down_revision: Union[str, None] = 'c97f5f6b7b37'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Add foreign key constraint between raw_transactions and categories."""
    
    # Add foreign key constraint from raw_transactions.category_id to categories.category_id
    # Note: La contrainte permet NULL (transactions sans catégorie)
    op.create_foreign_key(
        constraint_name='fk_raw_transactions_category_id',
        source_table='raw_transactions', 
        referent_table='categories',
        local_cols=['category_id'],
        remote_cols=['category_id'],
        ondelete='SET NULL',  # Si catégorie supprimée, mettre category_id à NULL
        onupdate='CASCADE'    # Si category_id change, mettre à jour
    )
    
    # Ajouter un index pour les performances sur les requêtes par catégorie
    op.create_index('ix_raw_transactions_category_id', 'raw_transactions', ['category_id'])


def downgrade() -> None:
    """Downgrade schema: Remove foreign key constraint."""
    
    op.drop_index('ix_raw_transactions_category_id', table_name='raw_transactions')
    op.drop_constraint('fk_raw_transactions_category_id', 'raw_transactions', type_='foreignkey')
