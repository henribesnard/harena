"""add_category_groups_table_and_description

Revision ID: 2014325daf3c
Revises: 3dbd0b3b084e
Create Date: 2025-10-02 23:20:04.859222

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2014325daf3c'
down_revision: Union[str, None] = '3dbd0b3b084e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Étape 1: Créer la table category_groups
    op.create_table(
        'category_groups',
        sa.Column('group_id', sa.Integer(), nullable=False, comment='Category group ID'),
        sa.Column('group_name', sa.String(length=255), nullable=False, comment='Category group name'),
        sa.Column('description', sa.Text(), nullable=True, comment='Group description'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('group_id')
    )

    # Étape 2: Insérer les groupes existants depuis la table categories
    op.execute("""
        INSERT INTO category_groups (group_id, group_name)
        SELECT DISTINCT group_id, group_name
        FROM categories
        ORDER BY group_id
        ON CONFLICT (group_id) DO NOTHING
    """)

    # Étape 3: Ajouter la colonne description à categories
    op.add_column('categories', sa.Column('description', sa.Text(), nullable=True, comment='Category description'))

    # Étape 4: Créer la foreign key entre categories et category_groups
    op.create_foreign_key(
        'fk_categories_group_id',
        'categories', 'category_groups',
        ['group_id'], ['group_id']
    )

    # Étape 5: Supprimer la colonne group_name de categories (maintenant dans category_groups)
    op.drop_column('categories', 'group_name')


def downgrade() -> None:
    """Downgrade schema."""
    # Étape 1: Ré-ajouter group_name à categories
    op.add_column('categories', sa.Column('group_name', sa.String(length=255), nullable=True, comment='Bridge API category group name'))

    # Étape 2: Remplir group_name depuis category_groups
    op.execute("""
        UPDATE categories
        SET group_name = cg.group_name
        FROM category_groups cg
        WHERE categories.group_id = cg.group_id
    """)

    # Étape 3: Rendre group_name NOT NULL
    op.alter_column('categories', 'group_name', nullable=False)

    # Étape 4: Supprimer la foreign key
    op.drop_constraint('fk_categories_group_id', 'categories', type_='foreignkey')

    # Étape 5: Supprimer la colonne description de categories
    op.drop_column('categories', 'description')

    # Étape 6: Supprimer la table category_groups
    op.drop_table('category_groups')
