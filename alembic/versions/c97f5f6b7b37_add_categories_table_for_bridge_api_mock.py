"""add_categories_table_for_bridge_api_mock

Revision ID: c97f5f6b7b37
Revises: a58a4daacbb8
Create Date: 2025-09-05 15:46:50.598747

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c97f5f6b7b37'
down_revision: Union[str, None] = 'a58a4daacbb8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Create categories table for Bridge API mock."""
    
    # Create categories table based on Bridge API structure
    op.create_table('categories',
        sa.Column('category_id', sa.Integer(), nullable=False, comment='Bridge API category ID'),
        sa.Column('category_name', sa.String(255), nullable=False, comment='Bridge API category name'),
        sa.Column('group_id', sa.Integer(), nullable=False, comment='Bridge API category group ID'),
        sa.Column('group_name', sa.String(255), nullable=False, comment='Bridge API category group name'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('category_id'),
        comment='Categories from Bridge API for transaction categorization'
    )
    
    # Create indexes for performance
    op.create_index('ix_categories_category_name', 'categories', ['category_name'])
    op.create_index('ix_categories_group_id', 'categories', ['group_id'])


def downgrade() -> None:
    """Downgrade schema: Drop categories table."""
    
    op.drop_index('ix_categories_group_id', table_name='categories')
    op.drop_index('ix_categories_category_name', table_name='categories')
    op.drop_table('categories')
