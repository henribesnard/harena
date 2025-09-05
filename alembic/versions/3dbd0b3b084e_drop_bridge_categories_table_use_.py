"""drop_bridge_categories_table_use_categories_instead

Revision ID: 3dbd0b3b084e
Revises: fd37009410f7
Create Date: 2025-09-05 16:39:08.905695

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3dbd0b3b084e'
down_revision: Union[str, None] = 'fd37009410f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Drop bridge_categories table (replaced by categories)."""
    
    # Drop the bridge_categories table as it's replaced by our categories table
    # which has better structure and is already populated with mock data
    op.drop_table('bridge_categories')


def downgrade() -> None:
    """Downgrade schema: Recreate bridge_categories table."""
    
    # Recreate bridge_categories table structure for rollback
    op.create_table('bridge_categories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bridge_category_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('parent_id', sa.Integer(), nullable=True),
        sa.Column('parent_name', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
