"""add_merchant_name_column_to_raw_transactions

Revision ID: 065c1780d509
Revises: 972cc48fabfc
Create Date: 2025-10-06 23:18:37.151529

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '065c1780d509'
down_revision: Union[str, None] = '972cc48fabfc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('raw_transactions', sa.Column('merchant_name', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('raw_transactions', 'merchant_name')
