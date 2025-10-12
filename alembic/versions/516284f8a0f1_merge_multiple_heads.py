"""merge_multiple_heads

Revision ID: 516284f8a0f1
Revises: 065c1780d509, f9a8b1234567, e28857c31199
Create Date: 2025-10-12 15:02:36.635711

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '516284f8a0f1'
down_revision: Union[str, None] = ('065c1780d509', 'f9a8b1234567', 'e28857c31199')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
