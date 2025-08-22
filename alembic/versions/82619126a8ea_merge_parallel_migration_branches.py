"""Merge parallel migration branches

Revision ID: 82619126a8ea
Revises: 5038c27e9983, c2805ce5b2d9
Create Date: 2025-08-22 10:22:38.755186

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '82619126a8ea'
down_revision: Union[str, None] = ('5038c27e9983', 'c2805ce5b2d9')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
