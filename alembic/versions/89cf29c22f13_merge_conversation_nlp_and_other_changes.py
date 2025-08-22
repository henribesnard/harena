"""merge conversation nlp and other changes

Revision ID: 89cf29c22f13
Revises: 1f72abc4d3e9, d40dee0675ee
Create Date: 2025-08-22 13:19:18.702737

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '89cf29c22f13'
down_revision: Union[str, None] = ('1f72abc4d3e9', 'd40dee0675ee')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
