"""enhance conversation for ai

Revision ID: 12d193f2a3ae
Revises: 4f2079db2694
Create Date: 2025-08-22 10:55:00.607115

"""
from typing import Sequence, Union

# Migration transformed into a no-op because the columns it introduced are
# already present from earlier revisions. This preserves the Alembic history
# without altering the schema a second time.

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401


# revision identifiers, used by Alembic.
revision: str = '12d193f2a3ae'
down_revision: Union[str, None] = '4f2079db2694'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op migration."""
    pass


def downgrade() -> None:
    """No-op downgrade."""
    pass
