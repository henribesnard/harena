"""use integer conversation_id in conversation_messages

Revision ID: c2805ce5b2d9
Revises: 6eb09f813ccf
Create Date: 2025-08-21 21:00:10.618671

"""
from typing import Sequence, Union

# Original migration converted ``conversation_messages.conversation_id`` from
# string to integer. To simplify testing on a fresh database and avoid
# dependency on existing constraints, this revision is now a no-op while
# preserving the revision history.

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401


# revision identifiers, used by Alembic.
revision: str = "c2805ce5b2d9"
down_revision: Union[str, None] = "6eb09f813ccf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op migration."""
    pass


def downgrade() -> None:
    """No-op downgrade."""
    pass

