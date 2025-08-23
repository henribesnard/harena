"""Modif conversation_service pour utiliser uniquement LLM

Revision ID: 44775ef78f2d
Revises: 93f0d886307b
Create Date: 2025-08-14 10:10:32.496646

"""
from typing import Sequence, Union

# This migration previously attempted to modify the ``conversation_turns``
# table by dropping columns that are now required by the application. To
# preserve schema integrity, the migration has been turned into a no-op.
# Keeping the revision in the history maintains Alembic's version chain
# while leaving the database structure unchanged.

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401
from sqlalchemy.dialects import postgresql  # noqa: F401

# revision identifiers, used by Alembic.
revision: str = '44775ef78f2d'
down_revision: Union[str, None] = '93f0d886307b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op migration to preserve required columns."""
    pass


def downgrade() -> None:
    """No-op downgrade."""
    pass
