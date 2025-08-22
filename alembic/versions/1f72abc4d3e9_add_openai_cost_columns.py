"""add openai cost and query time columns

Revision ID: 1f72abc4d3e9
Revises: c2805ce5b2d9
Create Date: 2025-01-29 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "1f72abc4d3e9"
down_revision: Union[str, None] = "c2805ce5b2d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add monitoring columns and grant permissions."""
    op.add_column("conversation_turns", sa.Column("openai_cost", sa.Float(), nullable=True))
    op.add_column("conversation_turns", sa.Column("db_query_time_ms", sa.Float(), nullable=True))

    # Grant read access on new columns to application role
    op.execute(
        "GRANT SELECT (openai_cost, db_query_time_ms) ON conversation_turns TO harena_user;"
    )

    # Verify that the grant has been applied
    bind = op.get_bind()
    check_sql = text(
        """
        SELECT 1 FROM information_schema.column_privileges
        WHERE table_name='conversation_turns'
          AND column_name='openai_cost'
          AND grantee='harena_user'
        """
    )
    if bind.execute(check_sql).scalar() is None:
        raise RuntimeError("Permissions not set for openai_cost on harena_user")


def downgrade() -> None:
    """Rollback monitoring columns."""
    op.drop_column("conversation_turns", "db_query_time_ms")
    op.drop_column("conversation_turns", "openai_cost")
