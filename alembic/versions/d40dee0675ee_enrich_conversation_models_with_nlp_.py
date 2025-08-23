"""enrich conversation models with nlp metadata

Revision ID: d40dee0675ee
Revises: 12d193f2a3ae
Create Date: 2025-08-22 13:02:16.023232

"""
from typing import Sequence, Union

# This revision originally added NLP-related columns that are already
# present in earlier migrations. To avoid dropping and re-adding these
# required fields, the migration has been replaced with a no-op.

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401


# revision identifiers, used by Alembic.
revision: str = "d40dee0675ee"
down_revision: Union[str, None] = "12d193f2a3ae"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op migration."""
    pass


def downgrade() -> None:
    """No-op downgrade."""
    pass

