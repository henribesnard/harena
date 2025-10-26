"""Add baseline_profile and spending_outliers to UserBudgetProfile

Revision ID: 85172b2f96a8
Revises: a5b2c4d8e9f0
Create Date: 2025-10-26 05:35:43.458493

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '85172b2f96a8'
down_revision: Union[str, None] = 'a5b2c4d8e9f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add baseline_profile and spending_outliers JSON columns to user_budget_profile table."""
    # Add baseline_profile column
    op.add_column('user_budget_profile',
        sa.Column('baseline_profile', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )

    # Add spending_outliers column
    op.add_column('user_budget_profile',
        sa.Column('spending_outliers', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    """Remove baseline_profile and spending_outliers columns from user_budget_profile table."""
    op.drop_column('user_budget_profile', 'spending_outliers')
    op.drop_column('user_budget_profile', 'baseline_profile')
