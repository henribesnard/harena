"""add_pre_computed_metrics_table

Revision ID: 26c811d62c1a
Revises: 516284f8a0f1
Create Date: 2025-10-12 15:02:45.134907

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '26c811d62c1a'
down_revision: Union[str, None] = '516284f8a0f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create pre_computed_metrics table for batch-calculated metrics"""
    op.create_table(
        'pre_computed_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('period', sa.String(length=20), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),

        # Metric values (JSON for flexibility)
        sa.Column('metric_value', sa.JSON(), nullable=False),

        # Metadata for monitoring
        sa.Column('computation_time_ms', sa.Integer(), nullable=True),
        sa.Column('data_points_count', sa.Integer(), nullable=True),
        sa.Column('cache_hit', sa.Boolean(), nullable=True, server_default='false'),

        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for fast metric retrieval
    op.create_index('idx_pre_computed_metrics_user_id', 'pre_computed_metrics', ['user_id'])
    op.create_index('idx_pre_computed_metrics_metric_type', 'pre_computed_metrics', ['metric_type'])
    op.create_index('idx_pre_computed_metrics_period', 'pre_computed_metrics', ['period'])
    op.create_index('idx_pre_computed_metrics_expires_at', 'pre_computed_metrics', ['expires_at'])

    # Composite index for common query pattern (user_id + metric_type + period)
    op.create_index(
        'idx_pre_computed_metrics_lookup',
        'pre_computed_metrics',
        ['user_id', 'metric_type', 'period'],
        unique=False
    )


def downgrade() -> None:
    """Drop pre_computed_metrics table"""
    op.drop_index('idx_pre_computed_metrics_lookup', table_name='pre_computed_metrics')
    op.drop_index('idx_pre_computed_metrics_expires_at', table_name='pre_computed_metrics')
    op.drop_index('idx_pre_computed_metrics_period', table_name='pre_computed_metrics')
    op.drop_index('idx_pre_computed_metrics_metric_type', table_name='pre_computed_metrics')
    op.drop_index('idx_pre_computed_metrics_user_id', table_name='pre_computed_metrics')
    op.drop_table('pre_computed_metrics')
