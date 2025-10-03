"""add_metrics_tables

Revision ID: 972cc48fabfc
Revises: 2014325daf3c
Create Date: 2025-10-03 13:13:05.738952

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '972cc48fabfc'
down_revision: Union[str, None] = '2014325daf3c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create metrics_cache table
    op.create_table(
        'metrics_cache',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('metric_key', sa.String(length=255), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_cache_lookup', 'metrics_cache', ['user_id', 'metric_type', 'metric_key'])
    op.create_index(op.f('ix_metrics_cache_user_id'), 'metrics_cache', ['user_id'])

    # Create metrics_history table
    op.create_table(
        'metrics_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_history_lookup', 'metrics_history', ['user_id', 'metric_type', 'period_start'])
    op.create_index(op.f('ix_metrics_history_user_id'), 'metrics_history', ['user_id'])

    # Create user_metrics_config table
    op.create_table(
        'user_metrics_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('savings_rate_target', sa.Float(), nullable=True, server_default='20.0'),
        sa.Column('expense_ratio_essentials', sa.Float(), nullable=True, server_default='50.0'),
        sa.Column('expense_ratio_lifestyle', sa.Float(), nullable=True, server_default='30.0'),
        sa.Column('expense_ratio_savings', sa.Float(), nullable=True, server_default='20.0'),
        sa.Column('enable_low_balance_alert', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('low_balance_threshold', sa.Float(), nullable=True, server_default='100.0'),
        sa.Column('enable_burn_rate_alert', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('burn_rate_runway_threshold', sa.Integer(), nullable=True, server_default='30'),
        sa.Column('forecast_default_periods', sa.Integer(), nullable=True, server_default='90'),
        sa.Column('recurring_detection_min_occurrences', sa.Integer(), nullable=True, server_default='3'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('user_metrics_config')
    op.drop_index('idx_history_lookup', table_name='metrics_history')
    op.drop_index(op.f('ix_metrics_history_user_id'), table_name='metrics_history')
    op.drop_table('metrics_history')
    op.drop_index('idx_cache_lookup', table_name='metrics_cache')
    op.drop_index(op.f('ix_metrics_cache_user_id'), table_name='metrics_cache')
    op.drop_table('metrics_cache')
