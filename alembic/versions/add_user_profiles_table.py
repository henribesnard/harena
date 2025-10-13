"""add user_profiles table

Revision ID: f9a8b1234567
Revises: a58a4daacbb8
Create Date: 2025-01-12 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'f9a8b1234567'
down_revision = 'a58a4daacbb8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create user_profiles table"""
    op.create_table(
        'user_profiles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),

        # Preferences
        sa.Column('preferred_categories', sa.JSON(), nullable=True),
        sa.Column('preferred_merchants', sa.JSON(), nullable=True),
        sa.Column('notification_preference', sa.String(length=50), nullable=True, server_default='important_only'),
        sa.Column('email_notifications', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('push_notifications', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('currency', sa.String(length=10), nullable=True, server_default='EUR'),
        sa.Column('date_format', sa.String(length=50), nullable=True, server_default='%Y-%m-%d'),
        sa.Column('language', sa.String(length=10), nullable=True, server_default='fr'),
        sa.Column('default_period', sa.String(length=20), nullable=True, server_default='month'),
        sa.Column('show_trends', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('show_insights', sa.Boolean(), nullable=True, server_default='true'),

        # Habits
        sa.Column('frequent_query_patterns', sa.JSON(), nullable=True),
        sa.Column('query_frequency', sa.JSON(), nullable=True),
        sa.Column('average_spending_by_category', sa.JSON(), nullable=True),
        sa.Column('peak_spending_days', sa.JSON(), nullable=True),
        sa.Column('peak_spending_months', sa.JSON(), nullable=True),
        sa.Column('preferred_visualization_types', sa.JSON(), nullable=True),
        sa.Column('average_session_duration_minutes', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('queries_per_session', sa.Float(), nullable=True, server_default='0.0'),

        # Interaction history
        sa.Column('accepted_recommendations', sa.JSON(), nullable=True),
        sa.Column('dismissed_recommendations', sa.JSON(), nullable=True),
        sa.Column('created_alerts', sa.JSON(), nullable=True),
        sa.Column('triggered_alerts_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('positive_feedback_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('negative_feedback_count', sa.Integer(), nullable=True, server_default='0'),

        # Metadata
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_updated', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('profile_completeness', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('total_queries', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_sessions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('last_active', sa.DateTime(), nullable=True),

        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', name='uq_user_profiles_user_id')
    )

    # Create indexes for better performance
    op.create_index('idx_user_profiles_user_id', 'user_profiles', ['user_id'])
    op.create_index('idx_user_profiles_last_active', 'user_profiles', ['last_active'])
    op.create_index('idx_user_profiles_completeness', 'user_profiles', ['profile_completeness'])


def downgrade() -> None:
    """Drop user_profiles table"""
    op.drop_index('idx_user_profiles_completeness', table_name='user_profiles')
    op.drop_index('idx_user_profiles_last_active', table_name='user_profiles')
    op.drop_index('idx_user_profiles_user_id', table_name='user_profiles')
    op.drop_table('user_profiles')
