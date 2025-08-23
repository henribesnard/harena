"""cleanup conversation columns

Revision ID: 69a9143f47a3
Revises: 48f2e3ebd17c
Create Date: 2025-08-23 19:35:37.333759

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '69a9143f47a3'
down_revision: Union[str, None] = '48f2e3ebd17c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('conversation_turns', schema=None) as batch_op:
        batch_op.drop_column('financial_context')
        batch_op.drop_column('user_preferences_ai')
        batch_op.drop_column('key_entities_history')

    with op.batch_alter_table('conversations', schema=None) as batch_op:
        batch_op.drop_column('intents')
        batch_op.drop_column('entities')
        batch_op.drop_column('prompt_tokens')
        batch_op.drop_column('completion_tokens')
        batch_op.drop_column('total_tokens')
        batch_op.drop_column('intent_classification')
        batch_op.drop_column('entities_extracted')
        batch_op.drop_column('intent_confidence')
        batch_op.drop_column('total_tokens_used')
        batch_op.drop_column('openai_usage_stats')
        batch_op.drop_column('openai_cost_usd')


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('conversations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('openai_cost_usd', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('openai_usage_stats', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('total_tokens_used', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('intent_confidence', sa.Numeric(5, 4), nullable=True))
        batch_op.add_column(sa.Column('entities_extracted', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('intent_classification', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('total_tokens', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('completion_tokens', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('prompt_tokens', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('entities', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('intents', sa.JSON(), nullable=True))

    with op.batch_alter_table('conversation_turns', schema=None) as batch_op:
        batch_op.add_column(sa.Column('key_entities_history', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('user_preferences_ai', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('financial_context', sa.JSON(), nullable=True))
