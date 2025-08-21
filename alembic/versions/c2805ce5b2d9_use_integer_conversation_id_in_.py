"""use integer conversation_id in conversation_messages

Revision ID: c2805ce5b2d9
Revises: 6eb09f813ccf
Create Date: 2025-08-21 21:00:10.618671

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2805ce5b2d9'
down_revision: Union[str, None] = '6eb09f813ccf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('conversation_messages') as batch_op:
        batch_op.drop_index('ix_conversation_messages_conversation_id')
        batch_op.drop_constraint(
            'conversation_messages_conversation_id_fkey', type_='foreignkey'
        )
        batch_op.drop_column('conversation_id')
        batch_op.add_column(sa.Column('conversation_id', sa.Integer(), nullable=False))
        batch_op.create_foreign_key(
            'conversation_messages_conversation_id_fkey',
            'conversations',
            ['conversation_id'],
            ['id'],
            ondelete='CASCADE',
        )
        batch_op.create_index(
            'ix_conversation_messages_conversation_id', ['conversation_id']
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('conversation_messages') as batch_op:
        batch_op.drop_index('ix_conversation_messages_conversation_id')
        batch_op.drop_constraint(
            'conversation_messages_conversation_id_fkey', type_='foreignkey'
        )
        batch_op.drop_column('conversation_id')
        batch_op.add_column(
            sa.Column('conversation_id', sa.String(length=255), nullable=False)
        )
        batch_op.create_foreign_key(
            'conversation_messages_conversation_id_fkey',
            'conversations',
            ['conversation_id'],
            ['conversation_id'],
            ondelete='CASCADE',
        )
        batch_op.create_index(
            'ix_conversation_messages_conversation_id', ['conversation_id']
        )
