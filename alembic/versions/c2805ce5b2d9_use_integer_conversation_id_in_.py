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
    """Upgrade schema: Change conversation_id from String to Integer."""
    
    # Étape 1: Ajouter une colonne temporaire pour le nouvel ID integer
    with op.batch_alter_table('conversation_messages') as batch_op:
        batch_op.add_column(sa.Column('conversation_id_temp', sa.Integer(), nullable=True))
    
    # Étape 2: Remplir la colonne temporaire avec les IDs integer correspondants
    # Joindre avec la table conversations pour obtenir l'ID integer
    op.execute("""
        UPDATE conversation_messages 
        SET conversation_id_temp = c.id 
        FROM conversations c 
        WHERE conversation_messages.conversation_id = c.conversation_id
    """)
    
    # Étape 3: Gérer les orphelins (messages sans conversation correspondante)
    # Supprimer les messages orphelins
    op.execute("""
        DELETE FROM conversation_messages 
        WHERE conversation_id_temp IS NULL
    """)
    
    # Étape 4: Supprimer l'ancienne colonne et ses contraintes
    with op.batch_alter_table('conversation_messages') as batch_op:
        # Supprimer l'index d'abord
        try:
            batch_op.drop_index('ix_conversation_messages_conversation_id')
        except Exception:
            # L'index n'existe peut-être pas
            pass
        
        # Supprimer la contrainte de clé étrangère
        try:
            batch_op.drop_constraint(
                'conversation_messages_conversation_id_fkey', 
                type_='foreignkey'
            )
        except Exception:
            # La contrainte n'existe peut-être pas
            pass
        
        # Supprimer l'ancienne colonne
        batch_op.drop_column('conversation_id')
    
    # Étape 5: Renommer la colonne temporaire et appliquer les contraintes
    with op.batch_alter_table('conversation_messages') as batch_op:
        # Renommer la colonne temporaire
        batch_op.alter_column(
            'conversation_id_temp', 
            new_column_name='conversation_id',
            nullable=False
        )
        
        # Créer la nouvelle contrainte de clé étrangère
        batch_op.create_foreign_key(
            'conversation_messages_conversation_id_fkey',
            'conversations',
            ['conversation_id'],
            ['id'],
            ondelete='CASCADE'
        )
        
        # Créer le nouvel index
        batch_op.create_index(
            'ix_conversation_messages_conversation_id', 
            ['conversation_id']
        )


def downgrade() -> None:
    """Downgrade schema: Change conversation_id from Integer back to String."""
    
    # Étape 1: Ajouter une colonne temporaire pour l'ancien ID string
    with op.batch_alter_table('conversation_messages') as batch_op:
        batch_op.add_column(sa.Column('conversation_id_temp', sa.String(length=255), nullable=True))
    
    # Étape 2: Remplir la colonne temporaire avec les conversation_id string
    op.execute("""
        UPDATE conversation_messages 
        SET conversation_id_temp = c.conversation_id 
        FROM conversations c 
        WHERE conversation_messages.conversation_id = c.id
    """)
    
    # Étape 3: Gérer les orphelins (supprimer les messages sans conversation)
    op.execute("""
        DELETE FROM conversation_messages 
        WHERE conversation_id_temp IS NULL
    """)
    
    # Étape 4: Supprimer l'ancienne colonne et ses contraintes
    with op.batch_alter_table('conversation_messages') as batch_op:
        # Supprimer l'index d'abord
        try:
            batch_op.drop_index('ix_conversation_messages_conversation_id')
        except Exception:
            pass
        
        # Supprimer la contrainte de clé étrangère
        try:
            batch_op.drop_constraint(
                'conversation_messages_conversation_id_fkey', 
                type_='foreignkey'
            )
        except Exception:
            pass
        
        # Supprimer l'ancienne colonne
        batch_op.drop_column('conversation_id')
    
    # Étape 5: Renommer la colonne temporaire et appliquer les contraintes
    with op.batch_alter_table('conversation_messages') as batch_op:
        # Renommer la colonne temporaire
        batch_op.alter_column(
            'conversation_id_temp', 
            new_column_name='conversation_id',
            nullable=False
        )
        
        # Créer la contrainte de clé étrangère vers conversation_id string
        batch_op.create_foreign_key(
            'conversation_messages_conversation_id_fkey',
            'conversations',
            ['conversation_id'],
            ['conversation_id'],
            ondelete='CASCADE'
        )
        
        # Créer l'index
        batch_op.create_index(
            'ix_conversation_messages_conversation_id', 
            ['conversation_id']
        )