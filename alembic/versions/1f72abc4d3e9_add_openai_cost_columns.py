"""add openai cost and query time columns

Revision ID: 1f72abc4d3e9
Revises: c2805ce5b2d9
Create Date: 2025-01-29 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

# revision identifiers, used by Alembic.
revision: str = "1f72abc4d3e9"
down_revision: Union[str, None] = "c2805ce5b2d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add monitoring columns and grant permissions if user exists."""
    
    # Étape 1 : Ajouter les colonnes de monitoring
    print("🔄 Ajout des colonnes de monitoring...")
    op.add_column("conversation_turns", sa.Column("openai_cost", sa.Float(), nullable=True))
    op.add_column("conversation_turns", sa.Column("db_query_time_ms", sa.Float(), nullable=True))
    print("✅ Colonnes ajoutées avec succès")

    # Étape 2 : Vérifier si l'utilisateur harena_user existe
    bind = op.get_bind()
    
    try:
        # Vérifier l'existence de l'utilisateur
        check_user_sql = text(
            """
            SELECT 1 FROM pg_roles WHERE rolname = 'harena_user'
            """
        )
        user_exists = bind.execute(check_user_sql).scalar()
        
        if user_exists:
            print("👤 Utilisateur harena_user détecté, application des permissions...")
            
            # Appliquer les permissions
            try:
                bind.execute(text(
                    "GRANT SELECT (openai_cost, db_query_time_ms) ON conversation_turns TO harena_user"
                ))
                
                # Vérifier que les permissions ont été appliquées
                check_permissions_sql = text(
                    """
                    SELECT 1 FROM information_schema.column_privileges
                    WHERE table_name='conversation_turns'
                      AND column_name='openai_cost'
                      AND grantee='harena_user'
                    """
                )
                
                if bind.execute(check_permissions_sql).scalar():
                    print("✅ Permissions accordées avec succès à harena_user")
                else:
                    print("⚠️  Permissions non vérifiables mais migration continue")
                    
            except ProgrammingError as e:
                print(f"⚠️  Impossible d'accorder les permissions: {e}")
                print("ℹ️  Migration continue sans permissions (OK pour dev local)")
        else:
            print("ℹ️  Utilisateur harena_user non trouvé (environnement local)")
            print("ℹ️  Permissions ignorées - OK pour développement local")
            
    except Exception as e:
        print(f"⚠️  Erreur lors de la vérification de l'utilisateur: {e}")
        print("ℹ️  Migration continue sans permissions")


def downgrade() -> None:
    """Rollback monitoring columns."""
    print("🔄 Suppression des colonnes de monitoring...")
    
    # Pas besoin de gérer les permissions au downgrade, 
    # elles disparaissent avec les colonnes
    op.drop_column("conversation_turns", "db_query_time_ms")
    op.drop_column("conversation_turns", "openai_cost")
    
    print("✅ Colonnes supprimées avec succès")