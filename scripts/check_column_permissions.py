"""Script to verify privileges on monitoring columns."""
from __future__ import annotations

from sqlalchemy import create_engine, text
from config_service.config import settings


def check_permissions() -> None:
    database_url = settings.DATABASE_URL or "sqlite://"
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(database_url)
    query = text(
        """
        SELECT grantee, privilege_type, column_name
        FROM information_schema.column_privileges
        WHERE table_name='conversation_turns'
          AND column_name IN ('openai_cost', 'db_query_time_ms')
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    for row in rows:
        print(f"{row.column_name}: {row.grantee} -> {row.privilege_type}")


if __name__ == "__main__":  # pragma: no cover
    check_permissions()
