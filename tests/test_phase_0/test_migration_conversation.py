import json
import os
import tempfile
from datetime import datetime, UTC
import pytest
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from alembic.runtime.migration import MigrationContext
from alembic.operations import Operations
import importlib.util
from pathlib import Path


def test_migration_conversation_preserves_data_and_defaults():
    # create temporary SQLite database file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        engine = create_engine(f"sqlite:///{db_path}")
        session = None
        try:
            # Pre-create minimal users table required by migrations
            with engine.begin() as conn:
                conn.execute(text(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email VARCHAR NOT NULL,
                        password_hash VARCHAR NOT NULL
                    );
                    """
                ))

            # Apply migration that introduces conversation tables
            migration_path = (
                Path(__file__).resolve().parents[2]
                / "alembic"
                / "versions"
                / "93f0d886307b_add_conversation_tables_with_proper_.py"
            )
            spec = importlib.util.spec_from_file_location("migration", migration_path)
            migration = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(migration)
            with engine.begin() as connection:
                connection.connection.create_function(
                    "now", 0, lambda: datetime.now(UTC).isoformat()
                )
                ctx = MigrationContext.configure(connection)
                migration.op = Operations(ctx)
                migration.upgrade()
            # Insert sample user and conversation data
            SessionLocal = sessionmaker(bind=engine)
            session = SessionLocal()
            session.execute(text("INSERT INTO users (email, password_hash) VALUES ('user@example.com', 'hash')"))

            from db_service.models.conversation import Conversation, ConversationTurn

            conv = Conversation(user_id=1)
            session.add(conv)
            session.commit()

            turn = ConversationTurn(
                conversation_id=conv.id,
                turn_number=1,
                user_message="bonjour",
                assistant_response="salut",
            )
            session.add(turn)
            session.commit()

            # Verify default values and JSON validity
            conv_db = session.get(Conversation, conv.id)
            assert conv_db.status == "active"
            assert conv_db.language == "fr"
            assert conv_db.conversation_metadata == {}
            assert conv_db.user_preferences == {}
            assert conv_db.session_metadata == {}
            json.dumps(conv_db.conversation_metadata)  # ensure serializable

            turn_db = session.get(ConversationTurn, turn.id)
            assert turn_db.agent_chain == []
            assert turn_db.turn_metadata == {}
            assert json.dumps(turn_db.turn_metadata) == "{}"

            # Ensure data is preserved after upgrade
            assert session.query(Conversation).count() == 1
            assert session.query(ConversationTurn).count() == 1

            # Downgrade and ensure schema is restored
            with engine.begin() as connection:
                ctx = MigrationContext.configure(connection)
                migration.op = Operations(ctx)
                migration.downgrade()
            inspector = inspect(engine)
            assert not inspector.has_table("conversations")
            assert not inspector.has_table("conversation_turns")
            assert not inspector.has_table("conversation_summaries")
        finally:
            if session is not None:
                session.close()
            engine.dispose()
