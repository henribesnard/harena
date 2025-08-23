import os
import tempfile
import subprocess

from sqlalchemy import create_engine, inspect, text


def test_alembic_upgrade_preserves_required_columns():
    """Run `alembic upgrade head` on a fresh DB and ensure critical columns exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db_url = f"sqlite:///{db_path}"

        # Environment required by the application's settings
        env = os.environ.copy()
        env["DATABASE_URL"] = db_url
        env["REDIS_URL"] = "redis://localhost:6379/0"

        # Pre-create minimal users table expected by migrations
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email VARCHAR NOT NULL,
                        password_hash VARCHAR NOT NULL
                    );
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE raw_transactions (
                        bridge_transaction_id INTEGER NOT NULL
                    );
                    """
                )
            )
        engine.dispose()

        repo_dir = os.path.join(os.path.dirname(__file__), "..", "..")

        # Apply migrations from the base revision through head
        subprocess.run(["alembic", "upgrade", "93f0d886307b"], cwd=repo_dir, env=env, check=True)
        subprocess.run(["alembic", "upgrade", "head"], cwd=repo_dir, env=env, check=True)

        # Inspect resulting schema for required columns
        engine = create_engine(db_url)
        inspector = inspect(engine)

        conv_cols = {col["name"] for col in inspector.get_columns("conversations")}
        turn_cols = {col["name"] for col in inspector.get_columns("conversation_turns")}

        for table_cols in (conv_cols, turn_cols):
            assert "entities_extracted" in table_cols
            assert "openai_usage_stats" in table_cols

        engine.dispose()

