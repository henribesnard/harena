"""Utility to create a timestamped database backup before running migrations."""
from __future__ import annotations

import datetime
import logging
import pathlib
import subprocess
from typing import Optional

from config_service.config import settings

logger = logging.getLogger("db_backup")


def backup_database(db_url: Optional[str] = None) -> pathlib.Path | None:
    """Create a compressed backup of the database using ``pg_dump``.

    Parameters
    ----------
    db_url:
        Database URL. If ``None`` uses ``settings.DATABASE_URL``.

    Returns
    -------
    pathlib.Path | None
        Path to the created backup file or ``None`` if backup failed.
    """
    url = db_url or settings.DATABASE_URL
    if not url:
        logger.error("No database URL provided; skipping backup")
        return None

    # Ensure URL is compatible with ``pg_dump``
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backups_dir = pathlib.Path("backups")
    backups_dir.mkdir(exist_ok=True)
    backup_path = backups_dir / f"backup_{timestamp}.sql.gz"

    cmd = f"pg_dump {url} | gzip > {backup_path}"
    logger.info("Creating database backup at %s", backup_path)
    try:
        subprocess.run(["bash", "-c", cmd], check=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Database backup failed: %s", exc)
        return None

    return backup_path


if __name__ == "__main__":  # pragma: no cover - manual execution
    backup_database()
