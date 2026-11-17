"""Acceso SQLite y aplicación ordenada de migraciones."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import Settings


MIGRATIONS_DIRECTORY = Path(__file__).resolve().parents[1] / "migrations"


def connect_database(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def apply_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied_versions = {
        row["version"]
        for row in connection.execute("SELECT version FROM schema_migrations")
    }
    for migration_path in sorted(MIGRATIONS_DIRECTORY.glob("*.sql")):
        version = migration_path.stem
        if version in applied_versions:
            continue
        migration_sql = migration_path.read_text(encoding="utf-8")
        with connection:
            connection.executescript(migration_sql)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
                (version,),
            )


def initialize_database(settings: Settings) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_database(settings.database_path)
    try:
        apply_migrations(connection)
    finally:
        connection.close()
