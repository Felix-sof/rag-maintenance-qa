"""
Plain-SQL migrations: apply db/migrations/NNN_*.sql in order, each exactly once.

    python -m db.migrate        # uses DATABASE_URL

Each file runs in the same transaction as the row that records it in
schema_migrations, so a migration that fails leaves no trace and is retried
next time. An advisory lock makes two processes starting at once (the API
and the UI) apply them one after the other instead of both at the same time.
"""

import os
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
LOCK_ID = 7_301_001  # any constant; only this code takes this advisory lock


def migrate(conn) -> list[str]:
    """Apply pending migrations on an open psycopg connection. Returns the versions applied."""
    applied_now = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.stem in done:
                continue
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            applied_now.append(path.stem)
    return applied_now


if __name__ == "__main__":
    import psycopg
    from dotenv import load_dotenv

    load_dotenv()
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        versions = migrate(connection)
    print(f"applied: {', '.join(versions)}" if versions else "up to date")
