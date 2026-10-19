"""
A local PostgreSQL + pgvector for development, without Docker or an installer.

    pip install pgserver
    python -m db.local_postgres        # prints the DATABASE_URL to put in .env

pgserver ships Postgres 16 binaries with pgvector. The cluster lives in
data/postgres/ (git-ignored) and keeps running after this script exits; run it
again after a reboot (the port may change, so copy the URL again).

The cluster is initialised with the C locale: initdb crashes on a Windows set
to a Turkish locale. Nothing depends on the database's locale - Turkish
lowercasing is done in Python before text reaches Postgres (see pg_store.py).
"""

from pathlib import Path

PGDATA = Path(__file__).resolve().parent.parent / "data" / "postgres"


def start() -> str:
    import pgserver
    from pgserver._commands import initdb

    if not (PGDATA / "PG_VERSION").exists():
        PGDATA.mkdir(parents=True, exist_ok=True)
        initdb(["--auth=trust", "--encoding=UTF8", "--locale=C", "-U", "postgres"], pgdata=PGDATA)
    server = pgserver.get_server(PGDATA, cleanup_mode=None)  # None: keep running after exit
    return server.get_uri()


if __name__ == "__main__":
    print(f"DATABASE_URL={start()}")
