"""
A local PostgreSQL + pgvector for development, without Docker or an installer.

    pip install pgserver
    python -m db.local_postgres                # prints the DATABASE_URL to put in .env
    python -m db.local_postgres D:/pgdata      # keep the database files somewhere else

pgserver ships Postgres 16 binaries with pgvector. The database files live in
~/.rag-maintenance-qa/postgres by default - outside the repository, and outside
folders that OneDrive or Dropbox sync, which don't mix with a live database.
The server keeps running after this script exits; run it again after a reboot
(the port may change, so copy the URL again).

The cluster is initialised with the C locale: initdb crashes on a Windows set
to a Turkish locale. Nothing depends on the database's locale - Turkish
lowercasing is done in Python before text reaches Postgres (see pg_store.py).
"""

import sys
from pathlib import Path

PGDATA = Path.home() / ".rag-maintenance-qa" / "postgres"


def start(pgdata: Path = PGDATA) -> str:
    import pgserver
    from pgserver._commands import initdb

    if not (pgdata / "PG_VERSION").exists():
        pgdata.mkdir(parents=True, exist_ok=True)
        initdb(["--auth=trust", "--encoding=UTF8", "--locale=C", "-U", "postgres"], pgdata=pgdata)
    server = pgserver.get_server(pgdata, cleanup_mode=None)  # None: keep running after exit
    return server.get_uri()


if __name__ == "__main__":
    print(f"DATABASE_URL={start(Path(sys.argv[1]) if len(sys.argv) > 1 else PGDATA)}")
