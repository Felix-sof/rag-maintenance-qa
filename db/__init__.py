"""PostgreSQL access shared by the pgvector store (pg_store.py) and the query log (query_log.py)."""


def connect(database_url: str, schema: str = None):
    """An autocommit connection with the schema migrated to the latest version.

    `schema` keeps every table in its own Postgres schema (the tests use one
    per test); the vector extension always goes to public, where every schema
    can see it.
    """
    import psycopg
    from psycopg import sql

    from db.migrate import migrate

    # connect_timeout: the query log writes during a request, so an unreachable
    # database must fail fast instead of holding the answer up
    conn = psycopg.connect(database_url, autocommit=True, connect_timeout=10)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # before search_path changes: lands in public
    if schema:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    migrate(conn)
    return conn
