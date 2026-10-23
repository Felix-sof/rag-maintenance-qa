"""
The analytics queries over the query log: one NN_name.sql file each, in this
folder, with a comment on what it measures. Plain SQL, so each one also runs
as-is in psql or the Neon console.

    python -m analytics                 # run them all against DATABASE_URL and print the results
    python -m analytics.replay          # put the eval questions through the pipeline to have data
"""

from pathlib import Path

QUERY_DIR = Path(__file__).parent


def queries() -> dict[str, str]:
    """name -> SQL, in file order. The name drops the number: 01_daily_volume.sql -> 'daily_volume'."""
    return {
        path.stem.split("_", 1)[1]: path.read_text(encoding="utf-8")
        for path in sorted(QUERY_DIR.glob("[0-9][0-9]_*.sql"))
    }


def run(query_log, name: str) -> list[dict]:
    """Rows of one query, as dicts, read through a query_log.QueryLog's connection."""
    return query_log.fetch(queries()[name])
