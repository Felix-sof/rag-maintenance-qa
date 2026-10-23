"""python -m analytics: run every analytics query against DATABASE_URL and print the results."""

import sys

import query_log
from analytics import queries, run


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    columns = list(rows[0])
    cells = [[("" if row[c] is None else str(row[c])) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths)), "  ".join("-" * w for w in widths)]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in cells]
    return "\n".join(lines)


if __name__ == "__main__":
    if not query_log.enabled():
        sys.exit("Set DATABASE_URL (and leave QUERY_LOG on) to read the query log.")
    log = query_log.get()
    for name in queries():
        print(f"\n## {name}\n")
        print(format_table(run(log, name)))
