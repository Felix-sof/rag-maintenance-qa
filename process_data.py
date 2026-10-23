"""
The AI4I dataset in Postgres, and the fixed questions the copilot may ask of it.

    python -m process_data          # load ai4i/ai4i2020.csv into DATABASE_URL (skipped if already there)

The copilot never writes SQL. It picks one of the named queries in QUERIES
and supplies typed parameters; the SQL is fixed in code and the driver binds
the parameters, so nothing a user types changes what runs. An LLM composing
SQL behind a public endpoint would be one prompt injection away from reading
or changing data it shouldn't.

similar_processes() is a nearest-neighbour search with pgvector: the recorded
processes closest to the one asked about - measured in the quantities the
failure rules use, not the raw readings - and how many of them failed. It scans the ~6,000 rows of one product type exactly; at this
size that takes milliseconds, and exact beats an approximate index.
"""

import os
import sys

from failure_rules import DATASET, Process, load_dataset

LOAD_LOCK_ID = 7_301_003
COLUMNS = (
    "udi", "product_id", "product_type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min", "machine_failure", "twf", "hdf", "pwf", "osf", "rnf",
)
BOOLEAN_COLUMNS = {"machine_failure", "twf", "hdf", "pwf", "osf", "rnf"}

TYPE_FILTER = "(%(product_type)s::text IS NULL OR product_type = %(product_type)s)"

QUERIES = {
    "failure_rate_by_type": {
        "description": "Processes, failures and failure rate per product type (L, M, H), with failures by mode.",
        "sql": f"""
            SELECT
                product_type,
                count(*)                                          AS processes,
                count(*) FILTER (WHERE machine_failure)           AS failures,
                round(avg(machine_failure::int)::numeric, 4)      AS failure_rate,
                count(*) FILTER (WHERE twf) AS twf, count(*) FILTER (WHERE hdf) AS hdf,
                count(*) FILTER (WHERE pwf) AS pwf, count(*) FILTER (WHERE osf) AS osf,
                count(*) FILTER (WHERE rnf) AS rnf
            FROM ai4i_processes
            WHERE {TYPE_FILTER}
            GROUP BY product_type
            ORDER BY failure_rate DESC
        """,
    },
    "failure_rate_by_tool_wear": {
        "description": "Failure rate in 20-minute tool-wear bands, with the TWF and OSF failures in each.",
        "sql": f"""
            SELECT
                (band - 1) * 20                                   AS wear_from_min,
                band * 20                                         AS wear_to_min,
                count(*)                                          AS processes,
                count(*) FILTER (WHERE machine_failure)           AS failures,
                round(avg(machine_failure::int)::numeric, 4)      AS failure_rate,
                count(*) FILTER (WHERE twf)                       AS twf,
                count(*) FILTER (WHERE osf)                       AS osf
            FROM (
                SELECT *, width_bucket(tool_wear_min, 0, 260, 13) AS band
                FROM ai4i_processes
                WHERE {TYPE_FILTER}
            ) banded
            GROUP BY band
            ORDER BY band
        """,
    },
    "sensor_averages_by_mode": {
        "description": "Average readings of the processes that failed in each mode, next to those that didn't fail.",
        # LATERAL (VALUES ...) unpivots the five mode columns into rows, so one
        # GROUP BY covers every mode; a process failing in two modes counts in both
        "sql": f"""
            SELECT
                m.mode,
                count(*)                                          AS processes,
                round(avg(p.temperature_difference_k)::numeric, 2) AS temperature_difference_k,
                round(avg(p.rotational_speed_rpm)::numeric)        AS rotational_speed_rpm,
                round(avg(p.torque_nm)::numeric, 1)                AS torque_nm,
                round(avg(p.tool_wear_min)::numeric)               AS tool_wear_min,
                round(avg(p.power_w)::numeric)                     AS power_w,
                round(avg(p.strain_min_nm)::numeric)               AS strain_min_nm
            FROM ai4i_processes p
            CROSS JOIN LATERAL (VALUES
                ('TWF', p.twf), ('HDF', p.hdf), ('PWF', p.pwf), ('OSF', p.osf), ('RNF', p.rnf),
                ('no failure', NOT p.machine_failure)
            ) AS m(mode, applies)
            WHERE m.applies AND {TYPE_FILTER}
            GROUP BY m.mode
            ORDER BY processes
        """,
    },
}

SIMILAR_SQL = """
    WITH readings AS (
        SELECT
            %(process)s - %(air)s                    AS diff,
            %(rpm)s::double precision                AS rpm,
            %(torque)s * %(rpm)s * 2 * pi() / 60     AS power,
            %(wear)s * %(torque)s                    AS strain,
            %(wear)s::double precision               AS wear
    ),
    query AS (
        SELECT ARRAY[
            (r.diff - s.diff_mean) / s.diff_sd,
            (r.rpm - s.rpm_mean) / s.rpm_sd,
            (r.power - s.power_mean) / s.power_sd,
            (r.strain - s.strain_mean) / s.strain_sd,
            (r.wear - s.wear_mean) / s.wear_sd
        ]::vector AS features
        FROM readings r, ai4i_scaling s
    )
    SELECT
        p.udi, p.product_id, p.air_temperature_k, p.process_temperature_k, p.rotational_speed_rpm,
        p.torque_nm, p.tool_wear_min, p.machine_failure, p.twf, p.hdf, p.pwf, p.osf, p.rnf,
        round((p.features <-> q.features)::numeric, 3) AS distance
    FROM ai4i_processes p, query q
    WHERE p.product_type = %(product_type)s
    ORDER BY p.features <-> q.features
    LIMIT %(k)s
"""


def load(conn, path=DATASET) -> int:
    """Load the CSV into ai4i_processes, unless it is already there. Returns the rows loaded."""
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOAD_LOCK_ID,))
        if conn.execute("SELECT count(*) FROM ai4i_processes").fetchone()[0] > 0:
            return 0
        records = load_dataset(path)
        with conn.cursor() as cur, cur.copy(f"COPY ai4i_processes ({', '.join(COLUMNS)}) FROM STDIN") as copy:
            for record in records:
                copy.write_row([bool(record[c]) if c in BOOLEAN_COLUMNS else record[c] for c in COLUMNS])
        conn.execute(
            """
            UPDATE ai4i_processes SET features = ARRAY[
                (temperature_difference_k - s.diff_mean) / s.diff_sd,
                (rotational_speed_rpm - s.rpm_mean) / s.rpm_sd,
                (power_w - s.power_mean) / s.power_sd,
                (strain_min_nm - s.strain_mean) / s.strain_sd,
                (tool_wear_min - s.wear_mean) / s.wear_sd
            ]::vector
            FROM ai4i_scaling s
            """
        )
    return len(records)


def ensure_loaded(conn) -> None:
    """Load on first use - except in a READ_ONLY deployment, where the deploy pipeline loads it."""
    if os.environ.get("READ_ONLY", "0") != "1":
        load(conn)


def run(conn, name: str, product_type: str = None) -> list[dict]:
    from psycopg.rows import dict_row

    if name not in QUERIES:
        raise ValueError(f"unknown query {name!r}; expected one of {sorted(QUERIES)}")
    if product_type is not None and product_type not in ("L", "M", "H"):
        raise ValueError(f"product type must be L, M or H, not {product_type!r}")
    ensure_loaded(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(QUERIES[name]["sql"], {"product_type": product_type}).fetchall()


def similar_processes(conn, process: Process, k: int = 25) -> dict:
    """The k recorded processes of the same product type nearest to `process`, and how they ended."""
    from psycopg.rows import dict_row

    ensure_loaded(conn)
    params = {
        "product_type": process.product_type, "k": max(1, min(int(k), 100)),
        "air": process.air_temperature_k, "process": process.process_temperature_k,
        "rpm": process.rotational_speed_rpm, "torque": process.torque_nm, "wear": process.tool_wear_min,
    }
    with conn.cursor(row_factory=dict_row) as cur:
        neighbours = cur.execute(SIMILAR_SQL, params).fetchall()
    failed = [n for n in neighbours if n["machine_failure"]]
    return {
        "product_type": process.product_type,
        "compared": len(neighbours),
        "failed": len(failed),
        "failure_rate": round(len(failed) / len(neighbours), 3) if neighbours else None,
        "failed_by_mode": {
            mode.upper(): sum(1 for n in failed if n[mode]) for mode in ("twf", "hdf", "pwf", "osf", "rnf")
        },
        "nearest": neighbours[:5],
    }


if __name__ == "__main__":
    import db
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("DATABASE_URL"):
        sys.exit("Set DATABASE_URL.")
    with db.connect(os.environ["DATABASE_URL"]) as connection:
        loaded = load(connection)
    print(f"loaded {loaded} processes" if loaded else "already loaded")
