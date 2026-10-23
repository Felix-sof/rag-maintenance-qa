"""The AI4I table and the copilot's data queries, against a real PostgreSQL (TEST_DATABASE_URL)."""

import os
import uuid

import pytest

import process_data
from failure_rules import Process, load_dataset

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")


def _schema_connection():
    import db

    schema = f"test_{uuid.uuid4().hex[:12]}"
    return db.connect(DATABASE_URL, schema=schema), schema


def _drop(conn, schema):
    from psycopg import sql

    conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    conn.close()


@pytest.fixture(scope="module")
def conn():
    connection, schema = _schema_connection()
    assert process_data.load(connection) == 10_000  # once for the whole module: it's read-only from here
    yield connection
    _drop(connection, schema)


@pytest.fixture(scope="module")
def records():
    return load_dataset()


def test_loading_twice_loads_once(conn):
    assert process_data.load(conn) == 0
    assert conn.execute("SELECT count(*) FROM ai4i_processes").fetchone()[0] == 10_000


def test_generated_columns_follow_the_documents(conn):
    diff, power, strain = conn.execute(
        "SELECT temperature_difference_k, power_w, strain_min_nm FROM ai4i_processes WHERE udi = 1"
    ).fetchone()
    # UDI 1: 298.1 K / 308.6 K, 1551 rpm, 42.8 Nm, 0 min
    assert diff == pytest.approx(10.5, abs=0.01)
    assert power == pytest.approx(42.8 * 1551 * 2 * 3.14159265 / 60, rel=1e-4)
    assert strain == 0


def test_features_are_standardised(conn):
    for i in range(1, 6):
        mean, sd = conn.execute(
            f"SELECT avg((features::real[])[{i}]), stddev_pop((features::real[])[{i}]) FROM ai4i_processes"
        ).fetchone()
        assert mean == pytest.approx(0, abs=1e-3) and sd == pytest.approx(1, abs=1e-3)


def test_failure_rate_by_type_adds_up(conn, records):
    rows = process_data.run(conn, "failure_rate_by_type")
    assert sum(r["processes"] for r in rows) == 10_000 and sum(r["failures"] for r in rows) == 339
    low = next(r for r in rows if r["product_type"] == "L")
    expected = sum(r["machine_failure"] for r in records if r["product_type"] == "L") / 6000
    assert float(low["failure_rate"]) == pytest.approx(expected, abs=1e-4)
    assert [r["product_type"] for r in process_data.run(conn, "failure_rate_by_type", product_type="H")] == ["H"]


def test_tool_wear_bands_cover_every_process(conn):
    bands = process_data.run(conn, "failure_rate_by_tool_wear")
    assert sum(b["processes"] for b in bands) == 10_000
    assert all(b["twf"] == 0 for b in bands if b["wear_to_min"] <= 180)  # TWF only at high wear
    assert [b["wear_from_min"] for b in bands] == sorted(b["wear_from_min"] for b in bands)


def test_sensor_averages_by_mode_agree_with_the_rules(conn):
    by_mode = {r["mode"]: r for r in process_data.run(conn, "sensor_averages_by_mode")}
    assert by_mode["no failure"]["processes"] == 9_661
    assert float(by_mode["HDF"]["temperature_difference_k"]) < 8.6  # every HDF is below the threshold
    assert float(by_mode["OSF"]["strain_min_nm"]) > 11_000


def test_a_recorded_process_is_its_own_nearest_neighbour(conn, records):
    hdf = next(r for r in records if r["hdf"])
    found = process_data.similar_processes(conn, Process(**{k: hdf[k] for k in (
        "product_type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    )}), k=10)
    assert found["nearest"][0]["udi"] == hdf["udi"] and float(found["nearest"][0]["distance"]) == 0
    assert found["compared"] == 10 and found["failed"] >= 1 and found["failed_by_mode"]["HDF"] >= 1
    assert all(n["distance"] >= found["nearest"][0]["distance"] for n in found["nearest"])


def test_only_named_queries_with_valid_parameters_run(conn):
    with pytest.raises(ValueError):
        process_data.run(conn, "DROP TABLE ai4i_processes")
    with pytest.raises(ValueError):
        process_data.run(conn, "failure_rate_by_type", product_type="L' OR '1'='1")


def test_a_read_only_deployment_does_not_load_the_data_itself(monkeypatch):
    monkeypatch.setenv("READ_ONLY", "1")
    connection, schema = _schema_connection()
    try:
        assert process_data.run(connection, "failure_rate_by_type") == []
    finally:
        _drop(connection, schema)


def test_copilot_tools_read_the_dataset(conn, records, monkeypatch):
    import copilot_tools

    monkeypatch.setattr(copilot_tools, "_connection", lambda: conn)
    hdf = next(r for r in records if r["hdf"])
    assessed = copilot_tools.call("assess_process", {k: hdf[k] for k in copilot_tools.READING_FIELDS})
    assert assessed["triggered"] == ["HDF"]
    similar = assessed["similar_recorded_processes"]
    assert similar["compared"] == 25 and len(similar["nearest"]) == 3
    assert similar["nearest"][0]["udi"] == hdf["udi"]
    assert isinstance(similar["nearest"][0]["distance"], float)  # JSON-able, not Decimal

    queried = copilot_tools.call("query_dataset", {"query": "failure_rate_by_type", "product_type": "H"})
    assert [r["product_type"] for r in queried["rows"]] == ["H"]
    assert isinstance(queried["rows"][0]["failure_rate"], float)
