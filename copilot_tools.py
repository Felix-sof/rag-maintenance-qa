"""
The tools the copilot can call: the failure rules and the AI4I dataset.

    assess_process   failure_rules.assess() on the readings the user gives, plus how
                     the most similar recorded processes ended (process_data, pgvector)
    query_dataset    one of the fixed statistics queries in process_data.QUERIES

rag.answer(..., tools=copilot_tools) hands TOOLS to the model and runs its
calls through call(). Every tool checks its arguments and returns a JSON-able
dict. A bad call returns {"error": ...} for the model to explain, never an
exception: a confused tool call must not cost the user the whole answer.

The dataset parts need DATABASE_URL. Without it, assess_process still applies
the rules, and query_dataset reports that the data isn't available.
"""

import os
import threading
from decimal import Decimal

import process_data
from failure_rules import DATA_RANGE, Process, assess

READING_FIELDS = ("product_type", *DATA_RANGE)

TOOLS = [
    {
        "name": "assess_process",
        "description": (
            "Assess one machining process of the milling machine. It applies the documented failure rules "
            "(HDF, PWF and OSF exactly, with the margin to each threshold; TWF and RNF as rates) and compares "
            "the process with the most similar recorded processes in the AI4I dataset. Call it whenever the "
            "user gives sensor readings or asks about the failure risk of specific values. Temperatures are in "
            "kelvin: add 273.15 to degrees Celsius. All six values are required; if the user didn't give one, "
            "don't call the tool - ask for the missing values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_type": {"type": "string", "enum": ["L", "M", "H"], "description": "Product quality type"},
                "air_temperature_k": {"type": "number", "description": "Air temperature, kelvin"},
                "process_temperature_k": {"type": "number", "description": "Process temperature, kelvin"},
                "rotational_speed_rpm": {"type": "number", "description": "Rotational speed, rpm"},
                "torque_nm": {"type": "number", "description": "Torque, newton metres"},
                "tool_wear_min": {"type": "number", "description": "Tool wear, minutes"},
            },
            "required": list(READING_FIELDS),
        },
    },
    {
        "name": "query_dataset",
        "description": (
            "Run one fixed statistics query over the AI4I 2020 dataset: 10,000 recorded (synthetic) machining "
            "processes and their failures. Use it for questions about how often failures happen, which product "
            "type or tool-wear level fails most, or what the readings of failed processes look like. "
            + " ".join(f"'{name}': {q['description']}" for name, q in process_data.QUERIES.items())
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "enum": sorted(process_data.QUERIES)},
                "product_type": {"type": "string", "enum": ["L", "M", "H"], "description": "Only this product type"},
            },
            "required": ["query"],
        },
    },
]

_conn = None
_lock = threading.Lock()  # one query at a time on the shared connection


def _connection():
    global _conn
    if not os.environ.get("DATABASE_URL"):
        return None
    if _conn is None or _conn.closed:
        import db

        _conn = db.connect(os.environ["DATABASE_URL"])
    return _conn


def _plain(value):
    """Database values made JSON-able: Decimal -> float, recursively."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _number(args: dict, key: str) -> float:
    value = args.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} is missing or not a number: {value!r}") from None


def assess_process(args: dict) -> dict:
    process = Process(str(args.get("product_type", "")), *(_number(args, key) for key in DATA_RANGE))
    result = assess(process)
    with _lock:
        conn = _connection()
        if conn is not None:
            similar = process_data.similar_processes(conn, process, k=25)
            similar["nearest"] = similar["nearest"][:3]
            result["similar_recorded_processes"] = similar
    return _plain(result)


def query_dataset(args: dict) -> dict:
    with _lock:
        conn = _connection()
        if conn is None:
            return {"error": "the dataset isn't available here (no database configured)"}
        rows = process_data.run(conn, str(args.get("query", "")), product_type=args.get("product_type"))
    return _plain({"query": args.get("query"), "rows": rows, "note": "AI4I 2020 is a synthetic dataset"})


HANDLERS = {"assess_process": assess_process, "query_dataset": query_dataset}


def call(name: str, args: dict) -> dict:
    """Run one tool call from the model. Never raises."""
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return handler(args)
    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - e.g. the database is unreachable: report it, don't crash the answer
        return {"error": f"the tool failed: {type(exc).__name__}"}
