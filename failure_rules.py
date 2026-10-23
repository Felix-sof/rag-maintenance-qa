"""
The AI4I 2020 failure rules (knowledge/failure_modes.md) as code: assess one process.

The knowledge base states exact rules for three failure modes, and the
dataset was generated from them. tests/test_failure_rules.py checks that on
all 10,000 rows of ai4i/ai4i2020.csv the rules reproduce every HDF, PWF
and OSF label, with no false alarm. The two other modes are random by
construction: TWF strikes at a random tool wear in a window, and RNF at a
0.1% chance per process. For TWF this module gives the rate measured in the
dataset at that wear level instead of a yes or no.

    python failure_rules.py L 300 308.6 1300 50 215

Every number in an assessment is either an input, derived from the inputs, or
a constant from the documents, so the copilot's answer can be checked
against it (grounding.py).
"""

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DATASET = Path(__file__).parent / "ai4i" / "ai4i2020.csv"

DOC = "failure_modes.md > Arıza Modları ve Oluşma Koşulları"
HDF_MAX_DIFF_K = 8.6
HDF_MAX_RPM = 1380
PWF_MIN_W, PWF_MAX_W = 3500, 9000
OSF_LIMITS = {"L": 11_000, "M": 12_000, "H": 13_000}
TWF_WINDOW_MIN = (200, 240)  # tool wear, minutes
RNF_PROBABILITY = 0.001

# Share of processes that ended in TWF, by tool wear, in ai4i/ai4i2020.csv.
# Checked against the CSV by tests/test_failure_rules.py.
TWF_RATE_BY_WEAR = [  # (wear up to, in minutes, rate)
    (180, 0.0),
    (200, 0.0022),
    (210, 0.0442),
    (220, 0.0489),
    (230, 0.0909),
    (240, 0.1613),
    (math.inf, 0.1818),
]

# The range each input spans in the dataset; outside it, the assessment extrapolates.
DATA_RANGE = {
    "air_temperature_k": (295.3, 304.5),
    "process_temperature_k": (305.7, 313.8),
    "rotational_speed_rpm": (1168, 2886),
    "torque_nm": (3.8, 76.6),
    "tool_wear_min": (0, 253),
}


@dataclass
class Process:
    product_type: str  # "L", "M" or "H"
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    def __post_init__(self):
        self.product_type = str(self.product_type).strip().upper()
        if self.product_type not in OSF_LIMITS:
            raise ValueError(f"product type must be L, M or H, not {self.product_type!r}")
        for name in DATA_RANGE:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or math.isnan(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, not {value!r}")

    @property
    def temperature_difference_k(self) -> float:
        return self.process_temperature_k - self.air_temperature_k

    @property
    def power_w(self) -> float:
        """Torque times angular speed: rpm * 2π / 60 is rad/s."""
        return self.torque_nm * self.rotational_speed_rpm * 2 * math.pi / 60

    @property
    def strain_min_nm(self) -> float:
        return self.tool_wear_min * self.torque_nm


def _condition(text: str, value: float, limit: float, met: bool) -> dict:
    return {"condition": text, "value": round(value, 1), "limit": limit, "met": met, "margin": round(abs(value - limit), 1)}


def load_dataset(path: Path = DATASET) -> list[dict]:
    """The 10,000 AI4I records, with plain column names and numbers parsed."""
    names = {
        "UDI": "udi", "Product ID": "product_id", "Type": "product_type",
        "Air temperature [K]": "air_temperature_k", "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm", "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min", "Machine failure": "machine_failure",
        "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf",
    }
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            row = {names[k]: v for k, v in raw.items()}
            for key in ("udi", "rotational_speed_rpm", "tool_wear_min", "machine_failure", "twf", "hdf", "pwf", "osf", "rnf"):
                row[key] = int(row[key])
            for key in ("air_temperature_k", "process_temperature_k", "torque_nm"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def process_of(record: dict) -> "Process":
    return Process(**{key: record[key] for key in ("product_type", *DATA_RANGE)})


def twf_rate(tool_wear_min: float) -> float:
    return next(rate for upper, rate in TWF_RATE_BY_WEAR if tool_wear_min <= upper)


def assess(process: Process) -> dict:
    """Which failure modes this process triggers, how close it is to the others, and why."""
    diff, rpm, power = process.temperature_difference_k, process.rotational_speed_rpm, process.power_w
    strain, osf_limit = process.strain_min_nm, OSF_LIMITS[process.product_type]

    hdf = [
        _condition(f"process - air temperature < {HDF_MAX_DIFF_K} K", diff, HDF_MAX_DIFF_K, diff < HDF_MAX_DIFF_K),
        _condition(f"rotational speed < {HDF_MAX_RPM} rpm", rpm, HDF_MAX_RPM, rpm < HDF_MAX_RPM),
    ]
    pwf = [
        _condition(f"power < {PWF_MIN_W} W", power, PWF_MIN_W, power < PWF_MIN_W),
        _condition(f"power > {PWF_MAX_W} W", power, PWF_MAX_W, power > PWF_MAX_W),
    ]
    osf = [_condition(f"tool wear x torque > {osf_limit} minNm (type {process.product_type})", strain, osf_limit, strain > osf_limit)]
    reached_window = process.tool_wear_min >= TWF_WINDOW_MIN[0]

    modes = {
        "HDF": {"triggered": all(c["met"] for c in hdf), "rule": "both conditions", "conditions": hdf, "source": f"{DOC} > HDF"},
        "PWF": {"triggered": any(c["met"] for c in pwf), "rule": "either condition", "conditions": pwf, "source": f"{DOC} > PWF"},
        "OSF": {"triggered": osf[0]["met"], "rule": "the condition", "conditions": osf, "source": f"{DOC} > OSF"},
        "TWF": {
            "triggered": None,  # random by construction: a rate, not a verdict
            "reached_window": reached_window,
            "window_min": list(TWF_WINDOW_MIN),
            "dataset_rate": twf_rate(process.tool_wear_min),
            "source": f"{DOC} > TWF",
        },
        "RNF": {"triggered": None, "probability": RNF_PROBABILITY, "source": f"{DOC} > RNF"},
    }
    triggered = [name for name, mode in modes.items() if mode["triggered"]]
    outside = [
        name for name, (low, high) in DATA_RANGE.items() if not low <= getattr(process, name) <= high
    ]
    return {
        "inputs": {"product_type": process.product_type, **{name: getattr(process, name) for name in DATA_RANGE}},
        "derived": {
            "temperature_difference_k": round(diff, 1),
            "power_w": round(power),
            "strain_min_nm": round(strain),
            "osf_limit_min_nm": osf_limit,
        },
        "modes": modes,
        "failure_expected": bool(triggered),
        "triggered": triggered,
        "outside_data_range": outside,  # the rules still apply, but no recorded process looked like this
    }


if __name__ == "__main__":
    import json

    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(0)
    type_, *values = sys.argv[1:]
    print(json.dumps(assess(Process(type_, *map(float, values))), indent=2, ensure_ascii=False))
