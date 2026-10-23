"""
The documented failure rules against a machine-learning model, on the AI4I dataset.

    python -m evals.failure_model_eval

Predicts `machine_failure` for each of the 10,000 processes three ways:

    rules            failure_rules.py: the HDF, PWF and OSF rules from the knowledge base
    gbm raw          gradient boosting on the six raw columns
    gbm + doc feat.  the same model, also given the quantities the documents
                     reason with: temperature difference, power, wear x torque
                     and its ratio to the type's limit

The models are scored with 5-fold stratified cross-validation, so every
prediction comes from a model that didn't see that row. Failures are 3.4%
of rows, so accuracy says little (always predicting "no failure" scores
96.6%): precision, recall and PR-AUC carry the comparison.

No method can catch every failure: TWF strikes at a random tool wear and
RNF at random, and 9 failures carry no mode at all. The last line says how
many failures a perfect predictor of the deterministic modes would find.

It also checks the space the copilot's "similar recorded processes" search
uses (process_data.py): for each failed process, how many of its 25 nearest
neighbours of the same product type failed in the same mode, when distance is
measured on the raw readings and on the documents' quantities.
"""

import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from failure_rules import OSF_LIMITS, assess, load_dataset, process_of

TYPE_CODE = {"L": 0, "M": 1, "H": 2}
RAW = ["product_type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
MODES = ("twf", "hdf", "pwf", "osf", "rnf")


def features(record: dict, with_doc_features: bool) -> list[float]:
    row = [TYPE_CODE[record["product_type"]], *(record[key] for key in RAW[1:])]
    if with_doc_features:
        diff = record["process_temperature_k"] - record["air_temperature_k"]
        power = record["torque_nm"] * record["rotational_speed_rpm"] * 2 * math.pi / 60
        strain = record["tool_wear_min"] * record["torque_nm"]
        row += [diff, power, strain, strain / OSF_LIMITS[record["product_type"]]]
    return row


def scores(y: np.ndarray, predicted: np.ndarray, probability: np.ndarray = None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision_score(y, probability) if probability is not None else None,
        "false_alarms": int(((predicted == 1) & (y == 0)).sum()),
    }


def missed_by_mode(records: list[dict], y: np.ndarray, predicted: np.ndarray) -> dict:
    missed = [r for r, label, guess in zip(records, y, predicted) if label == 1 and guess == 0]
    return {mode.upper(): sum(r[mode] for r in missed) for mode in MODES}


def evaluate() -> dict:
    records = load_dataset()
    y = np.array([r["machine_failure"] for r in records])
    results = {}

    rules = np.array([int(assess(process_of(r))["failure_expected"]) for r in records])
    results["rules"] = {**scores(y, rules), "missed": missed_by_mode(records, y, rules)}

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, with_doc_features in (("gbm raw", False), ("gbm + doc features", True)):
        x = np.array([features(r, with_doc_features) for r in records])
        model = HistGradientBoostingClassifier(random_state=0, categorical_features=[0])
        probability = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
        predicted = (probability >= 0.5).astype(int)
        results[name] = {**scores(y, predicted, probability), "missed": missed_by_mode(records, y, predicted)}

    deterministic = sum(1 for r in records if r["machine_failure"] and (r["hdf"] or r["pwf"] or r["osf"]))
    results["ceiling"] = {"deterministic_failures": deterministic, "failures": int(y.sum())}
    results["neighbours"] = {space: neighbour_agreement(records, space) for space in ("raw readings", "doc quantities")}
    return results


def neighbour_agreement(records: list[dict], space: str, k: int = 25) -> dict:
    """Per mode: the share of a failed process's k nearest same-type neighbours that failed in that mode."""
    if space == "raw readings":
        x = np.array([[r[key] for key in RAW[1:]] for r in records], dtype=float)
    else:  # the quantities process_data.py's search uses
        x = np.array([features(r, with_doc_features=True)[6:9] + [r["rotational_speed_rpm"], r["tool_wear_min"]] for r in records])
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    types = np.array([r["product_type"] for r in records])
    agreement = {}
    for mode in ("hdf", "pwf", "osf", "twf"):
        shares = []
        for i in (i for i, r in enumerate(records) if r[mode]):
            same_type = np.flatnonzero(types == types[i])
            nearest = same_type[np.argsort(np.linalg.norm(x[same_type] - x[i], axis=1))][1 : k + 1]  # not itself
            shares.append(np.mean([records[j][mode] for j in nearest]))
        agreement[mode.upper()] = float(np.mean(shares))
    return agreement


if __name__ == "__main__":
    results = evaluate()
    print(f"\n{'method':20s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'PR-AUC':>7s} {'false alarms':>13s}   missed failures by mode")
    print("-" * 110)
    for name in ("rules", "gbm raw", "gbm + doc features"):
        r = results[name]
        pr_auc = f"{r['pr_auc']:.3f}" if r["pr_auc"] is not None else "-"
        print(f"{name:20s} {r['precision']:9.3f} {r['recall']:7.3f} {r['f1']:6.3f} {pr_auc:>7s} {r['false_alarms']:13d}   {r['missed']}")
    ceiling = results["ceiling"]
    print(
        f"\n{ceiling['deterministic_failures']} of {ceiling['failures']} failures have a deterministic mode (HDF, PWF or OSF): "
        f"recall {ceiling['deterministic_failures'] / ceiling['failures']:.3f} is the most any method can reach without luck."
    )
    records = load_dataset()
    base = {mode.upper(): sum(r[mode] for r in records) / len(records) for mode in ("hdf", "pwf", "osf", "twf")}
    print("\nShare of a failed process's 25 nearest neighbours that failed in the same mode (base rate in brackets):")
    for space, agreement in results["neighbours"].items():
        print(f"  {space:15s} " + "  ".join(f"{m} {share:.3f} ({base[m]:.3f})" for m, share in agreement.items()))
