"""Compute paper-style metrics from saved per-example predictions."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _scores(rows: list[dict]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("no prediction rows")
    true = [str(row["answer"]) for row in rows]
    pred = [str(row["prediction"]) for row in rows]
    correct = [a == b for a, b in zip(true, pred, strict=True)]
    labels = sorted(set(true) | set(pred))
    f1s = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, pred, strict=True))
        fp = sum(a != label and b == label for a, b in zip(true, pred, strict=True))
        fn = sum(a == label and b != label for a, b in zip(true, pred, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    counts = Counter(pred)
    probabilities = [count / len(rows) for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probabilities)
    normalized_entropy = entropy / math.log(len(labels)) if len(labels) > 1 else 0.0

    confidences = [row.get("confidence") for row in rows]
    ece = None
    if all(value is not None for value in confidences):
        bins = [[] for _ in range(10)]
        for confidence, is_correct in zip(confidences, correct, strict=True):
            value = float(confidence)
            if not 0 <= value <= 1:
                raise ValueError("confidence outside [0, 1]")
            bins[min(9, int(value * 10))].append((value, is_correct))
        ece = sum(len(bucket) / len(rows) * abs(
            sum(item[0] for item in bucket) / len(bucket) -
            sum(item[1] for item in bucket) / len(bucket)
        ) for bucket in bins if bucket)
    return {
        "count": len(rows), "accuracy": sum(correct) / len(rows),
        "macro_f1": sum(f1s) / len(f1s), "ece": ece,
        "top_prediction_share": max(counts.values()) / len(rows),
        "normalized_prediction_entropy": normalized_entropy,
    }


def summarize_predictions(rows: list[dict]) -> dict[str, object]:
    families: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        families[str(row["question_type"])].append(row)
    return {"overall": _scores(rows),
            "by_question_type": {name: _scores(group) for name, group in sorted(families.items())}}


def collect_runs(
    output_root: Path, backbone: str, seeds: list[int], mode: str,
    *, cells: list[str] | None = None,
) -> dict[str, object]:
    cells = cells or ["PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace"]
    if not seeds or not cells:
        raise ValueError("at least one seed and cell required")
    reference: dict[str, tuple[str, str]] | None = None
    test_hash: str | None = None
    per_run = []
    for seed in seeds:
        for cell in cells:
            folder = output_root / backbone / f"seed{seed}" / cell / f"eval_{mode}"
            metadata = json.loads((folder / "meta.json").read_text())
            rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()
                    if line.strip()]
            membership = {str(row["question_id"]): (str(row["image_id"]), str(row["answer"]))
                          for row in rows}
            if len(membership) != len(rows):
                raise ValueError(f"duplicate question_id in {folder}")
            if reference is None:
                reference = membership
                test_hash = str(metadata["test_sha256"])
            elif membership != reference:
                raise ValueError(f"evaluation membership differs in {folder}")
            if str(metadata["test_sha256"]) != test_hash:
                raise ValueError(f"test sha256 differs in {folder}")
            per_run.append({"seed": seed, "cell": cell,
                            "metrics": summarize_predictions(rows)})

    aggregate = {}
    for cell in cells:
        runs = [entry["metrics"]["overall"] for entry in per_run if entry["cell"] == cell]
        aggregate[cell] = {}
        for key in ("accuracy", "macro_f1", "ece", "top_prediction_share",
                    "normalized_prediction_entropy"):
            values = [float(run[key]) for run in runs if run[key] is not None]
            aggregate[cell][key] = ({"mean": statistics.mean(values),
                                     "std": statistics.stdev(values) if len(values) > 1 else 0.0}
                                    if values else None)
    family_aggregate = {}
    for cell in cells:
        family_runs = [entry["metrics"]["by_question_type"] for entry in per_run if entry["cell"] == cell]
        names = sorted(set().union(*(set(run) for run in family_runs)))
        family_aggregate[cell] = {}
        for family in names:
            family_aggregate[cell][family] = {}
            for metric in ("accuracy", "macro_f1", "ece", "top_prediction_share", "normalized_prediction_entropy"):
                values = [float(run[family][metric]) for run in family_runs if run[family][metric] is not None]
                family_aggregate[cell][family][metric] = (
                    {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
                    if values else None)
    gap = None
    if {"PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace"}.issubset(cells):
        gaps = []
        for seed in seeds:
            accuracies = {entry["cell"]: entry["metrics"]["overall"]["accuracy"]
                          for entry in per_run if entry["seed"] == seed}
            value = ((accuracies["AA-Ans"] + accuracies["AA-Trace"])
                     - (accuracies["PC-Ans"] + accuracies["PC-Trace"])) / 2
            gaps.append({"seed": seed, "gap": value})
        values = [item["gap"] for item in gaps]
        gap = {"per_seed": gaps, "mean": statistics.mean(values),
               "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    return {"protocol": "ease2026-v1", "backbone": backbone,
            "mode": mode, "seeds": seeds, "test_sha256": test_hash,
            "per_run": per_run, "aggregate": aggregate,
            "aggregate_by_question_type": family_aggregate, "alignment_gap": gap}


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backbone", "mode", "cell", "metric", "mean", "std"])
        for cell, metrics in summary["aggregate"].items():
            for metric, value in metrics.items():
                writer.writerow([summary["backbone"], summary["mode"], cell, metric,
                                 value["mean"] if value else "", value["std"] if value else ""])
