"""Paired prediction and mask-evidence diagnostics for EASE experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def evidence_statistics(row: dict) -> dict:
    counts = row.get("mask_evidence", {}).get("area_pixels", {})
    areas = sorted((float(value) for value in counts.values()), reverse=True)
    if any(value < 0 for value in areas):
        raise ValueError("negative class pixel count")
    total = sum(areas)
    margin = (areas[0] - (areas[1] if len(areas) > 1 else 0)) / total if total else None
    if row.get("question_type") == "area_comparison" and total:
        margin = abs(float(counts.get("road_flooded", 0)) - float(counts.get("road_non_flooded", 0))) / total
    entropy = (-sum(value / total * math.log(value / total) for value in areas if value)
               / math.log(len(areas)) if total and len(areas) > 1 else None)
    support = {str(key): float(value) for key, value in row.get("trace_support", {}).items()}
    if support and (any(not math.isfinite(value) or not 0 <= value <= 1 for value in support.values())
                    or not math.isclose(sum(support.values()), 1.0, abs_tol=1e-6)):
        raise ValueError("trace support values must be probabilities that sum to one")
    ordered = sorted(support.values(), reverse=True)
    return {
        "mask_area_margin": margin, "normalized_area_entropy": entropy,
        "trace_gold_support": support.get(str(row.get("gold_answer"))),
        "trace_support_margin": ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
                                if ordered else None,
    }


def compare_predictions(runs: dict[str, list[dict]], records: list[dict] | None = None) -> dict:
    required = {"Zero-shot", "AA-Ans", "AA-Trace"}
    if not required.issubset(runs):
        raise ValueError(f"required prediction runs: {sorted(required)}")
    indexed = {}
    reference = None
    for name, rows in runs.items():
        if not rows:
            raise ValueError(f"empty prediction run: {name}")
        index = {str(row["question_id"]): row for row in rows}
        if len(index) != len(rows):
            raise ValueError(f"duplicate question_id in {name}")
        membership = {qid: (str(row["image_id"]), str(row["answer"]), str(row["question_type"]))
                      for qid, row in index.items()}
        if reference is None:
            reference = membership
        elif reference != membership:
            raise ValueError(f"prediction membership differs in {name}")
        indexed[name] = index
    record_by_id = {str(row["question_id"]): row for row in records or []}
    if records is not None:
        if len(record_by_id) != len(records) or set(record_by_id) != set(reference):
            raise ValueError("data and prediction membership differs")
        for qid, record in record_by_id.items():
            identity = (str(record["image_id"]), str(record["gold_answer"]), str(record["question_type"]))
            if identity != reference[qid]:
                raise ValueError(f"data and prediction membership differs at {qid}")

    totals = Counter()
    families = defaultdict(Counter)
    cases = []
    confusion = {}
    for name, index in indexed.items():
        matrix = defaultdict(lambda: defaultdict(Counter))
        for row in index.values():
            matrix[str(row["question_type"])][str(row["answer"])][str(row["prediction"])] += 1
        confusion[name] = {family: {gold: dict(sorted(pred.items())) for gold, pred in sorted(golds.items())}
                           for family, golds in sorted(matrix.items())}
    for qid, (image, gold, family) in sorted(reference.items()):
        labels = {name: str(index[qid]["prediction"]) for name, index in indexed.items()}
        aa = labels["AA-Ans"] == gold
        trace = labels["AA-Trace"] == gold
        events = {
            "aa_ans_fixes_zero_shot": not labels["Zero-shot"] == gold and aa,
            "trace_helps_aa_ans": not aa and trace,
            "trace_hurts_aa_ans": aa and not trace,
            "remaining_hard": not aa and not trace,
        }
        if "AA-Ans-second-seed" in labels:
            events["seed_disagreements"] = labels["AA-Ans"] != labels["AA-Ans-second-seed"]
        totals.update({name: int(value) for name, value in events.items()})
        families[family].update({name: int(value) for name, value in events.items()})
        case = {"question_id": qid, "image_id": image, "question_type": family, "answer": gold,
                **labels, **events}
        if qid in record_by_id:
            case.update(evidence_statistics(record_by_id[qid]))
        cases.append(case)

    result = {"count": len(cases), "counts": dict(totals),
              "by_question_type": {name: dict(values) for name, values in sorted(families.items())},
              "confusion_matrices": confusion, "cases": cases}
    if records is not None:
        dominant = [row for row in cases if row["question_type"] == "dominant_class"]
        groups = {}
        for name, correct in [("correct", True), ("incorrect", False)]:
            group = [row for row in dominant if (row["AA-Ans"] == row["answer"]) == correct]
            margins = [row["mask_area_margin"] for row in group if row["mask_area_margin"] is not None]
            entropies = [row["normalized_area_entropy"] for row in group if row["normalized_area_entropy"] is not None]
            groups[name] = {"count": len(group), "median_mask_area_margin": statistics.median(margins) if margins else None,
                            "mean_area_entropy": statistics.mean(entropies) if entropies else None,
                            "area_margin_below_0_10": sum(value < 0.10 for value in margins)}
        hurts = [row for row in cases if row["trace_hurts_aa_ans"]]
        area_hurts = [row for row in hurts if row["question_type"] in {"area_comparison", "dominant_class"}]
        result["evidence_analysis"] = {"dominant_class": groups,
            "area_related_trace_regressions": len(area_hurts),
            "area_regressions_below_0_10": sum(row["mask_area_margin"] is not None and row["mask_area_margin"] < 0.10 for row in area_hurts),
            "trace_regressions_support_below_0_25": sum(row["trace_support_margin"] is not None and row["trace_support_margin"] < 0.25 for row in hurts)}
    return result


def analyze_runs(output_root: Path, backbone: str, seed: int, mode: str,
                 output: Path, *, second_seed: int | None = None,
                 data_dir: Path | None = None) -> dict:
    runs = {}
    test_hash = None
    selected = [(cell, seed, cell) for cell in ("Zero-shot", "PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace")]
    if second_seed is not None:
        if second_seed == seed:
            raise ValueError("second_seed must differ from the primary seed")
        selected.append(("AA-Ans", second_seed, "AA-Ans-second-seed"))
    for cell, run_seed, name in selected:
        folder = output_root / backbone / f"seed{run_seed}" / cell / f"eval_{mode}"
        metadata = json.loads((folder / "meta.json").read_text())
        if test_hash is None:
            test_hash = metadata["test_sha256"]
        elif test_hash != metadata["test_sha256"]:
            raise ValueError(f"test sha256 differs in {folder}")
        runs[name] = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines() if line.strip()]
    records = None
    if data_dir is not None:
        from floodnet_rcmtd.ease2026.runner import verify_data_dir
        rows, manifest = verify_data_dir(data_dir)
        if manifest["files"]["test"]["sha256"] != test_hash:
            raise ValueError("data and evaluation test sha256 differ")
        records = rows["test"]
    result = compare_predictions(runs, records)
    result.update(backbone=backbone, seed=seed, second_seed=second_seed,
                  mode=mode, test_sha256=test_hash)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["cases"][0]))
        writer.writeheader()
        writer.writerows(result["cases"])
    return result
