from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

_QUESTION_COLUMNS = (
    "question_id",
    "image_id",
    "split",
    "question_type",
    "template_id",
    "question",
    "answer",
    "answer_space",
    "evidence",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="floodnet-rcmtd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_split = subparsers.add_parser("build-split")
    build_split.add_argument("--config", required=True, type=Path)
    build_questions = subparsers.add_parser("build-questions")
    build_questions.add_argument("--config", required=True, type=Path)
    audit_questions = subparsers.add_parser("audit-questions")
    audit_questions.add_argument("--config", required=True, type=Path)
    return parser


def _mask_class_histogram(mask_path: Path, class_ids: Sequence[int]) -> dict[int, int]:
    import numpy as np
    from PIL import Image

    with Image.open(mask_path) as mask:
        values = np.asarray(mask)
    counts = np.bincount(values.reshape(-1), minlength=max(class_ids, default=-1) + 1)
    return {
        class_id: int(counts[class_id]) if class_id < len(counts) else 0
        for class_id in class_ids
    }


def _sensitivity_summary(records, gap_seconds: float) -> dict[str, float | int]:
    from floodnet_rcmtd.data.sequences import group_sequences

    groups = group_sequences(records, gap_seconds)
    sizes = [len(group.records) for group in groups]
    return {
        "group_count": len(groups),
        "singleton_group_count": sum(size == 1 for size in sizes),
        "max_group_size": max(sizes, default=0),
        "mean_group_size": sum(sizes) / len(sizes) if sizes else 0.0,
    }


def _run_build_split(config_path: Path) -> None:
    import pandas as pd

    from floodnet_rcmtd.config import load_config
    from floodnet_rcmtd.data.layout import discover_pairs
    from floodnet_rcmtd.data.sequences import group_sequences
    from floodnet_rcmtd.data.split import assign_groups, count_cross_split_sequence_pairs
    from floodnet_rcmtd.manifest import write_manifest

    config = load_config(config_path)
    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    gap_seconds = float(config["sequence_gap_seconds"])
    class_ids = [int(class_id) for class_id in config["class_ids"]]

    records = discover_pairs(data_root)
    class_histograms = {
        record.image_id: _mask_class_histogram(record.mask_path, class_ids)
        for record in records
    }
    groups = group_sequences(records, gap_seconds)
    assignments = assign_groups(
        groups,
        ratios=tuple(float(value) for value in config["split_ratios"]),
        seed=int(config["seed"]),
        class_histograms=class_histograms,
    )

    group_by_image_id = {
        record.image_id: group for group in groups for record in group.records
    }
    image_rows = []
    for record in records:
        row = {
            "image_id": record.image_id,
            "group_id": group_by_image_id[record.image_id].group_id,
            "split": assignments[record.image_id],
            "original_split": record.original_split,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "timestamp": record.timestamp,
            "camera_model": record.camera_model,
        }
        row.update(
            {
                f"class_{class_id}_pixels": class_histograms[record.image_id][class_id]
                for class_id in class_ids
            }
        )
        image_rows.append(row)

    sequence_rows = [
        {
            "group_id": group.group_id,
            "split": assignments[group.records[0].image_id],
            "camera_model": group.camera_model,
            "start_time": group.start_time,
            "end_time": group.end_time,
            "image_count": len(group.records),
            "image_ids": [record.image_id for record in group.records],
        }
        for group in groups
    ]

    splits_root = output_root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(image_rows).to_parquet(splits_root / "images.parquet", index=False)
    pd.DataFrame(sequence_rows).to_parquet(splits_root / "sequences.parquet", index=False)
    persisted_image_rows = pd.read_parquet(splits_root / "images.parquet").to_dict(
        orient="records"
    )

    image_counts = Counter(assignments.values())
    group_counts = Counter(row["split"] for row in sequence_rows)
    audit = {
        "cross_split_sequence_pairs": count_cross_split_sequence_pairs(
            persisted_image_rows
        ),
        "split_image_counts": {
            split: image_counts[split] for split in ("train", "val", "test")
        },
        "split_group_counts": {
            split: group_counts[split] for split in ("train", "val", "test")
        },
        "sequence_gap_seconds": gap_seconds,
        "sensitivity_summaries": {
            str(int(seconds) if float(seconds).is_integer() else seconds): _sensitivity_summary(
                records, float(seconds)
            )
            for seconds in config["gap_sensitivity_seconds"]
        },
    }
    write_manifest(output_root / "audits" / "split_audit.json", audit)


def _serialized_question(record) -> dict[str, object]:
    row = asdict(record)
    row["answer_space"] = list(row["answer_space"])
    row["evidence"] = json.dumps(
        row["evidence"], sort_keys=True, separators=(",", ":")
    )
    return row


def _question_rows_from_images(image_rows, config) -> dict[str, list[dict[str, object]]]:
    import numpy as np
    from PIL import Image

    from floodnet_rcmtd.data.questions import build_evidence, generate_questions

    required_columns = {"image_id", "mask_path", "split"}
    missing_columns = sorted(required_columns - set(image_rows.columns))
    if missing_columns:
        raise ValueError(f"images.parquet is missing columns: {missing_columns}")

    rows_by_split: dict[str, list[dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for image_row in image_rows.to_dict(orient="records"):
        split = str(image_row["split"])
        if split not in rows_by_split:
            raise ValueError(f"images.parquet contains unknown split: {split!r}")
        with Image.open(Path(image_row["mask_path"])) as mask_image:
            mask = np.asarray(mask_image)
        evidence = build_evidence(
            mask,
            adjacency_radius=int(config.get("adjacency_radius", 1)),
            ratio_thresholds=config.get("ratio_thresholds"),
        )
        questions = generate_questions(
            str(image_row["image_id"]),
            evidence,
            seed=int(config["seed"]),
            split=split,
        )
        rows_by_split[split].extend(_serialized_question(row) for row in questions)
    return rows_by_split


def _run_build_questions(config_path: Path) -> None:
    import pandas as pd

    from floodnet_rcmtd.config import load_config
    from floodnet_rcmtd.manifest import write_manifest

    config = load_config(config_path)
    output_root = Path(config["output_root"])
    image_rows = pd.read_parquet(output_root / "splits" / "images.parquet")
    rows_by_split = _question_rows_from_images(image_rows, config)

    questions_root = output_root / "questions"
    questions_root.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        pd.DataFrame(rows, columns=_QUESTION_COLUMNS).to_parquet(
            questions_root / f"{split}.parquet",
            index=False,
        )
    write_manifest(
        output_root / "manifest.json",
        {
            "images": "splits/images.parquet",
            "questions": {
                split: f"questions/{split}.parquet"
                for split in ("train", "val", "test")
            },
        },
    )


def _run_audit_questions(config_path: Path) -> None:
    import pandas as pd

    from floodnet_rcmtd.config import load_config
    from floodnet_rcmtd.data.audit import (
        audit_questions,
        compare_regenerated_questions,
        manual_review_sample,
    )
    from floodnet_rcmtd.manifest import write_manifest

    config = load_config(config_path)
    output_root = Path(config["output_root"])
    questions_root = output_root / "questions"
    rows_by_split = {
        split: pd.read_parquet(questions_root / f"{split}.parquet").to_dict(
            orient="records"
        )
        for split in ("train", "val", "test")
    }
    image_rows = pd.read_parquet(output_root / "splits" / "images.parquet")
    regenerated_by_split = _question_rows_from_images(image_rows, config)
    compare_regenerated_questions(
        [row for rows in rows_by_split.values() for row in rows],
        [row for rows in regenerated_by_split.values() for row in rows],
    )
    distribution = audit_questions(
        rows_by_split,
        min_examples_per_type=int(config.get("min_examples_per_type", 100)),
        max_majority_answer_fraction=float(
            config.get("max_majority_answer_fraction", 0.95)
        ),
    )
    sample = manual_review_sample(
        rows_by_split,
        per_split=min(int(config.get("manual_review_per_split", 100)), 100),
        seed=int(config["seed"]),
    )

    audits_root = output_root / "audits"
    write_manifest(audits_root / "question_distribution.json", distribution)
    sample_rows = []
    for row in sample:
        serialized = dict(row)
        serialized["answer_space"] = json.dumps(
            list(serialized["answer_space"]), separators=(",", ":")
        )
        serialized["evidence"] = json.dumps(
            serialized["evidence"], sort_keys=True, separators=(",", ":")
        )
        sample_rows.append(serialized)
    audits_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows, columns=_QUESTION_COLUMNS).to_csv(
        audits_root / "manual_review_sample.csv",
        index=False,
    )



def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {"build-split": _run_build_split,
                "build-questions": _run_build_questions,
                "audit-questions": _run_audit_questions}
    handlers[args.command](args.config)


if __name__ == "__main__":
    main()
