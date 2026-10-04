import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from floodnet_rcmtd.cli import _run_build_split, build_parser, main
from floodnet_rcmtd.data.records import FloodNetRecord
from floodnet_rcmtd.data.sequences import SequenceGroup
from floodnet_rcmtd.data.split import assign_groups, count_cross_split_sequence_pairs


def _record(image_id: str, seconds: int, camera_model: str) -> FloodNetRecord:
    return FloodNetRecord(
        image_id=image_id,
        original_split="train",
        image_path=Path(f"{image_id}.jpg"),
        mask_path=Path(f"{image_id}_lab.png"),
        timestamp=datetime(2026, 6, 15, 12, 0, 0) + timedelta(seconds=seconds),
        camera_model=camera_model,
    )


def _group(group_id: str, camera_model: str, *records: FloodNetRecord) -> SequenceGroup:
    return SequenceGroup(
        group_id=group_id,
        camera_model=camera_model,
        start_time=records[0].timestamp,
        end_time=records[-1].timestamp,
        records=records,
    )


@pytest.fixture
def groups() -> list[SequenceGroup]:
    return [
        _group(
            "seq-00000",
            "Camera-A",
            _record("1", 0, "Camera-A"),
            _record("2", 10, "Camera-A"),
            _record("3", 20, "Camera-A"),
        ),
        _group(
            "seq-00001",
            "Camera-A",
            _record("4", 100, "Camera-A"),
            _record("5", 110, "Camera-A"),
        ),
        _group("seq-00002", "Camera-B", _record("6", 200, "Camera-B")),
        _group("seq-00003", "Camera-B", _record("7", 300, "Camera-B")),
    ]


def test_assign_groups_never_splits_a_sequence(groups):
    assignment = assign_groups(groups, ratios=(0.6, 0.2, 0.2), seed=20260615)
    group_splits = {
        group.group_id: {assignment[item.image_id] for item in group.records}
        for group in groups
    }
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert set(assignment.values()) == {"train", "val", "test"}


def test_assign_groups_is_deterministic_for_a_seed(groups):
    class_histograms = {
        "1": [8, 0],
        "2": [8, 0],
        "3": {0: 8},
        "4": [0, 8],
        "5": {1: 8},
        "6": [4, 4],
        "7": [4, 4],
    }

    first = assign_groups(groups, seed=17, class_histograms=class_histograms)
    second = assign_groups(groups, seed=17, class_histograms=class_histograms)

    assert first == second


def test_assign_groups_rejects_invalid_ratios(groups):
    with pytest.raises(ValueError, match="three non-negative"):
        assign_groups(groups, ratios=(0.8, -0.1, 0.3))


def test_assign_groups_handles_fewer_groups_than_active_splits(groups):
    assignment = assign_groups(groups[:1])

    assert set(assignment) == {"1", "2", "3"}
    assert len(set(assignment.values())) == 1


def test_count_cross_split_sequence_pairs_returns_zero_without_leakage():
    rows = [
        {"group_id": "seq-00000", "split": "train"},
        {"group_id": "seq-00000", "split": "train"},
        {"group_id": "seq-00001", "split": "val"},
    ]

    assert count_cross_split_sequence_pairs(rows) == 0


def test_count_cross_split_sequence_pairs_preserves_image_multiplicity():
    rows = [
        *[{"sequence_id": "seq-00000", "split": "train"} for _ in range(2)],
        *[{"sequence_id": "seq-00000", "split": "val"} for _ in range(3)],
    ]

    assert count_cross_split_sequence_pairs(rows) == 6


def test_build_split_parser_accepts_config_path():
    args = build_parser().parse_args(["build-split", "--config", "config.yaml"])

    assert args.command == "build-split"
    assert args.config == Path("config.yaml")


def test_cli_help_uses_argparse(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "usage: floodnet-rcmtd" in capsys.readouterr().out


def test_unknown_cli_command_reports_parser_error(capsys):
    with pytest.raises(SystemExit) as error:
        main(["future-command"])

    assert error.value.code == 2
    assert "invalid choice: 'future-command'" in capsys.readouterr().err


def test_run_build_split_writes_independently_audited_outputs(
    tiny_floodnet_root: Path,
    tmp_path: Path,
):
    output_root = tmp_path / "derived"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_root: {tiny_floodnet_root}",
                f"output_root: {output_root}",
                "sequence_gap_seconds: 30",
                "gap_sensitivity_seconds: [10, 30, 60]",
                "split_ratios: [0.6, 0.2, 0.2]",
                "class_ids: [0, 1]",
                "seed: 20260615",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run_build_split(config_path)

    images_path = output_root / "splits" / "images.parquet"
    sequences_path = output_root / "splits" / "sequences.parquet"
    audit_path = output_root / "audits" / "split_audit.json"
    assert images_path.exists()
    assert sequences_path.exists()
    assert audit_path.exists()

    image_rows = pd.read_parquet(images_path)
    sequence_rows = pd.read_parquet(sequences_path)
    assert {
        "image_id",
        "group_id",
        "split",
        "image_path",
        "mask_path",
    } <= set(image_rows.columns)
    assert {"group_id", "split"} <= set(sequence_rows.columns)
    assert len(image_rows) == 3

    persisted_rows = image_rows.to_dict(orient="records")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["cross_split_sequence_pairs"] == 0
    assert audit["cross_split_sequence_pairs"] == count_cross_split_sequence_pairs(
        persisted_rows
    )
