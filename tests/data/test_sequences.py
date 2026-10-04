from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from floodnet_rcmtd.data.records import FloodNetRecord
from floodnet_rcmtd.data.sequences import SequenceGroup, group_sequences


def _record(image_id: str, seconds: int, camera_model: str) -> FloodNetRecord:
    return FloodNetRecord(
        image_id=image_id,
        original_split="train",
        image_path=Path(f"{image_id}.jpg"),
        mask_path=Path(f"{image_id}_lab.png"),
        timestamp=datetime(2026, 6, 15, 12, 0, 0) + timedelta(seconds=seconds),
        camera_model=camera_model,
    )


@pytest.fixture
def records() -> list[FloodNetRecord]:
    return [
        _record("4", 50, "Camera-B"),
        _record("2", 10, "Camera-A"),
        _record("1", 10, "Camera-A"),
        _record("3", 45, "Camera-A"),
    ]


def test_group_sequences_splits_on_camera_or_time_gap(records):
    groups = group_sequences(records, max_gap_seconds=30)
    assert [[item.image_id for item in group.records] for group in groups] == [
        ["1", "2"],
        ["3"],
        ["4"],
    ]
    assert [group.group_id for group in groups] == ["seq-00000", "seq-00001", "seq-00002"]


def test_sequence_group_is_immutable(records):
    group = SequenceGroup(
        group_id="seq-00000",
        camera_model="Camera-A",
        start_time=records[1].timestamp,
        end_time=records[1].timestamp,
        records=(records[1],),
    )

    with pytest.raises(FrozenInstanceError):
        group.group_id = "changed"


@pytest.mark.parametrize("max_gap_seconds", [-1, float("nan")])
def test_group_sequences_rejects_invalid_max_gap(records, max_gap_seconds):
    with pytest.raises(ValueError, match="finite and non-negative"):
        group_sequences(records, max_gap_seconds=max_gap_seconds)
