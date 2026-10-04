import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from floodnet_rcmtd.data.records import FloodNetRecord


@dataclass(frozen=True)
class SequenceGroup:
    group_id: str
    camera_model: str
    start_time: datetime
    end_time: datetime
    records: tuple[FloodNetRecord, ...]


def group_sequences(
    records: Iterable[FloodNetRecord],
    max_gap_seconds: float,
) -> list[SequenceGroup]:
    if not math.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be finite and non-negative")

    ordered = sorted(records, key=lambda record: (record.timestamp, int(record.image_id)))
    grouped_records: list[list[FloodNetRecord]] = []

    for record in ordered:
        if not grouped_records:
            grouped_records.append([record])
            continue

        previous = grouped_records[-1][-1]
        gap_seconds = (record.timestamp - previous.timestamp).total_seconds()
        if record.camera_model != previous.camera_model or gap_seconds > max_gap_seconds:
            grouped_records.append([record])
        else:
            grouped_records[-1].append(record)

    return [
        SequenceGroup(
            group_id=f"seq-{index:05d}",
            camera_model=items[0].camera_model,
            start_time=items[0].timestamp,
            end_time=items[-1].timestamp,
            records=tuple(items),
        )
        for index, items in enumerate(grouped_records)
    ]
