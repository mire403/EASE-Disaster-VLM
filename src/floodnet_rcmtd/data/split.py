from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Union

from floodnet_rcmtd.data.sequences import SequenceGroup

ClassHistogram = Union[Mapping[int, int], Sequence[int]]

_SPLIT_NAMES = ("train", "val", "test")


def count_cross_split_sequence_pairs(image_rows: list[dict[str, object]]) -> int:
    split_counts_by_sequence: dict[object, Counter[object]] = defaultdict(Counter)
    for row in image_rows:
        sequence_id = row.get("sequence_id", row.get("group_id"))
        split_counts_by_sequence[sequence_id][row["split"]] += 1

    cross_split_pairs = 0
    for split_counts in split_counts_by_sequence.values():
        earlier_image_count = 0
        for image_count in split_counts.values():
            cross_split_pairs += earlier_image_count * image_count
            earlier_image_count += image_count
    return cross_split_pairs


def _histogram_counts(histogram: ClassHistogram) -> dict[int, float]:
    if isinstance(histogram, Mapping):
        return {int(class_id): float(count) for class_id, count in histogram.items()}
    return {class_id: float(count) for class_id, count in enumerate(histogram)}


def _normalized_deviation(
    counts: Mapping[str, float],
    total: float,
    ratios: Mapping[str, float],
) -> float:
    if total == 0:
        return 0.0
    return sum(abs(counts.get(split, 0.0) - total * ratios[split]) for split in ratios) / total


def assign_groups(
    groups: Sequence[SequenceGroup],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 20260615,
    class_histograms: Mapping[str, ClassHistogram] | None = None,
) -> dict[str, str]:
    if len(ratios) != len(_SPLIT_NAMES) or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    ratio_total = sum(ratios)
    if ratio_total <= 0:
        raise ValueError("ratios must contain three non-negative values with a positive sum")

    split_ratios = {
        split: ratio / ratio_total for split, ratio in zip(_SPLIT_NAMES, ratios)
    }
    active_splits = tuple(split for split in _SPLIT_NAMES if split_ratios[split] > 0)
    ordered_groups = sorted(groups, key=lambda group: (-len(group.records), group.group_id))
    enforce_split_coverage = len(ordered_groups) >= len(active_splits)
    total_images = sum(len(group.records) for group in ordered_groups)

    camera_totals = Counter()
    group_class_counts: dict[str, Counter[int]] = {}
    class_totals: Counter[int] = Counter()
    for group in ordered_groups:
        camera_totals[group.camera_model] += len(group.records)
        histogram = Counter()
        if class_histograms is not None:
            for record in group.records:
                histogram.update(_histogram_counts(class_histograms.get(record.image_id, ())))
        group_class_counts[group.group_id] = histogram
        class_totals.update(histogram)

    image_counts: Counter[str] = Counter()
    camera_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts: dict[int, Counter[str]] = defaultdict(Counter)
    group_assignments: dict[str, str] = {}
    rng = random.Random(seed)

    def score(group: SequenceGroup, candidate: str) -> float:
        hypothetical_images = image_counts.copy()
        hypothetical_images[candidate] += len(group.records)
        image_deviation = _normalized_deviation(
            hypothetical_images, total_images, split_ratios
        )

        camera_deviation = 0.0
        for camera_model, camera_total in camera_totals.items():
            hypothetical_camera = camera_counts[camera_model].copy()
            if camera_model == group.camera_model:
                hypothetical_camera[candidate] += len(group.records)
            camera_deviation += _normalized_deviation(
                hypothetical_camera, camera_total, split_ratios
            )
        if camera_totals:
            camera_deviation /= len(camera_totals)

        class_deviation = 0.0
        if class_histograms is not None and class_totals:
            group_histogram = group_class_counts[group.group_id]
            for class_id, class_total in class_totals.items():
                hypothetical_class = class_counts[class_id].copy()
                hypothetical_class[candidate] += group_histogram[class_id]
                class_deviation += _normalized_deviation(
                    hypothetical_class, class_total, split_ratios
                )
            class_deviation /= len(class_totals)

        return image_deviation + 0.25 * camera_deviation + 0.25 * class_deviation

    for index, group in enumerate(ordered_groups):
        remaining_groups = len(ordered_groups) - index - 1
        candidates = []
        for split in active_splits:
            used_after = set(group_assignments.values()) | {split}
            empty_splits_after = len(set(active_splits) - used_after)
            if not enforce_split_coverage or remaining_groups >= empty_splits_after:
                candidates.append(split)

        scored = [(score(group, split), split) for split in candidates]
        best_score = min(value for value, _ in scored)
        best_splits = [
            split
            for value, split in scored
            if math.isclose(value, best_score, rel_tol=1e-12, abs_tol=1e-12)
        ]
        selected = rng.choice(best_splits)
        group_assignments[group.group_id] = selected
        image_counts[selected] += len(group.records)
        camera_counts[group.camera_model][selected] += len(group.records)
        for class_id, count in group_class_counts[group.group_id].items():
            class_counts[class_id][selected] += count

    return {
        record.image_id: group_assignments[group.group_id]
        for group in ordered_groups
        for record in group.records
    }
