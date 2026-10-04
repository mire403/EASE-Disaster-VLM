"""Pure protocol checks shared by data preparation, training, and evaluation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

CELLS = ("PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace")


def validate_splits(rows_by_split: Mapping[str, Sequence[dict]]) -> dict[str, object]:
    """Fail closed on any question, image, or mask reused across splits."""

    owners: dict[str, dict[str, str]] = {
        "question_id": {}, "image_id": {}, "mask_sha256": {}, "group_id": {}
    }
    counts: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        counts[split] = len(rows)
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"row split mismatch for {row.get('question_id')!r}")
            question_id = str(row.get("question_id") or "")
            if question_id in owners["question_id"] and owners["question_id"][question_id] == split:
                raise ValueError(f"duplicate question_id in {split}: {question_id}")
            for key in owners:
                if key == "group_id" and not row.get(key):
                    continue
                value = str(row.get(key) or "")
                if not value:
                    raise ValueError(f"missing {key} in {split}")
                owner = owners[key].get(value)
                if owner is not None and owner != split:
                    raise ValueError(f"{key} overlap between {owner} and {split}: {value}")
                owners[key][value] = split
    return {"question_counts": counts, "image_counts": dict(Counter(owners["image_id"].values())),
            "mask_counts": dict(Counter(owners["mask_sha256"].values())), "cross_split_overlap": 0}


def balanced_sample(rows: Sequence[dict], limit_per_family: int, *, seed: int) -> list[dict]:
    if limit_per_family <= 0:
        raise ValueError("limit_per_family must be positive")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["question_type"])].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for family in sorted(groups):
        ordered = sorted(groups[family], key=lambda row: str(row["question_id"]))
        if len(ordered) < limit_per_family:
            raise ValueError(f"{family} has {len(ordered)} rows; need {limit_per_family}")
        rng.shuffle(ordered)
        selected.extend(ordered[:limit_per_family])
    return selected


def make_cells(train_rows: Sequence[dict], *, seed: int) -> dict[str, list[dict]]:
    """Apply a single example order to all four cells and validate trace agreement."""

    ordered = sorted(train_rows, key=lambda row: str(row["question_id"]))
    if len({str(row["question_id"]) for row in ordered}) != len(ordered):
        raise ValueError("duplicate question_id in training rows")
    random.Random(seed).shuffle(ordered)
    result: dict[str, list[dict]] = {}
    for cell in CELLS:
        support, target_type = cell.split("-")
        items = []
        for row in ordered:
            gold = str(row["gold_answer"])
            trace = str(row["trace_answer"])
            if trace != gold:
                raise ValueError(f"trace/gold disagreement at {row['question_id']}")
            answer_space = [str(value) for value in row["answer_space"]]
            if gold not in answer_space:
                raise ValueError(f"gold answer outside answer_space at {row['question_id']}")
            target = gold if target_type == "Ans" else trace
            item = dict(row)
            item.update(cell=cell, loss_support=support, target_source=target_type,
                        target_answer=target)
            items.append(item)
        result[cell] = items
    return result


def make_labels(
    input_ids: Sequence[int], prompt_ids: Sequence[int], pad_token_id: int | None,
    loss_support: str, *, attention_mask: Sequence[int] | None = None,
    answer_end: int | None = None,
) -> list[int]:
    """Return LM labels with the selected support; the prompt must be an exact prefix."""

    ids = list(input_ids)
    prefix = list(prompt_ids)
    if not prefix or ids[:len(prefix)] != prefix:
        raise ValueError("prompt token IDs are not an exact prefix of full token IDs")
    if loss_support not in {"PC", "AA"}:
        raise ValueError(f"unknown loss support: {loss_support}")
    if attention_mask is not None and len(attention_mask) != len(ids):
        raise ValueError("attention_mask length mismatch")
    end = len(ids) if answer_end is None else answer_end
    if not len(prefix) < end <= len(ids):
        raise ValueError("answer span must be non-empty and inside the input sequence")
    labels = ids.copy()
    for index, token in enumerate(ids):
        is_padding = (attention_mask[index] == 0 if attention_mask is not None
                      else pad_token_id is not None and token == pad_token_id)
        if is_padding or (loss_support == "AA" and (index < len(prefix) or index >= end)):
            labels[index] = -100
    if not any(value != -100 for value in labels[1:]):
        raise ValueError("no scoreable language tokens remain")
    return labels
