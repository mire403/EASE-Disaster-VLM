from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from floodnet_rcmtd.data.questions import QUESTION_FAMILIES, recompute_answer

_PERSISTED_SEMANTIC_FIELDS = (
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

_DIFFICULTY_BY_QUESTION_TYPE = {
    "presence": "binary",
    "damage_state": "binary",
    "spatial_adjacency": "binary",
    "area_comparison": "compositional",
    "logical_conjunction": "compositional",
    "dominant_class": "multiclass",
}


class QuestionAuditError(ValueError):
    pass


def _decoded_evidence(value: object) -> dict[str, object]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise QuestionAuditError("question evidence contains invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise QuestionAuditError("question evidence must decode to an object")
        return decoded
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    raise QuestionAuditError("question evidence must be an object or JSON object")


def _normalized_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows = []
    for split, split_rows in rows_by_split.items():
        for source_row in split_rows:
            row = dict(source_row)
            if str(row.get("split", split)) != split:
                raise QuestionAuditError(
                    f"row split {row.get('split')!r} does not match collection {split!r}"
                )
            row["split"] = split
            row["evidence"] = _decoded_evidence(row.get("evidence"))
            rows.append(row)
    return rows


def audit_questions(
    rows_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    min_examples_per_type: int = 100,
    max_majority_answer_fraction: float = 0.95,
) -> dict[str, object]:
    if min_examples_per_type < 0:
        raise ValueError("min_examples_per_type must be non-negative")
    if not 0 <= max_majority_answer_fraction <= 1:
        raise ValueError("max_majority_answer_fraction must be between 0 and 1")
    rows = _normalized_rows(rows_by_split)

    splits_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_id[str(row.get("question_id", ""))].add(str(row["split"]))
    duplicate_ids = sorted(
        question_id for question_id, splits in splits_by_id.items() if len(splits) > 1
    )
    if duplicate_ids:
        raise QuestionAuditError(
            f"question IDs occur across splits: {duplicate_ids[:5]}"
        )

    counts_by_split_and_type: dict[str, Counter[str]] = defaultdict(Counter)
    answers_by_split_and_type: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    templates_by_split_and_type: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in rows:
        split = str(row["split"])
        question_type = str(row.get("question_type", ""))
        counts_by_split_and_type[split][question_type] += 1
        answers_by_split_and_type[split][question_type][
            str(row.get("answer", ""))
        ] += 1
        templates_by_split_and_type[split][question_type][
            str(row.get("template_id", ""))
        ] += 1
    for split in ("val", "test"):
        for question_type in QUESTION_FAMILIES:
            count = counts_by_split_and_type[split][question_type]
            if count < min_examples_per_type:
                raise QuestionAuditError(
                    f"{split} has {count} {question_type} examples; "
                    f"minimum is {min_examples_per_type}"
                )

    for row in rows:
        evidence = row["evidence"]
        if not evidence:
            raise QuestionAuditError(
                f"question {row.get('question_id')!r} has empty evidence"
            )
        answer = str(row.get("answer", ""))
        try:
            recomputed_answer = recompute_answer(
                str(row.get("question_type", "")),
                evidence,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QuestionAuditError(
                f"cannot recompute answer for question "
                f"{row.get('question_id')!r}: {exc}"
            ) from exc
        if answer != recomputed_answer:
            raise QuestionAuditError(
                f"answer mismatch for question {row.get('question_id')!r}: "
                f"{answer!r} != {recomputed_answer!r}"
            )

    rows_by_template: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_template[str(row.get("template_id", ""))].append(row)
    for template_id, template_rows in rows_by_template.items():
        if len(template_rows) < 2:
            continue
        answers = {str(row.get("answer", "")) for row in template_rows}
        if len(answers) == 1:
            raise QuestionAuditError(
                f"template {template_id!r} maps to only one answer across "
                f"{len(template_rows)} rows"
            )

    by_split = Counter(str(row["split"]) for row in rows)
    by_type = Counter(str(row.get("question_type", "")) for row in rows)
    by_answer = Counter(str(row.get("answer", "")) for row in rows)
    majority_answer_fraction = {
        split: {
            question_type: (
                max(answers_by_split_and_type[split][question_type].values())
                / counts_by_split_and_type[split][question_type]
                if counts_by_split_and_type[split][question_type]
                else 0.0
            )
            for question_type in QUESTION_FAMILIES
        }
        for split in ("train", "val", "test")
    }
    warnings = [
        {
            "split": split,
            "question_type": question_type,
            "majority_answer_fraction": majority_answer_fraction[split][question_type],
            "max_majority_answer_fraction": max_majority_answer_fraction,
        }
        for split in ("train", "val", "test")
        for question_type in QUESTION_FAMILIES
        if counts_by_split_and_type[split][question_type]
        and (
            majority_answer_fraction[split][question_type]
            > max_majority_answer_fraction
        )
    ]
    return {
        "total_questions": len(rows),
        "by_split": {split: by_split[split] for split in ("train", "val", "test")},
        "by_type": dict(sorted(by_type.items())),
        "by_answer": dict(sorted(by_answer.items())),
        "by_split_and_type": {
            split: {
                question_type: counts_by_split_and_type[split][question_type]
                for question_type in QUESTION_FAMILIES
            }
            for split in ("train", "val", "test")
        },
        "by_split_and_type_and_answer": {
            split: {
                question_type: dict(
                    sorted(answers_by_split_and_type[split][question_type].items())
                )
                for question_type in QUESTION_FAMILIES
            }
            for split in ("train", "val", "test")
        },
        "by_split_and_type_and_template": {
            split: {
                question_type: dict(
                    sorted(templates_by_split_and_type[split][question_type].items())
                )
                for question_type in QUESTION_FAMILIES
            }
            for split in ("train", "val", "test")
        },
        "majority_answer_fraction": majority_answer_fraction,
        "difficulty_summary": {
            split: {
                question_type: (
                    {
                        _DIFFICULTY_BY_QUESTION_TYPE[question_type]:
                        counts_by_split_and_type[split][question_type]
                    }
                    if counts_by_split_and_type[split][question_type]
                    else {}
                )
                for question_type in QUESTION_FAMILIES
            }
            for split in ("train", "val", "test")
        },
        "warnings": warnings,
    }


def compare_regenerated_questions(
    persisted: Sequence[Mapping[str, object]],
    regenerated: Sequence[Mapping[str, object]],
) -> None:
    def indexed(
        rows: Sequence[Mapping[str, object]],
        source: str,
    ) -> dict[str, dict[str, object]]:
        result = {}
        for source_row in rows:
            row = dict(source_row)
            question_id = str(row.get("question_id", ""))
            if not question_id or question_id in result:
                raise QuestionAuditError(
                    f"{source} questions contain an empty or duplicate question_id"
                )
            answer_space = row.get("answer_space")
            if hasattr(answer_space, "tolist"):
                answer_space = answer_space.tolist()
            if isinstance(answer_space, (list, tuple)):
                row["answer_space"] = tuple(answer_space)
            row["evidence"] = _decoded_evidence(row.get("evidence"))
            result[question_id] = row
        return result

    persisted_by_id = indexed(persisted, "persisted")
    regenerated_by_id = indexed(regenerated, "regenerated")
    persisted_ids = set(persisted_by_id)
    regenerated_ids = set(regenerated_by_id)
    if persisted_ids != regenerated_ids:
        missing = sorted(regenerated_ids - persisted_ids)
        extra = sorted(persisted_ids - regenerated_ids)
        raise QuestionAuditError(
            "persisted questions differ from regenerated questions: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for question_id in sorted(persisted_ids):
        persisted_row = persisted_by_id[question_id]
        regenerated_row = regenerated_by_id[question_id]
        for field in _PERSISTED_SEMANTIC_FIELDS:
            if persisted_row.get(field) != regenerated_row.get(field):
                raise QuestionAuditError(
                    f"persisted question {question_id!r} differs from regenerated "
                    f"question in {field}"
                )


def manual_review_sample(
    rows_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    per_split: int = 100,
    seed: int = 20260615,
) -> list[dict[str, object]]:
    if per_split < 0:
        raise ValueError("per_split must be non-negative")
    limit = min(per_split, 100)
    normalized = _normalized_rows(rows_by_split)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["split"])].append(row)

    sample = []
    for split in ("train", "val", "test"):
        ordered = sorted(
            grouped[split],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{split}\0{row.get('question_id', '')}".encode()
            ).digest(),
        )
        sample.extend(ordered[:limit])
    return sample
