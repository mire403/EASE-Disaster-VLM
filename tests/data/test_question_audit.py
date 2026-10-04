from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from floodnet_rcmtd.cli import build_parser, main
from floodnet_rcmtd.data.audit import (
    QuestionAuditError,
    audit_questions,
    compare_regenerated_questions,
    manual_review_sample,
)
from floodnet_rcmtd.data.questions import (
    QUESTION_FAMILIES,
    build_evidence,
    generate_questions,
)


def _evidence(family: str, answer: str) -> dict[str, object]:
    if family == "presence":
        evidence = {"present": answer == "yes"}
    elif family == "damage_state":
        evidence = {
            "building_flooded_present": answer == "damage_present",
            "road_flooded_present": False,
        }
    elif family == "area_comparison":
        evidence = {
            "first_class": "road_flooded",
            "first_area_pixels": 2 if answer == "road_flooded" else 1,
            "second_class": "road_non_flooded",
            "second_area_pixels": 1 if answer == "road_flooded" else 2,
        }
    elif family == "spatial_adjacency":
        evidence = {"adjacent": answer == "yes"}
    elif family == "dominant_class":
        evidence = {"dominant_class": answer}
    else:
        evidence = {
            "road_flooded_present": answer == "yes",
            "road_flooded_water_adjacent": answer == "yes",
        }
    evidence["expected_answer"] = answer
    return evidence


def _valid_rows() -> dict[str, list[dict[str, object]]]:
    rows = {"train": [], "val": [], "test": []}
    for family in QUESTION_FAMILIES:
        val_answer = {
            "damage_state": "damage_present",
            "area_comparison": "road_flooded",
            "dominant_class": "water",
        }.get(family, "yes")
        test_answer = {
            "damage_state": "no_damage",
            "area_comparison": "road_non_flooded",
            "dominant_class": "background",
        }.get(family, "no")
        rows["val"].append(
            {
                "question_id": f"val-{family}",
                "image_id": f"val-{family}",
                "split": "val",
                "question_type": family,
                "template_id": f"{family}-template",
                "question": f"Validation question for {family}?",
                "answer": val_answer,
                "answer_space": ["yes", "no"],
                "evidence": _evidence(family, val_answer),
            }
        )
        rows["test"].append(
            {
                "question_id": f"test-{family}",
                "image_id": f"test-{family}",
                "split": "test",
                "question_type": family,
                "template_id": f"{family}-template",
                "question": f"Test question for {family}?",
                "answer": test_answer,
                "answer_space": ["yes", "no"],
                "evidence": _evidence(family, test_answer),
            }
        )
    return rows


def test_audit_accepts_valid_small_rows_and_reports_distributions():
    result = audit_questions(
        _valid_rows(),
        min_examples_per_type=1,
        max_majority_answer_fraction=1.0,
    )

    assert result["total_questions"] == 2 * len(QUESTION_FAMILIES)
    assert result["by_split"]["val"] == len(QUESTION_FAMILIES)
    assert result["by_split_and_type"]["test"]["presence"] == 1
    assert result["by_split_and_type_and_answer"]["val"]["presence"] == {"yes": 1}
    assert result["by_split_and_type_and_template"]["test"]["presence"] == {
        "presence-template": 1
    }
    assert result["majority_answer_fraction"]["val"]["presence"] == 1.0
    assert result["difficulty_summary"]["val"]["presence"] == {"binary": 1}
    assert result["difficulty_summary"]["test"]["area_comparison"] == {
        "compositional": 1
    }
    assert result["difficulty_summary"]["val"]["dominant_class"] == {
        "multiclass": 1
    }
    assert result["warnings"] == []


def test_audit_warns_when_majority_answer_fraction_exceeds_threshold():
    result = audit_questions(
        _valid_rows(),
        min_examples_per_type=1,
        max_majority_answer_fraction=0.95,
    )

    assert {
        (warning["split"], warning["question_type"])
        for warning in result["warnings"]
    } == {
        (split, question_type)
        for split in ("val", "test")
        for question_type in QUESTION_FAMILIES
    }
    assert all(
        warning["majority_answer_fraction"] == 1.0
        and warning["max_majority_answer_fraction"] == 0.95
        for warning in result["warnings"]
    )


def test_audit_rejects_duplicate_question_ids_across_splits():
    rows = _valid_rows()
    rows["test"][0]["question_id"] = rows["val"][0]["question_id"]

    with pytest.raises(QuestionAuditError, match="across splits"):
        audit_questions(rows, min_examples_per_type=1)


def test_audit_rejects_too_few_validation_or_test_examples_per_type():
    rows = _valid_rows()
    rows["val"] = [
        row for row in rows["val"] if row["question_type"] != "presence"
    ]

    with pytest.raises(QuestionAuditError, match="presence"):
        audit_questions(rows, min_examples_per_type=1)


def test_audit_rejects_templates_with_only_one_answer_when_repeated():
    rows = _valid_rows()
    rows["test"][0]["answer"] = "yes"
    rows["test"][0]["evidence"]["present"] = True
    rows["test"][0]["evidence"]["expected_answer"] = "yes"

    with pytest.raises(QuestionAuditError, match="one answer"):
        audit_questions(rows, min_examples_per_type=1)


def test_audit_rejects_empty_evidence():
    rows = _valid_rows()
    rows["val"][0]["evidence"] = {}

    with pytest.raises(QuestionAuditError, match="empty evidence"):
        audit_questions(rows, min_examples_per_type=1)


def test_audit_ignores_malicious_expected_answer_and_recomputes_from_predicates():
    rows = _valid_rows()
    rows["val"][0]["answer"] = "no"
    rows["val"][0]["evidence"]["expected_answer"] = "no"

    with pytest.raises(QuestionAuditError, match="answer mismatch"):
        audit_questions(rows, min_examples_per_type=1)


def _complete_question_row() -> dict[str, object]:
    return {
        "question_id": "q1",
        "image_id": "image-1",
        "split": "val",
        "question_type": "presence",
        "template_id": "presence-1",
        "question": "Is water present?",
        "answer": "yes",
        "answer_space": ["yes", "no"],
        "evidence": {"present": True, "expected_answer": "yes"},
    }


def test_compare_regenerated_questions_normalizes_answer_space_and_evidence():
    persisted = _complete_question_row()
    regenerated = deepcopy(persisted)
    persisted["answer_space"] = ("yes", "no")
    persisted["evidence"] = json.dumps(persisted["evidence"])

    compare_regenerated_questions([persisted], [regenerated])


@pytest.mark.parametrize(
    ("field", "corrupt_value"),
    [
        ("question_id", "q-corrupt"),
        ("image_id", "image-corrupt"),
        ("split", "test"),
        ("question_type", "damage_state"),
        ("template_id", "presence-corrupt"),
        ("question", "Corrupted question?"),
        ("answer", "no"),
        ("answer_space", ["yes", "no", "unknown"]),
        ("evidence", {"present": False, "expected_answer": "yes"}),
    ],
)
def test_compare_regenerated_questions_rejects_corrupted_semantic_fields(
    field,
    corrupt_value,
):
    persisted = _complete_question_row()
    regenerated = deepcopy(persisted)
    persisted[field] = corrupt_value

    with pytest.raises(QuestionAuditError, match="regenerated"):
        compare_regenerated_questions([persisted], [regenerated])


def test_compare_regenerated_questions_rejects_missing_or_extra_records():
    persisted = [_complete_question_row()]
    extra = _complete_question_row()
    extra["question_id"] = "q2"

    for regenerated in ([], persisted + [extra]):
        with pytest.raises(QuestionAuditError, match="regenerated"):
            compare_regenerated_questions(persisted, regenerated)


def test_manual_review_sample_is_deterministic_and_capped_per_split():
    rows = _valid_rows()

    first = manual_review_sample(rows, per_split=2, seed=19)
    second = manual_review_sample(deepcopy(rows), per_split=2, seed=19)

    assert first == second
    assert sum(row["split"] == "val" for row in first) == 2
    assert sum(row["split"] == "test" for row in first) == 2


def test_manual_review_sample_hard_caps_each_split_at_100():
    rows = {"train": [], "val": [], "test": []}
    for split in rows:
        rows[split] = [
            {
                "question_id": f"{split}-{index}",
                "split": split,
                "evidence": {"present": True},
            }
            for index in range(120)
        ]

    sample = manual_review_sample(rows, per_split=150)

    assert all(
        sum(row["split"] == split for row in sample) == 100
        for split in ("train", "val", "test")
    )


@pytest.mark.parametrize("command", ["build-questions", "audit-questions"])
def test_cli_parser_exposes_question_commands(command, tmp_path):
    args = build_parser().parse_args([command, "--config", str(tmp_path / "data.yaml")])

    assert args.command == command


def _mask_for_answer_profile(wet: bool) -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.uint8)
    if not wet:
        return mask
    mask[:, :] = 5
    mask[0, :8] = 3
    mask[1, 0] = 4
    for class_id in (1, 2, 6, 7, 8, 9):
        mask[1, class_id] = class_id
    return mask


def _ids_covering_templates(seed: int, prefix: str) -> list[str]:
    required = {
        (family, f"{family}-{template_index}")
        for family in QUESTION_FAMILIES
        for template_index in range(1, 4)
    }
    selected = []
    covered = set()
    evidence = build_evidence(_mask_for_answer_profile(wet=False))
    for index in range(500):
        image_id = f"{prefix}-{index}"
        records = generate_questions(image_id, evidence, seed=seed)
        contribution = {
            (record.question_type, record.template_id) for record in records
        } - covered
        if contribution:
            selected.append(image_id)
            covered.update(contribution)
        if covered == required:
            return selected
    raise AssertionError("failed to find deterministic template coverage")


def test_question_cli_roundtrip_outputs_and_detects_persisted_corruption(
    tiny_floodnet_root: Path,
    tmp_path: Path,
):
    output_root = tmp_path / "derived"
    config_path = tmp_path / "config.yaml"
    seed = 20260615
    config_path.write_text(
        "\n".join(
            [
                f"data_root: {tiny_floodnet_root}",
                f"output_root: {output_root}",
                "sequence_gap_seconds: 30",
                "gap_sensitivity_seconds: [10, 30, 60]",
                "split_ratios: [0.6, 0.2, 0.2]",
                "class_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
                f"seed: {seed}",
                "adjacency_radius: 1",
                "min_examples_per_type: 0",
                "manual_review_per_split: 2",
                "max_majority_answer_fraction: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    main(["build-split", "--config", str(config_path)])

    rows = []
    masks_root = tmp_path / "fixture-masks"
    masks_root.mkdir()
    for split in ("train", "val", "test"):
        for wet in (False, True):
            profile = "wet" if wet else "dry"
            for image_id in _ids_covering_templates(seed, f"{split}-{profile}"):
                mask_path = masks_root / f"{image_id}.png"
                Image.fromarray(_mask_for_answer_profile(wet)).save(mask_path)
                rows.append(
                    {
                        "image_id": image_id,
                        "split": split,
                        "mask_path": str(mask_path),
                    }
                )
    images_path = output_root / "splits" / "images.parquet"
    pd.DataFrame(rows).to_parquet(images_path, index=False)

    main(["build-questions", "--config", str(config_path)])
    main(["audit-questions", "--config", str(config_path)])

    questions_root = output_root / "questions"
    assert all(
        (questions_root / f"{split}.parquet").exists()
        for split in ("train", "val", "test")
    )
    distribution_path = output_root / "audits" / "question_distribution.json"
    sample_path = output_root / "audits" / "manual_review_sample.csv"
    distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
    assert distribution["warnings"] == []
    assert distribution["by_split_and_type_and_answer"]["val"]["presence"]
    assert sample_path.exists()
    assert len(pd.read_csv(sample_path)) == 6

    val_path = questions_root / "val.parquet"
    val_rows = pd.read_parquet(val_path)
    val_rows.loc[0, "question"] = "Persisted corruption?"
    val_rows.to_parquet(val_path, index=False)

    with pytest.raises(QuestionAuditError, match="question"):
        main(["audit-questions", "--config", str(config_path)])
