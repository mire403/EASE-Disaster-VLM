import numpy as np
import pytest

from floodnet_rcmtd.data.questions import (
    CLASS_NAMES,
    QUESTION_FAMILIES,
    build_evidence,
    generate_questions,
    ratio_category,
    recompute_answer,
)


def test_build_evidence_reports_exact_counts_ratios_and_adjacency():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 3
    mask[2:4, 4:6] = 5

    evidence = build_evidence(mask)

    assert evidence.image_shape == (8, 8)
    assert evidence.presence["road_flooded"] is True
    assert evidence.area_pixels["road_flooded"] == 4
    assert evidence.area_ratio["road_flooded"] == 4 / 64
    assert evidence.adjacency["road_flooded__water"] is True
    assert evidence.dominant_class == "background"


def test_build_evidence_treats_diagonal_contact_as_adjacent():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 3
    mask[1, 1] = 5

    evidence = build_evidence(mask, adjacency_radius=1)

    assert evidence.adjacency["road_flooded__water"] is True


def test_build_evidence_rejects_non_2d_and_unknown_labels():
    with pytest.raises(ValueError, match="2D"):
        build_evidence(np.zeros((2, 2, 1), dtype=np.uint8))

    with pytest.raises(ValueError, match="unknown labels"):
        build_evidence(np.array([[0, 10]], dtype=np.uint8))


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, "none"),
        (0.05, "low"),
        (0.050001, "medium"),
        (0.20, "medium"),
        (0.200001, "high"),
    ],
)
def test_ratio_category_uses_fixed_boundaries(ratio, expected):
    assert ratio_category(ratio) == expected


def test_ratio_category_supports_custom_boundaries():
    assert ratio_category(0.10, low_max=0.10, medium_max=0.30) == "low"
    assert ratio_category(0.20, low_max=0.10, medium_max=0.30) == "medium"
    assert ratio_category(0.40, low_max=0.10, medium_max=0.30) == "high"


@pytest.mark.parametrize(
    ("low_max", "medium_max"),
    [(-0.01, 0.20), (0.30, 0.20), (0.05, 1.01)],
)
def test_ratio_category_rejects_invalid_boundaries(low_max, medium_max):
    with pytest.raises(ValueError, match="threshold"):
        ratio_category(0.10, low_max=low_max, medium_max=medium_max)


def test_build_evidence_uses_custom_ratio_thresholds():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, :] = 3

    evidence = build_evidence(
        mask,
        ratio_thresholds={"low_max": 0.10, "medium_max": 0.30},
    )

    assert evidence.ratio_categories["road_flooded"] == "low"


@pytest.mark.parametrize(
    ("question_type", "evidence", "expected"),
    [
        ("presence", {"present": True}, "yes"),
        (
            "damage_state",
            {
                "building_flooded_present": False,
                "road_flooded_present": True,
            },
            "damage_present",
        ),
        (
            "area_comparison",
            {
                "first_class": "road_flooded",
                "first_area_pixels": 3,
                "second_class": "road_non_flooded",
                "second_area_pixels": 7,
            },
            "road_non_flooded",
        ),
        ("spatial_adjacency", {"adjacent": False}, "no"),
        ("dominant_class", {"dominant_class": "water"}, "water"),
        (
            "logical_conjunction",
            {
                "road_flooded_present": True,
                "road_flooded_water_adjacent": True,
            },
            "yes",
        ),
    ],
)
def test_recompute_answer_uses_raw_evidence(question_type, evidence, expected):
    assert recompute_answer(question_type, evidence) == expected


def test_generate_questions_is_deterministic_and_covers_all_families():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:2, :] = 1
    mask[2:4, :] = 2
    mask[4, 0:4] = 3
    mask[4, 4:8] = 5
    mask[5, :] = 4
    mask[6, 0:4] = 6
    mask[6, 4:8] = 7
    mask[7, 0:4] = 8
    mask[7, 4:8] = 9
    evidence = build_evidence(mask)

    first = generate_questions("image-1", evidence, seed=17, split="val")
    second = generate_questions("image-1", evidence, seed=17, split="val")

    assert first == second
    assert {record.question_type for record in first} == {
        "presence",
        "damage_state",
        "area_comparison",
        "spatial_adjacency",
        "dominant_class",
        "logical_conjunction",
    }
    assert all(record.evidence for record in first)
    assert all(record.image_id == "image-1" and record.split == "val" for record in first)
    assert len({record.question_id for record in first}) == len(first)
    assert len(first) == len(QUESTION_FAMILIES)


def test_question_evidence_is_compact_and_templates_do_not_depend_on_answers():
    dry_mask = np.zeros((8, 8), dtype=np.uint8)
    wet_mask = dry_mask.copy()
    wet_mask[0:4, :] = 3

    dry = generate_questions("same-image", build_evidence(dry_mask), seed=23)
    wet = generate_questions("same-image", build_evidence(wet_mask), seed=23)

    assert [record.template_id for record in dry] == [
        record.template_id for record in wet
    ]
    for record in wet:
        assert not any(isinstance(value, np.ndarray) for value in record.evidence.values())
        assert set(record.answer_space)


def test_class_mapping_uses_floodnet_domain_names():
    assert CLASS_NAMES == (
        "background",
        "building_flooded",
        "building_non_flooded",
        "road_flooded",
        "road_non_flooded",
        "water",
        "tree",
        "vehicle",
        "pool",
        "grass",
    )
