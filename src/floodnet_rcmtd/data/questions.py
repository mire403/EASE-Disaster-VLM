from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CLASS_NAMES = (
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

QUESTION_FAMILIES = (
    "presence",
    "damage_state",
    "area_comparison",
    "spatial_adjacency",
    "dominant_class",
    "logical_conjunction",
)

_YES_NO = ("yes", "no")
_DAMAGE_STATES = ("damage_present", "no_damage")
_AREA_ANSWERS = ("road_flooded", "road_non_flooded", "equal")

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "presence": (
        "Is {class_name} present in the scene?",
        "Does the segmentation contain {class_name}?",
        "Can {class_name} be found anywhere in this image?",
    ),
    "damage_state": (
        "Is flood damage visible on a building or road?",
        "Does the scene contain a flooded building or a flooded road?",
        "Is either built infrastructure class marked as flooded?",
    ),
    "area_comparison": (
        "Which covers more area: flooded road or non-flooded road?",
        "Is the larger road region flooded road or non-flooded road?",
        "Compare the road classes: which one occupies more pixels?",
    ),
    "spatial_adjacency": (
        "Is flooded road adjacent to water?",
        "Do the flooded-road and water regions lie within the adjacency radius?",
        "Are water and flooded road spatially neighboring?",
    ),
    "dominant_class": (
        "Which semantic class occupies the largest area?",
        "What is the dominant segmentation class in this image?",
        "Which class has the greatest pixel coverage?",
    ),
    "logical_conjunction": (
        "Is flooded road present and adjacent to water?",
        "Are both conditions true: flooded road exists and it neighbors water?",
        "Does the scene contain flooded road with nearby water?",
    ),
}


@dataclass(frozen=True)
class Evidence:
    """Mask evidence using a Chebyshev-radius square adjacency neighborhood."""

    image_shape: tuple[int, int]
    presence: dict[str, bool]
    area_pixels: dict[str, int]
    area_ratio: dict[str, float]
    ratio_categories: dict[str, str]
    adjacency: dict[str, bool]
    dominant_class: str
    adjacency_radius: int


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    image_id: str
    split: str
    question_type: str
    template_id: str
    question: str
    answer: str
    answer_space: tuple[str, ...]
    evidence: dict[str, object]


def ratio_category(
    ratio: float,
    low_max: float = 0.05,
    medium_max: float = 0.20,
) -> str:
    if not 0 <= low_max <= medium_max <= 1:
        raise ValueError(
            "ratio thresholds must satisfy 0 <= low_max <= medium_max <= 1"
        )
    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    if ratio == 0:
        return "none"
    if ratio <= low_max:
        return "low"
    if ratio <= medium_max:
        return "medium"
    return "high"


def _regions_are_adjacent(
    first: np.ndarray[Any, np.dtype[np.bool_]],
    second: np.ndarray[Any, np.dtype[np.bool_]],
    radius: int,
) -> bool:
    if radius == 0 or not first.any() or not second.any():
        return False
    padded = np.pad(second, radius, mode="constant", constant_values=False)
    height, width = first.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            neighborhood = padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            if np.any(first & neighborhood):
                return True
    return False


def build_evidence(
    mask: np.ndarray,
    adjacency_radius: int = 1,
    ratio_thresholds: Mapping[str, float] | None = None,
    *,
    low_max: float = 0.05,
    medium_max: float = 0.20,
) -> Evidence:
    """Extract mask evidence; adjacency uses a square Chebyshev neighborhood."""

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if not isinstance(adjacency_radius, int) or adjacency_radius < 0:
        raise ValueError("adjacency_radius must be a non-negative integer")
    if ratio_thresholds is not None:
        low_max = float(ratio_thresholds.get("low_max", low_max))
        medium_max = float(ratio_thresholds.get("medium_max", medium_max))
    ratio_category(0, low_max=low_max, medium_max=medium_max)

    unknown_labels = []
    for label in np.unique(values):
        try:
            integer_label = int(label)
        except (TypeError, ValueError, OverflowError):
            unknown_labels.append(str(label))
            continue
        if label != integer_label or integer_label not in range(len(CLASS_NAMES)):
            unknown_labels.append(str(label))
    if unknown_labels:
        raise ValueError(f"mask contains unknown labels: {sorted(unknown_labels)}")

    total_pixels = int(values.size)
    counts = np.bincount(values.reshape(-1).astype(np.int64), minlength=len(CLASS_NAMES))
    area_pixels = {
        class_name: int(counts[class_id])
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    area_ratio = {
        class_name: (pixel_count / total_pixels if total_pixels else 0.0)
        for class_name, pixel_count in area_pixels.items()
    }
    presence = {
        class_name: pixel_count > 0 for class_name, pixel_count in area_pixels.items()
    }
    ratio_categories = {
        class_name: ratio_category(
            ratio,
            low_max=low_max,
            medium_max=medium_max,
        )
        for class_name, ratio in area_ratio.items()
    }

    class_regions = [values == class_id for class_id in range(len(CLASS_NAMES))]
    adjacency: dict[str, bool] = {}
    for first_id, first_name in enumerate(CLASS_NAMES):
        for second_id in range(first_id + 1, len(CLASS_NAMES)):
            second_name = CLASS_NAMES[second_id]
            adjacency[f"{first_name}__{second_name}"] = _regions_are_adjacent(
                class_regions[first_id],
                class_regions[second_id],
                adjacency_radius,
            )

    dominant_class = CLASS_NAMES[int(np.argmax(counts))]
    return Evidence(
        image_shape=(int(values.shape[0]), int(values.shape[1])),
        presence=presence,
        area_pixels=area_pixels,
        area_ratio=area_ratio,
        ratio_categories=ratio_categories,
        adjacency=adjacency,
        dominant_class=dominant_class,
        adjacency_radius=adjacency_radius,
    )


def recompute_answer(question_type: str, evidence: Mapping[str, object]) -> str:
    if question_type == "presence":
        return "yes" if bool(evidence["present"]) else "no"
    if question_type == "damage_state":
        damage_present = bool(evidence["building_flooded_present"]) or bool(
            evidence["road_flooded_present"]
        )
        return "damage_present" if damage_present else "no_damage"
    if question_type == "area_comparison":
        first_pixels = int(evidence["first_area_pixels"])
        second_pixels = int(evidence["second_area_pixels"])
        if first_pixels > second_pixels:
            return str(evidence["first_class"])
        if second_pixels > first_pixels:
            return str(evidence["second_class"])
        return "equal"
    if question_type == "spatial_adjacency":
        return "yes" if bool(evidence["adjacent"]) else "no"
    if question_type == "dominant_class":
        return str(evidence["dominant_class"])
    if question_type == "logical_conjunction":
        conditions_met = bool(evidence["road_flooded_present"]) and bool(
            evidence["road_flooded_water_adjacent"]
        )
        return "yes" if conditions_met else "no"
    raise ValueError(f"unknown question type: {question_type!r}")


def _stable_index(
    image_id: str,
    seed: int,
    family: str,
    purpose: str,
    size: int,
) -> int:
    payload = f"{seed}\0{image_id}\0{family}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % size


def _question_id(image_id: str, split: str, family: str, template_id: str) -> str:
    payload = f"{image_id}\0{split}\0{family}\0{template_id}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"{family}-{digest}"


def _template(image_id: str, seed: int, family: str) -> tuple[str, str]:
    index = _stable_index(image_id, seed, family, "template", len(_TEMPLATES[family]))
    return f"{family}-{index + 1}", _TEMPLATES[family][index]


def generate_questions(
    image_id: str,
    evidence: Evidence,
    seed: int = 20260615,
    split: str = "train",
) -> tuple[QuestionRecord, ...]:
    """Generate one question per family for an image.

    The audit checks answer balance across the dataset while keeping one
    question per family for each image.
    """

    records = []
    for family in QUESTION_FAMILIES:
        template_id, template = _template(image_id, seed, family)

        if family == "presence":
            candidates = CLASS_NAMES[1:]
            class_name = candidates[
                _stable_index(image_id, seed, family, "class", len(candidates))
            ]
            present = evidence.presence[class_name]
            answer = "yes" if present else "no"
            question = template.format(class_name=class_name)
            answer_space = _YES_NO
            compact_evidence: dict[str, object] = {
                "class_name": class_name,
                "present": present,
                "area_ratio": evidence.area_ratio[class_name],
                "ratio_category": evidence.ratio_categories[class_name],
                "expected_answer": answer,
            }
        elif family == "damage_state":
            building_flooded = evidence.presence["building_flooded"]
            road_flooded = evidence.presence["road_flooded"]
            answer = (
                "damage_present" if building_flooded or road_flooded else "no_damage"
            )
            question = template
            answer_space = _DAMAGE_STATES
            compact_evidence = {
                "building_flooded_present": building_flooded,
                "road_flooded_present": road_flooded,
                "expected_answer": answer,
            }
        elif family == "area_comparison":
            flooded_pixels = evidence.area_pixels["road_flooded"]
            non_flooded_pixels = evidence.area_pixels["road_non_flooded"]
            if flooded_pixels > non_flooded_pixels:
                answer = "road_flooded"
            elif non_flooded_pixels > flooded_pixels:
                answer = "road_non_flooded"
            else:
                answer = "equal"
            question = template
            answer_space = _AREA_ANSWERS
            compact_evidence = {
                "first_class": "road_flooded",
                "first_area_pixels": flooded_pixels,
                "first_ratio_category": evidence.ratio_categories["road_flooded"],
                "second_class": "road_non_flooded",
                "second_area_pixels": non_flooded_pixels,
                "second_ratio_category": evidence.ratio_categories["road_non_flooded"],
                "expected_answer": answer,
            }
        elif family == "spatial_adjacency":
            adjacent = evidence.adjacency["road_flooded__water"]
            answer = "yes" if adjacent else "no"
            question = template
            answer_space = _YES_NO
            compact_evidence = {
                "first_class": "road_flooded",
                "second_class": "water",
                "adjacent": adjacent,
                "adjacency_radius": evidence.adjacency_radius,
                "expected_answer": answer,
            }
        elif family == "dominant_class":
            answer = evidence.dominant_class
            question = template
            answer_space = CLASS_NAMES
            compact_evidence = {
                "dominant_class": evidence.dominant_class,
                "dominant_area_pixels": evidence.area_pixels[evidence.dominant_class],
                "expected_answer": answer,
            }
        else:
            road_present = evidence.presence["road_flooded"]
            adjacent = evidence.adjacency["road_flooded__water"]
            answer = "yes" if road_present and adjacent else "no"
            question = template
            answer_space = _YES_NO
            compact_evidence = {
                "road_flooded_present": road_present,
                "road_flooded_water_adjacent": adjacent,
                "adjacency_radius": evidence.adjacency_radius,
                "expected_answer": answer,
            }

        records.append(
            QuestionRecord(
                question_id=_question_id(image_id, split, family, template_id),
                image_id=str(image_id),
                split=split,
                question_type=family,
                template_id=template_id,
                question=question,
                answer=answer,
                answer_space=answer_space,
                evidence=compact_evidence,
            )
        )
    return tuple(records)
