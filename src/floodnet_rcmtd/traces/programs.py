from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from floodnet_rcmtd.data.questions import CLASS_NAMES, Evidence, QuestionRecord
from floodnet_rcmtd.traces.schema import TraceRecord, TraceStep


@dataclass(frozen=True)
class ToolResult:
    output: object
    confidence: float
    evidence_ids: tuple[str, ...]
    latency_ms: float
    failure: str | None = None


class ToolSet(Protocol):
    def presence(self, class_name: str) -> ToolResult: ...

    def area_pixels(self, class_name: str) -> ToolResult: ...

    def adjacency(self, first_class: str, second_class: str) -> ToolResult: ...

    def dominant_class(self) -> ToolResult: ...


class EvidenceToolSet:
    def __init__(self, evidence: Evidence) -> None:
        self._evidence = evidence

    def presence(self, class_name: str) -> ToolResult:
        return ToolResult(
            output=self._evidence.presence[class_name],
            confidence=1.0,
            evidence_ids=(f"presence:{class_name}",),
            latency_ms=0.0,
        )

    def area_pixels(self, class_name: str) -> ToolResult:
        return ToolResult(
            output=self._evidence.area_pixels[class_name],
            confidence=1.0,
            evidence_ids=(f"area_pixels:{class_name}",),
            latency_ms=0.0,
        )

    def adjacency(self, first_class: str, second_class: str) -> ToolResult:
        first_index = CLASS_NAMES.index(first_class)
        second_index = CLASS_NAMES.index(second_class)
        if first_index > second_index:
            first_class, second_class = second_class, first_class
        pair = f"{first_class}__{second_class}"
        return ToolResult(
            output=self._evidence.adjacency[pair],
            confidence=1.0,
            evidence_ids=(f"adjacency:{pair}",),
            latency_ms=0.0,
        )

    def dominant_class(self) -> ToolResult:
        return ToolResult(
            output=self._evidence.dominant_class,
            confidence=1.0,
            evidence_ids=("dominant_class",),
            latency_ms=0.0,
        )


PROGRAM_VARIANTS: dict[str, tuple[str, ...]] = {
    "presence": ("direct", "area_check", "verified"),
    "damage_state": ("direct", "roads_first", "buildings_first"),
    "area_comparison": ("direct", "reverse_order", "ratio_route"),
    "spatial_adjacency": (
        "direct",
        "presence_then_relation",
        "relation_then_presence",
    ),
    "dominant_class": ("direct", "area_scan", "presence_area_scan"),
    "logical_conjunction": ("direct", "presence_first", "relation_first"),
}


def _step(
    tool: str,
    arguments: dict[str, object],
    output_type: str,
    result: ToolResult,
) -> TraceStep:
    return TraceStep(
        tool=tool,
        arguments=arguments,
        output_type=output_type,
        output=result.output,
        confidence=result.confidence,
        evidence_ids=list(result.evidence_ids),
        latency_ms=result.latency_ms,
        failure=result.failure,
    )


def _presence(steps: list[TraceStep], tools: ToolSet, class_name: str) -> ToolResult:
    result = tools.presence(class_name)
    steps.append(_step("presence", {"class_name": class_name}, "bool", result))
    return result


def _area(steps: list[TraceStep], tools: ToolSet, class_name: str) -> ToolResult:
    result = tools.area_pixels(class_name)
    steps.append(_step("area_pixels", {"class_name": class_name}, "int", result))
    return result


def _adjacency(
    steps: list[TraceStep],
    tools: ToolSet,
    first_class: str,
    second_class: str,
) -> ToolResult:
    result = tools.adjacency(first_class, second_class)
    steps.append(
        _step(
            "adjacency",
            {"first_class": first_class, "second_class": second_class},
            "bool",
            result,
        )
    )
    return result


def _dominant(steps: list[TraceStep], tools: ToolSet) -> ToolResult:
    result = tools.dominant_class()
    steps.append(_step("dominant_class", {}, "class_name", result))
    return result


def _mark_verification_failure(
    steps: list[TraceStep],
    tool: str,
    failure: str,
    *,
    class_name: str | None = None,
) -> None:
    for step in reversed(steps):
        if step.tool != tool:
            continue
        if class_name is not None and step.arguments.get("class_name") != class_name:
            continue
        step.failure = failure
        return
    raise RuntimeError(f"verification step {tool!r} was not recorded")


def _area_answer(first_class: str, first: ToolResult, second_class: str, second: ToolResult) -> str:
    if first.failure is not None or second.failure is not None:
        return "equal"
    first_area = int(first.output)
    second_area = int(second.output)
    if first_area > second_area:
        return first_class
    if second_area > first_area:
        return second_class
    return "equal"


def _execute_family(
    question: QuestionRecord,
    tools: ToolSet,
    variant: str,
    steps: list[TraceStep],
) -> str:
    evidence = question.evidence
    family = question.question_type

    if family == "presence":
        class_name = str(evidence["class_name"])
        if variant == "direct":
            return "yes" if bool(_presence(steps, tools, class_name).output) else "no"
        if variant == "area_check":
            area = _area(steps, tools, class_name)
            return "yes" if area.failure is None and int(area.output) > 0 else "no"
        if variant == "verified":
            present = _presence(steps, tools, class_name)
            area = _area(steps, tools, class_name)
            if (
                present.failure is None
                and area.failure is None
                and bool(present.output) != (int(area.output) > 0)
            ):
                _mark_verification_failure(
                    steps,
                    "area_pixels",
                    "presence/area contradiction",
                    class_name=class_name,
                )
            return "yes" if bool(present.output) else "no"

    elif family == "damage_state":
        if variant == "direct":
            building = _presence(steps, tools, "building_flooded")
            if building.failure is None and bool(building.output):
                return "damage_present"
            road = _presence(steps, tools, "road_flooded")
        elif variant == "roads_first":
            road = _presence(steps, tools, "road_flooded")
            building = _presence(steps, tools, "building_flooded")
        elif variant == "buildings_first":
            building = _presence(steps, tools, "building_flooded")
            road = _presence(steps, tools, "road_flooded")
        else:
            raise ValueError(f"unknown program variant {variant!r} for {family!r}")
        damaged = bool(building.output) or bool(road.output)
        return "damage_present" if damaged else "no_damage"

    elif family == "area_comparison":
        first_class = str(evidence["first_class"])
        second_class = str(evidence["second_class"])
        if variant == "reverse_order":
            second = _area(steps, tools, second_class)
            first = _area(steps, tools, first_class)
        elif variant == "direct":
            first = _area(steps, tools, first_class)
            second = _area(steps, tools, second_class)
        elif variant == "ratio_route":
            first_present = _presence(steps, tools, first_class)
            second_present = _presence(steps, tools, second_class)
            first = _area(steps, tools, first_class)
            second = _area(steps, tools, second_class)
            for class_name, present, area in (
                (first_class, first_present, first),
                (second_class, second_present, second),
            ):
                if (
                    present.failure is None
                    and area.failure is None
                    and bool(present.output) != (int(area.output) > 0)
                ):
                    _mark_verification_failure(
                        steps,
                        "area_pixels",
                        "inconsistent evidence",
                        class_name=class_name,
                    )
        else:
            raise ValueError(f"unknown program variant {variant!r} for {family!r}")
        return _area_answer(first_class, first, second_class, second)

    elif family == "spatial_adjacency":
        first_class = str(evidence["first_class"])
        second_class = str(evidence["second_class"])
        if variant == "direct":
            adjacent = _adjacency(steps, tools, first_class, second_class)
        elif variant == "presence_then_relation":
            first_present = _presence(steps, tools, first_class)
            second_present = _presence(steps, tools, second_class)
            adjacent = _adjacency(steps, tools, first_class, second_class)
        elif variant == "relation_then_presence":
            adjacent = _adjacency(steps, tools, first_class, second_class)
            first_present = _presence(steps, tools, first_class)
            second_present = _presence(steps, tools, second_class)
        else:
            raise ValueError(f"unknown program variant {variant!r} for {family!r}")
        if (
            variant != "direct"
            and adjacent.failure is None
            and first_present.failure is None
            and second_present.failure is None
            and bool(adjacent.output)
            and (not bool(first_present.output) or not bool(second_present.output))
        ):
            _mark_verification_failure(
                steps,
                "adjacency",
                "adjacency/presence contradiction",
            )
        return "yes" if bool(adjacent.output) else "no"

    elif family == "dominant_class":
        if variant == "direct":
            return str(_dominant(steps, tools).output)
        if variant == "area_scan":
            areas = {class_name: _area(steps, tools, class_name) for class_name in CLASS_NAMES}
            return max(
                CLASS_NAMES,
                key=lambda name: int(areas[name].output) if areas[name].failure is None else 0,
            )
        if variant == "presence_area_scan":
            presences = {
                class_name: _presence(steps, tools, class_name) for class_name in CLASS_NAMES
            }
            areas = {class_name: _area(steps, tools, class_name) for class_name in CLASS_NAMES}
            for class_name in CLASS_NAMES:
                present = presences[class_name]
                area = areas[class_name]
                if (
                    present.failure is None
                    and area.failure is None
                    and bool(present.output) != (int(area.output) > 0)
                ):
                    _mark_verification_failure(
                        steps,
                        "area_pixels",
                        "presence/area contradiction",
                        class_name=class_name,
                    )
                    break
            return max(
                CLASS_NAMES,
                key=lambda name: int(areas[name].output) if areas[name].failure is None else 0,
            )

    elif family == "logical_conjunction":
        first_class = "road_flooded"
        second_class = "water"
        if variant == "direct":
            area = _area(steps, tools, first_class)
            adjacent = _adjacency(steps, tools, first_class, second_class)
            present_output = area.failure is None and int(area.output) > 0
        elif variant == "presence_first":
            present = _presence(steps, tools, first_class)
            adjacent = _adjacency(steps, tools, first_class, second_class)
            present_output = bool(present.output)
        elif variant == "relation_first":
            adjacent = _adjacency(steps, tools, first_class, second_class)
            present = _presence(steps, tools, first_class)
            present_output = bool(present.output)
        else:
            raise ValueError(f"unknown program variant {variant!r} for {family!r}")
        return "yes" if present_output and bool(adjacent.output) else "no"

    raise ValueError(f"unknown program variant {variant!r} for {family!r}")


def _distribution(
    answer_space: tuple[str, ...],
    selected_answer: str,
    confidence: float,
    *,
    uniform: bool,
) -> dict[str, float]:
    if uniform:
        probability = 1.0 / len(answer_space)
        return {answer: probability for answer in answer_space}
    if len(answer_space) == 1:
        return {selected_answer: 1.0}

    uniform_probability = 1.0 / len(answer_space)
    if confidence <= uniform_probability:
        return {answer: uniform_probability for answer in answer_space}

    selected_probability = min(1.0, confidence)
    remainder = (1.0 - selected_probability) / (len(answer_space) - 1)
    return {
        answer: selected_probability if answer == selected_answer else remainder
        for answer in answer_space
    }


def _stable_trace_id(question_id: str, variant: str) -> str:
    digest = hashlib.sha256(f"{question_id}{variant}".encode()).digest()
    return f"trace-{digest.hex()[:20]}"


def _failure_answer(
    answer_space: tuple[str, ...],
    question_id: str,
    variant: str,
) -> str:
    digest = hashlib.sha256(f"{question_id}{variant}failure".encode()).digest()
    index = int.from_bytes(digest, "big") % len(answer_space)
    return answer_space[index]


def _generator_score(steps: list[TraceStep]) -> float:
    if not steps:
        return 0.0
    success_fraction = sum(step.failure is None for step in steps) / len(steps)
    mean_confidence = sum(step.confidence for step in steps) / len(steps)
    complexity_penalty = 1.0 / (1.0 + 0.05 * (len(steps) - 1))
    return success_fraction * mean_confidence * complexity_penalty


def execute_program(
    question: QuestionRecord,
    tools: ToolSet,
    program_variant: str,
) -> TraceRecord:
    variants = PROGRAM_VARIANTS.get(question.question_type)
    if variants is None or program_variant not in variants:
        raise ValueError(
            f"unknown program variant {program_variant!r} for {question.question_type!r}"
        )

    steps: list[TraceStep] = []
    selected_answer = _execute_family(question, tools, program_variant, steps)
    failures = [step.failure for step in steps if step.failure is not None]
    failure = failures[0] if failures else None
    if failure is not None:
        selected_answer = _failure_answer(
            question.answer_space,
            question.question_id,
            program_variant,
        )

    confidences = [step.confidence for step in steps]
    aggregate_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    distribution = _distribution(
        question.answer_space,
        selected_answer,
        aggregate_confidence,
        uniform=failure is not None,
    )
    evidence_ids = list(
        dict.fromkeys(evidence_id for step in steps for evidence_id in step.evidence_ids)
    )
    steps.append(
        TraceStep(
            tool="answer",
            arguments={},
            output_type="answer",
            output=selected_answer,
            confidence=distribution[selected_answer],
            evidence_ids=evidence_ids,
            latency_ms=0.0,
            failure=failure,
        )
    )
    trace_id = _stable_trace_id(
        question.question_id,
        program_variant,
    )
    return TraceRecord(
        trace_id=trace_id,
        question_id=question.question_id,
        source_family="evidence_rules",
        program_variant=program_variant,
        steps=steps,
        final_answer=selected_answer,
        answer_distribution=distribution,
        generator_score=_generator_score(steps[:-1]),
        total_latency_ms=sum(step.latency_ms for step in steps),
    )
