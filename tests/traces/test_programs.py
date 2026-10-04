from dataclasses import replace

import numpy as np
import pytest

from floodnet_rcmtd.data.questions import (
    CLASS_NAMES,
    Evidence,
    QuestionRecord,
    build_evidence,
    generate_questions,
)
from floodnet_rcmtd.traces.programs import (
    PROGRAM_VARIANTS,
    EvidenceToolSet,
    ToolResult,
    execute_program,
)


@pytest.fixture
def mask_evidence_fixture() -> Evidence:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:2, :] = 1
    mask[2, :] = 2
    mask[3:5, 0:3] = 3
    mask[3:5, 3:8] = 5
    mask[5, :] = 4
    mask[6, 0:4] = 6
    mask[6, 4:8] = 7
    mask[7, 0:4] = 8
    mask[7, 4:8] = 9
    return build_evidence(mask)


def _expected_answer(question: QuestionRecord, evidence: Evidence) -> str:
    if question.question_type == "presence":
        return "yes" if evidence.presence[str(question.evidence["class_name"])] else "no"
    if question.question_type == "damage_state":
        damaged = evidence.presence["building_flooded"] or evidence.presence["road_flooded"]
        return "damage_present" if damaged else "no_damage"
    if question.question_type == "area_comparison":
        flooded = evidence.area_pixels["road_flooded"]
        dry = evidence.area_pixels["road_non_flooded"]
        if flooded > dry:
            return "road_flooded"
        if dry > flooded:
            return "road_non_flooded"
        return "equal"
    if question.question_type == "spatial_adjacency":
        return "yes" if evidence.adjacency["road_flooded__water"] else "no"
    if question.question_type == "dominant_class":
        return evidence.dominant_class
    conditions_met = evidence.presence["road_flooded"] and evidence.adjacency["road_flooded__water"]
    return "yes" if conditions_met else "no"


def test_program_variants_cover_every_family_with_three_routes():
    assert set(PROGRAM_VARIANTS) == {
        "presence",
        "damage_state",
        "area_comparison",
        "spatial_adjacency",
        "dominant_class",
        "logical_conjunction",
    }
    assert all(len(variants) >= 3 for variants in PROGRAM_VARIANTS.values())
    assert all(len(set(variants)) == len(variants) for variants in PROGRAM_VARIANTS.values())


def test_evidence_tool_set_reads_all_supported_evidence(mask_evidence_fixture):
    tools = EvidenceToolSet(mask_evidence_fixture)

    assert tools.presence("road_flooded").output is True
    assert tools.area_pixels("road_flooded").output == 6
    assert tools.adjacency("road_flooded", "water").output is True
    assert tools.dominant_class().output == mask_evidence_fixture.dominant_class
    assert tools.presence("road_flooded").evidence_ids


def test_all_program_variants_compute_answers_and_explicit_final_steps(
    mask_evidence_fixture,
):
    questions = generate_questions("fixture", mask_evidence_fixture, seed=19)
    tools = EvidenceToolSet(mask_evidence_fixture)

    for question in questions:
        for variant in PROGRAM_VARIANTS[question.question_type]:
            trace = execute_program(question, tools, variant)

            assert trace.final_answer == _expected_answer(question, mask_evidence_fixture)
            assert trace.final_answer in question.answer_space
            assert trace.steps[-1].output_type == "answer"
            assert trace.steps[-1].output == trace.final_answer
            assert trace.source_family == "evidence_rules"
            assert trace.program_variant == variant
            assert trace.total_latency_ms == sum(step.latency_ms for step in trace.steps)
            assert trace.answer_distribution[trace.final_answer] > 0


def test_program_does_not_read_question_answer_or_expected_answer(mask_evidence_fixture):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]
    wrong_answer = next(answer for answer in question.answer_space if answer != question.answer)
    tampered = replace(
        question,
        answer=wrong_answer,
        evidence={**question.evidence, "expected_answer": wrong_answer},
    )

    trace = execute_program(
        tampered,
        EvidenceToolSet(mask_evidence_fixture),
        PROGRAM_VARIANTS[tampered.question_type][0],
    )

    assert trace.final_answer == _expected_answer(question, mask_evidence_fixture)
    assert trace.final_answer != wrong_answer


def test_trace_id_and_generator_score_are_deterministic(mask_evidence_fixture):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]
    tools = EvidenceToolSet(mask_evidence_fixture)

    first = execute_program(question, tools, "direct")
    second = execute_program(question, tools, "direct")

    assert first.trace_id == second.trace_id
    assert first.generator_score == second.generator_score
    assert 0 <= first.generator_score <= 1


def test_invalid_program_variant_raises_value_error(mask_evidence_fixture):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]

    with pytest.raises(ValueError, match="unknown program variant"):
        execute_program(question, EvidenceToolSet(mask_evidence_fixture), "missing")


class FailingPresenceTools(EvidenceToolSet):
    def presence(self, class_name: str) -> ToolResult:
        return ToolResult(
            output=None,
            confidence=0.0,
            evidence_ids=(f"presence:{class_name}",),
            latency_ms=1.0,
            failure="presence unavailable",
        )


class LowConfidenceTools(EvidenceToolSet):
    def presence(self, class_name: str) -> ToolResult:
        result = super().presence(class_name)
        return replace(result, confidence=0.2)


class ConflictingAreaComparisonTools(EvidenceToolSet):
    def presence(self, class_name: str) -> ToolResult:
        result = super().presence(class_name)
        if class_name == "road_flooded":
            return replace(result, output=False)
        return result


class ContradictoryPresenceTools(EvidenceToolSet):
    def area_pixels(self, class_name: str) -> ToolResult:
        result = super().area_pixels(class_name)
        return replace(result, output=0)


class ImpossibleAdjacencyTools(EvidenceToolSet):
    def presence(self, class_name: str) -> ToolResult:
        result = super().presence(class_name)
        return replace(result, output=False)

    def adjacency(self, first_class: str, second_class: str) -> ToolResult:
        result = super().adjacency(first_class, second_class)
        return replace(result, output=True)


class ConsistentNonAdjacencyTools(ImpossibleAdjacencyTools):
    def adjacency(self, first_class: str, second_class: str) -> ToolResult:
        result = super().adjacency(first_class, second_class)
        return replace(result, output=False)


def test_tool_failure_is_preserved_with_safe_answer_trace(mask_evidence_fixture):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]
    question = replace(
        question,
        question_type="presence",
        answer_space=("yes", "no"),
        evidence={"class_name": "road_flooded"},
    )

    trace = execute_program(question, FailingPresenceTools(mask_evidence_fixture), "direct")

    assert trace.final_answer in question.answer_space
    assert trace.steps[0].failure == "presence unavailable"
    assert trace.steps[-1].output_type == "answer"
    assert trace.steps[-1].failure == "presence unavailable"
    assert sum(trace.answer_distribution.values()) == pytest.approx(1.0)


def test_failure_fallback_is_deterministic_and_not_always_first_label(
    mask_evidence_fixture,
):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]
    tools = FailingPresenceTools(mask_evidence_fixture)
    answers = {
        execute_program(
            replace(question, question_id=f"question-{index}"),
            tools,
            "direct",
        ).final_answer
        for index in range(20)
    }

    assert answers == set(question.answer_space)


def test_confidence_distribution_uses_mean_and_allows_oracle_certainty(
    mask_evidence_fixture,
):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]

    low = execute_program(question, LowConfidenceTools(mask_evidence_fixture), "direct")
    oracle = execute_program(question, EvidenceToolSet(mask_evidence_fixture), "direct")

    assert list(low.answer_distribution.values()) == pytest.approx([0.5, 0.5])
    assert oracle.answer_distribution[oracle.final_answer] == 1.0


def test_generator_score_rewards_clean_confident_execution(mask_evidence_fixture):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]

    clean = execute_program(question, EvidenceToolSet(mask_evidence_fixture), "direct")
    low = execute_program(question, LowConfidenceTools(mask_evidence_fixture), "direct")
    failed = execute_program(question, FailingPresenceTools(mask_evidence_fixture), "direct")

    assert clean.generator_score > low.generator_score
    assert clean.generator_score > failed.generator_score


def test_presence_verified_rejects_presence_area_contradiction(mask_evidence_fixture):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]

    trace = execute_program(
        question,
        ContradictoryPresenceTools(mask_evidence_fixture),
        "verified",
    )

    assert trace.steps[-1].failure is not None


def test_area_ratio_route_rejects_presence_area_contradiction(mask_evidence_fixture):
    question = next(
        question
        for question in generate_questions("fixture", mask_evidence_fixture, seed=19)
        if question.question_type == "area_comparison"
    )

    trace = execute_program(
        question,
        ConflictingAreaComparisonTools(mask_evidence_fixture),
        "ratio_route",
    )

    assert trace.steps[-1].failure == "inconsistent evidence"


def test_area_ratio_route_presence_confidence_affects_distribution(mask_evidence_fixture):
    question = next(
        question
        for question in generate_questions("fixture", mask_evidence_fixture, seed=19)
        if question.question_type == "area_comparison"
    )

    trace = execute_program(
        question,
        LowConfidenceTools(mask_evidence_fixture),
        "ratio_route",
    )

    assert trace.steps[-1].failure is None
    assert trace.answer_distribution[trace.final_answer] == pytest.approx(0.6)


def test_presence_verified_accepts_consistent_negative_evidence(mask_evidence_fixture):
    question = generate_questions("fixture", mask_evidence_fixture, seed=19)[0]
    question = replace(question, evidence={"class_name": "missing"})

    class AbsentTools(EvidenceToolSet):
        def presence(self, class_name: str) -> ToolResult:
            return ToolResult(False, 1.0, ("presence:missing",), 0.0)

        def area_pixels(self, class_name: str) -> ToolResult:
            return ToolResult(0, 1.0, ("area_pixels:missing",), 0.0)

    trace = execute_program(question, AbsentTools(mask_evidence_fixture), "verified")

    assert trace.final_answer == "no"
    assert trace.steps[-1].failure is None


@pytest.mark.parametrize("variant", ["presence_then_relation", "relation_then_presence"])
def test_adjacency_presence_variants_reject_impossible_adjacency(
    mask_evidence_fixture,
    variant,
):
    question = next(
        question
        for question in generate_questions("fixture", mask_evidence_fixture, seed=19)
        if question.question_type == "spatial_adjacency"
    )

    trace = execute_program(
        question,
        ImpossibleAdjacencyTools(mask_evidence_fixture),
        variant,
    )

    assert trace.steps[-1].failure is not None


def test_adjacency_presence_variant_accepts_absent_nonadjacent_classes(
    mask_evidence_fixture,
):
    question = next(
        question
        for question in generate_questions("fixture", mask_evidence_fixture, seed=19)
        if question.question_type == "spatial_adjacency"
    )

    trace = execute_program(
        question,
        ConsistentNonAdjacencyTools(mask_evidence_fixture),
        "presence_then_relation",
    )

    assert trace.final_answer == "no"
    assert trace.steps[-1].failure is None


def test_dominant_presence_area_scan_rejects_presence_area_contradiction(
    mask_evidence_fixture,
):
    question = next(
        question
        for question in generate_questions("fixture", mask_evidence_fixture, seed=19)
        if question.question_type == "dominant_class"
    )

    trace = execute_program(
        question,
        ContradictoryPresenceTools(mask_evidence_fixture),
        "presence_area_scan",
    )

    assert trace.steps[-1].failure is not None


def test_dominant_area_scan_queries_every_class(mask_evidence_fixture):
    question = next(
        question
        for question in generate_questions("fixture", mask_evidence_fixture, seed=19)
        if question.question_type == "dominant_class"
    )

    trace = execute_program(
        question,
        EvidenceToolSet(mask_evidence_fixture),
        "area_scan",
    )

    queried = {
        str(step.arguments["class_name"]) for step in trace.steps if step.tool == "area_pixels"
    }
    assert queried == set(CLASS_NAMES)
