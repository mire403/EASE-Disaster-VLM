import pytest
from pydantic import ValidationError

from floodnet_rcmtd.traces.schema import TraceRecord, TraceStep


def _step(**overrides: object) -> TraceStep:
    values: dict[str, object] = {
        "tool": "presence",
        "arguments": {"class_name": "water"},
        "output_type": "bool",
        "output": True,
        "confidence": 0.9,
        "evidence_ids": ["presence:water"],
        "latency_ms": 2.5,
    }
    values.update(overrides)
    return TraceStep(**values)


def _record(**overrides: object) -> TraceRecord:
    values: dict[str, object] = {
        "trace_id": "trace-1",
        "question_id": "question-1",
        "source_family": "evidence_rules",
        "program_variant": "direct",
        "steps": [
            _step(),
            _step(
                tool="answer",
                output_type="answer",
                output="yes",
                latency_ms=0.0,
            ),
        ],
        "final_answer": "yes",
        "answer_distribution": {"yes": 0.9, "no": 0.1},
        "generator_score": 0.9,
        "total_latency_ms": 2.5,
    }
    values.update(overrides)
    return TraceRecord(**values)


def test_trace_record_json_roundtrip_preserves_equality():
    record = _record(offline_quality={"status": "accepted", "score": 0.8})

    assert TraceRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("field", ["confidence"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_trace_step_rejects_scores_outside_unit_interval(field, value):
    with pytest.raises(ValidationError):
        _step(**{field: value})


def test_trace_step_rejects_negative_latency():
    with pytest.raises(ValidationError):
        _step(latency_ms=-0.01)


@pytest.mark.parametrize(
    "distribution",
    [
        {},
        {"yes": -0.1, "no": 1.1},
        {"yes": 0.7, "no": 0.2},
    ],
)
def test_trace_record_rejects_invalid_answer_distributions(distribution):
    with pytest.raises(ValidationError):
        _record(answer_distribution=distribution)


@pytest.mark.parametrize("generator_score", [-0.01, 1.01])
def test_trace_record_rejects_generator_score_outside_unit_interval(
    generator_score,
):
    with pytest.raises(ValidationError):
        _record(generator_score=generator_score)


def test_trace_record_rejects_inconsistent_total_latency():
    with pytest.raises(ValidationError, match="total_latency_ms"):
        _record(total_latency_ms=3.0)


@pytest.mark.parametrize("model_factory", [_step, _record])
def test_trace_models_reject_extra_fields(model_factory):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_factory(unexpected="value")


def test_trace_record_rejects_empty_steps():
    with pytest.raises(ValidationError, match="steps"):
        _record(steps=[], total_latency_ms=0.0)


def test_trace_record_requires_last_step_to_be_answer():
    with pytest.raises(ValidationError, match="last step output_type"):
        _record(steps=[_step(tool="answer")])


def test_trace_record_requires_last_step_tool_to_be_answer():
    with pytest.raises(ValidationError, match="last step tool"):
        _record(
            steps=[
                _step(),
                _step(
                    tool="presence",
                    output_type="answer",
                    output="yes",
                    latency_ms=0.0,
                ),
            ]
        )


def test_trace_record_requires_last_step_output_to_match_final_answer():
    with pytest.raises(ValidationError, match="last step output"):
        _record(
            steps=[
                _step(),
                _step(
                    tool="answer",
                    output_type="answer",
                    output="no",
                    latency_ms=0.0,
                ),
            ]
        )


def test_trace_record_requires_final_answer_in_distribution():
    with pytest.raises(ValidationError, match="final_answer"):
        _record(answer_distribution={"no": 1.0})


def test_trace_record_requires_final_confidence_to_match_distribution():
    with pytest.raises(ValidationError, match="last step confidence"):
        _record(
            steps=[
                _step(),
                _step(
                    tool="answer",
                    output_type="answer",
                    output="yes",
                    confidence=0.8,
                    latency_ms=0.0,
                ),
            ]
        )


def test_trace_record_rejects_upstream_failure_hidden_by_successful_final_step():
    with pytest.raises(ValidationError, match="upstream step failure"):
        _record(
            steps=[
                _step(failure="presence unavailable"),
                _step(
                    tool="answer",
                    output_type="answer",
                    output="yes",
                    latency_ms=0.0,
                ),
            ]
        )


def test_successful_trace_requires_final_answer_to_be_distribution_argmax():
    with pytest.raises(ValidationError, match="maximum probability"):
        _record(
            steps=[
                _step(),
                _step(
                    tool="answer",
                    output_type="answer",
                    output="yes",
                    confidence=0.4,
                    latency_ms=0.0,
                ),
            ],
            answer_distribution={"yes": 0.4, "no": 0.6},
        )


def test_successful_trace_allows_final_answer_tied_for_distribution_argmax():
    record = _record(
        steps=[
            _step(),
            _step(
                tool="answer",
                output_type="answer",
                output="yes",
                confidence=0.5,
                latency_ms=0.0,
            ),
        ],
        answer_distribution={"yes": 0.5, "no": 0.5},
    )

    assert record.final_answer == "yes"


def test_failed_final_step_allows_uniform_distribution():
    record = _record(
        steps=[
            _step(failure="presence unavailable", confidence=0.0),
            _step(
                tool="answer",
                output_type="answer",
                output="yes",
                confidence=0.5,
                latency_ms=0.0,
                failure="presence unavailable",
            ),
        ],
        answer_distribution={"yes": 0.5, "no": 0.5},
    )

    assert record.steps[-1].failure == "presence unavailable"
