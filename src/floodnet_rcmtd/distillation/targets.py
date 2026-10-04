from __future__ import annotations

from collections import defaultdict

from floodnet_rcmtd.data.questions import QuestionRecord
from floodnet_rcmtd.traces.schema import TraceRecord

DISTILLATION_VARIANTS = ("task_only", "multi_trace_soft_answer")

def _best_trace(traces: list[TraceRecord]) -> TraceRecord:
    if not traces:
        raise ValueError("at least one trace is required")
    return max(
        traces,
        key=lambda trace: (
            trace.generator_score,
            trace.answer_distribution[trace.final_answer],
            trace.trace_id,
        ),
    )


def _soft_answer(traces: list[TraceRecord]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for trace in traces:
        for answer, probability in trace.answer_distribution.items():
            totals[answer] += probability
    total = sum(totals.values())
    return {answer: probability / total for answer, probability in sorted(totals.items())}


def build_target(
    variant: str,
    question: QuestionRecord,
    traces: list[TraceRecord],
) -> dict[str, object]:
    if variant not in DISTILLATION_VARIANTS:
        raise ValueError(f"unknown distillation variant: {variant!r}")
    best = _best_trace(traces)
    base: dict[str, object] = {
        "variant": variant,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "image_id": question.image_id,
        "question": question.question,
        "answer": question.answer if variant == "task_only" else best.final_answer,
        "answer_space": list(question.answer_space),
        "rationales": [],
        "soft_answer": None,
        "trace_weights": [],
        "evidence_target": None,
    }
    if variant == "multi_trace_soft_answer":
        base["soft_answer"] = _soft_answer(traces)
    return base
