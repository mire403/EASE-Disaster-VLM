from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, object]
    output_type: str
    output: object
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str]
    latency_ms: float = Field(ge=0.0)
    failure: str | None = None


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    question_id: str
    source_family: str
    program_variant: str
    steps: list[TraceStep]
    final_answer: str
    answer_distribution: dict[str, float]
    generator_score: float = Field(ge=0.0, le=1.0)
    total_latency_ms: float = Field(ge=0.0)
    offline_quality: dict[str, float | str] | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> TraceRecord:
        if not self.steps:
            raise ValueError("steps must not be empty")
        final_step = self.steps[-1]
        if final_step.tool != "answer":
            raise ValueError("last step tool must be 'answer'")
        if final_step.output_type != "answer":
            raise ValueError("last step output_type must be 'answer'")
        if final_step.output != self.final_answer:
            raise ValueError("last step output must equal final_answer")

        if not self.answer_distribution:
            raise ValueError("answer_distribution must not be empty")
        if any(value < 0.0 or value > 1.0 for value in self.answer_distribution.values()):
            raise ValueError("answer_distribution values must be between 0 and 1")
        if not math.isclose(
            sum(self.answer_distribution.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("answer_distribution must sum to 1")
        if self.final_answer not in self.answer_distribution:
            raise ValueError("final_answer must be present in answer_distribution")
        if not math.isclose(
            final_step.confidence,
            self.answer_distribution[self.final_answer],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "last step confidence must equal answer_distribution[final_answer]"
            )
        if final_step.failure is None and any(
            step.failure is not None for step in self.steps[:-1]
        ):
            raise ValueError("upstream step failure requires final step failure")
        if final_step.failure is None:
            maximum_probability = max(self.answer_distribution.values())
            if not math.isclose(
                self.answer_distribution[self.final_answer],
                maximum_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("successful final_answer must have maximum probability")

        step_latency = sum(step.latency_ms for step in self.steps)
        if not math.isclose(
            self.total_latency_ms,
            step_latency,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("total_latency_ms must equal the sum of step latencies")
        return self
