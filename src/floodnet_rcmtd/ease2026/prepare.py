"""Build an audited EASE dataset from split-specific question manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from floodnet_rcmtd.ease2026.protocol import balanced_sample, validate_splits


def _decode(value, default):
    if value is None:
        return default
    return json.loads(value) if isinstance(value, str) else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_records(
    question_rows_by_split: Mapping[str, Sequence[dict]], image_rows: Sequence[dict],
    trace_resolver: Callable[[dict, Path], tuple[str, dict[str, float]]],
) -> dict[str, list[dict]]:
    """Join questions to images while keeping the source split authoritative."""

    image_by_id = {str(row["image_id"]): row for row in image_rows}
    if len(image_by_id) != len(image_rows):
        raise ValueError("duplicate image_id in image manifest")
    mask_hashes: dict[Path, str] = {}
    assembled: dict[str, list[dict]] = {}
    for split, questions in question_rows_by_split.items():
        assembled[split] = []
        for question in questions:
            image_id = str(question["image_id"])
            if image_id not in image_by_id:
                raise ValueError(f"missing image_id {image_id}")
            image_row = image_by_id[image_id]
            if question.get("split") != split or image_row.get("split") != split:
                raise ValueError(f"question/image split mismatch for {question['question_id']}")
            image_path = Path(str(image_row["image_path"]))
            mask_path = Path(str(image_row["mask_path"]))
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"missing image or mask for {image_id}")
            if mask_path not in mask_hashes:
                mask_hashes[mask_path] = _sha256(mask_path)
            gold = str(question["answer"])
            answer_space = [str(value) for value in _decode(question["answer_space"], [])]
            trace_answer, support, *extra = trace_resolver(question, mask_path)
            if trace_answer not in answer_space or gold not in answer_space:
                raise ValueError(f"answer outside answer_space for {question['question_id']}")
            if str(trace_answer) != gold:
                raise ValueError(f"trace/gold disagreement for {question['question_id']}")
            assembled[split].append({
                "question_id": str(question["question_id"]),
                "image_id": image_id,
                "split": split,
                "question_type": str(question["question_type"]),
                "question": str(question["question"]),
                "image_path": str(image_path.resolve()),
                "mask_sha256": mask_hashes[mask_path],
                "group_id": str(image_row.get("group_id") or ""),
                "answer_space": answer_space,
                "gold_answer": gold,
                "trace_answer": str(trace_answer),
                "trace_support": {str(key): float(value) for key, value in support.items()},
                "trace_steps": extra[0] if extra else [],
                "mask_evidence": extra[1] if len(extra) > 1 else {},
            })
    validate_splits(assembled)
    return assembled


def trace_from_official_mask(question: dict, mask_path: Path) -> tuple[str, dict[str, float], list[dict], dict]:
    """Recompute deterministic trace support from the official mask, never model output."""

    import numpy as np
    from PIL import Image

    from floodnet_rcmtd.data.questions import QuestionRecord, build_evidence
    from floodnet_rcmtd.distillation.targets import build_target
    from floodnet_rcmtd.traces.programs import PROGRAM_VARIANTS, EvidenceToolSet, execute_program

    question_evidence = _decode(question["evidence"], {})
    with Image.open(mask_path) as mask:
        evidence = build_evidence(np.asarray(mask),
                                  adjacency_radius=int(question_evidence.get("adjacency_radius", 1)))
    record = QuestionRecord(
        question_id=str(question["question_id"]), image_id=str(question["image_id"]),
        split=str(question["split"]), question_type=str(question["question_type"]),
        template_id=str(question["template_id"]), question=str(question["question"]),
        answer=str(question["answer"]),
        answer_space=tuple(str(value) for value in _decode(question["answer_space"], [])),
        evidence=_decode(question["evidence"], {}),
    )
    variants = PROGRAM_VARIANTS[record.question_type]
    traces = [execute_program(record, EvidenceToolSet(evidence), variant) for variant in variants]
    target = build_target("multi_trace_soft_answer", record, traces)
    support = {str(key): float(value) for key, value in target["soft_answer"].items()}
    trace_answer = max(sorted(support), key=lambda key: support[key])
    steps = [{"tool": step.tool, "arguments": step.arguments, "output": step.output}
             for step in traces[0].steps[:-1]]
    return trace_answer, support, steps, {"area_pixels": evidence.area_pixels,
                                         "adjacency_radius": evidence.adjacency_radius}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_from_manifest(
    manifest_path: Path, output_dir: Path, *, train_per_family: int = 80,
    test_per_family: int = 80, seed: int = 20260618,
) -> dict[str, object]:
    import pandas as pd

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    image_rows = pd.read_parquet(_resolve(base, str(manifest["images"]))).to_dict("records")
    paths = {split: _resolve(base, str(value)) for split, value in manifest["questions"].items()}
    if not {"train", "test"}.issubset(paths):
        raise ValueError("manifest must contain separate train and test question files")
    question_rows = {split: pd.read_parquet(path).to_dict("records") for split, path in paths.items()}
    rows = assemble_records(question_rows, image_rows, trace_from_official_mask)
    from floodnet_rcmtd.data.questions import QUESTION_FAMILIES
    for split in ("train", "test"):
        families = {row["question_type"] for row in rows[split]}
        if families != set(QUESTION_FAMILIES):
            raise ValueError(f"{split} question families differ: missing={sorted(set(QUESTION_FAMILIES) - families)}, "
                             f"unexpected={sorted(families - set(QUESTION_FAMILIES))}")
    selected = dict(rows)
    selected["train"] = balanced_sample(rows["train"], train_per_family, seed=seed)
    selected["test"] = balanced_sample(rows["test"], test_per_family, seed=seed)
    audit = validate_splits(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in selected.items():
        write_jsonl(output_dir / f"{split}.jsonl", items)
    summary = {
        "source_manifest": str(manifest_path), "seed": seed,
        "train_per_family": train_per_family, "test_per_family": test_per_family,
        "split_audit": audit,
        "files": {split: {"path": f"{split}.jsonl", "sha256": _sha256(output_dir / f"{split}.jsonl")}
                  for split in selected},
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
