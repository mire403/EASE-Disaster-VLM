"""GPU training and evaluation for the EASE 2 x 2 protocol.

The module imports model libraries only after split and artifact checks pass.
"""

from __future__ import annotations

import hashlib
import copy
import csv
import importlib.metadata
import json
import math
import random
from itertools import cycle
from pathlib import Path

from floodnet_rcmtd.ease2026.protocol import CELLS, make_cells, make_labels, validate_splits

DEFAULT_MODELS = {"qwen": "Qwen/Qwen3.5-9B", "glm": "zai-org/GLM-4.6V-Flash"}


def resolve_local_model_path(model_path: str | None) -> str:
    if not model_path or not Path(model_path).is_dir():
        raise ValueError("a local model directory on the Linux GPU server is required")
    if not (Path(model_path) / "config.json").is_file():
        raise ValueError("local model directory must contain config.json")
    return str(Path(model_path).resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify_data_dir(data_dir: Path) -> tuple[dict[str, list[dict]], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    if not {"train", "test"}.issubset(files):
        raise ValueError("data manifest requires separate train and test files")
    rows = {}
    for split, record in files.items():
        path = data_dir / str(record["path"])
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"sha256 mismatch for {split}: {path}")
        rows[split] = _read_jsonl(path)
    audit = validate_splits(rows)
    if not rows["train"] or not rows["test"]:
        raise ValueError("train and test must both be non-empty")
    if audit["cross_split_overlap"] != 0:
        raise ValueError("cross-split overlap")
    return rows, manifest


def build_messages(row: dict, target_answer: str | None = None) -> list[dict]:
    candidates = ", ".join(str(answer) for answer in row["answer_space"])
    prompt = (
        "Answer the visual question using only the image. Choose exactly one label "
        "from the answer space and give no explanation.\n"
        f"Question: {row['question']}\nAnswer space: {candidates}"
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "path": str(row["image_path"])},
        {"type": "text", "text": prompt},
    ]}]
    if target_answer is not None:
        target = str(target_answer)
        if row.get("target_source") == "Trace":
            context = {"steps": row.get("trace_steps", []),
                       "candidate_support": row.get("trace_support", {})}
            target = "Evidence: " + json.dumps(context, sort_keys=True) + "\nAnswer:\n" + target
        messages.append({"role": "assistant", "content": target})
    return messages


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_model(model_path: str, *, quantize: bool):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForMultimodalLM as AutoVisionModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVisionModel

    if not torch.cuda.is_available():
        raise RuntimeError("EASE model execution requires a CUDA GPU on the Linux server")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True,
                                               local_files_only=True)
    quantization = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
                    if quantize else None)
    model = AutoVisionModel.from_pretrained(
        model_path, device_map="auto", dtype=torch.bfloat16,
        quantization_config=quantization, trust_remote_code=True,
        local_files_only=True,
    )
    return processor, model


def bounded_image(path: Path, max_pixels: int = 262144):
    from PIL import Image

    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / (image.width * image.height))
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _inputs(processor, messages: list[dict], *, generation: bool, continuation: bool = False):
    prepared = copy.deepcopy(messages)
    for message in prepared:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image" and "path" in block:
                    block["image"] = bounded_image(Path(block.pop("path")))
        elif isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
    inputs = processor.apply_chat_template(
        prepared, add_generation_prompt=generation, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
        continue_final_message=continuation,
    )
    inputs.pop("token_type_ids", None)
    return inputs


def _labels_for_example(processor, row: dict, support: str):
    full_messages = build_messages(row, row["target_answer"])
    prompt = _inputs(processor, full_messages[:-1], generation=True)
    full = _inputs(processor, full_messages, generation=False)
    continued = _inputs(processor, full_messages, generation=False, continuation=True)
    ids = full["input_ids"][0].tolist()
    prefix = prompt["input_ids"][0].tolist()
    continued_ids = continued["input_ids"][0].tolist()
    if ids[:len(prefix)] != prefix or ids[:len(continued_ids)] != continued_ids:
        raise ValueError("chat template does not preserve the prompt and answer token prefix")
    answer_ids = processor.tokenizer.encode(str(row["target_answer"]), add_special_tokens=False)
    if not answer_ids or continued_ids[-len(answer_ids):] != answer_ids:
        raise ValueError("answer tokens do not form the final assistant content span")
    answer_end = len(continued_ids)
    answer_start = answer_end - len(answer_ids)
    attention = full["attention_mask"][0].tolist()
    labels = make_labels(ids, ids[:answer_start], getattr(processor.tokenizer, "pad_token_id", None),
                         support, attention_mask=attention, answer_end=answer_end)
    return full, labels


def train_one(
    data_dir: Path, output_root: Path, *, backbone: str, model_path: str | None,
    seed: int, cell: str, updates: int = 480, accumulation: int = 16,
    learning_rate: float = 1e-4, quantize: bool = True,
) -> Path:
    if backbone not in DEFAULT_MODELS or cell not in CELLS:
        raise ValueError("unknown backbone or cell")
    if updates <= 0 or accumulation <= 0:
        raise ValueError("updates and accumulation must be positive")
    rows, manifest = verify_data_dir(data_dir)
    examples = make_cells(rows["train"], seed=seed)[cell]
    if not examples:
        raise ValueError("no training examples")
    run_dir = output_root / backbone / f"seed{seed}" / cell
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    _seed_everything(seed)
    resolved_model = resolve_local_model_path(model_path)
    processor, model = _load_model(resolved_model, quantize=quantize)
    if quantize:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    micro_steps = updates * accumulation
    for index, row in enumerate(cycle(examples)):
        if index >= micro_steps:
            break
        inputs, labels = _labels_for_example(processor, row, row["loss_support"])
        inputs["labels"] = torch.tensor([labels], dtype=torch.long)
        inputs = inputs.to(model.device)
        loss = model(**inputs).loss
        (loss / accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        if (index + 1) % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    run_dir.mkdir(parents=True)
    adapter = run_dir / "adapter"
    model.save_pretrained(adapter)
    processor.save_pretrained(run_dir / "processor")
    metadata = {
        "protocol": "ease2026-v1", "backbone": backbone,
        "model_path": resolved_model, "cell": cell, "seed": seed,
        "model_config_sha256": _sha256(Path(resolved_model) / "config.json"),
        "optimizer_updates": updates, "micro_steps": micro_steps,
        "gradient_accumulation": accumulation, "learning_rate": learning_rate,
        "quantize_4bit": quantize, "train_count": len(examples),
        "train_sha256": manifest["files"]["train"]["sha256"],
        "test_sha256": manifest["files"]["test"]["sha256"],
        "final_loss": losses[-1],
        "image_max_pixels": 262144, "enable_thinking": False,
        "environment": {name: importlib.metadata.version(name)
                        for name in ("torch", "transformers", "peft", "accelerate")},
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with (run_dir / "training_loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["micro_step", "optimizer_update", "loss"])
        writer.writerows((index + 1, (index + 1) // accumulation, loss)
                         for index, loss in enumerate(losses))
    return run_dir


def _candidate_scores(model, processor, row: dict) -> dict[str, float]:
    import torch

    scores = {}
    for answer in row["answer_space"]:
        full, labels = _labels_for_example(processor, {**row, "target_source": "Ans",
                                                      "target_answer": str(answer)}, "AA")
        inputs = full.to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[:, :-1, :]
            target = torch.tensor(labels[1:], device=logits.device)
            active = target != -100
            log_probs = torch.log_softmax(logits[0, active], dim=-1)
            chosen = log_probs.gather(1, target[active, None]).squeeze(1)
            scores[str(answer)] = float(chosen.mean().cpu())
    return scores


def evaluate_one(
    data_dir: Path, output_root: Path, *, backbone: str, model_path: str | None,
    seed: int, cell: str, mode: str = "candidate", quantize: bool = True,
    max_new_tokens: int = 32,
) -> Path:
    if backbone not in DEFAULT_MODELS or cell not in (*CELLS, "Zero-shot"):
        raise ValueError("unknown backbone or cell")
    if mode not in {"candidate", "generation"}:
        raise ValueError("mode must be candidate or generation")
    rows, manifest = verify_data_dir(data_dir)
    resolved_model = resolve_local_model_path(model_path)
    run_dir = output_root / backbone / f"seed{seed}" / cell
    if cell != "Zero-shot":
        metadata = json.loads((run_dir / "run.json").read_text())
        if metadata["test_sha256"] != manifest["files"]["test"]["sha256"]:
            raise ValueError("test manifest differs from training run")
        if metadata["model_config_sha256"] != _sha256(Path(resolved_model) / "config.json"):
            raise ValueError("model config differs from training run")
    eval_dir = run_dir / f"eval_{mode}"
    if eval_dir.exists():
        raise FileExistsError(f"evaluation directory already exists: {eval_dir}")

    import torch
    from peft import PeftModel
    from floodnet_rcmtd.ease2026.normalization import normalize_prediction

    _seed_everything(seed)
    processor, model = _load_model(resolved_model, quantize=quantize)
    if cell != "Zero-shot":
        model = PeftModel.from_pretrained(model, run_dir / "adapter")
    model.eval()
    predictions = []
    for row in rows["test"]:
        if mode == "candidate":
            scores = _candidate_scores(model, processor, row)
            prediction = max(scores, key=scores.get)
            largest = max(scores.values())
            weights = {key: math.exp(value - largest) for key, value in scores.items()}
            confidence = weights[prediction] / sum(weights.values())
            text = None
        else:
            inputs = _inputs(processor, build_messages(row), generation=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = processor.decode(generated[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            try:
                prediction = normalize_prediction(text, row["answer_space"])
            except ValueError:
                prediction = "__unmatched__"
            confidence = None
            scores = None
        predictions.append({
            "question_id": row["question_id"], "image_id": row["image_id"],
            "question_type": row["question_type"], "answer": row["gold_answer"],
            "prediction": prediction, "confidence": confidence,
            "candidate_scores": scores, "generated_text": text,
            "backbone": backbone, "cell": cell, "seed": seed, "mode": mode,
        })
    eval_dir.mkdir(parents=True)
    with (eval_dir / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (eval_dir / "meta.json").write_text(json.dumps({
        "protocol": "ease2026-v1", "backbone": backbone,
        "cell": cell, "seed": seed, "mode": mode,
        "test_sha256": manifest["files"]["test"]["sha256"],
        "count": len(predictions),
    }, indent=2, sort_keys=True) + "\n")
    return eval_dir
