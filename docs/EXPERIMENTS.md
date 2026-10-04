# EASE experiment guide

## Hardware workflow

Mac handles code, SSH control, lightweight tests, and analysis. Large data and models are downloaded to the Windows D: relay, checked by file count, size, and SHA256, and transferred directly to Linux with resumable transport. Linux performs CUDA training, inference, evaluation, and stores adapters and predictions.

Install `.[ease2026,analysis,dev]` in the Linux environment. Use model-compatible PyTorch, Transformers, PEFT, and bitsandbytes versions. The official [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) and [GLM-4.6V-Flash model card](https://huggingface.co/zai-org/GLM-4.6V-Flash) document their model interfaces. The runner requires existing local model directories and CUDA.

## Data preparation

Edit `configs/data/floodnet_v1.yaml` for the dataset and output directories, then run:

```bash
floodnet-rcmtd build-split --config configs/data/floodnet_v1.yaml
floodnet-rcmtd build-questions --config configs/data/floodnet_v1.yaml
floodnet-rcmtd audit-questions --config configs/data/floodnet_v1.yaml
ease2026 prepare --manifest /data/ease/derived/manifest.json --output /data/ease/vqa
ease2026 audit --data-dir /data/ease/vqa
```

The source manifest points to `images.parquet` (`image_id`, `image_path`, `mask_path`, `group_id`, `split`) and question files keyed by split. Question records include `question_id`, `image_id`, `split`, `question_type`, `question`, `answer`, `answer_space`, `template_id`, and `evidence`.

Preparation derives trace support from each semantic mask, checks the verified answer, and stores class pixel counts for offline diagnostics. It hashes masks and output files and rejects cross-split question, image, mask, and sequence overlap. The default train and test sample is 80 questions per family. Keep the resulting manifest with every run.

## Four supervision cells

| Cell | Loss support | Target content |
|---|---|---|
| PC-Ans | All non-padding sequence tokens | Verified direct answer |
| PC-Trace | All non-padding sequence tokens | Trace-derived answer with trace context |
| AA-Ans | Assistant answer span | Verified direct answer |
| AA-Trace | Assistant answer span | Trace-derived answer with masked trace context |

Every cell uses the same sampled training IDs, order, seed, optimizer settings, and update budget for a backbone. The user prompt is shared by Ans and Trace. Trace evidence and support appear before the answer in the assistant-side training target. Token prefix and suffix checks locate the exact answer span; AA ignores evidence, headers, and closing template tokens. PC includes all non-padding tokens. The defaults are 480 optimizer updates, 16 microsteps per update, learning rate `1e-4`, LoRA rank 16, alpha 32, and dropout 0.05. The runner caps image area at 262,144 pixels and disables thinking mode. Evaluation receives only the image, question, and answer choices.

## Complete experiment grids

```bash
bash scripts/run_experiments.sh qwen /data/ease/vqa /data/models/Qwen3.5-9B /data/ease/runs
bash scripts/run_experiments.sh glm /data/ease/vqa /data/models/GLM-4.6V-Flash /data/ease/runs
```

Qwen defaults to seeds `20260618 20260619 20260620`; GLM defaults to one seed. Supply trailing seed arguments to select another grid. Both commands run a zero-shot baseline, four training cells, generation and candidate evaluation, summaries, and paired diagnostics. Choose a fresh output root for a new grid; the runner refuses to overwrite run or evaluation directories.

## Evaluation and analysis

Candidate evaluation averages label-token log likelihood, selects the highest-scoring label, and records a softmax-derived confidence. Generation uses greedy decoding with answer normalization and records confidence as `null`.

```bash
ease2026 summarize --output-root /data/ease/runs --backbone qwen \
  --seeds 20260618 20260619 20260620 --mode candidate \
  --output /data/ease/runs/qwen_candidate_summary.json

ease2026 analyze --output-root /data/ease/runs --backbone qwen \
  --seed 20260618 --second-seed 20260619 --mode generation \
  --data-dir /data/ease/vqa --output /data/ease/runs/qwen_generation_diagnostics
```

Summary output includes overall and family accuracy, macro-F1, ECE where probabilities exist, prediction concentration, normalized entropy, and per-seed AA-versus-PC gaps. Analysis exports confusion matrices, all paired cases, help/hurt counts, seed disagreements, and area/support margin statistics. Both commands check that compared runs use identical held-out question membership and test hashes.

## Output layout

```text
vqa/
  manifest.json
  train.jsonl
  test.jsonl
  val.jsonl
runs/<backbone>/seed<seed>/<cell>/
  run.json
  training_loss.csv
  adapter/
  processor/
  eval_candidate/{meta.json,predictions.jsonl}
  eval_generation/{meta.json,predictions.jsonl}
runs/<backbone>_<mode>_summary.{json,csv}
runs/<backbone>_<mode>_diagnostics/{analysis.json,cases.csv}
runs/logs/
```

The `results/` directory contains the paper's published aggregate tables. For new experiments, retain the complete per-example outputs and recompute the tables from the corresponding run directories.
