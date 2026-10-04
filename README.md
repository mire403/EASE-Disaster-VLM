# EASE: Evidence-to-Answer Supervision for Disaster-Scene Vision-Language Adaptation

[![ACML 2026](https://img.shields.io/badge/ACML-2026-4b6cb7)](#citation) [![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776ab?logo=python&logoColor=white)](#installation) [![Task](https://img.shields.io/badge/Task-disaster--scene%20VQA-00897b)](#overview) [![Backbones](https://img.shields.io/badge/Backbones-Qwen%20%7C%20GLM-6f42c1)](#four-cell-workflow) [![QLoRA](https://img.shields.io/badge/Adaptation-4--bit%20QLoRA-orange)](#method)

Research code for **EASE**, by **Haoze Zheng** and **Yaping Han**, accepted to **ACML 2026**. The repository provides FloodNet data construction, mask-grounded evidence traces, four supervision combinations, Qwen/GLM adaptation, evaluation, and result analysis.

**Core finding:** aligning loss with the assistant answer span produces a larger and more stable improvement than changing direct answers to trace-derived targets. Across three Qwen seeds, the paper reports an AA-versus-PC accuracy gap of **23.58 ± 0.34 percentage points**.

## Contents

- [Overview](#overview)
- [Method](#method)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Dataset preparation](#dataset-preparation)
- [Experiment code](#experiment-code)
- [Four-cell workflow](#four-cell-workflow)
- [Results and analysis](#results-and-analysis)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Overview

Disaster-scene visual question answering requires concise, verifiable answers about aerial imagery. Semantic segmentation masks provide pixel-level evidence for constructing these questions, while the VLM must answer from the image at inference time. EASE studies the **evidence-to-answer alignment gap** that arises when prompts, evidence text, and answer labels are optimized together as a language-modeling sequence.

The study separates two supervision axes: **loss support**, either prompt-conditioned/full-sequence (PC) or assistant-answer-aligned (AA); and **target construction**, either verified direct answers (Ans) or trace-derived answers (Trace). The experiments cover Qwen3.5-9B and GLM-4.6V-Flash on six mask-verifiable question families.

## Method

1. Pair FloodNet images and semantic masks, then construct sequence-aware splits.
2. Extract mask evidence and generate six families of verified visual questions.
3. Execute evidence programs and construct direct or trace-derived answer targets.
4. Train the four supervision cells with shared data, example order, seeds, and adaptation settings.
5. Evaluate held-out questions and analyze prediction changes by family and case.

| Cell | Optimized token positions | Target construction |
|---|---|---|
| **PC-Ans** | Full non-padding sequence | Verified direct answer |
| **PC-Trace** | Full non-padding sequence | Trace-derived answer with evidence context |
| **AA-Ans** | Assistant answer span | Verified direct answer |
| **AA-Trace** | Assistant answer span | Trace-derived answer with masked evidence context |

Trace evidence and candidate support form the assistant-side training target. PC includes these evidence tokens in the loss; AA masks them and supervises only the final answer tokens. Evaluation prompts contain the image, question, and answer choices. The two target variants in the target builder are `task_only` and `multi_trace_soft_answer`.

![Figure 1: EASE overview and supervision design](figure/Figure1.png)

*Figure 1. EASE overview and the 2 × 2 supervision space.*

## Repository structure

```text
EASE_ACML2026_code/
├── README.md
├── CITATION.cff
├── pyproject.toml
├── figure/                       # Figure1.png ... Figure5.png
├── configs/data/                 # FloodNet construction settings
├── src/floodnet_rcmtd/
│   ├── data/                     # pairs, splits, evidence, questions, audits
│   ├── traces/                   # evidence programs and trace schema
│   ├── distillation/             # direct and trace-derived targets
│   ├── ease2026/                 # training, evaluation, summaries, diagnostics
│   ├── cli.py                    # data-construction commands
│   ├── config.py
│   └── manifest.py
├── scripts/
│   ├── run_experiments.sh        # Qwen/GLM experiment grids
│   └── plot_results.py           # CSV-based result plots
├── results/                      # paper aggregate tables and case examples
├── tests/                        # data, trace, protocol, and analysis tests
└── docs/                         # experiment instructions and module guide
```

See the [code inventory](docs/CODE_INVENTORY.md) and [experiment-to-result index](results/index.json) for a detailed map.

## Installation

Use **Python 3.11 or 3.12**. Model training and inference require a **CUDA Linux GPU** and local model weights.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ease2026,analysis,dev]'
pytest -q
```

For code inspection, data utilities, and lightweight tests, install `.[analysis,dev]`. Large datasets, model weights, and checkpoints stay on the Windows relay or Linux server. See [hardware workflow and environment setup](docs/EXPERIMENTS.md#hardware-workflow).

## Dataset preparation

The benchmark uses **FloodNet-Supervised v1.0**. The paper reports **2,343 image–mask pairs** and a balanced held-out evaluation set of **480 questions**, with **80 questions per family**. See [dataset_statistics.csv](results/dataset_statistics.csv) for the reported dataset and split checks.

| Family | Mask-derived evidence |
|---|---|
| Object presence | Class pixel count |
| Damage state | Flooded building or road presence |
| Area comparison | Flooded versus non-flooded road area |
| Spatial adjacency | Region proximity under the adjacency radius |
| Dominant class | Class with the greatest pixel coverage |
| Logical conjunction | Presence and adjacency conditions |

![Figure 2: FloodNet-derived dataset construction](figure/Figure2.png)

*Figure 2. Image–mask evidence is converted into questions with constrained answer spaces.*

Obtain FloodNet under the dataset provider's terms and edit `data_root` and `output_root` in [configs/data/floodnet_v1.yaml](configs/data/floodnet_v1.yaml) for the Linux server.

```bash
floodnet-rcmtd build-split --config configs/data/floodnet_v1.yaml
floodnet-rcmtd build-questions --config configs/data/floodnet_v1.yaml
floodnet-rcmtd audit-questions --config configs/data/floodnet_v1.yaml

ease2026 prepare --manifest /data/ease/derived/manifest.json --output /data/ease/vqa
ease2026 audit --data-dir /data/ease/vqa
```

Preparation writes separate train/test JSONL files and SHA256 records. Audits reject question, image, mask, or sequence-group overlap across splits and verify trace/gold agreement. Input schemas are documented in the [experiment guide](docs/EXPERIMENTS.md#data-preparation).

## Experiment code

| Stage | Implementation | Role |
|---|---|---|
| Data construction | `data/layout.py`, `data/sequences.py`, `data/split.py` | Pair images/masks and assign sequence groups |
| Grounded questions | `data/questions.py`, `data/audit.py` | Extract evidence, generate questions, verify labels |
| Trace targets | `traces/programs.py`, `distillation/targets.py` | Execute evidence programs and aggregate candidate support |
| Dataset export | `ease2026/prepare.py` | Export split-specific records, mask hashes, and area evidence |
| Loss support | `ease2026/protocol.py` | Apply PC/AA token labels and shared cell order |
| QLoRA and evaluation | `ease2026/runner.py` | Train adapters; run generation and candidate scoring |
| Answer normalization | `ease2026/normalization.py` | Map generated text to an allowed answer label |
| Metrics and seed analysis | `ease2026/summary.py` | Aggregate metrics, family results, entropy, and AA/PC gaps |
| Case diagnostics | `ease2026/diagnostics.py` | Export confusion matrices, help/hurt cases, and area/support margins |

## Four-cell workflow

The default adaptation settings are 4-bit NF4 QLoRA, LoRA rank **16**, alpha **32**, dropout **0.05**, learning rate **1e-4**, **480 optimizer updates**, and **16 accumulation microsteps**. Images are capped at **262,144 pixels** and thinking mode is disabled. All four cells use the same seed and ordered training examples within a run grid.

Run complete grids on Linux:

```bash
bash scripts/run_experiments.sh qwen /data/ease/vqa /data/models/Qwen3.5-9B /data/ease/runs
bash scripts/run_experiments.sh glm /data/ease/vqa /data/models/GLM-4.6V-Flash /data/ease/runs
```

The Qwen grid defaults to seeds `20260618 20260619 20260620`; the GLM grid uses one seed. Both grids include zero-shot evaluation, PC-Ans, PC-Trace, AA-Ans, AA-Trace, two decoding modes, metric aggregation, and paired diagnostics. Trailing seed arguments override the defaults.

For an individual experiment:

```bash
ease2026 train --data-dir /data/ease/vqa --output-root /data/ease/runs \
  --backbone qwen --model-path /data/models/Qwen3.5-9B --seed 20260618 --cell AA-Ans

ease2026 evaluate --data-dir /data/ease/vqa --output-root /data/ease/runs \
  --backbone qwen --model-path /data/models/Qwen3.5-9B --seed 20260618 \
  --cell AA-Ans --mode candidate
```

Use `--mode generation` for greedy short-answer decoding. Candidate scoring records answer probabilities for ECE; generation mode stores confidence as `null`. Training saves adapters, loss records, data hashes, and package versions. Evaluation saves raw text or candidate scores with every prediction.

## Results and analysis

The [results directory](results/README.md) contains machine-readable **paper-reported aggregate results** and representative cases. New run outputs are saved under the selected output root.

**Three-seed Qwen3.5-9B comparison.** Values are percentages; adapted entries are mean ± standard deviation.

| Method | Accuracy (%) | Macro-F1 (%) | Reported ECE (%) |
|---|---:|---:|---:|
| Zero-shot | 70.42 | 51.09 | 29.58 |
| PC-Ans | 69.10 ± 0.43 | 32.28 ± 0.62 | 31.17 ± 0.43 |
| PC-Trace | 70.14 ± 0.32 | 32.48 ± 0.71 | 29.87 ± 0.31 |
| AA-Ans | **93.54 ± 0.63** | **69.90 ± 4.22** | **6.46 ± 0.63** |
| AA-Trace | 92.85 ± 0.43 | 67.17 ± 1.72 | 7.12 ± 0.36 |

The paired AA-versus-PC gaps are **23.96, 23.44, and 23.33 percentage points**, averaging **23.58 ± 0.34**. The direct-versus-trace difference is smaller and seed-sensitive. CSV: [qwen_three_seed.csv](results/qwen_three_seed.csv).

![Figure 3: Primary-seed accuracy, macro-F1, and ECE](figure/Figure3.png)

*Figure 3. Primary-seed metric profile. The table above summarizes three seeds; the figure shows the primary run.*

The primary run reaches **94.17%** for AA-Ans and **92.71%** for AA-Trace under generation. Candidate scoring reaches **92.92%** and **92.29%**, respectively. The full decoding comparison is in [qwen_primary_seed.csv](results/qwen_primary_seed.csv).

**GLM-4.6V-Flash comparison.** The same loss-support pattern appears on the second backbone.

| Method | Accuracy (%) | Macro-F1 (%) | Reported ECE (%) |
|---|---:|---:|---:|
| Zero-shot | 68.54 | 49.63 | 30.36 |
| PC-Ans | 67.71 | 34.08 | 31.14 |
| PC-Trace | 69.17 | 35.11 | 29.71 |
| AA-Ans | **90.21** | 67.58 | 7.82 |
| AA-Trace | 89.79 | **68.02** | **7.55** |

The average AA accuracy exceeds the average PC accuracy by **21.56 percentage points**. CSV: [glm_single_seed.csv](results/glm_single_seed.csv).

**Reasoning families and prediction behavior.** On the primary Qwen run, AA-Ans improves spatial adjacency from **31.25% to 98.75%**, logical conjunction from **60.00% to 96.25%**, and area comparison from **67.50% to 93.75%**. PC predictions are more concentrated: normalized entropy falls from **0.74** for zero-shot to **0.43** for PC-Ans and **0.47** for PC-Trace.

![Figure 4: Accuracy by reasoning family](figure/Figure4.png)

*Figure 4. Primary-seed accuracy across six reasoning families.*

Dominant-class questions remain difficult when the two largest semantic regions have similar areas. The paper reports a median area margin of **0.058** for errors and **0.184** for correct cases; **8 of 11** errors have margins below **0.10**. See [family accuracy](results/question_family_accuracy.csv), [prediction behavior](results/prediction_behavior.csv), and [dominant-class diagnostics](results/dominant_class_diagnostics.csv).

**Trace support and case diagnostics.** Trace top-1 answers agree with verified labels on all **480** training examples; the reported median gold-label support is **0.92**. AA-Ans fixes **123** zero-shot errors. AA-Trace helps **9** AA-Ans cases and hurts **16**; **19** cases remain hard, and the two primary AA-Ans seeds disagree on **30** examples. These are separate diagnostics, rather than a partition of the evaluation set.

![Figure 5: Case-level diagnostics](figure/Figure5.png)

*Figure 5. Primary-run prediction changes and seed disagreements.*

Nine trace regressions involve dominant-class or area-comparison questions, and seven of those have area margins below 0.10. See [case counts](results/case_diagnostics.csv), [trace support](results/trace_support_diagnostics.csv), [trace regressions](results/trace_regression_diagnostics.csv), and [representative cases](results/qualitative_cases.csv).

Compute summaries and paired diagnostics from saved predictions:

```bash
ease2026 summarize --output-root /data/ease/runs --backbone qwen \
  --seeds 20260618 20260619 20260620 --mode candidate \
  --output /data/ease/runs/qwen_candidate_summary.json

ease2026 analyze --output-root /data/ease/runs --backbone qwen \
  --seed 20260618 --second-seed 20260619 --mode generation \
  --data-dir /data/ease/vqa --output /data/ease/runs/qwen_generation_diagnostics

python scripts/plot_results.py --results-dir results --output plots
```

## Citation

If EASE supports your research, please cite the accepted **ACML 2026** paper. Proceedings volume, pages, and DOI can be added when available.

```bibtex
@inproceedings{zheng2026ease,
  title     = {EASE: Evidence-to-Answer Supervision for Disaster-Scene Vision-Language Adaptation},
  author    = {Zheng, Haoze and Han, Yaping},
  booktitle = {Proceedings of the Asian Conference on Machine Learning},
  year      = {2026}
}
```

Machine-readable metadata: [CITATION.cff](CITATION.cff).

## Authors

- **Haoze Zheng** — School of Computer Science and Technology, Xinjiang University; `zhenghaoze@stu.xju.edu.cn`
- **Yaping Han** — School of Geography and Remote Sensing Science, Xinjiang University; `20231203205@stu.xju.edu.cn`

## Acknowledgments

We thank the **FloodNet** creators for aerial imagery and segmentation annotations, the **Qwen** and **GLM** teams for their open models, and the developers of PyTorch, Transformers, PEFT, and the Python research ecosystem. Follow the dataset and model licenses when obtaining their assets. For code reuse terms, contact the authors.
