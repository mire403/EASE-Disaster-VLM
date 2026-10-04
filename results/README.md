# Paper results

These files contain aggregate results reported in the EASE paper and representative qualitative cases. Metric columns ending in `_pct` use percentage units; accuracy differences ending in `_pp` use percentage points. Area margins, support, and entropy use normalized values.

| File | Experiment |
|---|---|
| `dataset_statistics.csv` | Dataset size, balanced evaluation set, and split checks reported in Section 3 |
| `qwen_three_seed.csv` | Table 4: three-seed Qwen3.5-9B comparison |
| `glm_single_seed.csv` | Table 5: GLM-4.6V-Flash comparison |
| `qwen_primary_seed.csv` | Primary-run generation and candidate scoring |
| `question_family_accuracy.csv` | Six-family accuracy and decoding comparison |
| `alignment_gap.csv` | Per-run and mean AA-versus-PC accuracy gaps |
| `prediction_behavior.csv` | Prediction concentration and normalized entropy |
| `dominant_class_diagnostics.csv` | Correct/error class-area margin and entropy |
| `trace_support_diagnostics.csv` | Trace/gold agreement and support margins |
| `trace_regression_diagnostics.csv` | Area-related and low-support trace regressions |
| `case_diagnostics.csv` | Five diagnostic counts used in Figure 5 |
| `qualitative_cases.csv` | Representative examples grouped by prediction changes |
| `index.json` | Mapping from paper experiments to code and result tables |

The three-seed summary and primary-run table describe different aggregates. Blank cells indicate quantities not reported in the corresponding table. The five case counts are separate diagnostics, rather than a partition of the evaluation set.

New experiment predictions are saved under `runs/<backbone>/seed<seed>/<cell>/eval_<mode>/predictions.jsonl`. Use `ease2026 summarize` and `ease2026 analyze` to compute metrics and paired case diagnostics from those predictions. Generation mode records no probability-based confidence; candidate mode supplies probabilities for ECE.
