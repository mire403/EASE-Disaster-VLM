# Code inventory

| Path | Purpose |
|---|---|
| `src/floodnet_rcmtd/data/` | Image–mask pairing, sequence-aware splitting, six question families, and question audits |
| `src/floodnet_rcmtd/traces/programs.py` | Deterministic mask-evidence trace programs |
| `src/floodnet_rcmtd/traces/schema.py` | Validated trace records and execution steps |
| `src/floodnet_rcmtd/distillation/targets.py` | Direct-answer and trace-derived target construction |
| `src/floodnet_rcmtd/ease2026/prepare.py` | Split-specific JSONL, mask hashes, trace support, and area statistics |
| `src/floodnet_rcmtd/ease2026/protocol.py` | Four-cell definitions, shared order, PC/AA labels, and split checks |
| `src/floodnet_rcmtd/ease2026/runner.py` | CUDA QLoRA training, bounded image preprocessing, and two evaluation modes |
| `src/floodnet_rcmtd/ease2026/normalization.py` | Mapping generated text to allowed answer labels |
| `src/floodnet_rcmtd/ease2026/summary.py` | Accuracy, macro-F1, ECE, family metrics, entropy, and paired alignment gaps |
| `src/floodnet_rcmtd/ease2026/diagnostics.py` | Confusion matrices, paired cases, seed disagreements, and area/support margins |
| `scripts/run_experiments.sh` | Qwen/GLM experiment grids and saved logs |
| `scripts/plot_results.py` | Metric, family, and diagnostic plots from CSV results |
| `results/` | Paper aggregate tables and representative cases |
| `tests/` | Data, trace, protocol, and analysis tests |

See [EXPERIMENTS.md](EXPERIMENTS.md) for commands and output formats, and [the result index](../results/index.json) for paper coverage.
