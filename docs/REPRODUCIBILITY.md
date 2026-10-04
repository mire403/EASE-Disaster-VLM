# Experiment records

Keep these records together for every experiment:

1. Dataset manifest, split assignments, question IDs, and SHA256 hashes.
2. Model and processor revisions, package versions, CUDA version, and GPU type.
3. Shared seeds, example order, PC/AA token labels, and optimizer settings.
4. `run.json`, `training_loss.csv`, adapters, and execution logs.
5. Per-example predictions and evaluation metadata for each mode.
6. Aggregate summaries, confusion matrices, paired case CSVs, and plot inputs.

Generation mode stores confidence as `null`. Use candidate mode for probability-based ECE. Compare runs only after the split and membership checks pass. See [EXPERIMENTS.md](EXPERIMENTS.md) for the complete workflow and [results/README.md](../results/README.md) for published table formats.
