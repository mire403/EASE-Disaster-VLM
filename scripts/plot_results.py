#!/usr/bin/env python3
"""Plot EASE aggregate tables without modifying paper image assets."""

import argparse
import csv
from pathlib import Path


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_results(results_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#718096", "#E69F00", "#CC79A7", "#0072B2", "#009E73"]
    methods = ["Zero-shot", "PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace"]
    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    rows = {row["method"]: row for row in read_rows(results_dir / "qwen_primary_seed.csv")
            if row["mode"] == "generation"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, key, title in zip(axes, ["accuracy_pct", "macro_f1_pct", "reported_ece_pct"],
                             ["Accuracy", "Macro-F1", "Reported ECE"]):
        values = [float(rows[method][key]) for method in methods]
        ax.bar(methods, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=40)
    save(fig, "primary_metrics")

    families = ["area_comparison", "damage_state", "dominant_class", "logical_conjunction", "presence", "spatial_adjacency"]
    rows = {(row["method"], row["question_family"]): row for row in read_rows(results_dir / "question_family_accuracy.csv")
            if row["mode"] == "generation"}
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for offset, method, color in [(-.24, "Zero-shot", colors[0]), (0, "AA-Ans", colors[3]), (.24, "AA-Trace", colors[4])]:
        values = [float(rows[(method, family)]["accuracy_pct"]) for family in families]
        ax.bar([index + offset for index in range(len(families))], values, width=.23, label=method, color=color)
    ax.set_xticks(range(len(families)), ["Area", "Damage", "Dominant", "Logic", "Presence", "Adjacency"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper center", ncols=3)
    save(fig, "family_accuracy")

    rows = read_rows(results_dir / "case_diagnostics.csv")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    labels = ["AA-Ans fixes\nzero-shot", "Trace helps\nAA-Ans", "Trace hurts\nAA-Ans", "Remaining\nhard", "Seed\ndisagreements"]
    bars = ax.bar(labels, [int(row["count"]) for row in rows], color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(int(row["count"]) for row in rows) * 1.18)
    ax.set_ylabel("Case count")
    save(fig, "case_diagnostics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.results_dir, args.output)
