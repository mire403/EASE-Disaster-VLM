"""Command line entry point for the EASE experiment protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ease2026")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="create split-safe train/test JSONL from FloodNet manifests")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--train-per-family", type=int, default=80)
    prepare.add_argument("--test-per-family", type=int, default=80)
    prepare.add_argument("--seed", type=int, default=20260618)

    audit = commands.add_parser("audit", help="verify hashes and zero cross-split overlap")
    audit.add_argument("--data-dir", type=Path, required=True)

    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--backbone", choices=("qwen", "glm"), required=True)
        command.add_argument("--model-path", required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--cell", choices=("PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace") +
                             (("Zero-shot",) if name == "evaluate" else ()), required=True)
        command.add_argument("--no-quantize", action="store_true")
        if name == "train":
            command.add_argument("--updates", type=int, default=480)
            command.add_argument("--accumulation", type=int, default=16)
            command.add_argument("--learning-rate", type=float, default=1e-4)
        else:
            command.add_argument("--mode", choices=("candidate", "generation"), default="candidate")
            command.add_argument("--max-new-tokens", type=int, default=32)

    summary = commands.add_parser("summarize")
    summary.add_argument("--output-root", type=Path, required=True)
    summary.add_argument("--backbone", choices=("qwen", "glm"), required=True)
    summary.add_argument("--seeds", type=int, nargs="+", required=True)
    summary.add_argument("--mode", choices=("candidate", "generation"), default="candidate")
    summary.add_argument("--cells", nargs="+", choices=("PC-Ans", "PC-Trace", "AA-Ans", "AA-Trace", "Zero-shot"), default=None)
    summary.add_argument("--output", type=Path, required=True)

    analyze = commands.add_parser("analyze", help="compare predictions and export case diagnostics")
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--backbone", choices=("qwen", "glm"), required=True)
    analyze.add_argument("--seed", type=int, required=True)
    analyze.add_argument("--second-seed", type=int)
    analyze.add_argument("--mode", choices=("candidate", "generation"), default="generation")
    analyze.add_argument("--data-dir", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from floodnet_rcmtd.ease2026.prepare import prepare_from_manifest
        result = prepare_from_manifest(args.manifest, args.output,
                                       train_per_family=args.train_per_family,
                                       test_per_family=args.test_per_family, seed=args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "audit":
        from floodnet_rcmtd.ease2026.runner import verify_data_dir
        _, manifest = verify_data_dir(args.data_dir)
        print(json.dumps(manifest["split_audit"], indent=2, sort_keys=True))
    elif args.command == "train":
        from floodnet_rcmtd.ease2026.runner import train_one
        print(train_one(args.data_dir, args.output_root, backbone=args.backbone,
                        model_path=args.model_path, seed=args.seed, cell=args.cell,
                        updates=args.updates, accumulation=args.accumulation,
                        learning_rate=args.learning_rate, quantize=not args.no_quantize))
    elif args.command == "evaluate":
        from floodnet_rcmtd.ease2026.runner import evaluate_one
        print(evaluate_one(args.data_dir, args.output_root, backbone=args.backbone,
                           model_path=args.model_path, seed=args.seed, cell=args.cell,
                           mode=args.mode, max_new_tokens=args.max_new_tokens,
                           quantize=not args.no_quantize))
    elif args.command == "summarize":
        from floodnet_rcmtd.ease2026.summary import collect_runs, write_summary
        result = collect_runs(args.output_root, args.backbone, args.seeds, args.mode,
                              cells=args.cells)
        write_summary(args.output, result)
        print(args.output)
    elif args.command == "analyze":
        from floodnet_rcmtd.ease2026.diagnostics import analyze_runs
        result = analyze_runs(args.output_root, args.backbone, args.seed, args.mode,
                              args.output, second_seed=args.second_seed, data_dir=args.data_dir)
        print(json.dumps(result["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
