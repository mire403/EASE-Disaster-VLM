#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <qwen|glm> <data-dir> <model-dir> <output-root> [seed ...]" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run EASE training and evaluation on the Linux GPU server." >&2
  exit 2
fi
backbone="$1"
data_dir="$2"
model_dir="$3"
output_root="$4"
shift 4
case "$backbone" in
  qwen) default_seeds=(20260618 20260619 20260620) ;;
  glm) default_seeds=(20260618) ;;
  *) echo "Backbone must be qwen or glm." >&2; exit 2 ;;
esac
seeds=("${default_seeds[@]}")
if [[ $# -gt 0 ]]; then seeds=("$@"); fi
primary_seed="${seeds[0]}"
mkdir -p "$output_root/logs"
run() {
  local log_name="$1"
  shift
  "$@" 2>&1 | tee "$output_root/logs/${backbone}_${log_name}.log"
}
run audit ease2026 audit --data-dir "$data_dir"
for mode in generation candidate; do
  run "zero_${mode}" ease2026 evaluate --data-dir "$data_dir" --output-root "$output_root" \
    --backbone "$backbone" --model-path "$model_dir" --seed "$primary_seed" --cell Zero-shot --mode "$mode"
done
for seed in "${seeds[@]}"; do
  for cell in PC-Ans PC-Trace AA-Ans AA-Trace; do
    run "${seed}_${cell}_train" ease2026 train --data-dir "$data_dir" --output-root "$output_root" \
      --backbone "$backbone" --model-path "$model_dir" --seed "$seed" --cell "$cell"
    for mode in generation candidate; do
      run "${seed}_${cell}_${mode}" ease2026 evaluate --data-dir "$data_dir" --output-root "$output_root" \
        --backbone "$backbone" --model-path "$model_dir" --seed "$seed" --cell "$cell" --mode "$mode"
    done
  done
done
for mode in generation candidate; do
  ease2026 summarize --output-root "$output_root" --backbone "$backbone" --seeds "${seeds[@]}" \
    --mode "$mode" --output "$output_root/${backbone}_${mode}_summary.json"
  ease2026 summarize --output-root "$output_root" --backbone "$backbone" --seeds "$primary_seed" \
    --mode "$mode" --cells Zero-shot --output "$output_root/${backbone}_${mode}_baseline.json"
  second_seed_args=()
  if [[ ${#seeds[@]} -gt 1 ]]; then second_seed_args=(--second-seed "${seeds[1]}"); fi
  ease2026 analyze --output-root "$output_root" --backbone "$backbone" --seed "$primary_seed" \
    "${second_seed_args[@]}" --mode "$mode" --data-dir "$data_dir" \
    --output "$output_root/${backbone}_${mode}_diagnostics"
done
