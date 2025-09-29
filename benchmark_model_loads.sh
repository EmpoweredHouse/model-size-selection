#!/usr/bin/env bash
set -euo pipefail

# Define variants here (3 values per line): run_name checkpoint_path merge_lora
VARIANTS=(
  "DistilBERT-full-false ./checkpoints/ex09_distilbert_full_ce_lr1.5e-5_wd0.02_wu8/best_checkpoint false"
  "DeBERTa_v2_xlarge-lora-false ./checkpoints/ex15_deberta_v2_xl_lora_stabilized_lr8e-5_wu10_r16/best_checkpoint false"
  "Gemma7B-lora-true ./checkpoints/ex08_gemma7b_lora_bigger_stable/best_checkpoint true"
  "RoBERTa-full-false ./checkpoints/ex12_roberta_full_baseline/best_checkpoint false"
  "Gemma2B-lora-false ./checkpoints/ex06_gemma2b_lora_bigger_lora/best_checkpoint false"
  "DistilBERT-lora-true ./checkpoints/ex10_distilbert_lora_r32_lr2e-4/best_checkpoint true"
  "DeBERTa_v2_xlarge-full-false ./checkpoints/ex14_deberta_v2_xl_fullft_baseline/best_checkpoint false"
  "Gemma7B-lora-false ./checkpoints/ex08_gemma7b_lora_bigger_stable/best_checkpoint false"
  "RoBERTa-lora-true ./checkpoints/ex11_roberta_lora_baseline/best_checkpoint true"
  "Gemma2B-full-false ./checkpoints/gemma-2b false"
  "DistilBERT-lora-false ./checkpoints/ex10_distilbert_lora_r32_lr2e-4/best_checkpoint false"
  "DeBERTa_v2_xlarge-lora-true ./checkpoints/ex15_deberta_v2_xl_lora_stabilized_lr8e-5_wu10_r16/best_checkpoint true"
  "Gemma7B-full-false ./checkpoints/gemma-7b false"
  "RoBERTa-lora-false ./checkpoints/ex11_roberta_lora_baseline/best_checkpoint false"
  "Gemma2B-lora-true ./checkpoints/ex06_gemma2b_lora_bigger_lora/best_checkpoint true"
)

RUNS=${RUNS:-11}

mkdir -p results
for i in $(seq 1 "$RUNS"); do
  echo "=== Iteration $i/$RUNS ==="
  for entry in "${VARIANTS[@]}"; do
    read -r RUN_NAME CKPT MERGE <<<"$entry"
    echo "--- Variant: $RUN_NAME (merge_lora=$MERGE) ---"
    BENCHMARK_RUNS=1 python benchmark_model_loads.py --run_name "$RUN_NAME" --checkpoint_path "$CKPT" --merge_lora "$MERGE"
    mv model_load_benchmark.json "results/${RUN_NAME//\//_}_iter_${i}.json"
  done
done

echo "Done. Aggregates can be produced with a separate script/jupyter."


