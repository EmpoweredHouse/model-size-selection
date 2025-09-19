#!/bin/sh
set -e

# MODE can be: api | single | batch | eval
MODE="${MODE:-api}"

if [ "$MODE" = "api" ]; then
  exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WORKERS:-1}"
elif [ "$MODE" = "single" ]; then
  # Placeholders; override via environment variables
  CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs:/PLACEHOLDER_RUN_ID/PLACEHOLDER_PATH}"
  PREMISE="${PREMISE:-A soccer game with multiple males playing.}"
  HYPOTHESIS="${HYPOTHESIS:-Some men are playing a sport.}"
  MERGE_LORA_FLAG=""
  if [ "${MERGE_LORA:-false}" = "true" ]; then MERGE_LORA_FLAG="--merge_lora"; fi
  exec python model_inference.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --mode single \
    --premise "$PREMISE" \
    --hypothesis "$HYPOTHESIS" \
    $MERGE_LORA_FLAG
elif [ "$MODE" = "batch" ]; then
  CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs:/PLACEHOLDER_RUN_ID/PLACEHOLDER_PATH}"
  BATCH_JSON="${BATCH_JSON:-/app/batch.jsonl}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  MERGE_LORA_FLAG=""
  if [ "${MERGE_LORA:-false}" = "true" ]; then MERGE_LORA_FLAG="--merge_lora"; fi
  exec python model_inference.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --mode batch \
    --batch_json "$BATCH_JSON" \
    --batch_size "$BATCH_SIZE" \
    $MERGE_LORA_FLAG
elif [ "$MODE" = "eval" ]; then
  CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs:/PLACEHOLDER_RUN_ID/PLACEHOLDER_PATH}"
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-200}"
  MERGE_LORA_FLAG=""
  if [ "${MERGE_LORA:-false}" = "true" ]; then MERGE_LORA_FLAG="--merge_lora"; fi
  exec python model_inference.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --mode eval \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    $MERGE_LORA_FLAG
else
  echo "Unknown MODE: $MODE" >&2
  exit 1
fi


