#!/usr/bin/env bash
set -e

PY=/media/caotulab/303A225B3A221DFA/envs/nina/bin/python

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORE_DIR="$PROJECT_ROOT/src/core"
DATA_DIR="$PROJECT_ROOT/data"

PREFIX="outcome"
MODEL_DIR="/media/caotulab/303A225B3A221DFA/Long/legal_outcome_classification_outputs/student_query_only"
TEST_DATA="$DATA_DIR/ALQAC2026_public_test.json"
OUTPUT="$PROJECT_ROOT/submissions/outcome_predictions.json"

GPU_IDS="0"
EVAL_BATCH_SIZE=4

cd "$CORE_DIR"

if [[ -n "${QUERY:-}" ]]; then
    "$PY" legal_outcome_classification_predict.py \
        --model-dir "$MODEL_DIR" \
        --gpu-ids "$GPU_IDS" \
        --query "$QUERY"
else
    "$PY" legal_outcome_classification_predict.py \
        --model-dir "$MODEL_DIR" \
        --data "$TEST_DATA" \
        --output "$OUTPUT" \
        --gpu-ids "$GPU_IDS" \
        --eval-batch-size "$EVAL_BATCH_SIZE"

    echo "Prediction on the test set finished. Output: $OUTPUT"
fi
