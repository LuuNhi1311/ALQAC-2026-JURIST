#!/usr/bin/env bash
set -e

PY=/media/caotulab/303A225B3A221DFA/envs/nina/bin/python

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORE_DIR="$PROJECT_ROOT/src/core"
DATA_DIR="$PROJECT_ROOT/data"

RAW_DATA="$DATA_DIR/ALQAC2026_public_test.json"
PREFIX="outcome"

TRAIN_DATA="$DATA_DIR/${PREFIX}_train.json"
VAL_DATA="$DATA_DIR/${PREFIX}_val.json"
TEST_DATA="$DATA_DIR/${PREFIX}_test.json"

OUTPUT_DIR="/media/caotulab/303A225B3A221DFA/Long/legal_outcome_classification_outputs"

TEACHER_MODEL="Qualcomm-AI-Research/BamiBERT"
STUDENT_MODEL="Qualcomm-AI-Research/BamiBERT"
TEACHER_MAX_LENGTH=2048
STUDENT_MAX_LENGTH=512

GPU_IDS="0"
BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRAD_ACCUM=4
TEACHER_EPOCHS=500
STUDENT_EPOCHS=500
TEACHER_LR=2e-5
STUDENT_LR=3e-5
WARMUP_RATIO=0.10
LR_SCHEDULER=cosine
PATIENCE=15
FREEZE_LAYERS=6
KFOLD=5

CE_WEIGHT=0.7
ALPHA=0.5
BETA=0.7
TEMPERATURE=2.0

SEED=43
TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15

WANDB_PROJECT="alqac-legal-outcome-distillation"
WANDB_RUN_NAME="teacher-student-distill"

cd "$CORE_DIR"

echo "[1/2] Chia train/val/test ..."
"$PY" legal_outcome_split_data.py \
    --data "$RAW_DATA" \
    --output-dir "$DATA_DIR" \
    --prefix "$PREFIX" \
    --seed "$SEED" \
    --train-ratio "$TRAIN_RATIO" \
    --valid-ratio "$VAL_RATIO" \
    --test-ratio "$TEST_RATIO"

echo "[2/2] Train teacher + distil student on train/val ..."
"$PY" legal_outcome_classification_train.py \
    --train-data "$TRAIN_DATA" \
    --valid-data "$VAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --teacher-model "$TEACHER_MODEL" \
    --student-model "$STUDENT_MODEL" \
    --teacher-max-length "$TEACHER_MAX_LENGTH" \
    --student-max-length "$STUDENT_MAX_LENGTH" \
    --gpu-ids "$GPU_IDS" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --teacher-epochs "$TEACHER_EPOCHS" \
    --student-epochs "$STUDENT_EPOCHS" \
    --teacher-lr "$TEACHER_LR" \
    --student-lr "$STUDENT_LR" \
    --warmup-ratio "$WARMUP_RATIO" \
    --lr-scheduler "$LR_SCHEDULER" \
    --early-stopping-patience "$PATIENCE" \
    --freeze-encoder-layers "$FREEZE_LAYERS" \
    --freeze-embeddings \
    --kfold "$KFOLD" \
    --ce-weight "$CE_WEIGHT" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --no-push \
    --wandb-project-name "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME"

echo "Training finished. Test split reserved for predict: $TEST_DATA"
echo "Student saved at: $OUTPUT_DIR/student_query_only"
