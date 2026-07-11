#!/usr/bin/env bash
set -uo pipefail

INDEX=1
PREDICTOR_KIND=llm
LIMIT=1

GPU_IDS=5

DENSE_MODEL=hiieu/halong_embedding
SPARSE_MODEL=Qdrant/bm25
RERANK_MODEL=AITeamVN/Vietnamese_Reranker
USE_RERANKER=true
LAW_RERANK_TOPK=8
LAW_RERANK_MIN_SCORE=0.5

LLM=VLSP2025-LegalSML/qwen3-4b-legal-pretrain
VLLM_API_BASE=http://localhost:8001/v1
VLLM_SERVED_NAME=vietnamese-law

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMISSION_DIR="/mnt/HDD6/longpm/alqac/submissions"

COLLECTION_NAME=alqac_private_law_corpus
INPUT_PATH=$ROOT/data/ALQAC_private_test.json
CORPUS_PATH=$ROOT/data/private_test_60_cases_extracted_corpus.json
QDRANT_LOCAL_PATH=$ROOT/.cache/qdrant_$COLLECTION_NAME

export HF_HOME=/mnt/HDD6/longpm/alqac/hf
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ "$LLM" == azure/* ]]; then
  PROVIDER=azure
  LLM_ARGS=(--llm_provider azure --llm_model_name "$LLM" --llm_min_interval 5)
else
  PROVIDER=vllm
  LLM_ARGS=(--llm_provider vllm --vllm_api_base "$VLLM_API_BASE" --vllm_model_name "$VLLM_SERVED_NAME" --llm_min_interval 0)
fi

ARGS=(
  --input-path "$INPUT_PATH"
  --corpus-path "$CORPUS_PATH"
  --collection_name "$COLLECTION_NAME"
  --qdrant_local_path "$QDRANT_LOCAL_PATH"
  --dense_model_name "$DENSE_MODEL"
  --sparse_model_name "$SPARSE_MODEL"
)

mkdir -p "$SUBMISSION_DIR"
cd "$ROOT/src/services"

RUN=(
  "${ARGS[@]}" "${LLM_ARGS[@]}"
  --gpu-ids "$GPU_IDS"
  --use_reranker "$USE_RERANKER"
  --rerank_model_name "$RERANK_MODEL"
  --law_rerank_topk "$LAW_RERANK_TOPK"
  --law_rerank_min_score "$LAW_RERANK_MIN_SCORE"
  --llm_temperature 0
  --llm_max_retries 5
  --case_api_url https://alqac-api.ngrok.pro/retrieve
  --api_min_interval 5.5
  --law_search_limit 15
  --max_retrieval_iterations 2
  --max_case_queries 16
  --max_law_evidence 12
  --predictor_kind "$PREDICTOR_KIND"
  --outcome_temperature 0.4
  --law_select true
  --law_evidence_target 8
  --trace_output "$ROOT/output.json"
  --output-path "$SUBMISSION_DIR/submission_deep_agents_${PROVIDER}.json"
)
[ "$INDEX" = 1 ] && RUN=(--index --recreate "${RUN[@]}")
[ -n "$LIMIT" ] && RUN+=(--limit "$LIMIT")

python deep_agents.py "${RUN[@]}"
