export HF_HOME=/mnt/HDD6/longpm/alqac/hf
PY=python
SERVICES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../services" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMISSION_DIR="/mnt/HDD6/longpm/alqac/submissions"

LLM=VLSP2025-LegalSML/qwen3-4b-legal-pretrain
VLLM_API_BASE=http://localhost:8001/v1
VLLM_SERVED_NAME=vietnamese-law

TEST_PATH="$ROOT/data/ALQAC_private_test.json"
LAW_PATH="$ROOT/data/private_test_60_cases_extracted_corpus.json"
CACHE_DIR="$ROOT/.cache"
VECTOR_PATH="$ROOT/.cache/milvus_deepsearcher.db"
COLLECTION=alqac_law_corpus
DENSE_MODEL=hiieu/halong_embedding

INDEX_ARGS=(
  --law-path "$LAW_PATH"
  --cache-dir "$CACHE_DIR"
  --vector-path "$VECTOR_PATH"
  --collection "$COLLECTION"
  --dense-model "$DENSE_MODEL"
)

LLM_ARGS=(
  --llm-model "$VLLM_SERVED_NAME"
  --llm-base-url "$VLLM_API_BASE"
  --llm-api-key EMPTY
  --no-fallback
)

MODE="${1:-search}"
mkdir -p "$SUBMISSION_DIR"
cd "$SERVICES"

case "$MODE" in
  index)
    $PY deep_searcher.py "${INDEX_ARGS[@]}" --index-only
    ;;
  search)
    $PY deep_searcher.py \
      "${LLM_ARGS[@]}" "${INDEX_ARGS[@]}" \
      --test-path "$TEST_PATH" \
      --output-path "$SUBMISSION_DIR/submission_deep_searcher.json"
    ;;
  *)
    echo "usage: bash deep_searcher.sh [index|search]"
    exit 1
    ;;
esac
