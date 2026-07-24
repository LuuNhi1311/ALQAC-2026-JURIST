export HF_HOME=/mnt/HDD6/longpm/alqac/hf
PY=python
SERVICES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../services" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMISSION_DIR="/mnt/HDD6/longpm/alqac/submissions"

LLM=VLSP2025-LegalSML/qwen3-4b-legal-pretrain
VLLM_API_BASE=http://localhost:8001/v1
VLLM_SERVED_NAME=vietnamese-law

LEADERBOARD="public"
INPUT_PATH=$ROOT/data/ALQAC2026_public_test.json
CORPUS_PATH=$ROOT/data/corpus_law_pub.json
GRAPH_DB_PATH=$ROOT/.cache/graph_db
COLLECTION_NAME=$ROOT/.cache/${GRAPH_DB_PATH}/alpac_${LEADERBOARD}_law_corpus.pkl

DATA_ARGS=(
  --corpus-path "$CORPUS_PATH"
  --input-path "$INPUT_PATH"
  --graph-db-path "$GRAPH_DB_PATH"
  --collection-name "$COLLECTION_NAME"
)

LLM_ARGS=(
  --llm-provider vllm
  --vllm-api-base "$VLLM_API_BASE"
  --vllm-model-name "$VLLM_SERVED_NAME"
)

MODE="${1:-search}"
mkdir -p "$SUBMISSION_DIR" "$GRAPH_DB_PATH"
cd "$SERVICES"

case "$MODE" in
  index)
    $PY legal_graph.py --index --recreate "${DATA_ARGS[@]}"
    ;;
  search)
    $PY legal_graph.py "${LLM_ARGS[@]}" "${DATA_ARGS[@]}" --output-path "$SUBMISSION_DIR/submission_${LEADERBOARD}_legal_graph.json"
    ;;
  *)
    echo "usage: bash legal_graph.sh [index|search]"
    exit 1
    ;;
esac
