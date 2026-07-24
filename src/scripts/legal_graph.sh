export HF_HOME=/mnt/HDD6/longpm/alqac/hf
PY=python
SERVICES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../services" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMISSION_DIR="/mnt/HDD6/longpm/alqac/submissions"

LLM=VLSP2025-LegalSML/qwen3-4b-legal-pretrain
VLLM_API_BASE=http://localhost:8001/v1
VLLM_SERVED_NAME=vietnamese-law

INPUT_PATH=$ROOT/data/ALQAC_private_test.json
CORPUS_PATH=$ROOT/data/private_test_60_cases_extracted_corpus.json
GRAPH_PATH=$ROOT/.cache/legal_graph_db.pkl

DATA_ARGS=(
  --corpus-path "$CORPUS_PATH"
  --input-path "$INPUT_PATH"
  --graph-path "$GRAPH_PATH"
)

LLM_ARGS=(
  --llm-provider vllm
  --vllm-api-base "$VLLM_API_BASE"
  --vllm-model-name "$VLLM_SERVED_NAME"
)

MODE="${1:-search}"
mkdir -p "$SUBMISSION_DIR"
cd "$SERVICES"

case "$MODE" in
  index)
    $PY legal_graph.py --index --recreate "${DATA_ARGS[@]}"
    ;;
  search)
    $PY legal_graph.py "${LLM_ARGS[@]}" "${DATA_ARGS[@]}" --output-path "$SUBMISSION_DIR/submission_legal_graph.json"
    ;;
  *)
    echo "usage: bash legal_graph.sh [index|search]"
    exit 1
    ;;
esac
