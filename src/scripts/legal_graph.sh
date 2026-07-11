export HF_HOME=/media/caotulab/303A225B3A221DFA/hf_cache
PY=/media/caotulab/303A225B3A221DFA/envs/nina/bin/python
SERVICES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../services" && pwd)"
SUBMISSIONS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../submissions" && pwd)"
cd "$SERVICES"
$PY legal_graph.py --index --recreate
$PY legal_graph.py --output-path "$SUBMISSIONS/submission_legal_graph.json"
