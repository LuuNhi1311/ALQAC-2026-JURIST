#!/usr/bin/env bash
set -e

PY=/media/caotulab/303A225B3A221DFA/envs/nina/bin/python

echo "[1/2] deep_searcher (retrieve-then-read, LLM = azure/gpt-4o @ 1 call/10s, Case API qua ALQAC_TOKEN) ..."
"$PY" ensemble.py submission_deep.json submission_graph.json submission_deep_agents.json --output-path submission_ensemble.json

echo "[2/2] Local scoring (case-recall reported as 0: not measurable offline; the leaderboard computes it) ..."
"$PY" evaluate.py submission_ensemble.json

echo "Done. Submission file: submission_ensemble.json"
