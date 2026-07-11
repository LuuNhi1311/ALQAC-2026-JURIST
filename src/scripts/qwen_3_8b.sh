#!/usr/bin/env bash
export HF_HOME=/media/caotulab/303A225B3A221DFA/hf-cache
export VLLM_USE_FLASHINFER_SAMPLER=0
ENV_BIN=/media/caotulab/303A225B3A221DFA/envs/nina/bin
"$ENV_BIN/vllm" serve Qwen/Qwen3-8B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --trust-remote-code
