#!/usr/bin/env bash

VLBIN=/home/longpm/miniconda3/envs/vllm-legal/bin
LOGDIR=/mnt/HDD6/longpm/alqac/logs
MODEL=VLSP2025-LegalSML/qwen3-4b-legal-pretrain
SERVED_NAME=vietnamese-law
GPU=3
PORT=8001
MQ_PORT=5555
HTTP_PORT=8081

export HF_HOME=/mnt/HDD6/longpm/alqac/hf
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then
  TOKEN=$(grep -E '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=' "$ENV_FILE" | tail -1 |
    sed -E 's/^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=//; s/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//')
  if [ -n "$TOKEN" ]; then
    export HF_TOKEN="$TOKEN"
    export HUGGING_FACE_HUB_TOKEN="$TOKEN"
  fi
  unset TOKEN
fi

mkdir -p "$LOGDIR" "$HF_HOME"

if [ -t 1 ]; then
  R=$'\033[0m'; B=$'\033[1m'; D=$'\033[2m'
  RED=$'\033[1;31m'; GRN=$'\033[1;32m'; YEL=$'\033[1;33m'
  BLU=$'\033[1;34m'; CYA=$'\033[1;36m'; MAG=$'\033[1;35m'
else
  R=""; B=""; D=""; RED=""; GRN=""; YEL=""; BLU=""; CYA=""; MAG=""
fi

SPIN='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
WIDTH=34

bar() {
  local pct=$1 label=$2 color=$3
  [ "$pct" -gt 100 ] 2>/dev/null && pct=100
  [ "$pct" -lt 0 ] 2>/dev/null && pct=0
  local filled=$(( pct * WIDTH / 100 ))
  local empty=$(( WIDTH - filled ))
  local fb="" eb=""
  [ "$filled" -gt 0 ] && fb=$(printf '━%.0s' $(seq 1 "$filled"))
  [ "$empty" -gt 0 ] && eb=$(printf '━%.0s' $(seq 1 "$empty"))
  printf "\r\033[K  ${color}%s${D}%s${R} ${B}%3d%%${R}  %s" "$fb" "$eb" "$pct" "$label"
}

spin_line() {
  local f=${SPIN:$(( $1 % 10 )):1}
  printf "\r\033[K  ${3}%s${R}  %s" "$f" "$2"
}

ok_line() { printf "\r\033[K  ${GRN}✔${R}  %s\n" "$1"; }
fail_line() { printf "\r\033[K  ${RED}✘${R}  %s\n" "$1"; }

model_bytes_total() {
  curl -s --max-time 15 "https://huggingface.co/api/models/${MODEL}?blobs=true" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = sum((f.get('size') or 0) for f in d.get('siblings', []) if f['rfilename'].endswith(('.safetensors', '.bin')))
    print(t)
except Exception:
    print(0)
" 2>/dev/null || echo 0
}

model_bytes_now() {
  local d
  d=$(find "$HF_HOME/hub" -maxdepth 1 -type d -name "*qwen3-4b-legal-pretrain*" 2>/dev/null | head -1)
  [ -z "$d" ] && { echo 0; return; }
  du -sb "$d" 2>/dev/null | cut -f1
}

human() { python3 -c "print(f'{$1/1e9:.1f} GB')" 2>/dev/null || echo "?"; }

printf "\n${MAG}╭────────────────────────────────────────────────────────────╮${R}\n"
printf "${MAG}│${R} ${B}vLLM serve${R}   ${CYA}%-46s${R}${MAG}│${R}\n" "$MODEL"
printf "${MAG}│${R} ${D}GPU %-3s · port %-5s · lmcache %s/%s${R}%*s${MAG}│${R}\n" "$GPU" "$PORT" "$MQ_PORT" "$HTTP_PORT" 21 ""
if [ -n "${HF_TOKEN:-}" ]; then
  printf "${MAG}│${R} ${D}HF_TOKEN ${GRN}loaded${R}${D} (%s…%s)${R}%*s${MAG}│${R}\n" "${HF_TOKEN:0:4}" "${HF_TOKEN: -3}" 27 ""
else
  printf "${MAG}│${R} ${D}HF_TOKEN ${YEL}missing${R}${D} — slower, rate limited${R}%*s${MAG}│${R}\n" 21 ""
fi
printf "${MAG}╰────────────────────────────────────────────────────────────╯${R}\n\n"

pkill -u "$(id -un)" -f "lmcache serv""er" 2>/dev/null
sleep 1

nohup "$VLBIN/lmcache" server \
  --host localhost \
  --port "$MQ_PORT" \
  --http-port "$HTTP_PORT" \
  --l1-size-gb 40 \
  --eviction-policy LRU \
  --chunk-size 256 \
  > "$LOGDIR/lmcache.log" 2>&1 &
LMC_PID=$!

i=0
while ! ss -ltn 2>/dev/null | grep -q ":${MQ_PORT}"; do
  if ! kill -0 "$LMC_PID" 2>/dev/null; then
    fail_line "lmcache died during startup"
    printf "${D}%s${R}\n" "$(tail -5 "$LOGDIR/lmcache.log")"
    exit 1
  fi
  spin_line "$i" "starting lmcache ..." "$YEL"
  i=$(( i + 1 ))
  sleep 0.4
done
ok_line "lmcache ready   ${D}zmq ${MQ_PORT} · http ${HTTP_PORT}${R}"

CUDA_VISIBLE_DEVICES="$GPU" nohup "$VLBIN/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port "$PORT" \
  > "$LOGDIR/vllm.log" 2>&1 &
VLLM_PID=$!

TOTAL=$(model_bytes_total)
i=0
phase=""

newphase() {
  if [ "$phase" != "$1" ]; then
    [ -n "$phase" ] && printf "\n"
    phase="$1"
  fi
}

while true; do
  if ss -ltn 2>/dev/null | grep -q ":${PORT}"; then
    [ -n "$phase" ] && printf "\n"
    ok_line "vLLM ready"
    break
  fi

  if ! kill -0 "$VLLM_PID" 2>/dev/null && ! pgrep -u "$(id -un)" -f "VLLM::" >/dev/null 2>&1; then
    [ -n "$phase" ] && printf "\n"
    fail_line "vLLM exited unexpectedly"
    printf "${D}%s${R}\n" "$(grep -E "Error|Traceback|OutOfMemory" "$LOGDIR/vllm.log" 2>/dev/null | tail -6)"
    exit 1
  fi

  if grep -q "Capturing CUDA graph\|Capturing cudagraph" "$LOGDIR/vllm.log" 2>/dev/null; then
    newphase graph
    spin_line "$i" "capturing CUDA graphs ..." "$BLU"
  elif grep -q "Loading safetensors checkpoint shards" "$LOGDIR/vllm.log" 2>/dev/null; then
    newphase load
    pct=$(tr '\r' '\n' < "$LOGDIR/vllm.log" | grep -o "Loading safetensors checkpoint shards: *[0-9]\+%" | tail -1 | grep -o "[0-9]\+" | tail -1)
    [ -z "$pct" ] && pct=0
    bar "$pct" "loading weights onto GPU ${GPU}" "$GRN"
  else
    NOW=$(model_bytes_now)
    if [ "${TOTAL:-0}" -gt 0 ] 2>/dev/null && [ "${NOW:-0}" -gt 0 ] 2>/dev/null; then
      newphase dl
      pct=$(( NOW * 100 / TOTAL ))
      bar "$pct" "downloading model   ${D}$(human "$NOW") / $(human "$TOTAL")${R}" "$CYA"
    else
      newphase init
      spin_line "$i" "initialising engine ..." "$YEL"
    fi
  fi

  i=$(( i + 1 ))
  sleep 0.5
done

MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU" 2>/dev/null)

printf "\n${GRN}╭────────────────────────────────────────────────────────────╮${R}\n"
printf "${GRN}│${R} ${B}READY${R}%*s${GRN}│${R}\n" 54 ""
printf "${GRN}├────────────────────────────────────────────────────────────┤${R}\n"
printf "${GRN}│${R} endpoint   ${CYA}%-46s${R}${GRN}│${R}\n" "http://localhost:${PORT}/v1"
printf "${GRN}│${R} model      ${YEL}%-46s${R}${GRN}│${R}\n" "$SERVED_NAME"
printf "${GRN}│${R} GPU ${GPU}      ${D}%-46s${R}${GRN}│${R}\n" "$MEM"
printf "${GRN}│${R} log        ${D}%-46s${R}${GRN}│${R}\n" "$LOGDIR/vllm.log"
printf "${GRN}╰────────────────────────────────────────────────────────────╯${R}\n\n"
