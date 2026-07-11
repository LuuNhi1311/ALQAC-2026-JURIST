#!/usr/bin/env bash

ME=$(id -un)
PAT="VLLM::|vllm ser""ve|lmcache serv""er"

if [ -t 1 ]; then
  R=$'\033[0m'; B=$'\033[1m'; D=$'\033[2m'
  RED=$'\033[1;31m'; GRN=$'\033[1;32m'; YEL=$'\033[1;33m'; CYA=$'\033[1;36m'
else
  R=""; B=""; D=""; RED=""; GRN=""; YEL=""; CYA=""
fi

SPIN='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

printf "\n${YEL}╭────────────────────────────────────────────────────────────╮${R}\n"
printf "${YEL}│${R} ${B}Stopping vLLM + lmcache${R}%*s${YEL}│${R}\n" 36 ""
printf "${YEL}╰────────────────────────────────────────────────────────────╯${R}\n\n"

N=$(pgrep -u "$ME" -f "$PAT" 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then
  printf "  ${GRN}✔${R}  nothing running\n\n"
else
  printf "  ${CYA}•${R}  found ${B}%s${R} process(es)\n" "$N"
  pkill -u "$ME" -f "VLLM::" 2>/dev/null
  pkill -u "$ME" -f "vllm ser""ve" 2>/dev/null
  pkill -u "$ME" -f "lmcache serv""er" 2>/dev/null

  i=0
  while pgrep -u "$ME" -f "$PAT" >/dev/null 2>&1 && [ "$i" -lt 30 ]; do
    printf "\r\033[K  ${YEL}%s${R}  stopping ..." "${SPIN:$(( i % 10 )):1}"
    i=$(( i + 1 ))
    sleep 0.5
  done

  if pgrep -u "$ME" -f "$PAT" >/dev/null 2>&1; then
    printf "\r\033[K  ${RED}!${R}  force killing (SIGKILL)\n"
    pkill -9 -u "$ME" -f "VLLM::" 2>/dev/null
    pkill -9 -u "$ME" -f "vllm ser""ve" 2>/dev/null
    pkill -9 -u "$ME" -f "lmcache serv""er" 2>/dev/null
    sleep 2
  fi

  if pgrep -u "$ME" -f "$PAT" >/dev/null 2>&1; then
    printf "\r\033[K  ${RED}✘${R}  still alive (likely state D, blocked on I/O)\n"
    pgrep -u "$ME" -af "$PAT" | sed 's/^/       /'
  else
    printf "\r\033[K  ${GRN}✔${R}  stopped cleanly\n"
  fi
  printf "\n"
fi

printf "  ${B}GPU${R}\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | while IFS=, read -r idx name used total; do
  u=$(echo "$used" | tr -dc '0-9')
  if [ "${u:-0}" -gt 1000 ]; then c="$YEL"; else c="$GRN"; fi
  printf "    ${c}GPU %s${R} ${D}%-22s${R} %10s /%10s\n" "$idx" "$name" "$used" "$total"
done

printf "\n  ${B}Processes using GPU${R}\n"
APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
if [ -z "$APPS" ]; then
  printf "    ${D}(none)${R}\n"
else
  echo "$APPS" | while IFS=, read -r pid mem; do
    pid=$(echo "$pid" | tr -d ' ')
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | cut -c1-38)
    if [ "$owner" = "$ME" ]; then c="$YEL"; tag="you"; else c="$CYA"; tag="$owner"; fi
    printf "    ${c}%-8s${R} %10s  ${D}%-8s %s${R}\n" "$pid" "$mem" "$tag" "$cmd"
  done
fi
printf "\n"
