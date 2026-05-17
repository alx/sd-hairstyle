#!/usr/bin/env bash
set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
ZIMAGE_MODEL="${ZIMAGE_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
ZIMAGE_CONTROLNET_ENABLED="${ZIMAGE_CONTROLNET_ENABLED:-0}"
ZIMAGE_CONTROLNET_MODEL="${ZIMAGE_CONTROLNET_MODEL:-alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1}"
API_HOST="0.0.0.0"
API_PORT="${API_PORT:-5000}"

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}!${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; }

echo ""
echo "Maison Coupe — pre-flight check (Z-Image backend)"
echo "──────────────────────────────────────────────────"

errors=0

# ── 1. uv ─────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  fail "uv not found — install from https://github.com/astral-sh/uv"
  (( errors++ ))
else
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
fi

# ── 2. Z-Image model (local path or HF repo ID) ───────────────────────────────
if [[ "$ZIMAGE_MODEL" == /* ]]; then
  # local path
  if [[ ! -d "$ZIMAGE_MODEL" ]]; then
    fail "ZIMAGE_MODEL local path not found: $ZIMAGE_MODEL"
    (( errors++ ))
  else
    ok "model     $ZIMAGE_MODEL (local)"
  fi
else
  ok "model     $ZIMAGE_MODEL (HuggingFace — downloaded on first run)"
  warn "First launch downloads ~10GB from HuggingFace Hub"
fi

# ── 3. ControlNet model (optional) ────────────────────────────────────────────
if [[ "$ZIMAGE_CONTROLNET_ENABLED" == "1" ]]; then
  if [[ "$ZIMAGE_CONTROLNET_MODEL" == /* ]]; then
    if [[ ! -d "$ZIMAGE_CONTROLNET_MODEL" && ! -f "$ZIMAGE_CONTROLNET_MODEL" ]]; then
      fail "ZIMAGE_CONTROLNET_MODEL not found: $ZIMAGE_CONTROLNET_MODEL"
      (( errors++ ))
    else
      ok "controlnet $ZIMAGE_CONTROLNET_MODEL (local)"
    fi
  else
    ok "controlnet $ZIMAGE_CONTROLNET_MODEL (HuggingFace)"
  fi
else
  ok "controlnet disabled (set ZIMAGE_CONTROLNET_ENABLED=1 to enable)"
fi

# ── 4. Tailscale ──────────────────────────────────────────────────────────────
TS_HOST=""
if ! command -v tailscale &>/dev/null; then
  fail "tailscale not found — install from https://tailscale.com/download"
  (( errors++ ))
else
  TS_JSON=$(tailscale status --json 2>/dev/null || echo "{}")
  TS_STATE=$(echo "$TS_JSON" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('BackendState',''))" 2>/dev/null || echo "")
  if [[ "$TS_STATE" != "Running" ]]; then
    fail "tailscale not connected (state: ${TS_STATE:-unknown}) — run: tailscale up"
    (( errors++ ))
  else
    TS_HOST=$(echo "$TS_JSON" | python3 -c \
      "import sys,json; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))" 2>/dev/null || echo "")
    ok "tailscale https://${TS_HOST}"
  fi
fi

echo "──────────────────────────────────────────────────"

if (( errors > 0 )); then
  echo -e "${RED}Pre-flight failed — fix the errors above before starting.${RESET}"
  echo ""
  exit 1
fi

# ── Supervisor setup ──────────────────────────────────────────────────────────
SUPERVISOR_LOG="./work/supervisor.log"

exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

echo -900 > /proc/self/oom_score_adj 2>/dev/null || true

# ── Tailscale HTTPS proxy ──────────────────────────────────────────────────────
tailscale serve --bg --https=443 "http://127.0.0.1:${API_PORT}"

SHUTDOWN=0
APP_PID=""
RESTART_COUNT=0
RESTART_DELAY=5

_ts() { date -u '+%H:%M:%S'; }

cleanup() {
  [[ "$SHUTDOWN" -eq 1 ]] && return
  SHUTDOWN=1
  echo ""
  echo "$(_ts) [supervisor] Shutting down…"
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    echo "$(_ts) [supervisor] Sending SIGTERM to process group $APP_PID…"
    kill -TERM -"$APP_PID" 2>/dev/null || kill -TERM "$APP_PID" 2>/dev/null || true
    local deadline=$((SECONDS + 15))
    while kill -0 "$APP_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    if kill -0 "$APP_PID" 2>/dev/null; then
      kill -KILL -"$APP_PID" 2>/dev/null || kill -KILL "$APP_PID" 2>/dev/null || true
    fi
  fi
  echo "$(_ts) [supervisor] Removing Tailscale HTTPS proxy…"
  tailscale serve --https=443 off 2>/dev/null || tailscale serve reset 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

# ── Supervisor loop ────────────────────────────────────────────────────────────
echo -e "${GREEN}All checks passed.${RESET}"
echo ""
echo -e "  Backend:   Z-Image Turbo (diffusers)"
echo -e "  Model:     ${ZIMAGE_MODEL}"
echo -e "  Local:     http://127.0.0.1:${API_PORT}"
echo -e "  Tailscale: https://${TS_HOST}"
echo -e "  Log:       $SUPERVISOR_LOG"
echo ""

export INFERENCE_BACKEND=zimage
export ZIMAGE_MODEL ZIMAGE_CONTROLNET_ENABLED ZIMAGE_CONTROLNET_MODEL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while true; do
  export APP_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  export APP_RESTART_COUNT="$RESTART_COUNT"

  echo -e "$(_ts) [supervisor] ${GREEN}Starting app.py (attempt $((RESTART_COUNT + 1)))…${RESET}"

  setsid uv run app.py &
  APP_PID=$!
  echo "$(_ts) [supervisor] app.py PID=$APP_PID"

  wait "$APP_PID"
  EXIT_CODE=$?
  APP_PID=""

  echo "$(_ts) [supervisor] app.py exited (code=${EXIT_CODE})"

  if (( SHUTDOWN )); then
    echo "$(_ts) [supervisor] Intentional shutdown — not restarting."
    break
  fi

  RESTART_COUNT=$(( RESTART_COUNT + 1 ))
  echo -e "$(_ts) [supervisor] ${YELLOW}Crashed — restart #${RESTART_COUNT} in ${RESTART_DELAY}s…${RESET}"
  sleep "$RESTART_DELAY"
done

echo "$(_ts) [supervisor] Exiting."
