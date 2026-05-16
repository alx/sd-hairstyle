#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SD_BINARY="${SD_BINARY:-./bin/sd-cli}"
SD_MODEL="${SD_MODEL:-./models/turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors}"
SD_VAE="${SD_VAE:-}"
API_HOST="0.0.0.0"
API_PORT="${API_PORT:-5000}"

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}!${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; }

echo ""
echo "Maison Coupe — pre-flight check"
echo "────────────────────────────────"

errors=0

# ── 1. uv ─────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  fail "uv not found — install from https://github.com/astral-sh/uv"
  (( errors++ ))
else
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
fi

# ── 2. sd-cli binary ──────────────────────────────────────────────────────────
if [[ ! -f "$SD_BINARY" ]]; then
  fail "sd-cli not found: $SD_BINARY"
  warn "Set SD_BINARY=/path/to/sd-cli or place it at ./bin/sd-cli"
  (( errors++ ))
elif [[ ! -x "$SD_BINARY" ]]; then
  fail "sd-cli is not executable: $SD_BINARY"
  warn "Run: chmod +x $SD_BINARY"
  (( errors++ ))
else
  ok "sd-cli    $SD_BINARY"
fi

# ── 3. model file ─────────────────────────────────────────────────────────────
if [[ ! -f "$SD_MODEL" ]]; then
  fail "Model not found: $SD_MODEL"
  warn "Set SD_MODEL=/path/to/model.safetensors or place it at ./models/"
  (( errors++ ))
else
  size=$(du -h "$SD_MODEL" 2>/dev/null | cut -f1)
  ok "model     $SD_MODEL ($size)"
fi

# ── 4. VAE ────────────────────────────────────────────────────────────────────
if [[ -n "$SD_VAE" ]]; then
  if [[ ! -f "$SD_VAE" ]]; then
    fail "VAE not found: $SD_VAE (SD_VAE is set but file is missing)"
    (( errors++ ))
  else
    size=$(du -h "$SD_VAE" 2>/dev/null | cut -f1)
    ok "vae       $SD_VAE ($size)"
  fi
else
  ok "vae       baked into model — no external VAE needed"
fi

# ── 5. Tailscale ──────────────────────────────────────────────────────────────
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

echo "────────────────────────────────"

if (( errors > 0 )); then
  echo -e "${RED}Pre-flight failed — fix the errors above before starting.${RESET}"
  echo ""
  exit 1
fi

# ── Tailscale HTTPS proxy ──────────────────────────────────────────────────────
tailscale serve --bg --https=443 "http://127.0.0.1:${API_PORT}"

cleanup() {
  echo ""
  echo "Removing Tailscale HTTPS proxy…"
  tailscale serve --https=443 off 2>/dev/null || tailscale serve reset 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Launch ────────────────────────────────────────────────────────────────────
echo -e "${GREEN}All checks passed.${RESET}"
echo ""
echo -e "  Local:     http://127.0.0.1:${API_PORT}"
echo -e "  Tailscale: https://${TS_HOST}"
echo ""

export LD_LIBRARY_PATH="$(dirname "$(realpath "$SD_BINARY")")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run app.py
