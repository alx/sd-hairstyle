#!/usr/bin/env bash
# Smoke test for Z-Image ControlNet Union pipeline.
# Downloads both the base model and ControlNet model on first run.
# Uses Canny edges from selfie.png as the control image.
#
# Usage:
#   bash tests/test_zimage_controlnet.sh
#   ZIMAGE_MODEL=Tongyi-MAI/Z-Image-Turbo \
#   ZIMAGE_CONTROLNET_MODEL=alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
#   bash tests/test_zimage_controlnet.sh

set -euo pipefail

SELFIE="$(cd "$(dirname "$0")/.." && pwd)/selfie.png"
MODEL="${ZIMAGE_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
CN_MODEL="${ZIMAGE_CONTROLNET_MODEL:-alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1}"
OUTDIR="$(mktemp -d /tmp/zimage_cn_test_XXXXX)"

echo "=== Z-Image ControlNet Union smoke test ==="
echo "Model:      $MODEL"
echo "ControlNet: $CN_MODEL"
echo "Selfie:     $SELFIE"
echo "Outdir:     $OUTDIR"
echo ""

[[ -f "$SELFIE" ]] || { echo "[FAIL] selfie.png not found at $SELFIE"; exit 1; }

# ── Generate Canny control image from selfie ───────────────────────────────────
echo "--- Step 1: generate Canny control image ---"
SELFIE_PATH="$SELFIE" OUTDIR="$OUTDIR" \
uv run python - <<'PYEOF'
import os, cv2, numpy as np
from PIL import Image

selfie = os.environ["SELFIE_PATH"]
outdir = os.environ["OUTDIR"]

img  = cv2.imread(selfie)
img  = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
control = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

out = f"{outdir}/canny.png"
cv2.imwrite(out, control)
assert os.path.isfile(out)
print(f"[OK] Canny control image: {out}")
PYEOF

echo ""

# ── Run ZImageControlNetPipeline ───────────────────────────────────────────────
# NOTE: guidance_scale must be 0.0 — CFG > 1 causes a shape mismatch in
# ZImageControlNetPipeline (known issue: huggingface/diffusers#13073).
echo "--- Step 2: ZImageControlNetPipeline ---"
MODEL_ID="$MODEL" CN_MODEL_ID="$CN_MODEL" OUTDIR="$OUTDIR" \
uv run python - <<'PYEOF'
import os, torch
from PIL import Image
from diffusers import ZImageControlNetPipeline, ZImageControlNetModel

model    = os.environ["MODEL_ID"]
cn_model = os.environ["CN_MODEL_ID"]
outdir   = os.environ["OUTDIR"]

control = Image.open(f"{outdir}/canny.png").convert("RGB")

cn   = ZImageControlNetModel.from_pretrained(cn_model, torch_dtype=torch.bfloat16)
pipe = ZImageControlNetPipeline.from_pretrained(model, controlnet=cn, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

result = pipe(
    "portrait photo, person with flowing auburn hair, photorealistic, studio lighting",
    control_image=control,
    controlnet_conditioning_scale=0.75,
    height=512,
    width=512,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

out = f"{outdir}/controlnet_result.png"
result.save(out)

img = Image.open(out)
assert img.size == (512, 512), f"Expected (512,512), got {img.size}"
print(f"[PASS] ControlNet result: {out}")
PYEOF

echo ""
echo "=== ControlNet test passed. Outputs: $OUTDIR ==="
