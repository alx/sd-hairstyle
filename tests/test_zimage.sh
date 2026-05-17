#!/usr/bin/env bash
# Standalone smoke test for Z-Image Turbo diffusers pipelines.
# Does NOT require the Flask app to be running.
# Downloads Tongyi-MAI/Z-Image-Turbo on first run (~10 GB).
#
# Usage:
#   bash tests/test_zimage.sh
#   ZIMAGE_MODEL=/local/path bash tests/test_zimage.sh

set -euo pipefail

SELFIE="$(cd "$(dirname "$0")/.." && pwd)/selfie.png"
MODEL="${ZIMAGE_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
OUTDIR="$(mktemp -d /tmp/zimage_test_XXXXX)"

echo "=== Z-Image Turbo smoke test ==="
echo "Model:  $MODEL"
echo "Selfie: $SELFIE"
echo "Outdir: $OUTDIR"
echo ""

[[ -f "$SELFIE" ]] || { echo "[FAIL] selfie.png not found at $SELFIE"; exit 1; }

# ── Test 1: img2img ────────────────────────────────────────────────────────────
echo "--- Test 1: img2img (ZImageImg2ImgPipeline) ---"
SELFIE_PATH="$SELFIE" MODEL_ID="$MODEL" OUTDIR="$OUTDIR" \
uv run python - <<'PYEOF'
import os, torch
from diffusers import ZImageImg2ImgPipeline
from PIL import Image

selfie = os.environ["SELFIE_PATH"]
model  = os.environ["MODEL_ID"]
outdir = os.environ["OUTDIR"]

pipe = ZImageImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

init = Image.open(selfie).convert("RGB").resize((512, 512))
result = pipe(
    "portrait photo, person with long dark wavy hair, photorealistic, studio lighting",
    image=init,
    strength=0.6,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

out = f"{outdir}/img2img.png"
result.save(out)

img = Image.open(out)
assert img.size == (512, 512), f"Expected (512,512), got {img.size}"
print(f"[PASS] img2img: {out}")
PYEOF

echo ""

# ── Test 2: txt2img ────────────────────────────────────────────────────────────
echo "--- Test 2: txt2img (ZImagePipeline) ---"
MODEL_ID="$MODEL" OUTDIR="$OUTDIR" \
uv run python - <<'PYEOF'
import os, torch
from diffusers import ZImagePipeline

model  = os.environ["MODEL_ID"]
outdir = os.environ["OUTDIR"]

pipe = ZImagePipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

result = pipe(
    "portrait photo, person with short blonde pixie cut, photorealistic, studio lighting",
    height=512,
    width=512,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

out = f"{outdir}/txt2img.png"
result.save(out)
print(f"[PASS] txt2img: {out}")
PYEOF

echo ""
echo "=== All tests passed. Outputs: $OUTDIR ==="
