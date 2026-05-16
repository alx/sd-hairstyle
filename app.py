"""
Haircut Transfer API — diffusers SDXL pipeline
POST /transfer-haircut  multipart/form-data
  selfie     : image file  (the subject)
  hairstyle  : image file  (the reference haircut)
  prompt     : str (optional extra prompt)
  strength   : float 0.0–1.0 (adaptive by target length; override to pin)
  steps      : int   (default 5, TurboVisionXL optimized)
  cfg_scale  : float (default 2.0, TurboVisionXL optimized)
  seed       : int   (default -1 = random)
  width      : int   (default 1024)
  height     : int   (default 1024)

Returns: JPEG image bytes  (Content-Type: image/jpeg)
"""

import gc
import hashlib
import io
import math
import os
import uuid
import threading
import logging

import torch

from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageStat

# ──────────────────────────────────────────────────────────────
# CONFIG  — edit these or use environment variables
# ──────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("SD_MODEL",     "./models/turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors")
VAE_PATH    = os.getenv("SD_VAE",       "")
WORK_DIR    = os.getenv("SD_WORK_DIR",  "./work")
os.makedirs(WORK_DIR, exist_ok=True)
HOST        = "0.0.0.0"
PORT        = int(os.getenv("API_PORT", "5000"))

# ── ControlNet ─────────────────────────────────────────────────
CONTROLNET_ENABLED  = os.getenv("CONTROLNET_ENABLED", "0") == "1"
CONTROLNET_MODEL    = os.getenv(
    "SD_CONTROLNET_MODEL",
    "/home/alx/Documents/models/diffusion/ControlNet/control-lora-depth-rank128.safetensors",
)
CONTROLNET_STRENGTH = float(os.getenv("SD_CONTROLNET_STRENGTH", "0.6"))
CONTROLNET_CPU      = os.getenv("SD_CONTROLNET_CPU", "0") == "1"
CANNY_LOW           = int(os.getenv("CANNY_THRESHOLD_LOW",  "50"))
CANNY_HIGH          = int(os.getenv("CANNY_THRESHOLD_HIGH", "150"))

# ── InsightFace / face swap ────────────────────────────────────
FACESWAP_ENABLED  = os.getenv("FACESWAP_ENABLED", "1") == "1"
INPAINT_ENABLED   = os.getenv("INPAINT_ENABLED",  "1") == "1"
INSIGHTFACE_ROOT  = os.getenv(
    "INSIGHTFACE_ROOT",
    "/home/alx/Documents/models/diffusion/insightface",
)
INSWAPPER_MODEL   = os.getenv(
    "INSWAPPER_MODEL",
    "/home/alx/Documents/models/diffusion/faceswap/inswapper_128.onnx",
)
# ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# File-based trace log — survives process crashes that swallow stderr
_TRACE_LOG = os.path.join(WORK_DIR, "inference_trace.log")
_fh = logging.FileHandler(_TRACE_LOG)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.setLevel(logging.DEBUG)
log.addHandler(_fh)

app = Flask(__name__, static_folder='public', static_url_path='')


# ── Diffusers pipeline singleton ──────────────────────────────
#
# VRAM budget (RTX 4070 Laptop, 7.6 GiB usable):
#   _pipe_base : ~6.5 GiB on CUDA (UNet + VAE + text encoders)
#   _pipe_cn   : independent pipeline with enable_sequential_cpu_offload().
#                Weights live in CPU RAM; only one transformer block is on
#                GPU at a time → peak VRAM ~1–2 GiB during inference.
#                Inference is ~5–10× slower than _pipe_base but always fits.
#
# from_single_file() fetches architecture JSON configs from HuggingFace on the
# first load (the .safetensors file stores weights only, not model structure).
# After the first run configs are cached in ~/.cache/huggingface/hub/.
# Set HF_HUB_OFFLINE=1 to use local cache only.

_pipe_lock    = threading.Lock()
_pipe_base    = None   # StableDiffusionXLImg2ImgPipeline — model CPU offload
_pipe_cn      = None   # StableDiffusionXLControlNetImg2ImgPipeline — sequential CPU offload
_pipe_inpaint = None   # StableDiffusionXLInpaintPipeline — for short→long hair transformations

_HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"


def _cuda_mem_stats() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc  = torch.cuda.memory_allocated()  / 1024**3
    reserv = torch.cuda.memory_reserved()   / 1024**3
    peak   = torch.cuda.max_memory_allocated() / 1024**3
    total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"alloc={alloc:.2f}G reserved={reserv:.2f}G peak={peak:.2f}G total={total:.2f}G"


def _free_gpu_memory() -> None:
    """Delete pipeline singletons and flush CUDA memory.

    enable_model_cpu_offload() registers forward hooks on every submodule
    via accelerate. Those hook closures hold references back into the pipeline,
    so _pipe_base = None alone does not drop the refcount to zero and the
    object is never garbage-collected. remove_hook_from_submodules() tears
    down the hook chain first so the subsequent None assignment actually frees
    the weights.
    """
    global _pipe_base, _pipe_cn, _pipe_inpaint
    try:
        from accelerate.hooks import remove_hook_from_submodules as _rm_hooks
    except ImportError:
        _rm_hooks = None

    with _pipe_lock:
        for pipe in (_pipe_base, _pipe_cn, _pipe_inpaint):
            if pipe is not None:
                if _rm_hooks is not None:
                    try:
                        _rm_hooks(pipe)
                    except Exception as exc:
                        log.debug("Hook removal: %s", exc)
        _pipe_base    = None
        _pipe_cn      = None
        _pipe_inpaint = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.debug("GPU memory after free: %s", _cuda_mem_stats())


def _load_control_lora(path, local_only):
    """Load a control-lora safetensors file as a ControlNetModel.

    control-lora weights split linear layers into LoRA .down/.up pairs instead
    of storing the full .weight tensor.  diffusers' from_single_file() uses
    "time_embed.0.weight" to detect the LDM controlnet format; when it's absent
    the converter takes the wrong branch and produces None-valued entries that
    crash the weight loader.  Injecting the two time-embed projections (computed
    from their LoRA factors) is enough to fix the detection path; the remaining
    LoRA pairs land in unexpected_keys and are silently skipped, which matches
    the original loading behaviour when the HF config URL was still reachable.
    """
    from diffusers import ControlNetModel
    from safetensors.torch import load_file as _load_sf

    ckpt = _load_sf(path)

    # Inject synthetic full weights for the time-embed projections so the
    # format-detection branch in convert_controlnet_checkpoint works correctly.
    for prefix in ("time_embed.0", "time_embed.2"):
        wk = f"{prefix}.weight"
        if wk not in ckpt and f"{prefix}.down" in ckpt:
            ckpt[wk] = (ckpt[f"{prefix}.up"] @ ckpt[f"{prefix}.down"]).to(torch.float16)

    # low_cpu_mem_usage=False: materialises all parameters to CPU before
    # applying the (partial) state dict.  With the default (True), diffusers
    # leaves meta-device tensors for any param that's absent from the state
    # dict; those meta tensors then crash enable_sequential_cpu_offload().
    return ControlNetModel.from_single_file(
        ckpt,
        config="diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        local_files_only=local_only,
        low_cpu_mem_usage=False,
    )


def _make_scheduler(config):
    from diffusers import DPMSolverSinglestepScheduler
    return DPMSolverSinglestepScheduler.from_config(
        config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )


def _get_pipeline(with_controlnet: bool = False, inpaint: bool = False):
    global _pipe_base, _pipe_cn, _pipe_inpaint
    from diffusers import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    )
    local_only = _HF_OFFLINE
    with _pipe_lock:
        if inpaint:
            if _pipe_inpaint is None:
                if not local_only:
                    log.info(
                        "First load: fetching SDXL architecture configs from HuggingFace "
                        "(cached after this — set HF_HUB_OFFLINE=1 to go fully offline)"
                    )
                log.info("Loading SDXL inpaint pipeline from %s", MODEL_PATH)
                _pipe_inpaint = StableDiffusionXLInpaintPipeline.from_single_file(
                    MODEL_PATH, torch_dtype=torch.float16, local_files_only=local_only,
                )
                _pipe_inpaint.scheduler = _make_scheduler(_pipe_inpaint.scheduler.config)
                _pipe_inpaint.vae.enable_tiling()
                _pipe_inpaint.vae.enable_slicing()
                _pipe_inpaint.enable_attention_slicing()
                _pipe_inpaint.enable_model_cpu_offload()
                log.info("Inpaint pipeline ready")
            return _pipe_inpaint

        if not with_controlnet:
            # ── base pipeline: model CPU offload keeps text encoders/VAE on
            # CPU until needed, so peak VRAM is ~3.5–4 GiB (UNet + activations)
            # rather than the full 6.5 GiB that .to("cuda") requires. Much
            # faster than sequential CPU offload.
            if _pipe_base is None:
                if not local_only:
                    log.info(
                        "First load: fetching SDXL architecture configs from HuggingFace "
                        "(cached after this — set HF_HUB_OFFLINE=1 to go fully offline)"
                    )
                log.info("Loading SDXL img2img pipeline from %s", MODEL_PATH)
                _pipe_base = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    MODEL_PATH, torch_dtype=torch.float16, local_files_only=local_only,
                )
                _pipe_base.scheduler = _make_scheduler(_pipe_base.scheduler.config)
                _pipe_base.vae.enable_tiling()
                _pipe_base.vae.enable_slicing()
                _pipe_base.enable_attention_slicing()
                _pipe_base.enable_model_cpu_offload()
                log.info("img2img pipeline ready")
            return _pipe_base

        # ── ControlNet pipeline: independent load + sequential CPU offload ──
        # enable_sequential_cpu_offload() moves each transformer block to GPU
        # one at a time, so peak VRAM is ~1–2 GiB regardless of model size.
        # Inference is ~5–10× slower than the base pipeline.
        if _pipe_cn is None:
            if not local_only:
                log.info(
                    "First load: fetching SDXL architecture configs from HuggingFace "
                    "(cached after this — set HF_HUB_OFFLINE=1 to go fully offline)"
                )
            log.info("Loading ControlNet model from %s", CONTROLNET_MODEL)
            controlnet = _load_control_lora(CONTROLNET_MODEL, local_only)
            log.info("Loading SDXL ControlNet pipeline from %s", MODEL_PATH)
            _pipe_cn = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                MODEL_PATH, controlnet=controlnet,
                torch_dtype=torch.float16, local_files_only=local_only,
            )
            _pipe_cn.scheduler = _make_scheduler(_pipe_cn.scheduler.config)
            _pipe_cn.vae.enable_tiling()
            _pipe_cn.vae.enable_slicing()
            _pipe_cn.enable_attention_slicing()
            # Do NOT call .to("cuda") — enable_sequential_cpu_offload manages devices.
            _pipe_cn.enable_sequential_cpu_offload()
            log.info("ControlNet pipeline ready (sequential CPU offload)")
        return _pipe_cn


# ── InsightFace singleton ──────────────────────────────────────

_face_app     = None
_face_swapper = None
_face_lock    = threading.Lock()


def _load_face_analyzer():
    """Lazy-load InsightFace FaceAnalysis (buffalo_l). Cached after first call."""
    global _face_app
    with _face_lock:
        if _face_app is None:
            import insightface  # deferred so server starts without insightface installed
            fa = insightface.app.FaceAnalysis(
                name="buffalo_l",
                root=INSIGHTFACE_ROOT,
                providers=["CPUExecutionProvider"],
            )
            fa.prepare(ctx_id=0, det_size=(640, 640))
            _face_app = fa
        return _face_app


def _load_face_swapper():
    """Lazy-load InsightFace FaceAnalysis + INSwapper. Cached after first call."""
    global _face_swapper
    fa = _load_face_analyzer()
    with _face_lock:
        if _face_swapper is None:
            import insightface
            sw = insightface.model_zoo.get_model(
                INSWAPPER_MODEL,
                providers=["CPUExecutionProvider"],
            )
            if sw is None:
                raise RuntimeError(f"INSwapper model could not be loaded from {INSWAPPER_MODEL}")
            _face_swapper = sw
        return fa, _face_swapper


def detect_face_bbox(image_path: str) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) pixel bbox of the largest face, or None."""
    try:
        import cv2
        fa  = _load_face_analyzer()
        img = cv2.imread(image_path)
        faces = fa.get(img)
        if not faces:
            log.warning("detect_face_bbox: no face found in %s", image_path)
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        log.debug("Face bbox: x1=%d y1=%d x2=%d y2=%d", x1, y1, x2, y2)
        return x1, y1, x2, y2
    except Exception as e:
        log.warning("Face bbox detection failed: %s", e)
        return None


def estimate_selfie_hair_length(
    face_bbox: tuple[int, int, int, int] | None,
) -> str:
    """Classify selfie hair length using face bbox proportions.

    Ratio = pixels above forehead / face height.
    Small ratio → little clearance above face → short hair.
    """
    if face_bbox is None:
        return "unknown"
    _, y1, _, y2 = face_bbox
    face_height = max(y2 - y1, 1)
    ratio = y1 / face_height
    if ratio < 0.15:
        return "short"
    if ratio < 0.5:
        return "medium-length"
    return "long"


# ── helpers ───────────────────────────────────────────────────

def describe_hairstyle(img: Image.Image) -> str:
    """
    Derive a brief textual description of the reference hairstyle
    by analysing the image attributes (color, length heuristic).
    No external model required — purely rule-based on pixel stats.
    For production, swap with a BLIP/LLaVA call.
    """
    # crop to upper-centre third where hair most likely lives
    w, h = img.size
    crop = img.crop((w // 4, 0, 3 * w // 4, h // 2))
    stat = ImageStat.Stat(crop)
    avg_r, avg_g, avg_b = stat.mean[:3]
    brightness = (avg_r + avg_g + avg_b) / 3

    # crude color label
    if brightness < 60:
        color = "black"
    elif avg_r > avg_b + 30 and avg_r > avg_g + 10:
        color = "auburn red" if brightness < 130 else "blonde"
    elif avg_r > avg_b + 15:
        color = "brown"
    elif avg_b > avg_r + 20:
        color = "blue-tinted dark"
    else:
        color = "dark brown"

    # very rough length heuristic: tall crop → short hair
    ratio = h / max(w, 1)
    length = "short" if ratio < 1.1 else "medium-length" if ratio < 1.6 else "long"

    return f"{color}, {length}"


_LENGTH_PROMPTS = {
    "long":          "long flowing hair past shoulders, voluminous long cascading locks",
    "medium-length": "medium-length hair to the shoulders, lob haircut",
    "short":         "short cropped hair, close-cut style",
}

_LENGTH_NEGATIVES = {
    "long":          "short hair, cropped hair, buzz cut, pixie cut, bald, thin hair",
    "medium-length": "long hair, buzz cut, bald",
    "short":         "long hair past shoulders, wavy extensions",
}


def build_prompt(
    hairstyle_desc: str,
    extra: str = "",
    current_length: str = "unknown",
) -> tuple[str, str]:
    parts  = hairstyle_desc.split(", ", 1)
    color  = parts[0]
    length = parts[-1]

    if extra:
        hair_part = extra
    else:
        length_phrase = _LENGTH_PROMPTS.get(length, f"{length} hairstyle")
        hair_part = f"{color} {length_phrase}"

    pos = (
        f"{hair_part}, portrait photo, photorealistic, "
        "high detail face, studio lighting, sharp focus, dramatic hair transformation"
    )
    neg_length = _LENGTH_NEGATIVES.get(length, "")
    if current_length == "short" and length == "long":
        neg_length = f"short hair, buzz cut, thin hair, {neg_length}"
    neg = (
        "deformed, blurry, bad anatomy, low quality, cartoon, painting, "
        "multiple faces, watermark, text, extra limbs, same hairstyle, unchanged hair"
    )
    if neg_length:
        neg = f"{neg_length}, {neg}"
    return pos, neg


_LENGTH_STRENGTH = {"long": 0.88, "medium-length": 0.78, "short": 0.70}


def _adaptive_strength(target_length: str, user_requested: float | None) -> float:
    """Return generation strength appropriate for the target hair length.

    Only overrides when the caller did not explicitly provide strength,
    so API clients that set strength retain full control.
    """
    if user_requested is not None:
        return user_requested
    return _LENGTH_STRENGTH.get(target_length, 0.78)


def resize_to_multiple(path: str, width: int, height: int) -> tuple[str, int, int]:
    """Center-crop selfie to target aspect ratio then resize to nearest multiple of 64.

    Plain resize would squish faces when the selfie aspect ratio differs from the
    target (e.g. portrait selfie → square 1024×1024 output).
    """
    w = (width  // 64) * 64
    h = (height // 64) * 64
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    target_ratio = w / h
    src_ratio    = src_w / src_h
    if src_ratio > target_ratio:
        crop_w = int(src_h * target_ratio)
        left   = (src_w - crop_w) // 2
        img    = img.crop((left, 0, left + crop_w, src_h))
    elif src_ratio < target_ratio:
        crop_h = int(src_w / target_ratio)
        top    = (src_h - crop_h) // 2
        img    = img.crop((0, top, src_w, top + crop_h))
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    out = path.replace(".png", "_resized.png")
    img.save(out)
    return out, w, h


def generate_canny_control_image(
    selfie_path: str,
    uid: str,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> str | None:
    """Apply Canny edge detection to the resized selfie for ControlNet conditioning.

    When face_bbox is provided, the hair region (rows 0..y1, i.e. everything
    above the detected forehead) is zeroed out so ControlNet conditions only on
    face structure and ignores the existing hairline edges.

    Returns path to a 3-channel edge PNG, or None on any failure.
    """
    try:
        import cv2
        img     = cv2.imread(selfie_path)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

        if face_bbox is not None:
            _, y1, _, _ = face_bbox
            edges[:y1, :] = 0
            log.debug("Hair region masked: rows 0..%d zeroed in control image", y1)

        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out = os.path.join(WORK_DIR, f"{uid}_control.png")
        cv2.imwrite(out, edges_rgb)
        log.info("Canny control image saved: %s", out)
        return out
    except Exception as e:
        log.warning("Canny control image failed: %s", e)
        return None


def generate_hair_mask(
    selfie_path: str,
    face_bbox: tuple[int, int, int, int],
    uid: str,
) -> str | None:
    """Create a white-on-black hair mask for SDXL inpainting.

    White (255) = repaint zone (hair), black (0) = preserve zone (face + body).
    Zones: above forehead, sides of face, narrow strip below jaw where long hair drapes.
    """
    try:
        import numpy as np
        img = Image.open(selfie_path)
        W, H = img.size
        x1, y1, x2, y2 = face_bbox
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:y1, :]  = 255                                    # above forehead
        side_bottom = min(H, int(y2 * 1.15))
        mask[y1:side_bottom, :x1] = 255                        # left of face
        mask[y1:side_bottom, x2:] = 255                        # right of face
        below_bot = min(H, y2 + int((y2 - y1) * 0.5))
        mask[y2:below_bot, :]     = 255                        # below jaw (drape zone)
        out = os.path.join(WORK_DIR, f"{uid}_hair_mask.png")
        Image.fromarray(mask, mode="L").save(out)
        log.info("Hair mask: %s (white px: %d / %d)", out, int((mask > 0).sum()), W * H)
        return out
    except Exception as e:
        log.warning("Hair mask generation failed: %s", e)
        return None


def swap_face(source_selfie_path: str, generated_path: str, output_path: str) -> bool:
    """Paste the face from the original selfie onto the generated result.

    Uses InsightFace buffalo_l for detection and inswapper_128 for blending.
    Returns True on success, False if any face is undetectable or deps missing.
    """
    try:
        import cv2
        face_app, swapper = _load_face_swapper()
        assert swapper is not None  # _load_face_swapper raises if model load fails

        src = cv2.imread(source_selfie_path)
        gen = cv2.imread(generated_path)

        src_faces = face_app.get(src)
        gen_faces = face_app.get(gen)

        if not src_faces or not gen_faces:
            log.warning(
                "swap_face: no face detected (src=%d gen=%d)",
                len(src_faces), len(gen_faces),
            )
            return False

        result = swapper.get(gen, gen_faces[0], src_faces[0], paste_back=True)
        cv2.imwrite(output_path, result)
        log.info("Face swap applied successfully")
        return True
    except Exception as e:
        log.warning("Face swap failed: %s", e)
        return False


def run_sd(selfie_path: str, output_path: str,
           prompt: str, neg_prompt: str,
           strength: float, steps: int, cfg: float,
           seed: int, width: int, height: int,
           control_image_path: str | None = None,
           mask_path: str | None = None) -> None:
    use_inpaint = mask_path is not None and os.path.isfile(mask_path)
    use_cn = (
        bool(control_image_path)
        and CONTROLNET_ENABLED
        and os.path.isfile(CONTROLNET_MODEL)
        and not use_inpaint
    )

    log.debug("run_sd start | use_inpaint=%s use_cn=%s seed=%d steps=%d strength=%.2f | %s",
              use_inpaint, use_cn, seed, steps, strength, _cuda_mem_stats())

    generator = (
        torch.Generator(device="cuda").manual_seed(seed)
        if seed != -1 else None
    )

    init_image = Image.open(selfie_path).convert("RGB").resize((width, height))

    pipe = _get_pipeline(with_controlnet=use_cn, inpaint=use_inpaint)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log.debug("Before pipe() | %s", _cuda_mem_stats())

    if use_inpaint:
        assert mask_path is not None
        mask_image = Image.open(mask_path).convert("L").resize((width, height))
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )
    elif use_cn:
        assert control_image_path is not None
        control_image = Image.open(control_image_path).convert("RGB").resize((width, height))
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            control_image=control_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=CONTROLNET_STRENGTH,
            generator=generator,
        )
    else:
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )

    log.debug("After pipe() | %s", _cuda_mem_stats())
    result.images[0].save(output_path)
    log.debug("run_sd done")


# ── routes ────────────────────────────────────────────────────

@app.route("/")
def landing():
    return send_from_directory("public", "index.html")


@app.route("/studio")
def studio():
    return send_from_directory("public", "studio.html")


@app.route("/transfer-haircut", methods=["POST"])
def transfer_haircut():
    # ── validate inputs ──────────────────────────────────────
    if "selfie" not in request.files:
        return jsonify(error="Missing 'selfie' file"), 400
    if "hairstyle" not in request.files:
        return jsonify(error="Missing 'hairstyle' file"), 400

    selfie_file    = request.files["selfie"]
    hairstyle_file = request.files["hairstyle"]

    selfie_bytes = selfie_file.read()
    selfie_hash  = hashlib.sha256(selfie_bytes).hexdigest()[:16]
    selfie_file.seek(0)

    hairstyle_id  = request.form.get("hairstyle_id", "")
    extra_prompt  = request.form.get("prompt",    "")
    _strength_raw = request.form.get("strength")
    strength      = float(_strength_raw) if _strength_raw is not None else None
    steps         = int(request.form.get("steps",       5))
    cfg_scale     = float(request.form.get("cfg_scale", 2.0))
    seed          = int(request.form.get("seed",        -1))
    width         = int(request.form.get("width",       1024))
    height        = int(request.form.get("height",      1024))

    # clamp steps and cfg; strength is clamped after adaptive computation below
    steps     = max(3,   min(12,  steps))
    cfg_scale = max(1.0, min(7.0, cfg_scale))

    # ── temp files ───────────────────────────────────────────
    uid          = uuid.uuid4().hex[:8]
    selfie_path  = os.path.join(WORK_DIR, f"{uid}_selfie.png")
    output_path  = os.path.join(WORK_DIR, f"{uid}_output.png")
    control_path = None
    cache_path   = os.path.join(WORK_DIR, f"{selfie_hash}_{hairstyle_id}.jpg") if hairstyle_id else None

    if cache_path and os.path.exists(cache_path):
        log.info("Cache hit: %s", cache_path)
        return send_file(cache_path, mimetype="image/jpeg", download_name="haircut_result.jpg")

    try:
        Image.open(selfie_file).convert("RGB").save(selfie_path)
        hairstyle_img = Image.open(hairstyle_file).convert("RGB")

        selfie_resized, actual_w, actual_h = resize_to_multiple(
            selfie_path, width, height
        )

        hair_desc = describe_hairstyle(hairstyle_img)
        log.info("Detected hairstyle: %s", hair_desc)

        _target_length = hair_desc.split(", ", 1)[-1]
        strength = _adaptive_strength(_target_length, strength)
        strength = max(0.1, min(0.98, strength))

        # detect selfie hair length for adaptive prompting and inpaint decision
        face_bbox = detect_face_bbox(selfie_resized)
        selfie_hair_length = estimate_selfie_hair_length(face_bbox)
        log.info("Selfie hair: %s → target: %s, strength=%.2f",
                 selfie_hair_length, _target_length, strength)

        # ── inpainting for short→long transformation ─────────────
        mask_path = None
        use_inpaint = (
            INPAINT_ENABLED
            and selfie_hair_length == "short"
            and _target_length == "long"
            and face_bbox is not None
        )
        if use_inpaint:
            assert face_bbox is not None  # guaranteed by use_inpaint condition above
            mask_path = generate_hair_mask(selfie_resized, face_bbox, uid)
            if mask_path:
                strength = 0.95
                log.info("Short→long: inpaint pipeline selected, strength=0.95")
            else:
                log.warning("Hair mask failed — falling back to img2img")
                mask_path = None

        # inflate steps now that final strength is known
        steps = min(20, math.ceil(steps / strength))
        log.info("Final: strength=%.2f steps=%d (inflated)", strength, steps)

        prompt, neg_prompt = build_prompt(hair_desc, extra_prompt, current_length=selfie_hair_length)
        log.info("Prompt: %s", prompt)

        # ── ControlNet: Canny edge conditioning with hair mask ───
        if CONTROLNET_ENABLED and not mask_path:
            if face_bbox:
                log.info("Face bbox for ControlNet hair masking: %s", face_bbox)
            else:
                log.warning("No face detected — control image will include hair edges")
            control_path = generate_canny_control_image(selfie_resized, uid, face_bbox)
            if control_path is None:
                log.warning("Canny generation failed, proceeding without ControlNet")

        # ── run SD with ControlNet/inpaint retry fallback ────
        try:
            run_sd(
                selfie_path        = selfie_resized,
                output_path        = output_path,
                prompt             = prompt,
                neg_prompt         = neg_prompt,
                strength           = strength,
                steps              = steps,
                cfg                = cfg_scale,
                seed               = seed,
                width              = actual_w,
                height             = actual_h,
                control_image_path = control_path,
                mask_path          = mask_path,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if control_path or mask_path:
                log.warning("Pipeline run failed, retrying without ControlNet/inpaint: %s", e)
                run_sd(
                    selfie_path        = selfie_resized,
                    output_path        = output_path,
                    prompt             = prompt,
                    neg_prompt         = neg_prompt,
                    strength           = strength,
                    steps              = steps,
                    cfg                = cfg_scale,
                    seed               = seed,
                    width              = actual_w,
                    height             = actual_h,
                    control_image_path = None,
                    mask_path          = None,
                )
            else:
                raise

        if not os.path.exists(output_path):
            return jsonify(error="Pipeline produced no output file"), 500

        # ── InsightFace: paste original face onto result ─────
        if FACESWAP_ENABLED:
            if not swap_face(selfie_path, output_path, output_path):
                log.warning("Face swap skipped, returning raw SD result")

        result_img = Image.open(output_path)
        if cache_path:
            result_img.save(cache_path, "JPEG", quality=92)
            log.info("Cached result: %s", cache_path)
        buf = io.BytesIO()
        result_img.save(buf, "JPEG", quality=92)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg",
                         download_name="haircut_result.jpg")

    except RuntimeError as e:
        log.error("Pipeline failed: %s | %s", e, _cuda_mem_stats(), exc_info=True)
        if "out of memory" in str(e).lower():
            log.warning("CUDA OOM — freeing GPU memory for next request")
            _free_gpu_memory()
        return jsonify(error=str(e)), 500
    except Exception as e:
        log.exception("Unexpected error")
        return jsonify(error=f"Server error: {e}"), 500
    finally:
        for p in [
            selfie_path,
            output_path,
            selfie_path.replace(".png", "_resized.png"),
            os.path.join(WORK_DIR, f"{uid}_control.png"),
            os.path.join(WORK_DIR, f"{uid}_hair_mask.png"),
        ]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        _free_gpu_memory()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        status             = "ok",
        inference_backend  = "diffusers",
        torch_device       = "cuda" if torch.cuda.is_available() else "cpu",
        model              = MODEL_PATH,
        controlnet_enabled = CONTROLNET_ENABLED,
        controlnet_model   = CONTROLNET_MODEL,
        faceswap_enabled   = FACESWAP_ENABLED,
    )


# ── entrypoint ────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Haircut Transfer API starting on %s:%s", HOST, PORT)
    log.info("Backend        : diffusers / torch %s / cuda=%s", torch.__version__, torch.cuda.is_available())
    log.info("Model          : %s", MODEL_PATH)
    log.info("ControlNet     : %s (enabled=%s)", CONTROLNET_MODEL, CONTROLNET_ENABLED)
    log.info("Face swap      : %s (enabled=%s)", INSWAPPER_MODEL, FACESWAP_ENABLED)
    app.run(host=HOST, port=PORT, debug=False)
