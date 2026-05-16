"""
Tests for sd-cli integration and Haircut Transfer API.

Run fast (no model required):
    pytest tests/ -v -k "not Integration"

Run all (requires model file):
    pytest tests/ -v
"""

import io
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from conftest import ROOT, _make_png, _png_bytes

HERE = Path(__file__).parent

SD_BINARY  = os.getenv("SD_BINARY",  str(ROOT / "bin" / "sd-cli"))
MODEL_PATH = os.getenv("SD_MODEL",   str(ROOT / "models" / "turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"))


# ════════════════════════════════════════════════════════════════════
# 1. Binary smoke tests
# ════════════════════════════════════════════════════════════════════

class TestSdCliBinary:
    def test_binary_exists(self):
        assert Path(SD_BINARY).exists(), f"sd-cli not found at {SD_BINARY}"

    def test_binary_is_executable(self):
        assert os.access(SD_BINARY, os.X_OK), f"sd-cli is not executable: {SD_BINARY}"

    def test_binary_responds(self):
        result = subprocess.run(
            [SD_BINARY, "--help"],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout + result.stderr
        assert len(output) > 0, "sd-cli --help produced no output"
        assert result.returncode in (0, 1), (
            f"sd-cli --help exited with unexpected code {result.returncode}\n{result.stderr}"
        )


# ════════════════════════════════════════════════════════════════════
# 2. Image generation tests (direct sd-cli invocation)
# ════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason=f"Model not found: {MODEL_PATH}",
)
class TestImageGeneration:
    """Invoke sd-cli directly and verify it produces valid image output."""

    def _run(self, out: Path, extra_args: list) -> subprocess.CompletedProcess:
        cmd = [
            SD_BINARY,
            "--mode", "img_gen",
            "-m",  MODEL_PATH,
            "-o",  str(out),
            "--steps",           "1",
            "-W",                "128",
            "-H",                "128",
            "--seed",            "42",
            "--cfg-scale",       "2.0",
            "--sampling-method", "dpm++2s_a",
            "--scheduler",       "karras",
        ] + extra_args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    def test_txt2img_exits_zero(self, tmp_path):
        r = self._run(tmp_path / "out.png", ["-p", "portrait photo, studio lighting"])
        assert r.returncode == 0, f"sd-cli failed (rc={r.returncode}):\n{r.stderr[-600:]}"

    def test_txt2img_creates_file(self, tmp_path):
        out = tmp_path / "out.png"
        r = self._run(out, ["-p", "portrait photo, studio lighting"])
        assert r.returncode == 0, r.stderr[-400:]
        assert out.exists(), "sd-cli produced no output file"

    def test_txt2img_output_is_valid_png(self, tmp_path):
        out = tmp_path / "out.png"
        r = self._run(out, ["-p", "portrait photo, studio lighting"])
        assert r.returncode == 0, r.stderr[-400:]
        img = Image.open(out)
        assert img.format == "PNG"

    def test_txt2img_output_dimensions(self, tmp_path):
        out = tmp_path / "out.png"
        r = self._run(out, ["-p", "portrait photo"])
        assert r.returncode == 0, r.stderr[-400:]
        assert Image.open(out).size == (128, 128)

    def test_txt2img_seed_determinism(self, tmp_path):
        """Two runs with the same seed must produce identical pixels."""
        out_a = tmp_path / "a.png"
        out_b = tmp_path / "b.png"
        args = ["-p", "portrait photo, studio lighting"]
        ra = self._run(out_a, args)
        rb = self._run(out_b, args)
        assert ra.returncode == 0, ra.stderr[-400:]
        assert rb.returncode == 0, rb.stderr[-400:]
        assert out_a.read_bytes() == out_b.read_bytes(), (
            "Same seed produced different outputs"
        )

    def test_img2img_exits_zero(self, tmp_path, small_png):
        out = tmp_path / "out.png"
        r = self._run(out, [
            "-p", "portrait photo with dark hair",
            "-i", small_png,
            "--strength", "0.65",
        ])
        assert r.returncode == 0, f"sd-cli img2img failed (rc={r.returncode}):\n{r.stderr[-600:]}"

    def test_img2img_creates_file(self, tmp_path, small_png):
        out = tmp_path / "out.png"
        r = self._run(out, ["-p", "portrait photo", "-i", small_png, "--strength", "0.65"])
        assert r.returncode == 0, r.stderr[-400:]
        assert out.exists(), "sd-cli img2img produced no output file"

    def test_img2img_output_is_valid_png(self, tmp_path, small_png):
        out = tmp_path / "out.png"
        r = self._run(out, ["-p", "portrait photo", "-i", small_png, "--strength", "0.65"])
        assert r.returncode == 0, r.stderr[-400:]
        img = Image.open(out)
        assert img.format == "PNG"
        assert img.size == (128, 128)

    def test_full_production_call(self, tmp_path):
        """Mirror the exact sd-cli command issued by run_sd() in app.py."""
        selfie = tmp_path / "selfie.png"
        out    = tmp_path / "output.png"
        _make_png(selfie, 1024, 1024, color=(180, 140, 110))

        prompt = (
            "portrait photo of a person with black, short hairstyle, "
            "photorealistic, high detail face, studio lighting, sharp focus, "
            "blunt, chin-length"
        )
        neg_prompt = (
            "deformed, blurry, bad anatomy, low quality, cartoon, painting, "
            "multiple faces, watermark, text, extra limbs"
        )
        cmd = [
            SD_BINARY,
            "--mode",            "img_gen",
            "-m",                MODEL_PATH,
            "-i",                str(selfie),
            "-o",                str(out),
            "-p",                prompt,
            "-n",                neg_prompt,
            "--strength",        "0.65",
            "--steps",           "5",
            "--cfg-scale",       "2.0",
            "--seed",            "-1",
            "-W",                "1024",
            "-H",                "1024",
            "--sampling-method", "dpm++2s_a",
            "--scheduler",       "karras",
            "--vae-tiling",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        assert result.returncode == 0, (
            f"sd-cli production call failed (rc={result.returncode}):\n{result.stderr[-600:]}"
        )
        assert out.exists(), "sd-cli produced no output file"
        img = Image.open(out)
        assert img.format == "PNG"
        assert img.size == (1024, 1024)


# ════════════════════════════════════════════════════════════════════
# 3. Unit tests: describe_hairstyle
# ════════════════════════════════════════════════════════════════════

class TestDescribeHairstyle:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(ROOT))
        import app as m
        self.fn = m.describe_hairstyle

    def _img(self, w, h, color):
        return Image.new("RGB", (w, h), color=color)

    def test_dark_image_yields_black(self):
        desc = self.fn(self._img(200, 200, (20, 20, 20)))
        assert desc.startswith("black"), f"Expected 'black', got: {desc}"

    def test_high_red_bright_yields_blonde(self):
        desc = self.fn(self._img(200, 200, (220, 180, 80)))
        assert "blonde" in desc, f"Expected 'blonde', got: {desc}"

    def test_short_hair_heuristic(self):
        desc = self.fn(self._img(400, 200, (100, 70, 50)))
        assert "short" in desc, f"Expected 'short', got: {desc}"

    def test_long_hair_heuristic(self):
        desc = self.fn(self._img(200, 500, (100, 70, 50)))
        assert "long" in desc, f"Expected 'long', got: {desc}"


# ════════════════════════════════════════════════════════════════════
# 4. Unit tests: build_prompt
# ════════════════════════════════════════════════════════════════════

class TestBuildPrompt:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(ROOT))
        import app as m
        self.fn = m.build_prompt

    def test_prompt_contains_descriptor(self):
        pos, _ = self.fn("auburn red, long")
        assert "auburn red" in pos and "long" in pos

    def test_extra_prompt_appended(self):
        pos, _ = self.fn("black, short", extra="cinematic lighting")
        assert "cinematic lighting" in pos

    def test_no_trailing_comma_without_extra(self):
        pos, _ = self.fn("brown, medium-length")
        assert not pos.endswith(",")
        assert not pos.endswith(", ")

    def test_negative_prompt_stable(self):
        _, neg = self.fn("black, short")
        assert "deformed" in neg
        assert "blurry" in neg


# ════════════════════════════════════════════════════════════════════
# 5. Unit tests: resize_to_multiple
# ════════════════════════════════════════════════════════════════════

class TestResizeToMultiple:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(ROOT))
        import app as m
        self.fn = m.resize_to_multiple

    def test_output_dimensions_multiple_of_64(self, small_png):
        _, w, h = self.fn(small_png, 300, 300)
        assert w % 64 == 0, f"Width {w} not a multiple of 64"
        assert h % 64 == 0, f"Height {h} not a multiple of 64"

    def test_output_file_created(self, small_png):
        out, _, _ = self.fn(small_png, 256, 256)
        assert Path(out).exists(), f"Resized file not found: {out}"

    def test_wide_image_cropped_to_square(self, tmp_path):
        p = _make_png(tmp_path / "wide.png", 512, 128, (100, 100, 100))
        out, w, h = self.fn(p, 256, 256)
        img = Image.open(out)
        assert img.size == (w, h)
        assert w == h

    def test_tall_image_cropped_to_square(self, tmp_path):
        p = _make_png(tmp_path / "tall.png", 128, 512, (100, 100, 100))
        out, w, h = self.fn(p, 256, 256)
        img = Image.open(out)
        assert img.size == (w, h)
        assert w == h


# ════════════════════════════════════════════════════════════════════
# 6. Unit tests: API validation (subprocess mocked)
# ════════════════════════════════════════════════════════════════════

class TestApiValidation:
    def test_health_returns_200(self, flask_client):
        resp = flask_client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "inference_backend" in data
        assert data["inference_backend"] == "diffusers"

    def test_health_reports_feature_flags(self, flask_client):
        resp = flask_client.get("/health")
        data = resp.get_json()
        assert "controlnet_enabled" in data
        assert "faceswap_enabled" in data

    def test_missing_selfie_returns_400(self, flask_client, hairstyle_png):
        with open(hairstyle_png, "rb") as f:
            resp = flask_client.post(
                "/transfer-haircut",
                data={"hairstyle": (f, "hairstyle.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 400
        assert "selfie" in resp.get_json()["error"].lower()

    def test_missing_hairstyle_returns_400(self, flask_client, selfie_png):
        with open(selfie_png, "rb") as f:
            resp = flask_client.post(
                "/transfer-haircut",
                data={"selfie": (f, "selfie.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 400
        assert "hairstyle" in resp.get_json()["error"].lower()

    def _mock_sd_run(self, output_path_holder):
        def side_effect(cmd, **_):
            idx = cmd.index("-o")
            out = cmd[idx + 1]
            output_path_holder.append(out)
            Image.new("RGB", (64, 64), (200, 100, 50)).save(out)
            r = MagicMock()
            r.returncode = 0
            r.stdout = "done"
            r.stderr = ""
            return r
        return side_effect

    def test_transfer_success_with_mocked_sd(self, flask_client, selfie_png, hairstyle_png):
        holder = []
        with patch("subprocess.run", side_effect=self._mock_sd_run(holder)):
            with open(selfie_png, "rb") as sf, open(hairstyle_png, "rb") as hf:
                resp = flask_client.post(
                    "/transfer-haircut",
                    data={
                        "selfie":    (sf, "selfie.png"),
                        "hairstyle": (hf, "hairstyle.png"),
                    },
                    content_type="multipart/form-data",
                )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.content_type == "image/jpeg"
        img = Image.open(io.BytesIO(resp.data))
        assert img.format == "JPEG"

    def test_parameter_clamping(self, flask_client, selfie_png, hairstyle_png):
        """Out-of-range params must be clamped before reaching sd-cli."""
        captured_cmd = []

        def capture(cmd, **_):
            captured_cmd.extend(cmd)
            idx = cmd.index("-o")
            out = cmd[idx + 1]
            Image.new("RGB", (64, 64), (0, 0, 0)).save(out)
            r = MagicMock()
            r.returncode = 0
            r.stdout = r.stderr = ""
            return r

        with patch("subprocess.run", side_effect=capture):
            with open(selfie_png, "rb") as sf, open(hairstyle_png, "rb") as hf:
                flask_client.post(
                    "/transfer-haircut",
                    data={
                        "selfie":    (sf, "selfie.png"),
                        "hairstyle": (hf, "hairstyle.png"),
                        "strength":  "5.0",
                        "steps":     "99",
                        "cfg_scale": "0.1",
                    },
                    content_type="multipart/form-data",
                )

        def _get_arg(flag):
            idx = captured_cmd.index(flag)
            return float(captured_cmd[idx + 1])

        assert _get_arg("--strength") <= 0.95
        assert int(_get_arg("--steps")) <= 8
        assert _get_arg("--cfg-scale") >= 1.0


# ════════════════════════════════════════════════════════════════════
# 7. Integration test (requires sd-cli binary + model)
# ════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason=f"Model not found: {MODEL_PATH}",
)
class TestIntegration:
    def test_full_transfer_returns_jpeg(self, flask_client, selfie_png, hairstyle_png):
        with open(selfie_png, "rb") as sf, open(hairstyle_png, "rb") as hf:
            resp = flask_client.post(
                "/transfer-haircut",
                data={
                    "selfie":    (sf, "selfie.png"),
                    "hairstyle": (hf, "hairstyle.png"),
                    "steps":     "3",
                    "seed":      "42",
                    "width":     "512",
                    "height":    "512",
                },
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.content_type == "image/jpeg"
        img = Image.open(io.BytesIO(resp.data))
        assert img.format == "JPEG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_production_params_web_client_receives_jpeg(self, flask_client):
        """Production parameters (1024×1024, 5 steps, strength 0.65)."""
        resp = flask_client.post(
            "/transfer-haircut",
            data={
                "selfie":    (io.BytesIO(_png_bytes(1024, 1024, (180, 140, 110))), "selfie.png"),
                "hairstyle": (io.BytesIO(_png_bytes(200,  300,  (30,  20,  20))),  "hairstyle.png"),
                "strength":  "0.65",
                "steps":     "5",
                "cfg_scale": "2.0",
                "seed":      "-1",
                "width":     "1024",
                "height":    "1024",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.content_type == "image/jpeg"
        assert len(resp.data) > 10_000, (
            f"Response body too small ({len(resp.data)} bytes) — likely empty or truncated"
        )
        img = Image.open(io.BytesIO(resp.data))
        assert img.format == "JPEG"
        assert img.size == (1024, 1024)
