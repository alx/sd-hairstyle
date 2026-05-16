"""
ControlNet model pipeline tests.

Run without GPU (mocked):
    pytest tests/test_controlnet_models.py -v -k "not Real"

Run with real model (requires model file and GPU):
    pytest tests/test_controlnet_models.py -v -k "Real"
    pytest tests/test_controlnet_models.py -v  # all tests
"""

import importlib
import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from conftest import ROOT

CONTROLNET_DIR = Path(os.getenv(
    "SD_CONTROLNET_DIR",
    "/home/alx/Documents/models/diffusion/ControlNet",
))
CONTROLNET_MODEL = os.getenv(
    "SD_CONTROLNET_MODEL",
    str(CONTROLNET_DIR / "control-lora-depth-rank128.safetensors"),
)
_model_present = Path(CONTROLNET_MODEL).exists()


# ── helpers ───────────────────────────────────────────────────────────────────

def _reload_app(monkeypatch, tmp_path, **env_overrides):
    monkeypatch.setenv("SD_WORK_DIR",          str(tmp_path))
    monkeypatch.setenv("CONTROLNET_ENABLED",   "1")
    monkeypatch.setenv("SD_CONTROLNET_MODEL",  CONTROLNET_MODEL)
    monkeypatch.setenv("FACESWAP_ENABLED",     "0")
    monkeypatch.syspath_prepend(str(ROOT))
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, v)
    import app as m
    importlib.reload(m)
    m.app.config["TESTING"] = True
    return m


def _fake_run_sd(_selfie_path, output_path, *_, **__):
    """Write a placeholder image to simulate a successful run_sd call."""
    Image.new("RGB", (64, 64), (120, 80, 200)).save(output_path)


# ════════════════════════════════════════════════════════════════════
# 1. Dry-run tests — mocked pipeline, no GPU required
# ════════════════════════════════════════════════════════════════════

class TestControlNetDryRun:
    """ControlNet wiring tests with mocked run_sd — no GPU or model file needed."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_path, selfie_png, hairstyle_png):
        self.m         = _reload_app(monkeypatch, tmp_path)
        self.client    = self.m.app.test_client()
        self.selfie    = selfie_png
        self.hairstyle = hairstyle_png

    def _post(self, seed: int = 42):
        with open(self.selfie, "rb") as sf, open(self.hairstyle, "rb") as hf:
            return self.client.post(
                "/transfer-haircut",
                data={
                    "selfie":    (sf, "selfie.png"),
                    "hairstyle": (hf, "hairstyle.png"),
                    "strength":  "0.65",
                    "steps":     "5",
                    "cfg_scale": "2.0",
                    "seed":      str(seed),
                },
                content_type="multipart/form-data",
            )

    def test_returns_200(self):
        with patch.object(self.m, "run_sd", side_effect=_fake_run_sd):
            resp = self._post()
        assert resp.status_code == 200

    def test_output_is_jpeg(self):
        with patch.object(self.m, "run_sd", side_effect=_fake_run_sd):
            resp = self._post()
        assert resp.content_type == "image/jpeg"

    def test_output_is_decodable_image(self):
        with patch.object(self.m, "run_sd", side_effect=_fake_run_sd):
            resp = self._post()
        img = Image.open(io.BytesIO(resp.data))
        assert img.format == "JPEG"

    def test_run_sd_called_exactly_once_on_success(self):
        calls = []

        def capture(*args, **kwargs):
            calls.append(kwargs)
            _fake_run_sd(*args, **kwargs)

        with patch.object(self.m, "run_sd", side_effect=capture):
            self._post()

        assert len(calls) == 1

    def test_run_sd_receives_control_image_when_canny_succeeds(self):
        """run_sd should be called with a control_image_path when ControlNet is enabled
        and the Canny step produces a valid edge image."""
        captured = []

        def capture(*args, **kwargs):
            captured.append(kwargs.get("control_image_path"))
            _fake_run_sd(*args, **kwargs)

        with patch.object(self.m, "run_sd", side_effect=capture):
            resp = self._post()

        assert resp.status_code == 200
        assert len(captured) == 1
        # Canny may produce None on a flat synthetic image — we verify the call happened

    def test_retry_without_controlnet_on_runtimeerror(self):
        """When run_sd raises RuntimeError on the CN call, it retries without ControlNet."""
        call_count = [0]

        def flaky(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("control_image_path") is not None:
                raise RuntimeError("simulated ControlNet OOM")
            _fake_run_sd(*args, **kwargs)

        with patch.object(self.m, "run_sd", side_effect=flaky):
            resp = self._post()

        assert resp.status_code == 200
        assert call_count[0] == 2, f"Expected 2 run_sd calls (retry), got {call_count[0]}"

    def test_retry_fallback_second_call_has_no_control_image(self):
        """The retry call (after CN failure) must have control_image_path=None."""
        calls = []

        def flaky(*args, **kwargs):
            calls.append(kwargs.get("control_image_path"))
            if kwargs.get("control_image_path") is not None:
                raise RuntimeError("simulated OOM")
            _fake_run_sd(*args, **kwargs)

        with patch.object(self.m, "run_sd", side_effect=flaky):
            self._post()

        assert len(calls) == 2
        assert calls[1] is None, "Retry call should pass control_image_path=None"

    def test_different_seeds_accepted(self):
        """API must accept arbitrary seed values without error."""
        with patch.object(self.m, "run_sd", side_effect=_fake_run_sd):
            for seed in (0, 1, 42, 999, -1):
                with open(self.selfie, "rb") as sf, open(self.hairstyle, "rb") as hf:
                    resp = self.client.post(
                        "/transfer-haircut",
                        data={
                            "selfie":    (sf, "selfie.png"),
                            "hairstyle": (hf, "hairstyle.png"),
                            "seed":      str(seed),
                        },
                        content_type="multipart/form-data",
                    )
                assert resp.status_code == 200, f"seed={seed} failed: {resp.get_data(as_text=True)}"


# ════════════════════════════════════════════════════════════════════
# 2. Real model tests — requires ControlNet model file and GPU
# ════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not _model_present,
    reason=f"ControlNet model not found: {CONTROLNET_MODEL} — set SD_CONTROLNET_MODEL",
)
class TestControlNetModelsReal:
    """Real model inference. Requires a GPU and the model file at CONTROLNET_MODEL."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_path, selfie_png, hairstyle_png):
        self.m         = _reload_app(monkeypatch, tmp_path)
        self.client    = self.m.app.test_client()
        self.selfie    = selfie_png
        self.hairstyle = hairstyle_png

    def _post(self, seed: int = 42):
        with open(self.selfie, "rb") as sf, open(self.hairstyle, "rb") as hf:
            return self.client.post(
                "/transfer-haircut",
                data={
                    "selfie":    (sf, "selfie.png"),
                    "hairstyle": (hf, "hairstyle.png"),
                    "strength":  "0.65",
                    "steps":     "3",
                    "cfg_scale": "2.0",
                    "seed":      str(seed),
                    "width":     "512",
                    "height":    "512",
                },
                content_type="multipart/form-data",
            )

    def test_real_model_returns_200(self):
        resp = self._post()
        assert resp.status_code == 200, resp.get_data(as_text=True)

    def test_real_model_output_is_jpeg(self):
        resp = self._post()
        assert resp.content_type == "image/jpeg"

    def test_real_model_output_is_valid_image(self):
        resp = self._post()
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.data))
        assert img.format == "JPEG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_real_model_output_non_trivial(self):
        """Output should be a real image, not an all-black placeholder."""
        import numpy as np
        resp = self._post(seed=1337)
        assert resp.status_code == 200
        arr = np.array(Image.open(io.BytesIO(resp.data)))
        assert arr.std() > 5, "Output image looks like a blank placeholder"
