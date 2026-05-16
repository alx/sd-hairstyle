"""
Tests for the ControlNet Canny conditioning and InsightFace face swap pipeline.

Unit tests (no model required, mocked):
    pytest tests/test_controlnet_faceswap.py -v -k "not Integration"

Integration tests (real assets, real models):
    pytest tests/test_controlnet_faceswap.py -v -k "Integration"
"""

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from conftest import ROOT

INSIGHTFACE_ROOT = os.getenv(
    "INSIGHTFACE_ROOT",
    "/home/alx/Documents/models/diffusion/insightface",
)
INSWAPPER_MODEL = os.getenv(
    "INSWAPPER_MODEL",
    "/home/alx/Documents/models/diffusion/faceswap/inswapper_128.onnx",
)
CONTROLNET_MODEL = os.getenv(
    "SD_CONTROLNET_MODEL",
    "/home/alx/Documents/models/diffusion/ControlNet/control-lora-canny-rank128.safetensors",
)

_models_present = (
    Path(INSIGHTFACE_ROOT, "models", "buffalo_l", "det_10g.onnx").exists()
    and Path(INSWAPPER_MODEL).exists()
)
_controlnet_present = Path(CONTROLNET_MODEL).exists()


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _reload_app(monkeypatch, tmp_path, **env_overrides):
    """Reload app module with isolated WORK_DIR and optional env overrides."""
    monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(ROOT))
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, v)
    import app as m
    importlib.reload(m)
    return m


def _fake_run_sd(selfie_path, output_path, *args, **kwargs):
    """Mock run_sd: write a 64×64 PNG to output_path."""
    Image.new("RGB", (64, 64), (200, 100, 50)).save(output_path)


# ════════════════════════════════════════════════════════════════════
# 1. Canny control image generation
# ════════════════════════════════════════════════════════════════════

class TestGenerateCannyControlImage:
    @pytest.fixture(autouse=True)
    def _mod(self, monkeypatch, tmp_path):
        self.m = _reload_app(monkeypatch, tmp_path)
        self.tmp = tmp_path

    def test_returns_path_string(self, real_selfie_resized):
        out = self.m.generate_canny_control_image(real_selfie_resized, "test01")
        assert out is not None
        assert isinstance(out, str)

    def test_output_file_exists(self, real_selfie_resized):
        out = self.m.generate_canny_control_image(real_selfie_resized, "test02")
        assert out is not None
        assert Path(out).exists(), f"Control image not written: {out}"

    def test_output_is_valid_png(self, real_selfie_resized):
        out = self.m.generate_canny_control_image(real_selfie_resized, "test03")
        assert out is not None
        img = Image.open(out)
        assert img.format == "PNG"

    def test_output_is_3_channel(self, real_selfie_resized):
        """sd-cli requires RGB, not single-channel grayscale."""
        out = self.m.generate_canny_control_image(real_selfie_resized, "test04")
        assert out is not None
        img = Image.open(out)
        assert img.mode == "RGB", f"Expected RGB, got {img.mode}"

    def test_output_matches_input_dimensions(self, real_selfie_resized):
        """Control image must be same size as the init image for ControlNet."""
        src = Image.open(real_selfie_resized)
        out = self.m.generate_canny_control_image(real_selfie_resized, "test05")
        assert out is not None
        ctl = Image.open(out)
        assert ctl.size == src.size, (
            f"Control image {ctl.size} != selfie {src.size}"
        )

    def test_edges_are_non_trivial(self, real_selfie_resized):
        """A real portrait should yield a meaningful number of edge pixels."""
        import numpy as np
        out = self.m.generate_canny_control_image(real_selfie_resized, "test06")
        assert out is not None
        arr = np.array(Image.open(out))
        edge_pixels = (arr > 0).any(axis=2).sum()
        total_pixels = arr.shape[0] * arr.shape[1]
        ratio = edge_pixels / total_pixels
        assert ratio > 0.01, f"Too few edge pixels ({ratio:.2%}) — Canny may have failed silently"
        assert ratio < 0.5,  f"Too many edge pixels ({ratio:.2%}) — thresholds may be too low"

    def test_bad_path_returns_none(self):
        import app as m
        out = m.generate_canny_control_image("/nonexistent/image.png", "bad")
        assert out is None

    def test_synthetic_image_produces_edges(self, tmp_path):
        """Even a synthetic flat image with a black square should produce edges."""
        import numpy as np, cv2
        p = tmp_path / "synthetic.png"
        canvas = np.full((256, 256, 3), 255, dtype="uint8")
        canvas[64:192, 64:192] = 0  # black square on white
        cv2.imwrite(str(p), canvas)
        import app as m
        out = m.generate_canny_control_image(str(p), "synth")
        assert out is not None
        arr = np.array(Image.open(out))
        assert (arr > 0).any(), "Synthetic high-contrast image should produce edges"


# ════════════════════════════════════════════════════════════════════
# 2. InsightFace face swap (unit — mocked models)
# ════════════════════════════════════════════════════════════════════

class TestSwapFaceMocked:
    """swap_face() plumbing tests with mocked InsightFace models."""

    @pytest.fixture(autouse=True)
    def _mod(self, monkeypatch, tmp_path):
        self.m = _reload_app(monkeypatch, tmp_path)
        self.tmp = tmp_path

    def _fake_face(self):
        f = MagicMock()
        f.bbox = [10, 10, 100, 100]
        return f

    def _gen_image(self, name="generated.png", size=(64, 64)):
        p = self.tmp / name
        Image.new("RGB", size, (200, 150, 100)).save(p)
        return str(p)

    def test_returns_true_when_faces_found(self, real_selfie):
        src_face = self._fake_face()
        gen_face = self._fake_face()
        gen_img = self._gen_image()
        out = str(self.tmp / "out.png")
        Image.new("RGB", (64, 64), (0, 0, 0)).save(out)

        fake_app = MagicMock()
        fake_app.get.side_effect = [[src_face], [gen_face]]

        fake_swapper = MagicMock()
        fake_swapper.get.return_value = __import__("numpy").zeros((64, 64, 3), dtype="uint8")

        with patch.object(self.m, "_load_face_swapper", return_value=(fake_app, fake_swapper)):
            result = self.m.swap_face(real_selfie, gen_img, out)

        assert result is True
        fake_swapper.get.assert_called_once()

    def test_returns_false_when_no_source_face(self, real_selfie):
        gen_img = self._gen_image()
        out = str(self.tmp / "out.png")
        Image.new("RGB", (64, 64)).save(out)

        fake_app = MagicMock()
        fake_app.get.side_effect = [[], [self._fake_face()]]  # no face in source
        fake_swapper = MagicMock()

        with patch.object(self.m, "_load_face_swapper", return_value=(fake_app, fake_swapper)):
            result = self.m.swap_face(real_selfie, gen_img, out)

        assert result is False
        fake_swapper.get.assert_not_called()

    def test_returns_false_when_no_generated_face(self, real_selfie):
        gen_img = self._gen_image()
        out = str(self.tmp / "out.png")
        Image.new("RGB", (64, 64)).save(out)

        fake_app = MagicMock()
        fake_app.get.side_effect = [[self._fake_face()], []]  # no face in generated
        fake_swapper = MagicMock()

        with patch.object(self.m, "_load_face_swapper", return_value=(fake_app, fake_swapper)):
            result = self.m.swap_face(real_selfie, gen_img, out)

        assert result is False
        fake_swapper.get.assert_not_called()

    def test_returns_false_on_exception(self, real_selfie):
        gen_img = self._gen_image()
        out = str(self.tmp / "out.png")
        Image.new("RGB", (64, 64)).save(out)

        with patch.object(self.m, "_load_face_swapper", side_effect=RuntimeError("model missing")):
            result = self.m.swap_face(real_selfie, gen_img, out)

        assert result is False

    def test_output_file_written_on_success(self, real_selfie):
        gen_img = self._gen_image()
        out = str(self.tmp / "swapped.png")

        import numpy as np
        fake_result = np.zeros((64, 64, 3), dtype="uint8")
        fake_app = MagicMock()
        fake_app.get.side_effect = [[self._fake_face()], [self._fake_face()]]
        fake_swapper = MagicMock()
        fake_swapper.get.return_value = fake_result

        with patch.object(self.m, "_load_face_swapper", return_value=(fake_app, fake_swapper)):
            self.m.swap_face(real_selfie, gen_img, out)

        assert Path(out).exists(), "swap_face should write the output file on success"


# ════════════════════════════════════════════════════════════════════
# 3. API pipeline — ControlNet flag wiring
# ════════════════════════════════════════════════════════════════════

class TestApiControlNetFlags:
    """Verify ControlNet is wired correctly through run_sd."""

    def _post(self, client, selfie_png, hairstyle_png):
        with open(selfie_png, "rb") as sf, open(hairstyle_png, "rb") as hf:
            return client.post(
                "/transfer-haircut",
                data={"selfie": (sf, "selfie.png"), "hairstyle": (hf, "hairstyle.png")},
                content_type="multipart/form-data",
            )

    def test_controlnet_used_when_enabled(
        self, monkeypatch, tmp_path, selfie_png, hairstyle_png
    ):
        monkeypatch.setenv("CONTROLNET_ENABLED", "1")
        monkeypatch.setenv("FACESWAP_ENABLED", "0")
        monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
        monkeypatch.syspath_prepend(str(ROOT))
        import app as m
        importlib.reload(m)
        m.app.config["TESTING"] = True

        calls = []

        def capture(*args, **kwargs):
            calls.append(kwargs.get("control_image_path", args[10] if len(args) > 10 else None))
            _fake_run_sd(*args, **kwargs)

        with m.app.test_client() as client:
            with patch.object(m, "run_sd", side_effect=capture):
                self._post(client, selfie_png, hairstyle_png)

        assert len(calls) == 1
        assert calls[0] is not None, "Expected control_image_path to be set when CONTROLNET_ENABLED=1"

    def test_controlnet_absent_when_disabled(
        self, monkeypatch, tmp_path, selfie_png, hairstyle_png
    ):
        monkeypatch.setenv("CONTROLNET_ENABLED", "0")
        monkeypatch.setenv("FACESWAP_ENABLED", "0")
        monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
        monkeypatch.syspath_prepend(str(ROOT))
        import app as m
        importlib.reload(m)
        m.app.config["TESTING"] = True

        calls = []

        def capture(*args, **kwargs):
            calls.append(kwargs.get("control_image_path", args[10] if len(args) > 10 else None))
            _fake_run_sd(*args, **kwargs)

        with m.app.test_client() as client:
            with patch.object(m, "run_sd", side_effect=capture):
                self._post(client, selfie_png, hairstyle_png)

        assert len(calls) == 1
        assert calls[0] is None, "Expected control_image_path=None when CONTROLNET_ENABLED=0"

    def test_controlnet_strength_value_is_correct(
        self, monkeypatch, tmp_path, selfie_png, hairstyle_png
    ):
        monkeypatch.setenv("CONTROLNET_ENABLED", "1")
        monkeypatch.setenv("SD_CONTROLNET_STRENGTH", "0.42")
        monkeypatch.setenv("FACESWAP_ENABLED", "0")
        monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
        monkeypatch.syspath_prepend(str(ROOT))
        import app as m
        importlib.reload(m)
        m.app.config["TESTING"] = True

        with m.app.test_client() as client:
            with patch.object(m, "run_sd", side_effect=_fake_run_sd):
                self._post(client, selfie_png, hairstyle_png)

        assert m.CONTROLNET_STRENGTH == pytest.approx(0.42)

    def test_controlnet_retry_without_on_failure(
        self, monkeypatch, tmp_path, selfie_png, hairstyle_png
    ):
        """If run_sd raises RuntimeError with a control_image_path, retry without ControlNet."""
        monkeypatch.setenv("CONTROLNET_ENABLED", "1")
        monkeypatch.setenv("FACESWAP_ENABLED", "0")
        monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
        monkeypatch.syspath_prepend(str(ROOT))
        import app as m
        importlib.reload(m)
        m.app.config["TESTING"] = True

        call_count = [0]

        def flaky(*args, **kwargs):
            call_count[0] += 1
            control = kwargs.get("control_image_path", args[10] if len(args) > 10 else None)
            if control is not None:
                raise RuntimeError("control net model failed to load")
            _fake_run_sd(*args, **kwargs)

        with m.app.test_client() as client:
            with patch.object(m, "run_sd", side_effect=flaky):
                resp = self._post(client, selfie_png, hairstyle_png)

        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert call_count[0] == 2, f"Expected 2 run_sd calls (retry), got {call_count[0]}"


# ════════════════════════════════════════════════════════════════════
# 4. API pipeline — face swap wiring
# ════════════════════════════════════════════════════════════════════

class TestApiFaceSwapWiring:
    """Verify swap_face is called/skipped correctly via the API."""

    def _post(self, client, selfie_png, hairstyle_png):
        with open(selfie_png, "rb") as sf, open(hairstyle_png, "rb") as hf:
            return client.post(
                "/transfer-haircut",
                data={"selfie": (sf, "selfie.png"), "hairstyle": (hf, "hairstyle.png")},
                content_type="multipart/form-data",
            )

    def test_swap_face_called_when_enabled(
        self, monkeypatch, tmp_path, selfie_png, hairstyle_png
    ):
        monkeypatch.setenv("FACESWAP_ENABLED", "1")
        monkeypatch.setenv("CONTROLNET_ENABLED", "0")
        monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
        monkeypatch.syspath_prepend(str(ROOT))
        import app as m
        importlib.reload(m)
        m.app.config["TESTING"] = True

        swap_calls = []

        def fake_swap(src, gen, out):
            swap_calls.append((src, gen, out))
            return False  # skip actual model, just record the call

        with m.app.test_client() as client:
            with patch.object(m, "run_sd", side_effect=_fake_run_sd), \
                 patch.object(m, "swap_face", side_effect=fake_swap):
                resp = self._post(client, selfie_png, hairstyle_png)

        assert resp.status_code == 200
        assert len(swap_calls) == 1, "swap_face should be called exactly once"

    def test_swap_face_not_called_when_disabled(
        self, monkeypatch, tmp_path, selfie_png, hairstyle_png
    ):
        monkeypatch.setenv("FACESWAP_ENABLED", "0")
        monkeypatch.setenv("CONTROLNET_ENABLED", "0")
        monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
        monkeypatch.syspath_prepend(str(ROOT))
        import app as m
        importlib.reload(m)
        m.app.config["TESTING"] = True

        swap_calls = []

        def fake_swap(*_):
            swap_calls.append(True)
            return False

        with m.app.test_client() as client:
            with patch.object(m, "run_sd", side_effect=_fake_run_sd), \
                 patch.object(m, "swap_face", side_effect=fake_swap):
                resp = self._post(client, selfie_png, hairstyle_png)

        assert resp.status_code == 200
        assert len(swap_calls) == 0, "swap_face should NOT be called when FACESWAP_ENABLED=0"

    def test_api_returns_jpeg_even_when_swap_fails(
        self, monkeypatch, tmp_path, selfie_png, hairstyle_png
    ):
        """Pipeline must still return a result image if face swap finds no face."""
        monkeypatch.setenv("FACESWAP_ENABLED", "1")
        monkeypatch.setenv("CONTROLNET_ENABLED", "0")
        monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
        monkeypatch.syspath_prepend(str(ROOT))
        import app as m
        importlib.reload(m)
        m.app.config["TESTING"] = True

        with m.app.test_client() as client:
            with patch.object(m, "run_sd", side_effect=_fake_run_sd), \
                 patch.object(m, "swap_face", return_value=False):
                resp = self._post(client, selfie_png, hairstyle_png)

        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"


# ════════════════════════════════════════════════════════════════════
# 5. Integration: real Canny + real InsightFace on real selfie
# ════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not _models_present,
    reason="InsightFace models not found — set INSIGHTFACE_ROOT / INSWAPPER_MODEL",
)
class TestIntegrationFaceSwap:
    """End-to-end face swap using real InsightFace models and real selfie."""

    @pytest.fixture(autouse=True)
    def _mod(self, monkeypatch, tmp_path):
        self.m = _reload_app(monkeypatch, tmp_path)
        self.tmp = tmp_path

    def test_swap_face_on_real_selfie(self, real_selfie, real_selfie_resized):
        """Swap original selfie face onto the resized version of itself."""
        out = str(self.tmp / "swapped.png")
        result = self.m.swap_face(real_selfie, real_selfie_resized, out)
        assert result is True, "swap_face returned False — check model paths and face detection"
        assert Path(out).exists(), "No output file written"
        img = Image.open(out)
        assert img.size == Image.open(real_selfie_resized).size, (
            "Output dimensions changed unexpectedly"
        )

    def test_swapped_image_is_valid_rgb(self, real_selfie, real_selfie_resized):
        out = str(self.tmp / "swapped_rgb.png")
        result = self.m.swap_face(real_selfie, real_selfie_resized, out)
        assert result is True
        img = Image.open(out)
        assert img.mode in ("RGB", "BGR"), f"Unexpected mode: {img.mode}"

    def test_loader_caches_models(self):
        """Second call to _load_face_swapper should return the same objects."""
        a1, s1 = self.m._load_face_swapper()
        a2, s2 = self.m._load_face_swapper()
        assert a1 is a2, "FaceAnalysis not cached between calls"
        assert s1 is s2, "INSwapper not cached between calls"


@pytest.mark.skipif(
    not _controlnet_present,
    reason="ControlNet model not found — set SD_CONTROLNET_MODEL",
)
class TestIntegrationCanny:
    """Real Canny pipeline on the real selfie asset."""

    @pytest.fixture(autouse=True)
    def _mod(self, monkeypatch, tmp_path):
        self.m = _reload_app(monkeypatch, tmp_path)
        self.tmp = tmp_path

    def test_canny_on_real_selfie_resized(self, real_selfie_resized):
        out = self.m.generate_canny_control_image(real_selfie_resized, "intg01")
        assert out is not None
        assert Path(out).exists()
        img = Image.open(out)
        assert img.mode == "RGB"
        assert img.size == Image.open(real_selfie_resized).size
