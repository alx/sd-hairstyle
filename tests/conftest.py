"""
Shared fixtures for all test modules.
"""

import importlib
import io
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent          # project root (where app.py lives)
WORK = ROOT / "work"


def _make_png(path: Path, width=256, height=256, color=(120, 80, 60)) -> str:
    Image.new("RGB", (width, height), color=color).save(path)
    return str(path)


def _png_bytes(width=64, height=64, color=(100, 150, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


# ── real test assets (from work/) ─────────────────────────────────────

@pytest.fixture(scope="session")
def real_selfie():
    """960×1080 portrait selfie — real image, not synthetic."""
    p = WORK / "6e67091c_selfie.png"
    if not p.exists():
        pytest.skip(f"Test asset not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def real_selfie_resized():
    """1024×1024 resized selfie — already processed by resize_to_multiple."""
    p = WORK / "6e67091c_selfie_resized.png"
    if not p.exists():
        pytest.skip(f"Test asset not found: {p}")
    return str(p)


# ── synthetic image fixtures ───────────────────────────────────────────

@pytest.fixture
def small_png(tmp_path):
    return _make_png(tmp_path / "test.png")


@pytest.fixture
def selfie_png(tmp_path):
    return _make_png(tmp_path / "selfie.png", 256, 256, (180, 140, 110))


@pytest.fixture
def hairstyle_png(tmp_path):
    return _make_png(tmp_path / "hairstyle.png", 200, 300, (30, 20, 20))


# ── Flask test client ──────────────────────────────────────────────────

@pytest.fixture
def flask_client(monkeypatch, tmp_path):
    """Flask test client with WORK_DIR isolated to tmp_path."""
    monkeypatch.setenv("SD_WORK_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(ROOT))
    import app as app_module
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client
