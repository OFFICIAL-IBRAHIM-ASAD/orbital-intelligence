"""
conftest.py – Session-wide stubs for unavailable heavy packages.

This file runs before any test module is collected. It installs lightweight
stubs into sys.modules so that ee, torch, cv2, etc. can be imported by the
project source without errors in a minimal CI environment.

Individual test files must NOT replace backend.satellite_client or
ml_core.inference with MagicMock stubs; they should use @patch decorators
within each test instead.
"""

import sys
import os
import types
from unittest.mock import MagicMock
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _already_real(name):
    """Return True if the module is already loaded as a real (non-stub) module."""
    existing = sys.modules.get(name)
    return existing is not None and not isinstance(existing, MagicMock)


# ---------------------------------------------------------------------------
# 1. Google Earth Engine stub
# ---------------------------------------------------------------------------

if not _already_real("ee"):
    _mod("ee")  # bare stub; tests patch `backend.satellite_client.ee` as needed


# ---------------------------------------------------------------------------
# 2. PyTorch stubs
# ---------------------------------------------------------------------------

if not _already_real("torch"):
    _dummy_arr_256 = np.zeros((256, 256), dtype=np.float32)

    class _FakeTensor:
        shape = (1, 1, 256, 256)
        def unsqueeze(self, _): return self
        def to(self, _): return self
        def squeeze(self): return _dummy_arr_256
        def cpu(self): return self
        def numpy(self): return _dummy_arr_256

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch_stub = _mod("torch")
    torch_stub.cuda = MagicMock()
    torch_stub.cuda.is_available = MagicMock(return_value=False)
    torch_stub.device = MagicMock(side_effect=lambda x: x)
    torch_stub.no_grad = MagicMock(return_value=_NoGrad())
    torch_stub.sigmoid = MagicMock(return_value=_FakeTensor())
    torch_stub.load = MagicMock(return_value={})
    torch_stub.from_numpy = MagicMock(return_value=_FakeTensor())

    nn_stub = _mod("torch.nn")
    class _FakeNNModule:
        def __init__(self): pass
        def parameters(self): return iter([MagicMock()])
        def eval(self): return self
        def to(self, _): return self
        def load_state_dict(self, d): pass
        def __call__(self, x): return _FakeTensor()
    nn_stub.Module = _FakeNNModule
    torch_stub.nn = nn_stub


# ---------------------------------------------------------------------------
# 3. segmentation_models_pytorch + timm stubs
# ---------------------------------------------------------------------------

if not _already_real("segmentation_models_pytorch"):
    _dummy_arr_256 = np.zeros((256, 256), dtype=np.float32)

    class _FakeTensor:
        def unsqueeze(self, _): return self
        def to(self, _): return self
        def squeeze(self): return _dummy_arr_256
        def cpu(self): return self
        def numpy(self): return _dummy_arr_256

    smp_stub = _mod("segmentation_models_pytorch")
    smp_stub.Unet = MagicMock(return_value=MagicMock(
        __call__=MagicMock(return_value=_FakeTensor())
    ))

for _n in ("timm", "torchvision", "torchvision.transforms"):
    if not _already_real(_n):
        _mod(_n)


# ---------------------------------------------------------------------------
# 4. OpenCV stub
# ---------------------------------------------------------------------------

if not _already_real("cv2"):
    _dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    _clean_mask = np.zeros((256, 256), dtype=np.float32)

    cv2_stub = _mod("cv2")
    cv2_stub.imread = MagicMock(return_value=_dummy_img)
    cv2_stub.cvtColor = MagicMock(return_value=_dummy_img)
    cv2_stub.resize = MagicMock(return_value=_dummy_img)
    cv2_stub.COLOR_BGR2RGB = 4
    cv2_stub.MORPH_OPEN = 2
    cv2_stub.MORPH_CLOSE = 3
    cv2_stub.morphologyEx = MagicMock(return_value=_clean_mask)
    cv2_stub.ones = MagicMock(return_value=np.ones((5, 5), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 5. Matplotlib stub
# ---------------------------------------------------------------------------

if not _already_real("matplotlib"):
    mpl_stub = _mod("matplotlib")
    plt_stub = _mod("matplotlib.pyplot")
    plt_stub.figure = MagicMock()
    plt_stub.imshow = MagicMock()
    plt_stub.axis = MagicMock()
    plt_stub.savefig = MagicMock()
    plt_stub.close = MagicMock()
    mpl_stub.pyplot = plt_stub


# ---------------------------------------------------------------------------
# 6. Pre-load ml_core.inference with weight-check patched
#    (avoids FileNotFoundError from module-level engine = DamageInferenceEngine())
# ---------------------------------------------------------------------------

from unittest.mock import patch as _patch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if "ml_core.inference" not in sys.modules:
    with _patch("os.path.exists", return_value=True):
        import ml_core.inference  # noqa: F401 – side-effect import to cache it

# Also ensure backend.satellite_client is loaded as the REAL module (not a stub)
if "backend.satellite_client" not in sys.modules or isinstance(
    sys.modules.get("backend.satellite_client"), MagicMock
):
    # Force real import (ee is already stubbed above)
    for _key in list(sys.modules.keys()):
        if "satellite_client" in _key:
            del sys.modules[_key]
    import backend.satellite_client  # noqa: F401
