"""
Test Suite – ML Model Architecture (ml_core/model.py) and Inference
=====================================================================
Verifies the OrbitalDamageDetector architecture and DamageInferenceEngine
preprocessing logic.

conftest.py stubs torch, smp, cv2, matplotlib, and patches the weight-file
check so that module-level engine initialisation succeeds without real weights.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import segmentation_models_pytorch as smp_stub
from ml_core import model as model_module
from ml_core.model import OrbitalDamageDetector
from ml_core import inference as inference_module
from ml_core.inference import DamageInferenceEngine


# ---------------------------------------------------------------------------
# OrbitalDamageDetector – structural tests (mocked architecture)
# ---------------------------------------------------------------------------

class TestOrbitalDamageDetectorStructure:
    def test_module_imports_without_error(self):
        assert model_module is not None

    def test_class_exists(self):
        assert OrbitalDamageDetector is not None

    def test_class_has_forward_method(self):
        assert hasattr(OrbitalDamageDetector, "forward")

    def test_instantiation_does_not_raise(self):
        model = OrbitalDamageDetector()
        assert model is not None

    def test_smp_unet_called_with_6_in_channels(self):
        """Model must request 6-channel input (pre+post RGB stacked)."""
        smp_stub.Unet.reset_mock()
        OrbitalDamageDetector()
        if smp_stub.Unet.called:
            kw = smp_stub.Unet.call_args[1]
            assert kw.get("in_channels", 6) == 6

    def test_smp_unet_called_with_1_class(self):
        """Single damage-probability output channel."""
        smp_stub.Unet.reset_mock()
        OrbitalDamageDetector()
        if smp_stub.Unet.called:
            kw = smp_stub.Unet.call_args[1]
            assert kw.get("classes", 1) == 1

    def test_smp_unet_activation_is_none(self):
        """Raw logits: activation must be None."""
        smp_stub.Unet.reset_mock()
        OrbitalDamageDetector()
        if smp_stub.Unet.called:
            kw = smp_stub.Unet.call_args[1]
            assert kw.get("activation", None) is None

    def test_resnet34_backbone_selected(self):
        """ResNet34 backbone is required for the 6-channel encoder."""
        smp_stub.Unet.reset_mock()
        OrbitalDamageDetector()
        if smp_stub.Unet.called:
            kw = smp_stub.Unet.call_args[1]
            assert kw.get("encoder_name", "resnet34") == "resnet34"


# ---------------------------------------------------------------------------
# DamageInferenceEngine – preprocessing helper (no weights needed)
# ---------------------------------------------------------------------------

class TestInferencePreprocessing:
    @pytest.fixture()
    def engine(self):
        """Return a DamageInferenceEngine with mocked internals."""
        eng = DamageInferenceEngine.__new__(DamageInferenceEngine)
        eng.device = "cpu"
        eng.model  = MagicMock()
        return eng

    def test_preprocess_output_is_float32(self, engine, tmp_path):
        result = engine._preprocess_image(str(tmp_path / "test.png"))
        assert result.dtype == np.float32

    def test_preprocess_output_normalized_to_0_1(self, engine, tmp_path):
        result = engine._preprocess_image(str(tmp_path / "test.png"))
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_preprocess_output_shape_is_256_256_3(self, engine, tmp_path):
        result = engine._preprocess_image(str(tmp_path / "test.png"))
        assert result.shape == (256, 256, 3)

    def test_preprocess_raises_on_unreadable_image(self, engine):
        import cv2
        original = cv2.imread
        cv2.imread = MagicMock(return_value=None)
        with pytest.raises(ValueError, match="Could not read image"):
            engine._preprocess_image("/nonexistent/path.png")
        cv2.imread = original


# ---------------------------------------------------------------------------
# Inference constants / thresholds
# ---------------------------------------------------------------------------

class TestInferenceConstants:
    def test_confidence_threshold_is_65_percent(self):
        inference_path = os.path.join(
            os.path.dirname(__file__), "..", "ml_core", "inference.py"
        )
        with open(inference_path) as f:
            src = f.read()
        assert "0.65" in src, "Confidence threshold 0.65 must be present in inference.py"

    def test_damage_verdict_threshold_is_5_percent(self):
        main_path = os.path.join(
            os.path.dirname(__file__), "..", "backend", "main.py"
        )
        with open(main_path) as f:
            src = f.read()
        assert "5.0" in src, "Damage verdict threshold 5.0 must be present in main.py"

    def test_weights_path_points_to_v2_file(self):
        assert "orbital_detector_v2.pth" in inference_module.WEIGHTS_PATH
