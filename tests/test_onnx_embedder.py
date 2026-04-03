"""Tests for diarization/onnx_embedder.py — ONNX speaker embedding extraction.

These tests verify the OnnxSpeakerEmbedder interface without requiring an
actual ONNX model file. Model-dependent tests (accuracy, comparison with
PyTorch) are in test_diarization_accuracy.py.
"""

import numpy as np
import pytest

from diarization.onnx_embedder import OnnxSpeakerEmbedder, ONNX_MODELS


class TestOnnxEmbedderConfig:

    def test_known_models(self):
        """All registered models should have required fields."""
        for name, (model_id, filename, embed_dim) in ONNX_MODELS.items():
            assert isinstance(model_id, str)
            assert filename.endswith(".onnx")
            assert isinstance(embed_dim, int) and embed_dim > 0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            OnnxSpeakerEmbedder(model_name="nonexistent_model")

    def test_not_loaded_returns_none(self):
        """extract() should return None when model isn't loaded."""
        embedder = OnnxSpeakerEmbedder(model_name="eres2net_large")
        audio = np.random.randn(32000).astype(np.float32) * 0.1
        result = embedder.extract(audio)
        assert result is None

    def test_short_audio_returns_none(self):
        """extract() should return None for audio shorter than 1.0s."""
        embedder = OnnxSpeakerEmbedder(model_name="eres2net_large")
        # Fake load state to test the audio length check
        embedder._loaded = True
        embedder._session = "dummy"  # won't actually be called
        audio = np.random.randn(8000).astype(np.float32)  # 0.5s < 1.0s min
        result = embedder.extract(audio)
        assert result is None

    def test_embedding_dim_property(self):
        """embedding_dim should reflect the model's configured output."""
        embedder = OnnxSpeakerEmbedder(model_name="eres2net_large")
        # Before load, dim comes from model init
        assert embedder.embedding_dim == 0  # not loaded yet
        # After manual setup
        embedder._embedding_dim = 192
        assert embedder.embedding_dim == 192

    def test_default_model_is_eres2net(self):
        from diarization.onnx_embedder import DEFAULT_MODEL
        assert DEFAULT_MODEL == "eres2net_large"


class TestOnnxEmbedderWeighted:

    def test_weighted_not_loaded_returns_none(self):
        embedder = OnnxSpeakerEmbedder(model_name="eres2net_large")
        audio = np.random.randn(32000).astype(np.float32)
        weights = np.ones(100, dtype=np.float32)
        result = embedder.extract_weighted(audio, weights)
        assert result is None
