"""ONNX-based speaker embedding extraction (no PyTorch dependency).

Loads a 3D-Speaker model exported to ONNX format and extracts speaker
embeddings using pure-numpy feature extraction + onnxruntime inference.
This runs safely in the main process alongside MLX — no subprocess needed.

Supported models (via export_onnx.py):
  - ERes2Net-large (512-dim, best accuracy)
  - ERes2Netv2 (192-dim, smaller variant)
  - CAM++ (512-dim, lighter/faster)

Follows the same ONNX pattern as audio/vad.py (Silero VAD).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from diarization.fbank import compute_fbank

log = logging.getLogger(__name__)

# Model registry: model_name → (ModelScope model_id, filename, embedding_dim)
ONNX_MODELS = {
    "eres2net_large": (
        "iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
        "eres2net_large.onnx",
        512,
    ),
    "eres2netv2": (
        "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "eres2netv2.onnx",
        192,
    ),
    "campplus": (
        "iic/speech_campplus_sv_zh-cn_16k-common",
        "campplus.onnx",
        512,
    ),
}

DEFAULT_MODEL = "eres2net_large"
CACHE_DIR = Path.home() / ".cache" / "3dspeaker"


class OnnxSpeakerEmbedder:
    """Extract speaker embeddings via ONNX Runtime.

    No PyTorch dependency. Uses pure-numpy Fbank features and onnxruntime
    for inference. Safe to run in the same process as MLX.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
    ):
        self.model_name = model_name
        self._cache_dir = cache_dir or CACHE_DIR
        self._session = None
        self._loaded = False
        self._embedding_dim: int = 0

        if model_name not in ONNX_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(ONNX_MODELS.keys())}"
            )

    def load(self) -> None:
        """Load the ONNX model. Call once before extract()."""
        import onnxruntime

        model_id, filename, embed_dim = ONNX_MODELS[self.model_name]
        self._embedding_dim = embed_dim

        model_path = self._cache_dir / self.model_name / filename
        if not model_path.exists():
            model_path = self._try_export(model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {model_path}. "
                    f"Run: python -m scripts.export_onnx --model {self.model_name}"
                )

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2  # slightly more than VAD since embeddings are heavier
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._loaded = True
        log.info(
            "Loaded ONNX speaker model: %s (%d-dim embeddings)",
            self.model_name, self._embedding_dim,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Extract a speaker embedding from audio.

        Args:
            audio: 1-D float32 array, values in [-1, 1].
            sample_rate: Audio sample rate (default 16000).

        Returns:
            L2-normalized embedding of shape (embedding_dim,), or None if
            the audio is too short or the model isn't loaded.
        """
        if not self._loaded or self._session is None:
            return None

        audio = np.asarray(audio, dtype=np.float32).ravel()
        if len(audio) < 16000:  # 1.5s minimum
            return None

        # Compute Fbank features (pure numpy, no PyTorch)
        feats = compute_fbank(audio, sample_rate=sample_rate)
        if feats.shape[0] == 0:
            return None

        # ONNX inference: input shape (1, num_frames, 80)
        feats_input = feats[np.newaxis, :, :].astype(np.float32)

        outputs = self._session.run(
            None,
            {"feature": feats_input},
        )
        embedding = outputs[0].squeeze()  # (embedding_dim,)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def extract_weighted(
        self,
        audio: np.ndarray,
        frame_weights: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Extract embedding with feature-level weighting (for overlap-aware diarization).

        Args:
            audio: 1-D float32 array.
            frame_weights: Per-frame weights from segmentation model.
            sample_rate: Audio sample rate.

        Returns:
            L2-normalized weighted embedding, or None.
        """
        if not self._loaded or self._session is None:
            return None

        audio = np.asarray(audio, dtype=np.float32).ravel()
        if len(audio) < 16000:
            return None

        feats = compute_fbank(audio, sample_rate=sample_rate)
        if feats.shape[0] == 0:
            return None

        # Upsample segmentation weights (~17ms) to Fbank frame level (~10ms)
        n_fbank = feats.shape[0]
        seg_dur = 270 / sample_rate
        fbank_dur = 160 / sample_rate
        fbank_weights = np.ones(n_fbank, dtype=np.float32)
        for i in range(n_fbank):
            seg_idx = min(int(i * fbank_dur / seg_dur), len(frame_weights) - 1)
            fbank_weights[i] = frame_weights[seg_idx]

        feats = feats * fbank_weights[:, None]

        feats_input = feats[np.newaxis, :, :].astype(np.float32)
        outputs = self._session.run(None, {"feature": feats_input})
        embedding = outputs[0].squeeze()

        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def _try_export(self, target_path: Path) -> Path:
        """Attempt to export the model to ONNX on-the-fly.

        This requires PyTorch + speakerlab to be installed. If they're not
        available (e.g. in the main process), returns the path as-is and
        the caller should raise FileNotFoundError.
        """
        try:
            from scripts.export_onnx import export_model
            export_model(self.model_name, target_path)
            return target_path
        except ImportError:
            log.warning(
                "Cannot auto-export ONNX model (speakerlab/torch not available). "
                "Run: python -m scripts.export_onnx --model %s",
                self.model_name,
            )
            return target_path
        except Exception as e:
            log.warning("ONNX auto-export failed: %s", e)
            return target_path
