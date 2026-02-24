"""SentenceTransformer-compatible acoustic encoder module.

Wraps any HuggingFace audio model whose encoder outputs ``last_hidden_state``
of shape ``[B, T, H]`` — including WavLM, wav2vec2, HuBERT, and Whisper
(encoder half only).

Pipeline position::

    HFAcousticEncoder → Pooling → [MultiAxisProjection]
    sets                sets
    "token_embeddings"  "sentence_embedding"

This follows the same pattern as the text ``Transformer`` module: the encoder
produces per-frame hidden states and the downstream ``Pooling`` module
collapses them to a single vector.  Pooling strategy (mean, cls, etc.) is
therefore configured on the ``Pooling`` module, not here.

Audio input formats accepted by ``tokenize()``
----------------------------------------------
* ``list[np.ndarray]``  — raw waveforms assumed to be at ``self.sampling_rate``
* ``list[dict]``        — HuggingFace datasets audio column format:
  ``{"array": np.ndarray, "sampling_rate": int}``

Both formats are resampled to the feature extractor's native rate if needed.
"""

from __future__ import annotations

from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import Tensor
from transformers import AutoFeatureExtractor, AutoModel

from .base import AcousticEncoder


class HFAcousticEncoder(AcousticEncoder):
    """SentenceTransformer module wrapping a HuggingFace audio encoder.

    Designed for models that consume raw waveforms (WavLM, wav2vec2, HuBERT)
    or mel spectrograms (Whisper).  Uses ``AutoFeatureExtractor`` and
    ``AutoModel`` so the same class handles all of them without subclassing.

    For seq2seq models such as Whisper only the encoder half is used; the
    decoder is discarded.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        sampling_rate: Expected sample rate of incoming waveforms.  If the
            feature extractor requires a different rate, audio is resampled
            automatically.  Defaults to 16 000 Hz.
    """

    config_keys: list[str] = ["sampling_rate"]
    save_in_root: bool = True  # mirrors Transformer; saves HF files to model root

    def __init__(
        self,
        model_name_or_path: str,
        sampling_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.sampling_rate = sampling_rate

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        _model = AutoModel.from_pretrained(model_name_or_path)
        # For seq2seq models (e.g. Whisper) keep only the encoder.
        self.encoder = getattr(_model, "encoder", _model)
        if self.encoder is not _model:
            del _model

    # ------------------------------------------------------------------
    # Tokenize (feature extraction) — called by SentenceTransformer.encode()
    # ------------------------------------------------------------------

    def tokenize(
        self,
        audio_list: list[np.ndarray | dict[str, Any]],
        **kwargs,
    ) -> dict[str, Tensor]:
        """Extract features from a batch of audio waveforms.

        Resamples each waveform to ``self.feature_extractor.sampling_rate``
        if it differs from the source rate.

        Args:
            audio_list: Waveforms as raw ``np.ndarray`` arrays or as HF
                datasets ``{"array": ..., "sampling_rate": ...}`` dicts.

        Returns:
            Feature dict with ``input_values`` (wav2vec2 / WavLM / HuBERT)
            or ``input_features`` (Whisper), plus ``attention_mask`` where
            applicable, as ``Tensor`` values.
        """
        target_sr = self.feature_extractor.sampling_rate
        waveforms: list[np.ndarray] = []

        for item in audio_list:
            if isinstance(item, dict):
                waveform = np.asarray(item["array"], dtype=np.float32)
                src_sr = int(item.get("sampling_rate", self.sampling_rate))
            else:
                waveform = np.asarray(item, dtype=np.float32)
                src_sr = self.sampling_rate

            if src_sr != target_sr:
                waveform = _resample(waveform, src_sr, target_sr)

            waveforms.append(waveform)

        batch = self.feature_extractor(
            waveforms,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        return dict(batch)

    # ------------------------------------------------------------------
    # Forward — called by SentenceTransformer.forward()
    # ------------------------------------------------------------------

    def forward(
        self, features: dict[str, Tensor | Any], **kwargs
    ) -> dict[str, Tensor | Any]:
        """Run the encoder and write ``token_embeddings`` into features.

        Handles both ``input_values`` (wav2vec2 / WavLM / HuBERT) and
        ``input_features`` (Whisper) transparently.

        The ``attention_mask`` expected by downstream ``Pooling`` is taken
        from the feature extractor output when present, or synthesised as
        all-ones for models that do not produce one (e.g. Whisper).
        """
        if "input_values" in features:
            encoder_inputs = {"input_values": features["input_values"]}
            if "attention_mask" in features:
                encoder_inputs["attention_mask"] = features["attention_mask"]
        elif "input_features" in features:
            encoder_inputs = {"input_features": features["input_features"]}
        else:
            raise ValueError(
                f"HFAcousticEncoder.forward: expected 'input_values' or "
                f"'input_features' in features; got {list(features.keys())}"
            )

        outputs = self.encoder(**encoder_inputs)
        hidden = outputs.last_hidden_state  # [B, T, H]

        features["token_embeddings"] = hidden

        # Pooling needs attention_mask [B, T].  Whisper doesn't produce one,
        # so we fall back to all-ones (treat every frame as valid).
        if "attention_mask" not in features:
            features["attention_mask"] = torch.ones(
                hidden.shape[:2], dtype=torch.long, device=hidden.device
            )

        return features

    # ------------------------------------------------------------------
    # Dimension helper — used by Pooling and MultiAxisProjection
    # ------------------------------------------------------------------

    def get_word_embedding_dimension(self) -> int:
        return self.encoder.config.hidden_size

    # ------------------------------------------------------------------
    # Save / load  (mirrors sentence_transformers.models.Transformer)
    # ------------------------------------------------------------------

    def save(
        self,
        output_path: str,
        *args,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        self.feature_extractor.save_pretrained(output_path)
        self.encoder.save_pretrained(
            output_path, safe_serialization=safe_serialization
        )
        self.save_config(output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = dict(
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        # model_name_or_path is the directory containing saved HF model files.
        config["model_name_or_path"] = model_name_or_path
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"HFAcousticEncoder("
            f"model='{self.model_name_or_path}', "
            f"sampling_rate={self.sampling_rate}, "
            f"hidden_size={self.get_word_embedding_dimension()})"
        )


# ------------------------------------------------------------------
# Resampling helper (lazy import so librosa is optional)
# ------------------------------------------------------------------

def _resample(waveform: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    try:
        import librosa
        return librosa.resample(waveform, orig_sr=src_sr, target_sr=target_sr)
    except ImportError:
        pass
    try:
        import torchaudio.functional as F
        t = torch.from_numpy(waveform).unsqueeze(0)
        r = F.resample(t, src_sr, target_sr)
        return r.squeeze(0).numpy()
    except ImportError:
        pass
    raise ImportError(
        "Audio resampling requires either `librosa` or `torchaudio`. "
        "Install one with:\n"
        "    pip install librosa\n"
        "or\n"
        "    pip install torchaudio"
    )
