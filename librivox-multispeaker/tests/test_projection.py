from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from sentence_transformers.models.Module import Module

from spoken_sentence_transformers import MultiAxisProjection, MultiAxisSentenceTransformer
from spoken_sentence_transformers import MultiAxisInfoNCELoss, MultiAxisNoDuplicatesBatchSampler
from spoken_sentence_transformers.encoders import AcousticEncoder, HFAcousticEncoder


AXES = {"content": 32, "speaker": 32, "accent": 16, "production": 16}
IN_FEATURES = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyEmbedder(Module):
    """Mimics a Transformer+Pooling stack: returns random sentence_embedding."""

    def __init__(self, dim: int = IN_FEATURES) -> None:
        super().__init__()
        self.dim = dim
        # Need at least one parameter so save/load doesn't complain
        self.dummy = nn.Linear(1, 1)

    def forward(self, features, **kwargs):
        tokens = features["input_ids"]
        batch_size = tokens.shape[0]
        features["sentence_embedding"] = torch.randn(
            batch_size, self.dim, device=tokens.device
        )
        return features

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        self.save_config(output_path)

    def tokenize(self, texts, **kwargs):
        return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}

    config_keys = ["dim"]


# ---------------------------------------------------------------------------
# MultiAxisProjection (module-level) tests
# ---------------------------------------------------------------------------


class TestMultiAxisProjection:
    def _make_module(self, **kwargs):
        defaults = dict(in_features=IN_FEATURES, axes=AXES)
        defaults.update(kwargs)
        return MultiAxisProjection(**defaults)

    def test_forward_concatenated_shape(self):
        proj = self._make_module()
        features = {"sentence_embedding": torch.randn(4, IN_FEATURES)}
        out = proj(features)

        total_dim = sum(AXES.values())
        assert out["sentence_embedding"].shape == (4, total_dim)

    def test_forward_writes_all_axis_keys(self):
        proj = self._make_module()
        features = {"sentence_embedding": torch.randn(2, IN_FEATURES)}
        out = proj(features)

        for name, dim in AXES.items():
            key = f"embedding_{name}"
            assert key in out, f"Missing key {key}"
            assert out[key].shape == (2, dim)

    def test_forward_with_axis_kwarg(self):
        proj = self._make_module()
        features = {"sentence_embedding": torch.randn(3, IN_FEATURES)}
        out = proj(features, axis="speaker")

        assert out["sentence_embedding"].shape == (3, AXES["speaker"])

    def test_forward_with_default_axis(self):
        proj = self._make_module(default_axis="accent")
        features = {"sentence_embedding": torch.randn(3, IN_FEATURES)}
        out = proj(features)

        assert out["sentence_embedding"].shape == (3, AXES["accent"])

    def test_axis_kwarg_overrides_default(self):
        proj = self._make_module(default_axis="accent")
        features = {"sentence_embedding": torch.randn(2, IN_FEATURES)}
        out = proj(features, axis="content")

        assert out["sentence_embedding"].shape == (2, AXES["content"])

    def test_invalid_default_axis_raises(self):
        with pytest.raises(ValueError, match="default_axis"):
            self._make_module(default_axis="nonexistent")

    def test_invalid_axis_kwarg_raises(self):
        proj = self._make_module()
        features = {"sentence_embedding": torch.randn(1, IN_FEATURES)}
        with pytest.raises(ValueError, match="Unknown axis"):
            proj(features, axis="nonexistent")

    def test_hidden_dim_creates_mlp(self):
        proj = self._make_module(hidden_dim=128)
        for name in AXES:
            head = proj.heads[name]
            assert isinstance(head, nn.Sequential)
            assert len(head) == 3  # Linear, ReLU, Linear

        features = {"sentence_embedding": torch.randn(2, IN_FEATURES)}
        out = proj(features)
        total_dim = sum(AXES.values())
        assert out["sentence_embedding"].shape == (2, total_dim)

    def test_no_hidden_dim_creates_linear(self):
        proj = self._make_module()
        for name in AXES:
            assert isinstance(proj.heads[name], nn.Linear)

    def test_get_sentence_embedding_dimension_concat(self):
        proj = self._make_module()
        assert proj.get_sentence_embedding_dimension() == sum(AXES.values())

    def test_get_sentence_embedding_dimension_default_axis(self):
        proj = self._make_module(default_axis="production")
        assert proj.get_sentence_embedding_dimension() == AXES["production"]

    def test_get_config_dict(self):
        proj = self._make_module(hidden_dim=64, default_axis="speaker")
        config = proj.get_config_dict()
        assert config["in_features"] == IN_FEATURES
        assert config["axes"] == AXES
        assert config["hidden_dim"] == 64
        assert config["default_axis"] == "speaker"

    def test_save_and_load_roundtrip(self):
        proj = self._make_module(hidden_dim=64)
        features = {"sentence_embedding": torch.randn(2, IN_FEATURES)}
        with torch.no_grad():
            original_out = proj(features.copy())

        with tempfile.TemporaryDirectory() as tmpdir:
            proj.save(tmpdir)

            # Check config file was written
            config_path = os.path.join(tmpdir, "config.json")
            assert os.path.exists(config_path)
            with open(config_path) as f:
                saved_config = json.load(f)
            assert saved_config["axes"] == AXES
            assert saved_config["hidden_dim"] == 64

            # Check weights file was written
            assert os.path.exists(
                os.path.join(tmpdir, "model.safetensors")
            )

            loaded = MultiAxisProjection.load(tmpdir)

        with torch.no_grad():
            loaded_out = loaded(features.copy())

        torch.testing.assert_close(
            original_out["sentence_embedding"],
            loaded_out["sentence_embedding"],
        )
        for name in AXES:
            torch.testing.assert_close(
                original_out[f"embedding_{name}"],
                loaded_out[f"embedding_{name}"],
            )

    def test_repr(self):
        proj = self._make_module()
        r = repr(proj)
        assert "MultiAxisProjection" in r
        assert "content" in r


# ---------------------------------------------------------------------------
# MultiAxisSentenceTransformer tests
# ---------------------------------------------------------------------------


class TestMultiAxisSentenceTransformer:
    @pytest.fixture()
    def model(self):
        embedder = DummyEmbedder(dim=IN_FEATURES)
        projection = MultiAxisProjection(
            in_features=IN_FEATURES, axes=AXES
        )
        return MultiAxisSentenceTransformer(modules=[embedder, projection])

    @pytest.fixture()
    def model_default_axis(self):
        embedder = DummyEmbedder(dim=IN_FEATURES)
        projection = MultiAxisProjection(
            in_features=IN_FEATURES,
            axes=AXES,
            default_axis="speaker",
        )
        return MultiAxisSentenceTransformer(modules=[embedder, projection])

    def test_axes_property(self, model):
        assert model.axes == sorted(AXES.keys())

    def test_encode_returns_concatenated(self, model):
        emb = model.encode(["hello", "world"])
        total_dim = sum(AXES.values())
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (2, total_dim)

    def test_encode_default_axis(self, model_default_axis):
        emb = model_default_axis.encode(["hello", "world"])
        assert emb.shape == (2, AXES["speaker"])

    def test_encode_single_string(self, model):
        emb = model.encode("hello")
        total_dim = sum(AXES.values())
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (total_dim,)

    def test_encode_output_value_none(self, model):
        features_list = model.encode(
            ["hello", "world"], output_value=None
        )
        assert isinstance(features_list, list)
        assert len(features_list) == 2
        for f in features_list:
            assert isinstance(f, dict)
            for name in AXES:
                assert f"embedding_{name}" in f

    def test_encode_axis(self, model):
        emb = model.encode_axis(["a", "b", "c"], axis="content")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (3, AXES["content"])

    def test_encode_axis_single_string(self, model):
        emb = model.encode_axis("hello", axis="speaker")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (AXES["speaker"],)

    def test_encode_axis_as_tensor(self, model):
        emb = model.encode_axis(
            ["a", "b"],
            axis="accent",
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        assert isinstance(emb, Tensor)
        assert emb.shape == (2, AXES["accent"])

    def test_encode_all_axes(self, model):
        result = model.encode_all_axes(["a", "b"])
        assert isinstance(result, dict)
        assert set(result.keys()) == set(AXES.keys())
        for name, dim in AXES.items():
            assert result[name].shape == (2, dim)
            assert isinstance(result[name], np.ndarray)

    def test_encode_all_axes_single_sentence(self, model):
        """Single string still returns [1, D] arrays (not squeezed)."""
        result = model.encode_all_axes("hello")
        for name, dim in AXES.items():
            assert result[name].shape == (1, dim)

    def test_similarity_vector(self, model):
        a = model.encode_all_axes(["x", "y", "z"])
        b = model.encode_all_axes(["p", "q"])
        sim = model.similarity_vector(a, b)

        assert isinstance(sim, dict)
        assert set(sim.keys()) == set(AXES.keys())
        for name in AXES:
            assert sim[name].shape == (3, 2)

    def test_similarity_vector_self(self, model):
        a = model.encode_all_axes(["x", "y"])
        sim = model.similarity_vector(a, a)
        for name in AXES:
            assert sim[name].shape == (2, 2)

    def test_no_projection_module_raises(self):
        embedder = DummyEmbedder(dim=IN_FEATURES)
        model = MultiAxisSentenceTransformer(modules=[embedder])
        with pytest.raises(ValueError, match="No MultiAxisProjection"):
            _ = model.axes


# ---------------------------------------------------------------------------
# MultiAxisNoDuplicatesBatchSampler stub tests
# ---------------------------------------------------------------------------


class TestMultiAxisNoDuplicatesBatchSampler:
    def test_import(self):
        assert MultiAxisNoDuplicatesBatchSampler is not None

    def test_instantiation_with_mock_dataset(self):
        dataset = MagicMock()
        dataset.column_names = ["content_label", "speaker_label"]
        dataset.__len__ = MagicMock(return_value=100)
        dataset.__getitem__ = MagicMock(return_value={"content_label": "a", "speaker_label": "b"})
        dataset["content_label"] = ["label_" + str(i % 10) for i in range(100)]
        dataset["speaker_label"] = ["spk_" + str(i % 5) for i in range(100)]

        sampler = MultiAxisNoDuplicatesBatchSampler(
            dataset,
            batch_size=8,
            drop_last=False,
            valid_label_columns=["content_label", "speaker_label"],
        )
        assert sampler.batch_size == 8
        assert len(sampler.axis_label_columns) == 2


# ---------------------------------------------------------------------------
# MultiAxisInfoNCELoss stub tests
# ---------------------------------------------------------------------------


class TestMultiAxisInfoNCELoss:
    def test_import(self):
        assert MultiAxisInfoNCELoss is not None

    def test_forward(self):
        embedder = DummyEmbedder(dim=IN_FEATURES)
        projection = MultiAxisProjection(in_features=IN_FEATURES, axes=AXES)
        model = MultiAxisSentenceTransformer(modules=[embedder, projection])
        model = model.to("cpu")

        loss_fn = MultiAxisInfoNCELoss(model)

        batch_size = 4
        named_features = {
            "anchor": {"input_ids": torch.zeros(batch_size, 1, dtype=torch.long)},
            "content_pos": {"input_ids": torch.zeros(batch_size, 1, dtype=torch.long)},
            "speaker_pos": {"input_ids": torch.zeros(batch_size, 1, dtype=torch.long)},
        }
        losses = loss_fn(named_features, labels=None)

        assert isinstance(losses, dict)
        assert "content" in losses
        assert "speaker" in losses
        for v in losses.values():
            assert isinstance(v, Tensor)
            assert v.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# AcousticEncoder / HFAcousticEncoder stub tests
# ---------------------------------------------------------------------------


class TestAcousticEncoderInterface:
    def test_abstract_base_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AcousticEncoder()

    def test_hf_encoder_is_subclass(self):
        assert issubclass(HFAcousticEncoder, AcousticEncoder)

    @patch("spoken_sentence_transformers.encoders.hf.AutoFeatureExtractor")
    @patch("spoken_sentence_transformers.encoders.hf.AutoModel")
    def test_hf_encoder_init(self, mock_model_cls, mock_fe_cls):
        mock_fe = MagicMock()
        mock_fe.sampling_rate = 16000
        mock_fe_cls.from_pretrained.return_value = mock_fe

        mock_model = MagicMock()
        mock_model.encoder = MagicMock()
        mock_model.encoder.config.hidden_size = 768
        mock_model_cls.from_pretrained.return_value = mock_model

        enc = HFAcousticEncoder("fake-model")
        assert enc.sampling_rate == 16000
        assert enc.get_word_embedding_dimension() == 768
