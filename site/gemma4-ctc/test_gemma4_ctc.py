"""Sanity checks for Gemma4ForCTC loaded as a remote-code model repo."""

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


MODEL_DIR = Path(__file__).resolve().parent


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers.models.gemma4") is None,
    reason="Gemma 4 support requires a newer transformers build",
)


def test_remote_code_model_instantiates_without_encoder_download(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "hf_home"))

    config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        "modeling_gemma4_ctc.Gemma4ForCTC",
        str(MODEL_DIR),
    )
    model = model_class(config, _skip_encoder_download=True)

    assert type(model.gemma4_audio_encoder.output_proj).__name__ == "Identity"
    assert model.lm_head.in_features == 1024

    batch, time, n_mels = 2, 400, 128
    input_features = torch.zeros(batch, time, n_mels, dtype=torch.float32)
    attention_mask = torch.ones(batch, time, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        out = model(input_features=input_features, attention_mask=attention_mask)

    expected_time = time // 4
    assert out.logits.shape == (batch, expected_time, config.vocab_size)
