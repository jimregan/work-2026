
Here's the refined prompt:

## Instructions for Claude Code

### Goal

Create a `Gemma4ForCTC` model class for CTC-based ASR training, using the Gemma 4 audio encoder as the backbone. The implementation must be a direct fork of the `Wav2Vec2ForCTC` code from HuggingFace Transformers, edited minimally to replace the Wav2Vec2 encoder with the Gemma 4 audio encoder, skipping the encoder's final upsampling projection layer. The code lives in a **standalone repository** (not inside the `transformers` library), and will be loaded via `trust_remote_code=True` from a HuggingFace model hub repo.

---

### Step 0 – Understand the source files before writing any code

Read and understand these files in full before touching anything:

1. `transformers/src/transformers/models/wav2vec2/modeling_wav2vec2.py` — focus on `Wav2Vec2ForCTC`, `Wav2Vec2PreTrainedModel`, and how `lm_head`, `dropout`, and the CTC loss are wired up.
2. `transformers/src/transformers/models/gemma4/modeling_gemma4.py` — focus on the audio encoder class and any upsampling/projection layers that come after the conformer stack (e.g. `output_proj`, or any linear that projects from conformer hidden size 1536 to LLM embedding size 2560).
3. `transformers/src/transformers/models/gemma4/configuration_gemma4.py` — specifically `Gemma4AudioConfig` and its fields.

Key facts you will discover and must use:

- The Gemma 4 audio encoder is a 12-layer USM-style conformer producing `(batch, time/4, 1536)`.
- It then applies an `output_proj` linear projecting 1536 → 2560 for LLM use. **This is the layer to skip.**
- The correct way to load the encoder with trained weights is `AutoModelForMultimodalLM.from_pretrained(...)` then extracting `model.model.audio_tower` — **not** `AutoModel`, which silently uses random weights due to a key prefix mismatch.

---

### Step 1 – Repository structure

Create a self-contained HuggingFace custom model repo with this layout:

```
gemma4-ctc/
    config.json              ← will be generated/filled in manually, not written by code
    configuration_gemma4_ctc.py
    modeling_gemma4_ctc.py
```

No `__init__.py` is needed. HuggingFace's `trust_remote_code` mechanism discovers files by name.

The top of `modeling_gemma4_ctc.py` must declare:

```python
AUTO_MAP = {
    "AutoConfig": "configuration_gemma4_ctc.Gemma4CTCConfig",
    "AutoModelForCTC": "modeling_gemma4_ctc.Gemma4ForCTC",
}
```

And the top of `configuration_gemma4_ctc.py` must declare:

```python
AUTO_MAP = {
    "AutoConfig": "configuration_gemma4_ctc.Gemma4CTCConfig",
}
```

---

### Step 2 – Create `configuration_gemma4_ctc.py`

Start from an exact copy of `Wav2Vec2Config` from `configuration_wav2vec2.py`. Then make these targeted edits only:

- Rename the class to `Gemma4CTCConfig`.
- Set `model_type = "gemma4_ctc"`.
- Remove all Wav2Vec2-specific fields relating to the feature extractor (conv layers, quantizer, gumbel softmax, codebook, etc.).
- Add a `gemma4_audio_model_id: str` field (default `"google/gemma-4-E2B-it"`) recording which checkpoint the audio encoder comes from.
- Keep: `vocab_size`, `ctc_loss_reduction`, `ctc_zero_infinity`, `pad_token_id`, `final_dropout`.
- Set `hidden_size` to default to `1536` (the conformer output dim, **not** the projected 2560).

---

### Step 3 – Create `modeling_gemma4_ctc.py`

Start from an **exact copy** of the relevant sections of `modeling_wav2vec2.py`, then apply only the diffs described below. Preserve all docstrings, type hints, and output dataclass structures where they still apply.

**Classes to copy and rename:**

- `Wav2Vec2PreTrainedModel` → `Gemma4CTCPreTrainedModel`
- `Wav2Vec2ForCTC` → `Gemma4ForCTC`

**Changes to `Gemma4CTCPreTrainedModel`:**

- Set `config_class = Gemma4CTCConfig`.
- Set `base_model_prefix = "gemma4_audio_encoder"`.
- Update or remove `_keys_to_ignore_on_load_missing` / `_keys_to_ignore_on_load_unexpected` to remove Wav2Vec2 references; add an entry to ignore `output_proj.*` since it is replaced with `nn.Identity()`.

**Changes to `Gemma4ForCTC.__init__`:**

Replace `self.wav2vec2 = Wav2Vec2Model(config)` with the following block that extracts the pre-trained audio tower:

```python
from transformers import AutoModelForMultimodalLM

_full_model = AutoModelForMultimodalLM.from_pretrained(
    config.gemma4_audio_model_id,
    torch_dtype=torch.bfloat16,
    device_map=None,  # let caller handle device placement
    low_cpu_mem_usage=True,
)
self.gemma4_audio_encoder = _full_model.model.audio_tower
del _full_model  # free remaining model memory
```

Immediately after, skip the final upsampling projection by replacing it with a no-op. The exact attribute name must be verified from Step 0, but it is expected to be `output_proj`:

```python
# Disable the projection to LLM embedding space (1536 → 2560); we want raw conformer output.
self.gemma4_audio_encoder.output_proj = nn.Identity()
```

Keep these lines unchanged:

```python
self.dropout = nn.Dropout(config.final_dropout)
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)  # 1536 → vocab_size
```

**Changes to `Gemma4ForCTC.forward`:**

- Replace `input_values` with `input_features` (shape `(batch, time, 128)`) and `attention_mask` with `input_features_mask` (shape `(batch, time)`) to match `Gemma4AudioFeatureExtractor`'s outputs. Verify exact argument names against the Gemma4 audio encoder's `forward` in Step 0.
- Replace `outputs = self.wav2vec2(input_values, ...)` with:

```python
hidden_states, output_mask = self.gemma4_audio_encoder(
    input_features, input_features_mask
)
```

Verify the exact return signature (positional tuple vs dataclass) in Step 0 and adjust accordingly.

- `hidden_states` will be `(batch, time/4, 1536)` — no further shape manipulation needed.
- For the CTC loss: replace Wav2Vec2's `_get_feat_extract_output_lengths` length calculation with a direct derivation from `output_mask` (e.g. `input_lengths = output_mask.sum(dim=-1).long()`), since the encoder already accounts for the 4× temporal downsampling internally.
- Keep everything else unchanged: `dropout`, `lm_head`, the CTC loss block, and the `CausalLMOutput` return.

---

### Step 4 – Freeze helpers

Add two convenience methods matching the Wav2Vec2ForCTC API:

```python
def freeze_audio_encoder(self):
    """Freeze all Gemma4 conformer parameters."""
    for param in self.gemma4_audio_encoder.parameters():
        param.requires_grad = False

def freeze_audio_encoder_except_norm(self):
    """Freeze encoder but leave final layer norm trainable."""
    for name, param in self.gemma4_audio_encoder.named_parameters():
        if "norm_out" not in name:  # verify attribute name in Step 0
            param.requires_grad = False
```

---

### Step 5 – Sanity check script

Write a standalone script `test_gemma4_ctc.py` (not a pytest file) that can be run directly. It should:

1. Load config and model using the `trust_remote_code` path:

```python
from transformers import AutoConfig, AutoModelForCTC

config = AutoConfig.from_pretrained("./gemma4-ctc", trust_remote_code=True)
model = AutoModelForCTC.from_pretrained("./gemma4-ctc", trust_remote_code=True)
```

2. Assert `type(model.gemma4_audio_encoder.output_proj).__name__ == "Identity"`.
3. Create dummy `input_features` of shape `(2, 400, 128)` and `input_features_mask` of shape `(2, 400)`.
4. Run a forward pass and assert `logits.shape == (2, 100, config.vocab_size)`.
5. Assert `model.lm_head.in_features == 1536`.
6. Print a summary of trainable vs frozen parameter counts.

---

### Important constraints

- **All imports from `transformers` use the public API** — no relative imports into the transformers source tree.
- **Do not use `AutoModel.from_pretrained`** to load the audio tower — it will silently produce random weights. Always go via `AutoModelForMultimodalLM` → `.model.audio_tower`.
- **`transformers >= 5.5.0`** is required for Gemma4; add a version check at the top of `modeling_gemma4_ctc.py`:

```python
from transformers import __version__ as _transformers_version
assert tuple(int(x) for x in _transformers_version.split(".")[:2]) >= (5, 5), \
    "transformers >= 5.5.0 required for Gemma4 support"
```

- The `output_proj` attribute name and the audio encoder's `forward` return signature **must be verified against the actual source in Step 0** before hardcoding anything.