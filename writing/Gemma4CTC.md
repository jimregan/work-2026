
## Gemma4ForCTC — implementation notes

### What was built

A standalone HuggingFace `trust_remote_code` repo in `gemma4-ctc/` for
CTC-based ASR using the Gemma 4 audio encoder as the backbone.

```
gemma4-ctc/
    config.json                  ← vocab_size=58, hidden_size=1024
    configuration_gemma4_ctc.py  ← Gemma4CTCConfig
    modeling_gemma4_ctc.py       ← Gemma4CTCPreTrainedModel, Gemma4ForCTC
    vocab.json                   ← Swedish phoneme vocab (58 tokens)
    tokenizer_config.json        ← Wav2Vec2CTCTokenizer, | as word delimiter
    test_gemma4_ctc.py           ← sanity-check script
```

---

### Corrections to the original spec

The original instructions contained several factual errors about the Gemma 4
audio encoder, discovered by reading the transformers source before writing
any code (transformers commit `b9f0fbf`).

#### 1. Conformer hidden size is 1024, not 1536

`Gemma4AudioConfig` has:
- `hidden_size: int = 1024` — the conformer's internal dimension
- `output_proj_dims: int = 1536` — the target dimension of `output_proj`

The conformer stack produces `(batch, time/4, 1024)`.  `output_proj` maps
`1024 → 1536` (not `1536 → 2560` as the spec stated).  The LLM text
`hidden_size` is `2304`, not `2560`.

**Impact:** `hidden_size` in `Gemma4CTCConfig` defaults to `1024`.  After
replacing `output_proj` with `nn.Identity()`, the CTC head is
`nn.Linear(1024, vocab_size)`.

#### 2. `Gemma4AudioModel.forward` argument name

The encoder's forward signature is:
```python
def forward(self, input_features, attention_mask=None, ...)
```
The mask parameter is `attention_mask`, not `input_features_mask`.  The
model is called as:
```python
outputs = self.gemma4_audio_encoder(
    input_features, attention_mask=attention_mask, return_dict=True
)
```

#### 3. Forward return is a dataclass, not a tuple

`Gemma4AudioModel.forward` returns `Gemma4AudioModelOutput` (a
`BaseModelOutputWithPooling` subclass).  The mask is at
`outputs.attention_mask`, not a second positional element.  Calling with
`return_dict=True` and accessing named fields is the safe approach.

#### 4. `norm_out` is per-layer, not a single final norm

`norm_out` is an attribute of each `Gemma4AudioLayer` (all 12 layers), not
a top-level encoder norm.  `freeze_audio_encoder_except_norm` checks
`"norm_out" not in name`, which leaves all twelve layer norms trainable —
more than the docstring's "final layer norm" implies, but likely the right
behaviour.

---

### Key implementation details

**Audio tower extraction**

`AutoModelForMultimodalLM` does include `gemma4` (via the
`IMAGE_TEXT_TO_TEXT` mapping), so the load path is:
```python
_full_model = AutoModelForMultimodalLM.from_pretrained(
    config.gemma4_audio_model_id,
    torch_dtype=torch.bfloat16,
    device_map=None,
    low_cpu_mem_usage=True,
)
self.gemma4_audio_encoder = _full_model.model.audio_tower
del _full_model
self.gemma4_audio_encoder.output_proj = nn.Identity()
```

This only runs on cold initialisation.  After the first `save_pretrained`,
subsequent `from_pretrained` loads skip it and reload the saved encoder
weights directly.

**CTC input lengths**

The encoder's subsampler applies two stride-2 convolutions, so the output
is `time/4` frames.  The output mask is returned as
`outputs.attention_mask` (bool, True = valid frame).  Input lengths for
`ctc_loss` are derived as:
```python
input_lengths = output_mask.sum(dim=-1).long()
```

**Tokenizer**

Swedish phoneme vocabulary, 58 tokens.  `<pad>` (id 0) is the CTC blank.
`|` (id 57) is the word delimiter.  The empty-string token at id 4 is the
word-internal silence/connector — verify its role against the original
fairseq `dict.ltr.txt`.

---

### Still needed

- A `DataCollatorCTCWithPadding` (padding `input_features` + `-100` label
  padding)
- An `accelerate`-based training script
- Feature extractor config (reference `Gemma4AudioFeatureExtractor` from
  the upstream model or copy `preprocessor_config.json`)
