# gemma4-ctc

Standalone HuggingFace `trust_remote_code` repo for CTC-based ASR using the
Gemma 4 audio encoder (USM-style conformer) as a frozen backbone.

## Architecture

- **Encoder**: `Gemma4AudioModel` — 12-layer conformer, `hidden_size=1024`,
  4× temporal downsampling via two stride-2 conv layers.
- **`output_proj` suppressed**: the upstream encoder projects 1024→1536 for
  LLM use; we replace that with `nn.Identity()` to get raw conformer output.
- **CTC head**: `nn.Linear(1024, 58)` — Swedish phoneme vocab, `<pad>`=0 as
  the CTC blank token.

## Files

| File | Purpose |
|------|---------|
| `configuration_gemma4_ctc.py` | `Gemma4CTCConfig` |
| `modeling_gemma4_ctc.py` | `Gemma4ForCTC`, `Gemma4CTCPreTrainedModel` |
| `collator.py` | `DataCollatorCTCWithPadding` |
| `train.py` | `accelerate`-based training script |
| `test_gemma4_ctc.py` | Sanity-check script (run directly, not pytest) |
| `config.json` | Repo config (vocab_size=58, hidden_size=1024) |
| `vocab.json` | Swedish phoneme vocab, 58 tokens |
| `tokenizer_config.json` | `Wav2Vec2CTCTokenizer`, `\|` as word delimiter |
| `preprocessor_config.json` | `Gemma4AudioFeatureExtractor` defaults (16kHz, 128 mel, 20ms frame) |

## Key decisions

**`__init__` downloads the upstream model by default.** On first use it
fetches the Gemma 4 multimodal checkpoint via `AutoModelForMultimodalLM`
(accepts a Hub repo ID or a local path via `config.gemma4_audio_model_id`),
extracts `model.audio_tower`, and discards the rest.

**Saved checkpoints are self-contained.** `from_pretrained` is overridden:
if the target path is a local directory containing safetensors/pytorch_model
files, `_skip_encoder_download=True` is passed to `__init__` so the upstream
model is never fetched again. The encoder weights come from the checkpoint
instead.

**`input_features_mask` alias.** `Gemma4AudioFeatureExtractor` outputs the
mask under the key `input_features_mask`; `Gemma4AudioModel.forward` takes
`attention_mask`. The model's `forward` accepts both names so that
`model(**batch)` works directly without renaming keys in the collator.

**`output_proj` keys ignored on load.** `_keys_to_ignore_on_load_unexpected`
covers `gemma4_audio_encoder.output_proj.*` so checkpoints saved before the
Identity swap don't error.

## Corrected facts (vs. original spec)

- Conformer `hidden_size` is **1024**, not 1536. `output_proj_dims=1536` is
  the projection target (suppressed here). LLM text `hidden_size` is 2304.
- `Gemma4AudioModel.forward` mask arg is `attention_mask`, not
  `input_features_mask`.
- Forward returns `Gemma4AudioModelOutput` (dataclass); mask is at
  `.attention_mask`, not a positional tuple element.
- `norm_out` is per-layer (each of the 12 `Gemma4AudioLayer`s), not a single
  final norm.

## Training

First run — downloads Gemma 4 encoder, trains, saves self-contained checkpoint:
```
accelerate launch train.py --model_dir ./gemma4-ctc --output_dir ./run1 \
    --dataset_name <name> --audio_column audio --text_column phonemes
```

Subsequent runs — reloads from saved checkpoint, no upstream dependency:
```
accelerate launch train.py --model_dir ./run1 --output_dir ./run1 \
    --dataset_name <name> --audio_column audio --text_column phonemes
```

Encoder is **frozen by default**. Pass `--unfreeze_norms` to also train the
12 `norm_out` layer norms.

To use a local copy of the Gemma 4 checkpoint instead of downloading, set
`gemma4_audio_model_id` in `config.json` to the local path.

## Vocab note

The empty-string token at id 4 in `vocab.json` is the word-internal
silence/connector from the original fairseq training. Verify its role against
the original `dict.ltr.txt` before using the tokenizer for decoding.
