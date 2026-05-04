"""Run the decoder over a HuggingFace dataset in streaming mode.

Usage example::

    python -m dysfluent_wfst.hf_stream \\
        --dataset mozilla-foundation/common_voice_17_0 \\
        --subset en \\
        --split test \\
        --text-field sentence \\
        --model-id facebook/wav2vec2-base-960h \\
        --lexicon lexicon.tsv \\
        --output results.jsonl

The ``--lexicon`` argument is a TSV file (word TAB p1 p2 p3).
Alternatively, if your dataset already has a pronunciation column, pass
``--pron-field`` with the column name and omit ``--lexicon``.  In that
case each row must contain both a text field and a pronunciation field
whose value is a space-separated phoneme string.

Field mapping
-------------
HuggingFace speech datasets typically have an ``audio`` column that is
a dict with keys ``array`` (numpy float32, 16 kHz) and ``sampling_rate``.
If yours uses a different layout, subclass ``HFStreamRunner`` and override
``get_audio_array``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch


def _resample(array: np.ndarray, src_rate: int, tgt_rate: int = 16000) -> np.ndarray:
    if src_rate == tgt_rate:
        return array
    import torchaudio
    t = torch.from_numpy(array).float().unsqueeze(0)
    t = torchaudio.functional.resample(t, src_rate, tgt_rate)
    return t.squeeze(0).numpy()


class HFStreamRunner:
    """Wraps a ``Decoder`` and iterates over a streaming HF dataset.

    Args:
        model_id: HuggingFace model ID for the wav2vec2 CTC model.
        lexicon_path: Path to TSV lexicon (word TAB phonemes). Mutually
            exclusive with ``lexicon_entries``.
        lexicon_entries: Pre-built list of (word, phoneme_string) tuples.
            Pass this when the lexicon comes from a HF dataset or another
            in-memory source.
        rules_path: Optional YAML phonetic rules file.
        sim_matrix_path: Optional ``.npy`` similarity matrix.
        device: Torch device string.
        beta, back, skip, sub, output_beam: Decoder parameters.
    """

    def __init__(
        self,
        model_id: str,
        lexicon_path: Optional[str] = None,
        lexicon_entries: Optional[list[tuple[str, str]]] = None,
        rules_path: Optional[str] = None,
        sim_matrix_path: Optional[str] = None,
        device: str = "cpu",
        beta: float = 5.0,
        back: bool = True,
        skip: bool = False,
        sub: bool = True,
        output_beam: float = 25.0,
    ):
        from .decoder import Decoder

        self.decoder = Decoder(
            model_id=model_id,
            lexicon_path=lexicon_path,
            lexicon_entries=lexicon_entries,
            rules_path=rules_path,
            sim_matrix_path=sim_matrix_path,
            device=device,
        )
        self._model_id = model_id
        self._device = device
        self.beta = beta
        self.back = back
        self.skip = skip
        self.sub = sub
        self.output_beam = output_beam

        # Lazy-load the processor (reuse if already in Decoder internals)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self._model_id)
        return self._processor

    def _get_model(self):
        if not hasattr(self, "_model"):
            from transformers import Wav2Vec2ForCTC
            self._model = Wav2Vec2ForCTC.from_pretrained(self._model_id)
            self._model = self._model.to(self._device)
            self._model.eval()
        return self._model

    def get_audio_array(self, row: dict) -> np.ndarray:
        """Extract a 16 kHz float32 numpy array from a dataset row.

        Override this if your dataset has a non-standard audio layout.
        The default handles the standard HF ``audio`` column format.
        """
        audio = row["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        return _resample(array, sr, 16000)

    def get_ref_text(self, row: dict, text_field: str) -> str:
        """Extract reference text from a dataset row."""
        return str(row[text_field])

    def get_ref_phonemes_from_pron_field(
        self, row: dict, pron_field: str
    ) -> list[str]:
        """Extract phonemes directly from a dataset pronunciation column."""
        return str(row[pron_field]).split()

    def _text_to_phonemes(self, text: str) -> list[str]:
        """Look up phonemes for a text string using the loaded lexicon."""
        if not hasattr(self, "_word_to_prons"):
            self._word_to_prons: dict[str, list[str]] = {}
            for word, pron in self.decoder.lexicon_entries:
                self._word_to_prons.setdefault(word.lower(), []).append(pron)

        phonemes: list[str] = []
        for word in text.lower().strip().split():
            clean = word.strip(".,!?;:")
            if clean in self._word_to_prons:
                phonemes.extend(self._word_to_prons[clean][0].split())
            else:
                print(f"Warning: '{clean}' not in lexicon", file=sys.stderr)
        return phonemes

    def infer(self, array: np.ndarray) -> tuple[torch.Tensor, int, float]:
        """Run wav2vec2 inference on a 16 kHz float32 array.

        Returns ``(log_probs, length, frame_shift_ms)`` where ``log_probs`` has shape ``(T, C)``.
        """
        processor = self._get_processor()
        model = self._get_model()
        inputs = processor(
            array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self._device)
        with torch.no_grad():
            logits = model(input_values).logits  # (1, T, C)
        length = logits.shape[1]
        frame_shift_ms = (len(array) / 16000.0) * 1000.0 / length if length else 0.0
        return logits.squeeze(0), length, frame_shift_ms

    def run(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "test",
        text_field: str = "sentence",
        pron_field: Optional[str] = None,
        utt_id_field: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """Iterate over a streaming HF dataset and yield alignment results.

        Args:
            dataset_name: HuggingFace dataset name (e.g. ``"mozilla-foundation/common_voice_17_0"``).
            subset: Dataset configuration/subset (e.g. ``"en"``).
            split: Dataset split (default ``"test"``).
            text_field: Column containing the reference transcript.
            pron_field: If set, read phonemes directly from this column
                instead of looking them up in the lexicon.
            utt_id_field: Column to use as utterance ID. Defaults to row index.
            max_samples: Stop after this many rows (None = no limit).

        Yields:
            ``dict`` representations of ``UtteranceAlignment`` objects.
        """
        from datasets import load_dataset

        ds = load_dataset(
            dataset_name,
            subset,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break

            utt_id = (
                str(row[utt_id_field])
                if utt_id_field and utt_id_field in row
                else f"utt_{i:06d}"
            )
            ref_text = self.get_ref_text(row, text_field)

            if pron_field:
                ref_phonemes = self.get_ref_phonemes_from_pron_field(row, pron_field)
            else:
                ref_phonemes = self._text_to_phonemes(ref_text)

            if not ref_phonemes:
                print(
                    f"Skipping {utt_id}: no phonemes for '{ref_text}'",
                    file=sys.stderr,
                )
                continue

            try:
                array = self.get_audio_array(row)
                logits, length, frame_shift_ms = self.infer(array)
                alignment = self.decoder.decode_utterance(
                    logits=logits,
                    length=length,
                    ref_phonemes=ref_phonemes,
                    utterance_id=utt_id,
                    ref_text=ref_text,
                    beta=self.beta,
                    back=self.back,
                    skip=self.skip,
                    sub=self.sub,
                    output_beam=self.output_beam,
                    frame_shift_ms=frame_shift_ms,
                )
                yield asdict(alignment)
            except Exception as exc:
                print(f"Error on {utt_id}: {exc}", file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stream a HF dataset through the WFST decoder")
    p.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    p.add_argument("--subset", default=None, help="Dataset configuration/subset")
    p.add_argument("--split", default="test")
    p.add_argument("--text-field", default="sentence", help="Column with reference text")
    p.add_argument("--pron-field", default=None, help="Column with pronunciation (space-separated phonemes); skips lexicon lookup")
    p.add_argument("--utt-id-field", default=None, help="Column to use as utterance ID")
    p.add_argument("--model-id", required=True)
    p.add_argument("--lexicon", default=None, help="TSV lexicon file (required unless --pron-field is set)")
    p.add_argument("--rules", default=None)
    p.add_argument("--sim-matrix", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--beam", type=float, default=25.0)
    p.add_argument("--back", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sub", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default=None, help="Output JSONL path (default: stdout)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.lexicon is None and args.pron_field is None:
        print("Error: provide --lexicon or --pron-field", file=sys.stderr)
        sys.exit(1)

    runner = HFStreamRunner(
        model_id=args.model_id,
        lexicon_path=args.lexicon,
        rules_path=args.rules,
        sim_matrix_path=args.sim_matrix,
        device=args.device,
        beta=args.beta,
        back=args.back,
        skip=args.skip,
        sub=args.sub,
        output_beam=args.beam,
    )

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for result in runner.run(
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            text_field=args.text_field,
            pron_field=args.pron_field,
            utt_id_field=args.utt_id_field,
            max_samples=args.max_samples,
        ):
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
