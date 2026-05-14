"""CLI entry point for dysfluent-wfst phonetic variation decoding."""

from __future__ import annotations

import argparse
import json
import sys

import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phonetic variation detection via WFST decoding"
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID for the wav2vec2 CTC model",
    )
    parser.add_argument(
        "--lexicon",
        required=True,
        help="Path to TSV lexicon file (word<TAB>p1 p2 p3)",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Optional path to YAML phonetic rules file (MFA format)",
    )
    parser.add_argument(
        "--sim-matrix",
        default=None,
        help="Optional path to phoneme similarity matrix (.npy)",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file, or JSONL manifest for batch mode",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Reference text for single-file mode",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for alignment JSON (default: stdout)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="Variation tolerance (default: 5.0)",
    )
    parser.add_argument(
        "--beam",
        type=float,
        default=25.0,
        help="Output beam width for k2 dense intersection (default: 25.0)",
    )
    parser.add_argument(
        "--back",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow back/repetition arcs (default: enabled)",
    )
    parser.add_argument(
        "--skip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow skip/deletion arcs (default: disabled)",
    )
    parser.add_argument(
        "--sub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow substitution arcs (default: enabled)",
    )
    return parser.parse_args(argv)


def load_audio(audio_path: str, model_id: str, device: str):
    """Load audio and run wav2vec2 inference.

    Returns (logits, length, processor, frame_shift_ms) where ``logits`` is a FloatTensor of
    shape ``(T, C)`` containing raw (unnormalized) scores.
    """
    import torchaudio
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    inputs = processor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits  # (1, T, C)

    length = logits.shape[1]
    num_samples = waveform.shape[-1]
    frame_shift_ms = (num_samples / 16000.0) * 1000.0 / length if length else 0.0
    return logits.squeeze(0), length, processor, frame_shift_ms


def get_phonemes_from_text(
    ref_text: str,
    lexicon_path: str,
) -> list[str]:
    """Look up reference text phonemes from the lexicon.

    Splits the reference text into words and looks up each word
    in the TSV lexicon. Returns concatenated phoneme list.
    """
    from dysfluent_wfst.lexicon import load_lexicon

    entries = load_lexicon(lexicon_path)
    word_to_prons: dict[str, list[str]] = {}
    for word, pron in entries:
        word_to_prons.setdefault(word.lower(), []).append(pron)

    phonemes: list[str] = []
    words = ref_text.lower().strip().split()
    for word in words:
        # Strip punctuation
        clean = word.strip(".,!?;:")
        if clean in word_to_prons:
            # Use first pronunciation
            pron = word_to_prons[clean][0]
            phonemes.extend(pron.split())
        else:
            print(f"Warning: word '{clean}' not found in lexicon", file=sys.stderr)
    return phonemes


def process_single(args: argparse.Namespace) -> dict:
    """Process a single audio file."""
    from dysfluent_wfst.alignment import save_alignment
    from dysfluent_wfst.decoder import Decoder

    # Load audio and run inference
    log_probs, length, processor, frame_shift_ms = load_audio(
        args.audio, args.model_id, args.device
    )

    # Get reference phonemes
    if args.ref_text is None:
        print("Error: --ref-text is required for single-file mode", file=sys.stderr)
        sys.exit(1)

    ref_phonemes = get_phonemes_from_text(args.ref_text, args.lexicon)
    if not ref_phonemes:
        print("Error: no phonemes found for reference text", file=sys.stderr)
        sys.exit(1)

    # Build decoder and run
    decoder = Decoder(
        model_id=args.model_id,
        lexicon_path=args.lexicon,
        rules_path=args.rules,
        sim_matrix_path=args.sim_matrix,
        device=args.device,
    )

    alignment = decoder.decode_utterance(
        log_probs=log_probs,
        length=length,
        ref_phonemes=ref_phonemes,
        audio_path=args.audio,
        ref_text=args.ref_text,
        beta=args.beta,
        back=args.back,
        skip=args.skip,
        sub=args.sub,
        output_beam=args.beam,
        frame_shift_ms=frame_shift_ms,
    )

    if args.output:
        save_alignment(alignment, args.output)
    else:
        from dataclasses import asdict
        print(json.dumps(asdict(alignment), indent=2, ensure_ascii=False))

    return asdict(alignment)


def process_batch(args: argparse.Namespace) -> list[dict]:
    """Process a JSONL manifest of audio files."""
    from dataclasses import asdict

    from dysfluent_wfst.alignment import save_alignment
    from dysfluent_wfst.decoder import Decoder

    decoder = Decoder(
        model_id=args.model_id,
        lexicon_path=args.lexicon,
        rules_path=args.rules,
        sim_matrix_path=args.sim_matrix,
        device=args.device,
    )

    results = []
    with open(args.audio, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: skipping line {line_num}: {e}",
                    file=sys.stderr,
                )
                continue

            audio_path = item["audio_path"]
            ref_text = item.get("ref_text", "")
            utterance_id = item.get("id", f"utt_{line_num}")

            log_probs, length, _, frame_shift_ms = load_audio(
                audio_path, args.model_id, args.device
            )
            ref_phonemes = get_phonemes_from_text(ref_text, args.lexicon)
            if not ref_phonemes:
                print(
                    f"Warning: no phonemes for '{ref_text}', skipping",
                    file=sys.stderr,
                )
                continue

            alignment = decoder.decode_utterance(
                log_probs=log_probs,
                length=length,
                ref_phonemes=ref_phonemes,
                utterance_id=utterance_id,
                audio_path=audio_path,
                ref_text=ref_text,
                beta=args.beta,
                back=args.back,
                skip=args.skip,
                sub=args.sub,
                output_beam=args.beam,
                frame_shift_ms=frame_shift_ms,
            )
            results.append(asdict(alignment))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))

    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Determine mode: single file or batch manifest
    if args.audio.endswith(".jsonl"):
        process_batch(args)
    else:
        process_single(args)


if __name__ == "__main__":
    main()
