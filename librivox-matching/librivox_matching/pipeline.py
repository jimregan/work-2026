"""End-to-end pipeline and CLI entry point."""

import argparse
import json
import logging
import os

from .vibevoice import parse_vibevoice, merge_chunks
from .chunk_matcher import match_all_chunks
from .ctc_align import load_model, extract_audio_segment, align_with_transcript, get_word_timestamps

logger = logging.getLogger(__name__)


def run_pipeline(
    vibevoice_json_path: str,
    etext_path: str,
    audio_path: str,
    model_name: str = None,
) -> dict:
    """Run the full alignment pipeline.

    Args:
        vibevoice_json_path: path to VibeVoice JSON file
        etext_path: path to plain text etext file
        audio_path: path to audio file
        model_name: optional wav2vec2 model name

    Returns:
        dict with keys: source_etext, audio_file, segments, mismatches
    """
    # Parse and merge VibeVoice chunks
    logger.info("Parsing VibeVoice JSON: %s", vibevoice_json_path)
    chunks = parse_vibevoice(vibevoice_json_path)
    logger.info("Parsed %d chunks", len(chunks))

    merged = merge_chunks(chunks)
    logger.info("Merged into %d chunks", len(merged))

    # Read etext
    with open(etext_path) as f:
        etext = f.read()

    # Stage 1: chunk-to-etext matching
    logger.info("Stage 1: matching chunks to etext")
    segments, mismatches = match_all_chunks(merged, etext)
    logger.info(
        "Matched %d segments, %d mismatches",
        len(segments),
        len(mismatches),
    )

    # Stage 2: CTC word-level alignment
    logger.info("Stage 2: loading model for CTC alignment")
    model, processor = load_model(model_name)

    etext_words = etext.split()

    for seg in segments:
        logger.info(
            "Aligning segment %.2f-%.2f (%s)",
            seg["start"],
            seg["end"],
            seg["match_method"],
        )
        audio_array = extract_audio_segment(audio_path, seg["start"], seg["end"])

        if seg["boilerplate"] or seg["match_method"] == "none":
            # No etext match — use model's own decoding
            word_timestamps = get_word_timestamps(audio_array, model, processor)
        else:
            # Use etext words as ground truth
            gt_words = seg["etext"].split()
            if gt_words:
                word_timestamps = align_with_transcript(
                    audio_array, gt_words, model, processor
                )
            else:
                word_timestamps = get_word_timestamps(
                    audio_array, model, processor
                )

        # Offset timestamps to absolute time
        for wt in word_timestamps:
            wt["start"] = round(wt["start"] + seg["start"], 3)
            wt["end"] = round(wt["end"] + seg["start"], 3)

        seg["words"] = word_timestamps

    # Build output
    output_segments = []
    for seg in segments:
        output_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "etext": seg["etext"],
            "vibevoice": seg["vibevoice"],
            "words": seg.get("words", []),
            "boilerplate": seg["boilerplate"],
        })

    return {
        "source_etext": os.path.basename(etext_path),
        "audio_file": os.path.basename(audio_path),
        "segments": output_segments,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Align LibriVox audio to etext using VibeVoice transcripts"
    )
    parser.add_argument(
        "--vibevoice", required=True, help="Path to VibeVoice JSON file"
    )
    parser.add_argument(
        "--etext", required=True, help="Path to plain text etext file"
    )
    parser.add_argument(
        "--audio", required=True, help="Path to audio file (mp3, wav, etc.)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write output JSON"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="wav2vec2 model name (default: jonatasgrosman/wav2vec2-large-xlsr-53-english)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run_pipeline(
        vibevoice_json_path=args.vibevoice,
        etext_path=args.etext,
        audio_path=args.audio,
        model_name=args.model,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Output written to %s", args.output)


if __name__ == "__main__":
    main()
