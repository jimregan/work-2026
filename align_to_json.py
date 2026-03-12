#!/usr/bin/env python3
"""
align_to_json.py

Align a WhisperX or HuggingFace ASR JSON hypothesis with a sentence-split
reference and output one JSON object per sentence (JSONL).

Each output line:
  {
    "id": "<file_id>_<sentence_num>",
    "reference": "<original reference text>",
    "asr": "<ASR words assigned to this sentence>",
    "start": <float>,
    "end": <float>,
    "speaker": "<id>",   // only if --speaker is given
    "gender": "<m/f>"    // only if --gender is given
  }

A second ASR output (--hyp2) can be provided as a tie-breaker: for each
sentence the alignment with more correct words wins. Timing always comes
from the primary hypothesis (--hyp), which must have word-level timestamps.

Usage:
    align_to_json.py --hyp hyp.json --ref r2 --ref-format tsv-sentences
    align_to_json.py --hyp hyp.json --ref r1 --ref-format numbered --speaker S001 --gender f
    align_to_json.py --hyp hyp.json --hyp2 hyp.vv.json --ref r2 --ref-format tsv-sentences
    align_to_json.py --hyp-dir json_dir/ --ref r2 --ref-format tsv-sentences --output out.jsonl
"""

from __future__ import print_function
import argparse
import json
import logging
import sys
from pathlib import Path

from align_whisper_ref import (
    smith_waterman_alignment,
    load_hyp,
    read_ref_tsv_sentences,
    read_ref_numbered,
    build_flat_ref,
    _make_align_arrays,
)

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(pathname)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s"
))
_handler.setLevel(logging.WARNING)
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)


def collect_sentence_data(alignment_output, ctm_array, sentence_indices,
                           sentence_list, eps_symbol="<eps>"):
    """Collect per-sentence ASR words and timing from the primary hypothesis.

    Returns list of dicts with keys: ref_words, asr_words, start, end.
    Timing comes from ctm_array (word-level timestamps required).
    """
    n = len(sentence_list)
    sentences = [{"ref_words": sentence_list[i], "asr_words": [], "start": None, "end": None}
                 for i in range(n)]
    current_sentence = 0

    for (_rw, _hw, ref_prev_i, hyp_prev_i, ref_i, hyp_i) in alignment_output:
        if ref_i > ref_prev_i:
            current_sentence = sentence_indices[ref_prev_i]
        if hyp_i > hyp_prev_i:
            ctm_pos = hyp_prev_i
            if ctm_pos >= len(ctm_array):
                continue
            start, duration, word, _conf = ctm_array[ctm_pos]
            if word == eps_symbol:
                continue
            sent = sentences[current_sentence]
            sent["asr_words"].append(word)
            if sent["start"] is None:
                sent["start"] = start
            sent["end"] = start + duration

    return sentences


def score_and_collect_words_by_sentence(alignment_output, word_list,
                                         sentence_indices, n_sentences,
                                         eps_symbol="<eps>"):
    """Collect words and count correct alignments per sentence.

    alignment_output was computed on normalized arrays, so ref_word==hyp_word
    means a normalized match.

    Returns list of (words: list[str], score: float) indexed by sentence.
    """
    per_words = [[] for _ in range(n_sentences)]
    cor = [0] * n_sentences
    total = [0] * n_sentences
    current_sentence = 0

    for (ref_word, hyp_word, ref_prev_i, hyp_prev_i, ref_i, hyp_i) in alignment_output:
        if ref_i > ref_prev_i:
            current_sentence = sentence_indices[ref_prev_i]
        total[current_sentence] += 1
        if hyp_i > hyp_prev_i:
            word = word_list[hyp_prev_i]
            if word != eps_symbol:
                per_words[current_sentence].append(word)
            if ref_word == hyp_word and ref_word != eps_symbol:
                cor[current_sentence] += 1

    results = []
    for i in range(n_sentences):
        score = cor[i] / total[i] if total[i] > 0 else 0.0
        results.append((per_words[i], score))
    return results


def format_sentence(file_id, sent_idx, sent_data, speaker=None, gender=None):
    obj = {
        "id": f"{file_id}_{sent_idx + 1}",
        "reference": " ".join(sent_data["ref_words"]),
        "asr": " ".join(sent_data["asr_words"]),
        "start": sent_data["start"],
        "end": sent_data["end"],
    }
    if speaker is not None:
        obj["speaker"] = speaker
    if gender is not None:
        obj["gender"] = gender
    return json.dumps(obj, ensure_ascii=False)


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    hyp_group = parser.add_mutually_exclusive_group(required=True)
    hyp_group.add_argument("--hyp", metavar="FILE",
                            help="Primary hypothesis JSON file (must have word-level timestamps)")
    hyp_group.add_argument("--hyp-dir", metavar="DIR",
                            help="Directory of primary hypothesis JSON files")

    parser.add_argument("--hyp-format", choices=["whisperx", "hfjson", "vv", "auto"],
                        default="auto",
                        help="Primary hypothesis format (default: auto-detect)")
    parser.add_argument("--hyp2", metavar="FILE", default=None,
                        help="Secondary hypothesis JSON file for tie-breaking (used with --hyp)")
    parser.add_argument("--hyp2-dir", metavar="DIR", default=None,
                        help="Directory of secondary hypothesis JSON files (used with --hyp-dir)")
    parser.add_argument("--hyp2-format", choices=["whisperx", "hfjson", "vv", "auto"],
                        default="auto",
                        help="Secondary hypothesis format (default: auto-detect)")
    parser.add_argument("--ref", required=True, type=argparse.FileType("r"),
                        help="Reference file")
    parser.add_argument("--ref-format",
                        choices=["tsv-sentences", "numbered"],
                        default="tsv-sentences",
                        help=(
                            "tsv-sentences: <ID>\\t<text> with same ID repeated per sentence; "
                            "numbered: <num>\\t<text> for a single file "
                            "(default: tsv-sentences)"))
    parser.add_argument("--output", default="-",
                        help="Output JSONL file (default: stdout)")
    parser.add_argument("--speaker", default=None,
                        help="Speaker ID to include in every output record")
    parser.add_argument("--gender", default=None,
                        help="Speaker gender to include in every output record")
    parser.add_argument("--eps-symbol", default="<eps>")
    parser.add_argument("--correct-score", type=int, default=1)
    parser.add_argument("--substitution-penalty", type=int, default=1)
    parser.add_argument("--deletion-penalty", type=int, default=1)
    parser.add_argument("--insertion-penalty", type=int, default=1)
    parser.add_argument("--no-align-full-hyp", dest="align_full_hyp",
                        action="store_false", default=True)
    parser.add_argument("--no-normalize", dest="normalize",
                        action="store_false", default=True)
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    return parser.parse_args()


def _align(align_ref, align_hyp, similarity_score, del_score, ins_score,
           eps_symbol, align_full_hyp):
    return smith_waterman_alignment(
        align_ref, align_hyp,
        similarity_score_function=similarity_score,
        del_score=del_score, ins_score=ins_score,
        eps_symbol=eps_symbol,
        align_full_hyp=align_full_hyp)


def run(args):
    if args.verbose > 0:
        _handler.setLevel(logging.DEBUG)

    correct_score = args.correct_score
    sub_penalty = args.substitution_penalty
    del_score = -args.deletion_penalty
    ins_score = -args.insertion_penalty

    def similarity_score(x, y):
        return correct_score if x == y else -sub_penalty

    if args.ref_format == "numbered":
        sentence_list_for = {None: read_ref_numbered(args.ref)}
    else:
        sentence_list_for = read_ref_tsv_sentences(args.ref)
    args.ref.close()

    if args.hyp:
        hyp_files = [args.hyp]
    else:
        hyp_files = sorted(Path(args.hyp_dir).glob("*.json"))
        if not hyp_files:
            raise RuntimeError(f"No JSON files found in {args.hyp_dir}")

    out = open(args.output, "w") if args.output != "-" else sys.stdout

    num_done = num_err = 0
    try:
        for hyp_path in hyp_files:
            file_id, ctm_array = load_hyp(hyp_path, args.hyp_format)

            if args.ref_format == "numbered":
                sentence_list = sentence_list_for[None]
            else:
                if file_id not in sentence_list_for:
                    logger.warning("ID '%s' not found in reference; skipping", file_id)
                    num_err += 1
                    continue
                sentence_list = sentence_list_for[file_id]

            if not ctm_array:
                logger.warning("No words in hypothesis for '%s'; skipping", file_id)
                num_err += 1
                continue

            original_ref, sentence_indices = build_flat_ref(sentence_list)
            align_ref = _make_align_arrays(original_ref, args.normalize)

            hyp1_words = [row[2] for row in ctm_array]
            align_hyp1 = _make_align_arrays(hyp1_words, args.normalize)

            logger.info("Aligning %s: %d hyp words, %d ref words across %d sentences",
                        file_id, len(hyp1_words), len(original_ref), len(sentence_list))

            alignment1, _ = _align(align_ref, align_hyp1, similarity_score,
                                    del_score, ins_score, args.eps_symbol,
                                    args.align_full_hyp)

            sentences = collect_sentence_data(
                alignment1, ctm_array, sentence_indices, sentence_list,
                eps_symbol=args.eps_symbol)

            if args.hyp2:
                hyp2_path = args.hyp2
            elif args.hyp2_dir:
                hyp2_path = Path(args.hyp2_dir) / Path(hyp_path).name
                if not hyp2_path.exists():
                    logger.warning("No hyp2 file for '%s' in %s; skipping tie-break",
                                   file_id, args.hyp2_dir)
                    hyp2_path = None
            else:
                hyp2_path = None

            if hyp2_path:
                _, ctm_array2 = load_hyp(hyp2_path, args.hyp2_format)
                hyp2_words = [row[2] for row in ctm_array2]
                align_hyp2 = _make_align_arrays(hyp2_words, args.normalize)

                alignment2, _ = _align(align_ref, align_hyp2, similarity_score,
                                        del_score, ins_score, args.eps_symbol,
                                        args.align_full_hyp)

                scored1 = score_and_collect_words_by_sentence(
                    alignment1, hyp1_words, sentence_indices,
                    len(sentence_list), args.eps_symbol)
                scored2 = score_and_collect_words_by_sentence(
                    alignment2, hyp2_words, sentence_indices,
                    len(sentence_list), args.eps_symbol)

                for i, sent in enumerate(sentences):
                    words2, score2 = scored2[i]
                    _words1, score1 = scored1[i]
                    if score2 > score1:
                        logger.debug("Sentence %d: hyp2 wins (%.2f > %.2f)", i + 1, score2, score1)
                        sent["asr_words"] = words2

            for i, sent_data in enumerate(sentences):
                print(format_sentence(file_id, i, sent_data,
                                       speaker=args.speaker, gender=args.gender),
                      file=out)

            num_done += 1
    finally:
        if args.output != "-":
            out.close()

    logger.info("Done: %d succeeded, %d failed/skipped", num_done, num_err)
    if num_done == 0:
        raise RuntimeError("Processed 0 files successfully.")


def main():
    args = get_args()
    try:
        run(args)
    except Exception:
        logger.error("Fatal error", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
