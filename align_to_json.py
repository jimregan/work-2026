#!/usr/bin/env python3
"""
align_to_json.py

Align a WhisperX or HuggingFace ASR JSON hypothesis with a sentence-split
reference and output one JSON object per sentence (JSONL).

Each output line:
  {
    "id": "<file_id>_<line_num>",
    "reference": "<original reference text>",
    "asr": "<ASR words assigned to this sentence>",
    "start": <float>,
    "end": <float>,
    "speaker": "<id>",              // only if --speaker is given
    "gender": "<m/f>",              // only if --gender is given
    "secondary_validation": "<name>" // when sentence is validated (see below)
  }

A second ASR output (--hyp2) acts as a tie-breaker: for each sentence the
alignment with more correct words wins. Timing always comes from the primary
hypothesis (--hyp), which must have word-level timestamps.

Validation: the "secondary_validation" field (value = --secondary-validator name)
is added when:
  - the primary ASR perfectly matches the reference (normalized), OR
  - the secondary ASR scores at or above --secondary-threshold

A normalizations file (--normalizations) records known equivalences in TSV
format: ref<TAB>hyp<TAB>type (e.g. "LibriVox.org<TAB>librivox dot org<TAB>URL").
Multi-word hyp sequences are collapsed before alignment. New normalizations
detected during processing are appended to the file.

Usage:
    align_to_json.py --hyp hyp.json --ref r2 --ref-format tsv-sentences
    align_to_json.py --hyp hyp.json --ref r1 --ref-format numbered \\
        --speaker S001 --gender f
    align_to_json.py --hyp hyp.json --hyp2 hyp.vv.json --ref r2 \\
        --ref-format tsv-sentences --secondary-validator "VibeVoice ASR" \\
        --normalizations norms.tsv
"""

from __future__ import print_function
import argparse
import json
import logging
import os
import string
import sys
from pathlib import Path

from align_whisper_ref import (
    smith_waterman_alignment,
    load_hyp,
    read_ref_tsv_sentences,
    read_ref_numbered,
    build_flat_ref,
    normalize_word,
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


def load_normalizations(path):
    """Load a normalizations TSV (ref<TAB>hyp<TAB>type).

    Returns a dict mapping tuple(normalize_word(w) for w in hyp.split())
    to (norm_ref_word, orig_ref, orig_hyp, type).
    """
    norm_map = {}
    if not os.path.exists(path):
        return norm_map
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            orig_ref = parts[0].strip()
            orig_hyp = parts[1].strip()
            ntype = parts[2].strip() if len(parts) >= 3 else "normalization"
            norm_ref = normalize_word(orig_ref)
            norm_hyp_tuple = tuple(normalize_word(w) for w in orig_hyp.split())
            if norm_hyp_tuple:
                norm_map[norm_hyp_tuple] = (norm_ref, orig_ref, orig_hyp, ntype)
    return norm_map


def apply_normalizations(align_hyp, hyp_words, ctm_array, norm_map):
    """Merge known multi-word hyp sequences into single tokens before alignment.

    Single-word entries that normalize_word already handles are no-ops.
    Multi-word entries (e.g. "librivox dot org" → "librivox.org") are collapsed:
    the merged ctm entry spans the original words, the merged hyp word is the
    original words joined by space (preserving ASR output in the asr field).

    Returns (new_align_hyp, new_hyp_words, new_ctm_array, fired_positions) where
    fired_positions maps new array index → (orig_ref, orig_hyp, ntype) for each
    multi-word merge that was applied.
    """
    if not norm_map:
        return align_hyp, hyp_words, ctm_array, {}

    multi = {k: v for k, v in norm_map.items() if len(k) > 1}
    if not multi:
        return align_hyp, hyp_words, ctm_array, {}

    max_len = max(len(k) for k in multi)
    new_align, new_words, new_ctm = [], [], []
    fired_positions = {}
    i, n = 0, len(align_hyp)

    while i < n:
        matched = False
        for length in range(min(max_len, n - i), 1, -1):
            pattern = tuple(align_hyp[i:i + length])
            if pattern in multi:
                norm_ref, orig_ref, orig_hyp, ntype = multi[pattern]
                merged_word = " ".join(hyp_words[i:i + length])
                start = ctm_array[i][0]
                end = ctm_array[i + length - 1][0] + ctm_array[i + length - 1][1]
                conf = sum(ctm_array[j][3] for j in range(i, i + length)) / length
                fired_positions[len(new_align)] = (orig_ref, orig_hyp, ntype)
                new_align.append(norm_ref)
                new_words.append(merged_word)
                new_ctm.append([start, end - start, merged_word, conf])
                i += length
                matched = True
                break
        if not matched:
            new_align.append(align_hyp[i])
            new_words.append(hyp_words[i])
            new_ctm.append(ctm_array[i])
            i += 1

    return new_align, new_words, new_ctm, fired_positions


def detect_normalization_type(ref_word, hyp_word):
    """Auto-classify what kind of normalization made ref_word and hyp_word match."""
    rl, hl = ref_word.lower(), hyp_word.lower()
    if rl == hl:
        return "case"
    rl2 = rl.replace("-", "").replace("'", "").replace("\u2019", "")
    hl2 = hl.replace("-", "").replace("'", "").replace("\u2019", "")
    if rl2 == hl2:
        return "hyphen-apostrophe"
    if rl2.strip(string.punctuation) == hl2.strip(string.punctuation):
        return "punctuation"
    return "normalization"


def collect_sentence_data(alignment_output, ctm_array, original_ref,
                           sentence_indices, sentence_list,
                           norm_map=None, fired_positions=None,
                           eps_symbol="<eps>"):
    """Collect per-sentence ASR words, timing, scores, and normalizations.

    Returns (sentences, new_norms) where:
      sentences: list of dicts with ref_words, asr_words, start, end, score,
                 normalizations (list of {ref, hyp, type} from known norm_map)
      new_norms: dict of {(orig_ref, orig_hyp): type} for newly detected pairs
    """
    n = len(sentence_list)
    sentences = [{"ref_words": sentence_list[i], "asr_words": [],
                  "start": None, "end": None, "cor": 0, "total": 0,
                  "normalizations": []}
                 for i in range(n)]
    new_norms = {}
    if fired_positions is None:
        fired_positions = {}
    known_lookup = {(v[1], v[2]): v[3] for v in norm_map.values()} if norm_map else {}
    current_sentence = 0

    for (ref_word, hyp_word, ref_prev_i, hyp_prev_i, ref_i, hyp_i) in alignment_output:
        if ref_i > ref_prev_i:
            current_sentence = sentence_indices[ref_prev_i]

        sent = sentences[current_sentence]
        sent["total"] += 1

        is_cor = ref_word == hyp_word and ref_word != eps_symbol
        if is_cor:
            sent["cor"] += 1

        if hyp_i > hyp_prev_i:
            ctm_pos = hyp_prev_i
            if ctm_pos >= len(ctm_array):
                continue
            start, duration, word, _conf = ctm_array[ctm_pos]
            if word == eps_symbol:
                continue
            sent["asr_words"].append(word)
            if sent["start"] is None:
                sent["start"] = start
            sent["end"] = start + duration

            if is_cor:
                if ctm_pos in fired_positions:
                    orig_ref, orig_hyp, ntype = fired_positions[ctm_pos]
                    sent["normalizations"].append(
                        {"ref": orig_ref, "hyp": orig_hyp, "type": ntype})
                elif ref_i > ref_prev_i:
                    orig_ref = original_ref[ref_prev_i]
                    orig_hyp = word
                    if orig_ref != orig_hyp:
                        key = (orig_ref, orig_hyp)
                        if key in known_lookup:
                            sent["normalizations"].append(
                                {"ref": orig_ref, "hyp": orig_hyp,
                                 "type": known_lookup[key]})
                        else:
                            new_norms[key] = detect_normalization_type(orig_ref, orig_hyp)

    for sent in sentences:
        total = sent.pop("total")
        cor = sent.pop("cor")
        sent["score"] = cor / total if total > 0 else 0.0

    return sentences, new_norms


def score_and_collect_words_by_sentence(alignment_output, word_list,
                                         sentence_indices, n_sentences,
                                         eps_symbol="<eps>"):
    """Collect words and score per sentence from alignment output.

    alignment_output was computed on normalized arrays so ref==hyp means match.
    Returns list of (words: list[str], score: float).
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

    return [(per_words[i], cor[i] / total[i] if total[i] > 0 else 0.0)
            for i in range(n_sentences)]


def format_sentence(file_id, sent_num, sent_data,
                     speaker=None, gender=None, validator=None,
                     secondary_threshold=1.0, score2=None):
    obj = {
        "id": f"{file_id}_{sent_num}",
        "reference": " ".join(sent_data["ref_words"]),
        "asr": " ".join(sent_data["asr_words"]),
        "start": sent_data["start"],
        "end": sent_data["end"],
    }
    if speaker is not None:
        obj["speaker"] = speaker
    if gender is not None:
        obj["gender"] = gender
    if validator is not None:
        primary_perfect = sent_data["score"] == 1.0
        secondary_ok = score2 is not None and score2 >= secondary_threshold
        if primary_perfect or secondary_ok:
            obj["secondary_validation"] = validator
    return json.dumps(obj, ensure_ascii=False)


def write_normalizations(path, existing_map, new_norms):
    """Append newly detected normalizations to the TSV file."""
    existing_keys = {(v[1], v[2]) for v in existing_map.values()}
    to_write = [(ref, hyp, t) for (ref, hyp), t in new_norms.items()
                if (ref, hyp) not in existing_keys]
    if not to_write:
        return
    with open(path, "a", encoding="utf-8") as f:
        for ref, hyp, t in to_write:
            f.write(f"{ref}\t{hyp}\t{t}\n")
    logger.info("Appended %d new normalizations to %s", len(to_write), path)


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    hyp_group = parser.add_mutually_exclusive_group(required=True)
    hyp_group.add_argument("--hyp", metavar="FILE",
                            help="Primary hypothesis JSON (must have word-level timestamps)")
    hyp_group.add_argument("--hyp-dir", metavar="DIR",
                            help="Directory of primary hypothesis JSON files")

    parser.add_argument("--hyp-format", choices=["whisperx", "hfjson", "vv", "auto"],
                        default="auto")
    parser.add_argument("--hyp2", metavar="FILE", default=None,
                        help="Secondary hypothesis for tie-breaking (used with --hyp)")
    parser.add_argument("--hyp2-dir", metavar="DIR", default=None,
                        help="Directory of secondary hypothesis files (used with --hyp-dir)")
    parser.add_argument("--hyp2-format", choices=["whisperx", "hfjson", "vv", "auto"],
                        default="auto")
    parser.add_argument("--ref", required=True, type=argparse.FileType("r"))
    parser.add_argument("--ref-format", choices=["tsv-sentences", "numbered"],
                        default="tsv-sentences",
                        help=(
                            "tsv-sentences: <ID>\\t<text> repeated per sentence; "
                            "numbered: <num>\\t<text> for a single file"))
    parser.add_argument("--output", default="-",
                        help="Output JSONL file (default: stdout)")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--secondary-validator", default=None, metavar="NAME",
                        help="Validator name for the secondary_validation field "
                             "(e.g. 'VibeVoice ASR'). Field is added when primary "
                             "is perfect or secondary scores >= --secondary-threshold.")
    parser.add_argument("--secondary-threshold", type=float, default=1.0, metavar="FLOAT",
                        help="Min secondary score to count as validation (default: 1.0)")
    parser.add_argument("--normalizations", metavar="FILE", default=None,
                        help="TSV file of known normalizations (ref<TAB>hyp<TAB>type). "
                             "Read at start to improve alignment; new detections are appended.")
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


def _do_align(align_ref, align_hyp, correct_score, sub_penalty,
              del_score, ins_score, eps_symbol, align_full_hyp):
    def similarity_score(x, y):
        return correct_score if x == y else -sub_penalty
    return smith_waterman_alignment(
        align_ref, align_hyp,
        similarity_score_function=similarity_score,
        del_score=del_score, ins_score=ins_score,
        eps_symbol=eps_symbol,
        align_full_hyp=align_full_hyp)


def run(args):
    if args.verbose > 0:
        _handler.setLevel(logging.DEBUG)

    norm_map = load_normalizations(args.normalizations) if args.normalizations else {}
    all_new_norms = {}

    if args.ref_format == "numbered":
        numbered = read_ref_numbered(args.ref)
        sentence_nums_for = {None: [num for num, _ in numbered]}
        sentence_list_for = {None: [words for _, words in numbered]}
    else:
        raw = read_ref_tsv_sentences(args.ref)
        sentence_list_for = dict(raw)
        sentence_nums_for = {k: [str(i + 1) for i in range(len(v))]
                             for k, v in sentence_list_for.items()}
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
                sentence_nums = sentence_nums_for[None]
            else:
                if file_id not in sentence_list_for:
                    logger.warning("ID '%s' not found in reference; skipping", file_id)
                    num_err += 1
                    continue
                sentence_list = sentence_list_for[file_id]
                sentence_nums = sentence_nums_for[file_id]

            if not ctm_array:
                logger.warning("No words in hypothesis for '%s'; skipping", file_id)
                num_err += 1
                continue

            original_ref, sentence_indices = build_flat_ref(sentence_list)
            hyp1_words = [row[2] for row in ctm_array]

            align_ref = _make_align_arrays(original_ref, args.normalize)
            align_hyp1 = _make_align_arrays(hyp1_words, args.normalize)
            align_hyp1, hyp1_words, ctm_array = apply_normalizations(
                align_hyp1, hyp1_words, ctm_array, norm_map)

            logger.info("Aligning %s: %d hyp words, %d ref words across %d sentences",
                        file_id, len(hyp1_words), len(original_ref), len(sentence_list))

            alignment1, _ = _do_align(align_ref, align_hyp1,
                                       args.correct_score, args.substitution_penalty,
                                       -args.deletion_penalty, -args.insertion_penalty,
                                       args.eps_symbol, args.align_full_hyp)

            sentences, new_norms = collect_sentence_data(
                alignment1, ctm_array, original_ref, sentence_indices, sentence_list,
                norm_map=norm_map, eps_symbol=args.eps_symbol)
            all_new_norms.update(new_norms)

            scores2 = [None] * len(sentence_list)

            if args.hyp2:
                hyp2_path = args.hyp2
            elif args.hyp2_dir:
                hyp2_path = Path(args.hyp2_dir) / Path(hyp_path).name
                if not Path(hyp2_path).exists():
                    logger.warning("No hyp2 for '%s' in %s", file_id, args.hyp2_dir)
                    hyp2_path = None
            else:
                hyp2_path = None

            if hyp2_path:
                _, ctm2 = load_hyp(hyp2_path, args.hyp2_format)
                hyp2_words = [row[2] for row in ctm2]
                align_hyp2 = _make_align_arrays(hyp2_words, args.normalize)
                align_hyp2, hyp2_words, ctm2 = apply_normalizations(
                    align_hyp2, hyp2_words, ctm2, norm_map)

                alignment2, _ = _do_align(align_ref, align_hyp2,
                                           args.correct_score, args.substitution_penalty,
                                           -args.deletion_penalty, -args.insertion_penalty,
                                           args.eps_symbol, args.align_full_hyp)

                scored1 = score_and_collect_words_by_sentence(
                    alignment1, hyp1_words, sentence_indices,
                    len(sentence_list), args.eps_symbol)
                scored2 = score_and_collect_words_by_sentence(
                    alignment2, hyp2_words, sentence_indices,
                    len(sentence_list), args.eps_symbol)

                for i, sent in enumerate(sentences):
                    words2, s2 = scored2[i]
                    _words1, s1 = scored1[i]
                    scores2[i] = s2
                    if s2 > s1:
                        logger.debug("Sentence %s: hyp2 wins (%.2f > %.2f)",
                                     sentence_nums[i], s2, s1)
                        sent["asr_words"] = words2

            for i, sent_data in enumerate(sentences):
                print(format_sentence(
                    file_id, sentence_nums[i], sent_data,
                    speaker=args.speaker, gender=args.gender,
                    validator=args.secondary_validator,
                    secondary_threshold=args.secondary_threshold,
                    score2=scores2[i]),
                    file=out)

            num_done += 1
    finally:
        if args.output != "-":
            out.close()

    if args.normalizations and all_new_norms:
        write_normalizations(args.normalizations, norm_map, all_new_norms)

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
