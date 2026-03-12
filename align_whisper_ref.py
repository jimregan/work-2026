#!/usr/bin/env python3
"""
align_whisper_ref.py

Align a WhisperX or HuggingFace ASR JSON hypothesis with a reference text,
outputting CTM-edit format using Smith-Waterman alignment.

Core alignment algorithm from Kaldi's align_ctm_ref.py
(Copyright 2016 Vimal Manohar, 2020 Dongji Gao, Apache 2.0).

Usage:
    # Single file, TSV reference:
    align_whisper_ref.py --hyp hyp.json --ref refs.tsv --output out.ctm-edit

    # Directory of JSON files, Kaldi-format reference:
    align_whisper_ref.py --hyp-dir json_dir/ --ref refs.txt --ref-format kaldi --output out.ctm-edit

    # Single file, sentence-split TSV (same ID repeated per sentence):
    align_whisper_ref.py --hyp hyp.json --ref refs.tsv --ref-format tsv-sentences --output out.ctm-edit

    # Single file, numbered sentence reference (r1 style):
    align_whisper_ref.py --hyp hyp.json --ref sents.txt --ref-format numbered --output out.ctm-edit
"""

from __future__ import print_function
import argparse
import json
import logging
import string
import sys
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(pathname)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s"
))
_handler.setLevel(logging.WARNING)
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)


def normalize_word(word):
    """Normalize for comparison: remove internal hyphens/apostrophes, strip
    remaining edge punctuation, lowercase.

    Handles old-spelling variants like To-night→tonight, Hallowe'en→halloween.
    Does NOT split on hyphens.
    """
    word = word.replace("-", "").replace("'", "").replace("\u2019", "")
    return word.strip(string.punctuation).lower()


def smith_waterman_alignment(ref, hyp, similarity_score_function,
                             del_score, ins_score,
                             eps_symbol="<eps>", align_full_hyp=True):
    """Smith-Waterman alignment of reference and hypothesis sequences.

    If align_full_hyp is True, traceback starts at the end of the hypothesis
    (aligns the full hypothesis to the best-matching sub-sequence of ref).
    If False, traceback starts at the max-score cell (sub-sequence of both).

    Returns (output, max_score) where output is a list of tuples:
        (ref_word, hyp_word, ref_from, hyp_from, ref_to, hyp_to)
    """
    output = []
    ref_len = len(ref)
    hyp_len = len(hyp)

    bp = [[] for _ in range(ref_len + 1)]
    H = [[] for _ in range(ref_len + 1)]

    for ref_index in range(ref_len + 1):
        if align_full_hyp:
            H[ref_index] = [-(hyp_len + 2)] * (hyp_len + 1)
            H[ref_index][0] = 0
        else:
            H[ref_index] = [0] * (hyp_len + 1)
        bp[ref_index] = [(0, 0)] * (hyp_len + 1)

        if align_full_hyp and ref_index == 0:
            for hyp_index in range(1, hyp_len + 1):
                H[0][hyp_index] = H[0][hyp_index - 1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index - 1)

    max_score = -float("inf")
    max_score_element = (0, 0)

    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            sub_or_ok = (H[ref_index - 1][hyp_index - 1]
                         + similarity_score_function(ref[ref_index - 1],
                                                     hyp[hyp_index - 1]))
            if ((not align_full_hyp and sub_or_ok > 0)
                    or (align_full_hyp and sub_or_ok >= H[ref_index][hyp_index])):
                H[ref_index][hyp_index] = sub_or_ok
                bp[ref_index][hyp_index] = (ref_index - 1, hyp_index - 1)

            if H[ref_index - 1][hyp_index] + del_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index - 1][hyp_index] + del_score
                bp[ref_index][hyp_index] = (ref_index - 1, hyp_index)

            if H[ref_index][hyp_index - 1] + ins_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index][hyp_index - 1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index - 1)

            if ((not align_full_hyp or hyp_index == hyp_len)
                    and H[ref_index][hyp_index] >= max_score):
                max_score = H[ref_index][hyp_index]
                max_score_element = (ref_index, hyp_index)

    ref_index, hyp_index = max_score_element
    score = max_score

    while ((not align_full_hyp and score >= 0)
           or (align_full_hyp and hyp_index > 0)):
        try:
            prev_ref_index, prev_hyp_index = bp[ref_index][hyp_index]
            if ((prev_ref_index, prev_hyp_index) == (ref_index, hyp_index)
                    or (prev_ref_index, prev_hyp_index) == (0, 0)):
                score = H[ref_index][hyp_index]
                if score != 0:
                    ref_word = ref[ref_index - 1] if ref_index > 0 else eps_symbol
                    hyp_word = hyp[hyp_index - 1] if hyp_index > 0 else eps_symbol
                    output.append((ref_word, hyp_word, prev_ref_index,
                                   prev_hyp_index, ref_index, hyp_index))
                    ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
                    score = H[ref_index][hyp_index]
                break

            if ref_index == prev_ref_index + 1 and hyp_index == prev_hyp_index + 1:
                output.append((
                    ref[ref_index - 1] if ref_index > 0 else eps_symbol,
                    hyp[hyp_index - 1] if hyp_index > 0 else eps_symbol,
                    prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            elif prev_hyp_index == hyp_index:
                assert prev_ref_index == ref_index - 1
                output.append((
                    ref[ref_index - 1] if ref_index > 0 else eps_symbol,
                    eps_symbol,
                    prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            elif prev_ref_index == ref_index:
                assert prev_hyp_index == hyp_index - 1
                output.append((
                    eps_symbol,
                    hyp[hyp_index - 1] if hyp_index > 0 else eps_symbol,
                    prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            else:
                raise RuntimeError("Unexpected backpointer state")

            ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
            score = H[ref_index][hyp_index]
        except Exception:
            logger.error("Unexpected entry (%d,%d) -> (%d,%d)",
                         prev_ref_index, prev_hyp_index, ref_index, hyp_index)
            raise RuntimeError("Unexpected result: bug in alignment code")

    assert align_full_hyp or score == 0
    output.reverse()
    return output, max_score


def get_edit_type(hyp_word, ref_word, duration=-1, eps_symbol="<eps>",
                  oov_word=None, symbol_table=None):
    if hyp_word == ref_word and hyp_word != eps_symbol:
        return "cor"
    if hyp_word != eps_symbol and ref_word == eps_symbol:
        return "ins"
    if hyp_word == eps_symbol and ref_word != eps_symbol and duration == 0.0:
        return "del"
    if (hyp_word == oov_word and symbol_table is not None
            and len(symbol_table) > 0 and ref_word not in symbol_table):
        return "cor"
    if hyp_word == eps_symbol and ref_word == eps_symbol and duration > 0.0:
        return "sil"
    assert hyp_word != eps_symbol and ref_word != eps_symbol
    return "sub"


def get_ctm_edits(alignment_output, ctm_array, original_ref,
                  eps_symbol="<eps>", oov_word=None, symbol_table=None,
                  normalize=False):
    """Map alignment output onto ctm_array timing using original word arrays.

    alignment_output may have been computed on normalized arrays; original_ref
    and ctm_array supply the original text for output.

    ctm_array: list of [start, duration, word, confidence]
    original_ref: list of original reference words (same length as normalized ref)
    Returns list of [start, duration, hyp_word, confidence, ref_word, edit_type]
    """
    ctm_edits = []
    ctm_len = len(ctm_array)
    current_time = ctm_array[0][0] if ctm_len > 0 else 0.0

    for (_rw, _hw, ref_prev_i, hyp_prev_i, ref_i, hyp_i) in alignment_output:
        orig_ref_word = original_ref[ref_prev_i] if ref_i > ref_prev_i else eps_symbol
        try:
            ctm_pos = hyp_prev_i
            assert ctm_pos < ctm_len

            if hyp_prev_i == hyp_i:
                edit_type = get_edit_type(
                    eps_symbol, orig_ref_word, duration=0.0,
                    eps_symbol=eps_symbol, oov_word=oov_word,
                    symbol_table=symbol_table)
                ctm_edits.append([current_time, 0.0, eps_symbol, 1.0,
                                   orig_ref_word, edit_type])
            else:
                assert hyp_i == hyp_prev_i + 1
                assert len(ctm_array[ctm_pos]) == 4
                ctm_line = list(ctm_array[ctm_pos])
                orig_hyp_word = ctm_line[2]

                if orig_hyp_word == eps_symbol and orig_ref_word != eps_symbol:
                    ctm_edits.append([current_time, 0.0, eps_symbol, 1.0,
                                       orig_ref_word, "del"])
                    ctm_line.extend([eps_symbol, "sil"])
                    ctm_edits.append(ctm_line)
                else:
                    cmp_hyp = normalize_word(orig_hyp_word) if normalize and orig_hyp_word != eps_symbol else orig_hyp_word
                    cmp_ref = normalize_word(orig_ref_word) if normalize and orig_ref_word != eps_symbol else orig_ref_word
                    edit_type = get_edit_type(
                        cmp_hyp, cmp_ref, duration=ctm_line[1],
                        eps_symbol=eps_symbol, oov_word=oov_word,
                        symbol_table=symbol_table)
                    ctm_line.extend([orig_ref_word, edit_type])
                    ctm_edits.append(ctm_line)

                current_time = ctm_array[ctm_pos][0] + ctm_array[ctm_pos][1]
        except Exception:
            logger.error("get_ctm_edits failed at ctm[%d]=%s",
                         ctm_pos,
                         ctm_array[ctm_pos] if ctm_pos < ctm_len else "NONE")
            raise
    return ctm_edits


def get_ctm_edits_with_sentences(alignment_output, ctm_array, original_ref,
                                  sentence_indices, eps_symbol="<eps>",
                                  oov_word=None, symbol_table=None,
                                  normalize=False):
    """Like get_ctm_edits but annotates each edit with its sentence index.

    sentence_indices[i] gives the sentence index (0-based) for flat ref word i.
    Returns list of (sentence_idx, [start, duration, hyp_word, conf, ref_word, edit_type])
    """
    ctm_edits = []
    ctm_len = len(ctm_array)
    current_time = ctm_array[0][0] if ctm_len > 0 else 0.0
    current_sentence = 0

    for (_rw, _hw, ref_prev_i, hyp_prev_i, ref_i, hyp_i) in alignment_output:
        if ref_i > ref_prev_i:
            current_sentence = sentence_indices[ref_prev_i]
        orig_ref_word = original_ref[ref_prev_i] if ref_i > ref_prev_i else eps_symbol
        try:
            ctm_pos = hyp_prev_i
            assert ctm_pos < ctm_len

            if hyp_prev_i == hyp_i:
                edit_type = get_edit_type(
                    eps_symbol, orig_ref_word, duration=0.0,
                    eps_symbol=eps_symbol, oov_word=oov_word,
                    symbol_table=symbol_table)
                ctm_edits.append((current_sentence,
                                   [current_time, 0.0, eps_symbol, 1.0,
                                    orig_ref_word, edit_type]))
            else:
                assert hyp_i == hyp_prev_i + 1
                assert len(ctm_array[ctm_pos]) == 4
                ctm_line = list(ctm_array[ctm_pos])
                orig_hyp_word = ctm_line[2]

                if orig_hyp_word == eps_symbol and orig_ref_word != eps_symbol:
                    ctm_edits.append((current_sentence,
                                       [current_time, 0.0, eps_symbol, 1.0,
                                        orig_ref_word, "del"]))
                    ctm_line.extend([eps_symbol, "sil"])
                    ctm_edits.append((current_sentence, ctm_line))
                else:
                    cmp_hyp = normalize_word(orig_hyp_word) if normalize and orig_hyp_word != eps_symbol else orig_hyp_word
                    cmp_ref = normalize_word(orig_ref_word) if normalize and orig_ref_word != eps_symbol else orig_ref_word
                    edit_type = get_edit_type(
                        cmp_hyp, cmp_ref, duration=ctm_line[1],
                        eps_symbol=eps_symbol, oov_word=oov_word,
                        symbol_table=symbol_table)
                    ctm_line.extend([orig_ref_word, edit_type])
                    ctm_edits.append((current_sentence, ctm_line))

                current_time = ctm_array[ctm_pos][0] + ctm_array[ctm_pos][1]
        except Exception:
            logger.error("get_ctm_edits_with_sentences failed at ctm[%d]=%s",
                         ctm_pos,
                         ctm_array[ctm_pos] if ctm_pos < ctm_len else "NONE")
            raise
    return ctm_edits


def read_whisperx_json(data):
    """WhisperX format: top-level 'segments' list, each with 'words' list."""
    ctm = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            start = w.get("start", 0.0)
            end = w.get("end", start)
            score = w.get("score", 1.0)
            if word:
                ctm.append([start, end - start, word, score])
    return ctm


def read_vv_json(data):
    """VibeVoice format: list of {Start, End, Content, Speaker}.

    No word-level timestamps; time is distributed evenly across words.
    Segments with no real content (e.g. [Silence]) are skipped.
    """
    ctm = []
    for seg in data:
        start = float(seg.get("Start", 0.0))
        end = float(seg.get("End", start))
        content = seg.get("Content", "").strip()
        words = content.split()
        if not words:
            continue
        dur_per_word = (end - start) / len(words)
        for i, w in enumerate(words):
            ctm.append([start + i * dur_per_word, dur_per_word, w, 1.0])
    return ctm


def read_hfjson_chunks(data):
    """HuggingFace pipeline format: top-level 'chunks' list with timestamps."""
    ctm = []
    for chunk in data.get("chunks", []):
        text = chunk.get("text", "").strip()
        ts = chunk.get("timestamp", [None, None])
        start = ts[0] if ts[0] is not None else 0.0
        end = ts[1] if ts[1] is not None else start
        if text:
            ctm.append([start, end - start, text, 1.0])
    return ctm


def autodetect_and_read(data, path=""):
    if isinstance(data, list) and data and "Content" in data[0]:
        logger.info("Detected VibeVoice format for %s", path)
        return read_vv_json(data)
    elif "segments" in data:
        logger.info("Detected WhisperX format for %s", path)
        return read_whisperx_json(data)
    elif "chunks" in data:
        logger.info("Detected HuggingFace JSON format for %s", path)
        return read_hfjson_chunks(data)
    else:
        raise ValueError(
            f"Cannot detect format for {path!r}: "
            "expected VibeVoice list, 'segments' (WhisperX), or 'chunks' (HuggingFace)")


def load_hyp(path, fmt):
    """Load one JSON hypothesis file. Returns (file_id, ctm_array)."""
    file_id = Path(path).stem
    with open(path) as f:
        data = json.load(f)
    if fmt == "whisperx":
        ctm = read_whisperx_json(data)
    elif fmt == "hfjson":
        ctm = read_hfjson_chunks(data)
    elif fmt == "vv":
        ctm = read_vv_json(data)
    else:
        ctm = autodetect_and_read(data, path)
    return file_id, ctm


def read_ref_tsv(ref_file):
    """TSV format: <ID>\\t<text>"""
    for line in ref_file:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        id_, _, text = line.partition("\t")
        words = text.strip().split()
        if words:
            yield id_.strip(), words


def read_ref_kaldi(ref_file):
    """Kaldi text format: <ID> word1 word2 ..."""
    for line in ref_file:
        parts = line.strip().split()
        if len(parts) >= 2:
            yield parts[0], parts[1:]


def read_ref_tsv_sentences(ref_file):
    """TSV multi-sentence format: <ID>\\t<text> where the same ID may repeat.

    Returns OrderedDict mapping ID -> list of word lists (one per sentence).
    """
    result = OrderedDict()
    for line in ref_file:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        id_, _, text = line.partition("\t")
        id_ = id_.strip()
        words = text.strip().split()
        if words:
            if id_ not in result:
                result[id_] = []
            result[id_].append(words)
    return result


def read_ref_numbered(ref_file):
    """Numbered sentence format: <num>\\t<text> for a single file.

    Returns list of word lists (one per sentence, in order).
    """
    sentences = []
    for line in ref_file:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        _num, _, text = line.partition("\t")
        words = text.strip().split()
        if words:
            sentences.append(words)
    return sentences


def build_flat_ref(sentence_list):
    """Flatten list of sentence word lists.

    Returns (flat_words, sentence_indices) where sentence_indices[i] is the
    0-based sentence index for flat_words[i].
    """
    flat_words = []
    sentence_indices = []
    for i, words in enumerate(sentence_list):
        for w in words:
            flat_words.append(w)
            sentence_indices.append(i)
    return flat_words, sentence_indices


def format_ctm_edit(file_id, row):
    start, duration, hyp_word, confidence, ref_word, edit_type = row
    return f"{file_id} 1 {start:.3f} {duration:.3f} {hyp_word} {confidence:.2f} {ref_word} {edit_type}"


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    hyp_group = parser.add_mutually_exclusive_group(required=True)
    hyp_group.add_argument("--hyp", metavar="FILE",
                           help="Single hypothesis JSON file")
    hyp_group.add_argument("--hyp-dir", metavar="DIR",
                           help="Directory of hypothesis JSON files")

    parser.add_argument("--hyp-format", choices=["whisperx", "hfjson", "auto"],
                        default="auto",
                        help="Hypothesis format (default: auto-detect)")
    parser.add_argument("--ref", required=True, type=argparse.FileType("r"),
                        help="Reference file")
    parser.add_argument("--ref-format",
                        choices=["tsv", "kaldi", "tsv-sentences", "numbered"],
                        default="tsv",
                        help=(
                            "Reference format: "
                            "tsv (<ID>\\t<text>), "
                            "kaldi (<ID> word...), "
                            "tsv-sentences (<ID>\\t<text> with same ID repeated per sentence), "
                            "numbered (<num>\\t<text> for a single file) "
                            "(default: tsv)"))
    parser.add_argument("--output", default="-",
                        help="Output file (default: stdout)")
    parser.add_argument("--eps-symbol", default="<eps>",
                        help="Epsilon/gap symbol (default: <eps>)")
    parser.add_argument("--correct-score", type=int, default=1)
    parser.add_argument("--substitution-penalty", type=int, default=1)
    parser.add_argument("--deletion-penalty", type=int, default=1)
    parser.add_argument("--insertion-penalty", type=int, default=1)
    parser.add_argument("--no-align-full-hyp", dest="align_full_hyp",
                        action="store_false", default=True,
                        help="Use standard Smith-Waterman (sub-sequence) instead of full-hyp mode")
    parser.add_argument("--no-normalize", dest="normalize",
                        action="store_false", default=True,
                        help="Disable punctuation/case normalization for word comparison")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    return parser.parse_args()


def _make_align_arrays(words, normalize):
    """Return normalized word list for alignment (or original if not normalizing)."""
    if normalize:
        return [normalize_word(w) for w in words]
    return words


def run(args):
    if args.verbose > 0:
        _handler.setLevel(logging.DEBUG)

    correct_score = args.correct_score
    sub_penalty = args.substitution_penalty
    del_score = -args.deletion_penalty
    ins_score = -args.insertion_penalty

    def similarity_score(x, y):
        return correct_score if x == y else -sub_penalty

    out = open(args.output, "w") if args.output != "-" else sys.stdout

    if args.ref_format in ("tsv-sentences", "numbered"):
        _run_sentenced(args, similarity_score, del_score, ins_score, out)
        return

    if args.hyp:
        hyp_files = [args.hyp]
    else:
        hyp_files = sorted(Path(args.hyp_dir).glob("*.json"))
        if not hyp_files:
            raise RuntimeError(f"No JSON files found in {args.hyp_dir}")

    if args.ref_format == "tsv":
        refs = dict(read_ref_tsv(args.ref))
    else:
        refs = dict(read_ref_kaldi(args.ref))
    args.ref.close()

    num_done = num_err = 0
    try:
        for hyp_path in hyp_files:
            file_id, ctm_array = load_hyp(hyp_path, args.hyp_format)

            if file_id not in refs:
                logger.warning("ID '%s' not found in reference; skipping", file_id)
                num_err += 1
                continue

            if not ctm_array:
                logger.warning("No words in hypothesis for '%s'; skipping", file_id)
                num_err += 1
                continue

            original_ref = refs[file_id]
            hyp_words = [row[2] for row in ctm_array]

            align_ref = _make_align_arrays(original_ref, args.normalize)
            align_hyp = _make_align_arrays(hyp_words, args.normalize)

            logger.info("Aligning %s: %d hyp words, %d ref words",
                        file_id, len(hyp_words), len(original_ref))

            alignment, _score = smith_waterman_alignment(
                align_ref, align_hyp,
                similarity_score_function=similarity_score,
                del_score=del_score, ins_score=ins_score,
                eps_symbol=args.eps_symbol,
                align_full_hyp=args.align_full_hyp)

            ctm_edits = get_ctm_edits(alignment, ctm_array, original_ref,
                                       eps_symbol=args.eps_symbol,
                                       normalize=args.normalize)

            for row in ctm_edits:
                print(format_ctm_edit(file_id, row), file=out)

            num_done += 1
    finally:
        if args.output != "-":
            out.close()

    logger.info("Done: %d succeeded, %d failed/skipped", num_done, num_err)
    if num_done == 0:
        raise RuntimeError("Processed 0 files successfully.")


def _run_sentenced(args, similarity_score, del_score, ins_score, out):
    """Handle tsv-sentences and numbered formats with per-sentence output."""
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
            hyp_words = [row[2] for row in ctm_array]

            align_ref = _make_align_arrays(original_ref, args.normalize)
            align_hyp = _make_align_arrays(hyp_words, args.normalize)

            logger.info("Aligning %s: %d hyp words, %d ref words across %d sentences",
                        file_id, len(hyp_words), len(original_ref), len(sentence_list))

            alignment, _score = smith_waterman_alignment(
                align_ref, align_hyp,
                similarity_score_function=similarity_score,
                del_score=del_score, ins_score=ins_score,
                eps_symbol=args.eps_symbol,
                align_full_hyp=args.align_full_hyp)

            ctm_edits = get_ctm_edits_with_sentences(
                alignment, ctm_array, original_ref, sentence_indices,
                eps_symbol=args.eps_symbol, normalize=args.normalize)

            for sent_idx, row in ctm_edits:
                sent_id = f"{file_id}_{sent_idx + 1}"
                print(format_ctm_edit(sent_id, row), file=out)

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
