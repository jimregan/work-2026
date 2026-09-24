"""Decoder class orchestrating the full pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pynini
import torch

from .alignment import AlignmentSegment, UtteranceAlignment
from .ctc_topo import build_ctc_topo
from .k2_bridge import create_dense_fsa_vec, fst_to_k2_str
from .lexicon import build_lexicon_fst, build_utterance_fst, load_lexicon
from .ref_fst import build_ref_fst
from .rules import compile_rules
from .symbols import build_output_symbol_table, build_symbol_table, extract_vocab
from .variation import (
    build_state_trajectory,
    collapse_label_runs,
    merge_trans_markers,
)


class Decoder:
    """WFST decoder for phonetic variation detection.

    Orchestrates: symbol tables, lexicon FST, CTC topology, reference
    FST construction, pynini composition, k2 dense intersection, and
    variation analysis.

    The CTC topology is built once and reused across utterances.
    The reference FST and output symbol table are rebuilt per utterance
    since each utterance adds different ``<trans>`` markers.

    Args:
        model_id: HuggingFace model ID for the wav2vec2 CTC model.
        lexicon_path: Path to the TSV lexicon file.
        rules_path: Optional path to a YAML phonetic rules file
            (MFA format: segment, replacement, contexts).
        sim_matrix_path: Optional path to a numpy phoneme similarity
            matrix file (``.npy``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_id: str,
        lexicon_path: Optional[str] = None,
        lexicon_entries: Optional[list[tuple[str, str]]] = None,
        rules_path: Optional[str] = None,
        sim_matrix_path: Optional[str] = None,
        device: str = "cpu",
    ):
        if lexicon_path is None and lexicon_entries is None:
            raise ValueError("Provide either lexicon_path or lexicon_entries")
        self.device = device

        # 1. Extract vocab and build symbol tables
        self.vocab = extract_vocab(model_id)
        self.input_syms = build_symbol_table(self.vocab)
        self.num_tokens = self.input_syms.num_symbols()

        # 2. Build the lexicon list (for index-to-string lookups)
        #    and the lexicon FST
        self.lexicon_entries = (
            lexicon_entries if lexicon_entries is not None
            else load_lexicon(lexicon_path)
        )
        self.lexicon_fst = build_lexicon_fst(
            self.lexicon_entries,
            input_token_type="utf8",
            output_syms=self.input_syms,
        )

        # Build a flat list for phoneme-id-to-string lookup
        # (mirrors original's self.lexicon usage)
        self._build_lexicon_list()

        # 3. Compile optional phonetic rules
        self.rules_fst = compile_rules(rules_path, self.input_syms)

        # 4. Load optional similarity matrix
        self.similarity_matrix = None
        self.phn2idx: Optional[dict[str, int]] = None
        if sim_matrix_path is not None:
            self.similarity_matrix = np.load(sim_matrix_path)
            self.similarity_matrix = torch.from_numpy(self.similarity_matrix)
            # Default CMU/ARPAbet phn2idx (41 phonemes)
            self.phn2idx = {
                "|": 0, "OW": 1, "UW": 2, "EY": 3, "AW": 4, "AH": 5,
                "AO": 6, "AY": 7, "EH": 8, "K": 9, "NG": 10, "F": 11,
                "JH": 12, "M": 13, "CH": 14, "IH": 15, "UH": 16,
                "HH": 17, "L": 18, "AA": 19, "R": 20, "TH": 21,
                "AE": 22, "D": 23, "Z": 24, "OY": 25, "DH": 26,
                "IY": 27, "B": 28, "W": 29, "S": 30, "T": 31,
                "SH": 32, "ZH": 33, "ER": 34, "V": 35, "Y": 36,
                "N": 37, "G": 38, "P": 39, "-": 40,
            }

        # 5. Build CTC topology once
        self.ctc_topo = build_ctc_topo(self.num_tokens, self.input_syms)

    def _build_lexicon_list(self) -> None:
        """Build a list mapping symbol indices to symbol strings."""
        self.lexicon_list: list[str] = []
        for idx in range(self.input_syms.num_symbols()):
            sym = self.input_syms.find(idx)
            self.lexicon_list.append(sym if sym else "")

    def _get_phoneme_id(self, phoneme: str) -> int:
        """Map a phoneme string to its symbol table index."""
        idx = self.input_syms.find(phoneme)
        if idx == -1:
            return 0  # fall back to epsilon/blank for unknown symbols
        return idx

    def _get_phoneme_ids(self, phonemes: list[str]) -> list[int]:
        """Map a list of phoneme strings to symbol table indices."""
        return [self._get_phoneme_id(p) for p in phonemes]

    def _tokenize_words(self, ref_text: str) -> list[str]:
        """Split reference text into lexicon lookup tokens."""
        words: list[str] = []
        for word in ref_text.lower().strip().split():
            clean = word.strip(".,!?;:")
            if clean:
                words.append(clean)
        return words

    def _enumerate_label_paths(
        self,
        fst: pynini.Fst,
        max_paths: int = 256,
    ) -> list[list[int]]:
        """Enumerate output-label paths from a finite pronunciation lattice."""
        start_state = fst.start()
        no_state_id = getattr(pynini, "NO_STATE_ID", -1)
        if start_state == no_state_id:
            return []

        zero = pynini.Weight.zero(fst.weight_type())
        paths: list[list[int]] = []

        def dfs(state: int, labels: list[int], stack: set[int]) -> None:
            if len(paths) >= max_paths or state in stack:
                return

            if fst.final(state) != zero:
                paths.append(list(labels))
                if len(paths) >= max_paths:
                    return

            next_stack = set(stack)
            next_stack.add(state)
            for arc in fst.arcs(state):
                label = arc.olabel if arc.olabel > 0 else arc.ilabel
                if label > 0:
                    labels.append(label)
                dfs(arc.nextstate, labels, next_stack)
                if label > 0:
                    labels.pop()

        dfs(start_state, [], set())
        return paths

    def _candidate_phoneme_sequences(
        self,
        ref_phonemes: list[str],
        ref_text: str,
    ) -> list[list[int]]:
        """Build one or more candidate pronunciation sequences for decoding."""
        fallback = [self._get_phoneme_ids(ref_phonemes)]

        if self.rules_fst is None or not ref_text:
            return fallback

        words = self._tokenize_words(ref_text)
        if not words:
            return fallback

        try:
            utt_fst = build_utterance_fst(
                words=words,
                lexicon_fst=self.lexicon_fst,
                rules_fst=self.rules_fst,
            )
            paths = self._enumerate_label_paths(utt_fst)
        except Exception:
            return fallback

        if not paths:
            return fallback

        deduped: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for path in paths:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                deduped.append(path)
        return deduped or fallback

    def decode_utterance(
        self,
        logits: torch.Tensor,
        length: int,
        ref_phonemes: list[str],
        utterance_id: str = "",
        audio_path: str = "",
        ref_text: str = "",
        beta: float = 5.0,
        back: bool = True,
        skip: bool = False,
        sub: bool = True,
        output_beam: float = 25.0,
        frame_shift_ms: float = 20.0,
        sample_rate: int = 16000,
    ) -> UtteranceAlignment:
        """Run the full decoding pipeline for a single utterance.

        Args:
            logits: Raw CTC logits tensor of shape (T, C) (before softmax).
            length: Number of valid frames.
            ref_phonemes: Reference phoneme sequence (list of strings).
            utterance_id: Identifier for this utterance.
            audio_path: Path to the audio file.
            ref_text: Original reference text.
            beta: Variation tolerance parameter.
            back: Allow back/repetition arcs.
            skip: Allow skip/deletion arcs.
            sub: Allow substitution arcs.
            output_beam: Beam width for k2.intersect_dense.
            frame_shift_ms: Duration represented by one decoder frame.
            sample_rate: Audio sample rate for the source waveform.

        Returns:
            An UtteranceAlignment with decoded phonemes and variation info.
        """
        import k2

        dev = torch.device(self.device)

        # 1. Build dense FSA from CTC posteriors
        emission = logits.unsqueeze(0).to(dev)
        lengths_t = torch.tensor([length], dtype=torch.int32)
        dense_fsa = create_dense_fsa_vec(emission, lengths_t).to(dev)

        # 2. Build per-utterance output symbol table (fresh copy)
        output_syms = build_output_symbol_table(self.input_syms)

        # 3. Build one or more candidate pronunciation paths.
        candidate_paths = self._candidate_phoneme_sequences(ref_phonemes, ref_text)

        # 4. Build reference FST with variation arcs.
        ref_fsts = [
            build_ref_fst(
                phoneme_ids=phoneme_ids,
                beta=beta,
                input_syms=self.input_syms,
                output_syms=output_syms,
                skip=skip,
                back=back,
                sub=sub,
                similarity_matrix=self.similarity_matrix,
                phn2idx=self.phn2idx,
                lexicon=self.lexicon_list,
            )
            for phoneme_ids in candidate_paths
        ]
        ref = pynini.union(*ref_fsts).optimize()

        # 5. Compose CTC topology with reference FST
        ctc_copy = self.ctc_topo.copy()
        ref.arcsort("ilabel")
        composed = pynini.compose(ctc_copy, ref)

        # 6. Export to k2 text format
        fst_str = fst_to_k2_str(composed, to_log_probs=True)

        # 7. Import into k2, intersect, shortest path
        composed_k2 = k2.Fsa.from_str(fst_str, acceptor=False)
        composed_k2 = k2.arc_sort(composed_k2).to(dev)

        lattice = k2.intersect_dense(
            composed_k2, dense_fsa, output_beam=output_beam
        )
        shortest = k2.shortest_path(lattice, use_double_scores=True)

        # 8. Extract output labels
        aux_labels = shortest[0].aux_labels.tolist()
        if aux_labels and aux_labels[-1] == -1:
            aux_labels = aux_labels[:-1]

        # 9. Variation analysis
        # Collapse per-frame labels into runs so timing survives deduplication.
        phoneme_seq = collapse_label_runs(aux_labels, output_syms)
        # Merge consecutive <trans> markers
        merged_seq = merge_trans_markers(phoneme_seq)
        # Build state trajectory and classify
        variation_info = build_state_trajectory(merged_seq, ref_phonemes)

        # 10. Build alignment result
        decoded_phonemes = [item["phoneme"] for item in variation_info]
        segments = [
            AlignmentSegment(
                phoneme=item["phoneme"],
                ref_phoneme=(
                    ref_phonemes[item["start_state"]]
                    if isinstance(item.get("start_state"), int)
                    and 0 <= item["start_state"] < len(ref_phonemes)
                    else None
                ),
                start_frame=item.get("start_frame", 0),
                end_frame=item.get("end_frame", 0),
                start_time_s=item.get("start_frame", 0) * frame_shift_ms / 1000.0,
                end_time_s=item.get("end_frame", 0) * frame_shift_ms / 1000.0,
                variation_type=item["variation_type"],
                ref_state=item["start_state"],
            )
            for item in variation_info
        ]

        alignment = UtteranceAlignment(
            utterance_id=utterance_id,
            audio_path=audio_path,
            ref_text=ref_text,
            sample_rate=sample_rate,
            frame_shift_ms=frame_shift_ms,
            ref_phonemes=ref_phonemes,
            decoded_phonemes=decoded_phonemes,
            segments=segments,
            variation_info=variation_info,
        )

        # Cleanup
        del lattice, shortest, dense_fsa, composed_k2
        if self.device != "cpu":
            torch.cuda.empty_cache()

        return alignment

    def decode_batch(
        self,
        batch: dict,
        beta: float = 5.0,
        back: bool = True,
        skip: bool = False,
        sub: bool = True,
        output_beam: float = 25.0,
    ) -> list[UtteranceAlignment]:
        """Decode a batch of utterances.

        Args:
            batch: Dict with keys ``"id"``, ``"tensor"``, ``"ref_phonemes"``,
                ``"lengths"``, and optionally ``"ref_text"``, ``"audio_path"``.
            beta: Variation tolerance parameter.
            back: Allow back/repetition arcs.
            skip: Allow skip/deletion arcs.
            sub: Allow substitution arcs.
            output_beam: Beam width for k2.intersect_dense.

        Returns:
            List of UtteranceAlignment results, one per utterance.
        """
        ids = batch["id"]
        emissions = batch["tensor"]
        ref_phonemes_list = batch["ref_phonemes"]
        lengths = batch["lengths"]
        ref_texts = batch.get("ref_text", [""] * len(ids))
        audio_paths = batch.get("audio_path", [""] * len(ids))

        results = []
        for idx, sample_id in enumerate(ids):
            utt_len = int(lengths[idx])
            emission = emissions[idx, :utt_len]
            alignment = self.decode_utterance(
                logits=emission,
                length=utt_len,
                ref_phonemes=ref_phonemes_list[idx],
                utterance_id=sample_id,
                audio_path=audio_paths[idx] if idx < len(audio_paths) else "",
                ref_text=ref_texts[idx] if idx < len(ref_texts) else "",
                beta=beta,
                back=back,
                skip=skip,
                sub=sub,
                output_beam=output_beam,
            )
            results.append(alignment)

        return results
