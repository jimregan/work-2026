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
    deduplicate_and_filter,
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
        rules_path: Optional path to a Python module exporting
            ``build_rules(sigma_star) -> pynini.Fst``.
        sim_matrix_path: Optional path to a numpy phoneme similarity
            matrix file (``.npy``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_id: str,
        lexicon_path: str,
        rules_path: Optional[str] = None,
        sim_matrix_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device

        # 1. Extract vocab and build symbol tables
        self.vocab = extract_vocab(model_id)
        self.input_syms = build_symbol_table(self.vocab)
        self.num_tokens = self.input_syms.num_symbols()

        # 2. Build the lexicon list (for index-to-string lookups)
        #    and the lexicon FST
        self.lexicon_entries = load_lexicon(lexicon_path)
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

    def decode_utterance(
        self,
        log_probs: torch.Tensor,
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
    ) -> UtteranceAlignment:
        """Run the full decoding pipeline for a single utterance.

        Args:
            log_probs: CTC logits tensor of shape (T, C).
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

        Returns:
            An UtteranceAlignment with decoded phonemes and variation info.
        """
        import k2

        dev = torch.device(self.device)

        # 1. Build dense FSA from CTC posteriors
        emission = log_probs.unsqueeze(0).to(dev)
        lengths_t = torch.tensor([length], dtype=torch.int32)
        dense_fsa = create_dense_fsa_vec(emission, lengths_t).to(dev)

        # 2. Build per-utterance output symbol table (fresh copy)
        output_syms = build_output_symbol_table(self.input_syms)

        # 3. Get phoneme IDs for the reference sequence
        phoneme_ids = self._get_phoneme_ids(ref_phonemes)

        # 4. Build reference FST with variation arcs
        ref = build_ref_fst(
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

        # 5. Compose CTC topology with reference FST
        ctc_copy = self.ctc_topo.copy()
        ctc_copy.arcsort("olabel")
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
        # Extend the lexicon list for any new <trans> markers
        extended_lexicon = list(self.lexicon_list)
        for idx in range(len(extended_lexicon), output_syms.num_symbols()):
            sym = output_syms.find(idx)
            extended_lexicon.append(sym if sym else "")

        # Deduplicate and filter
        phoneme_seq = deduplicate_and_filter(aux_labels, output_syms)
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
                    if item["start_state"] < len(ref_phonemes)
                    else None
                ),
                start_frame=0,
                end_frame=0,
                variation_type=item["variation_type"],
                ref_state=item["start_state"],
            )
            for item in variation_info
        ]

        alignment = UtteranceAlignment(
            utterance_id=utterance_id,
            audio_path=audio_path,
            ref_text=ref_text,
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
            emission = emissions[idx, : lengths[idx]]
            alignment = self.decode_utterance(
                log_probs=emission,
                length=lengths[idx],
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
