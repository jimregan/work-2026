"""Alignment dataclasses and JSON serialisation.

Format matches CLAUDE.md Section 9.2 (the Praat stage contract).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class AlignmentSegment:
    """A single phoneme segment in the alignment."""

    phoneme: str
    ref_phoneme: Optional[str] = None
    start_frame: int = 0
    end_frame: int = 0
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    variation_type: str = "normal"
    ref_state: int = 0
    lattice_score: Optional[float] = None


@dataclass
class UtteranceAlignment:
    """Full alignment result for one utterance."""

    utterance_id: str = ""
    audio_path: str = ""
    ref_text: str = ""
    sample_rate: int = 16000
    frame_shift_ms: float = 20.0
    ref_phonemes: list[str] = field(default_factory=list)
    decoded_phonemes: list[str] = field(default_factory=list)
    segments: list[AlignmentSegment] = field(default_factory=list)
    variation_info: list[dict] = field(default_factory=list)


def save_alignment(alignment: UtteranceAlignment, path: str) -> None:
    """Save an UtteranceAlignment to a JSON file."""
    data = asdict(alignment)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_alignment(path: str) -> UtteranceAlignment:
    """Load an UtteranceAlignment from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segments = [AlignmentSegment(**seg) for seg in data.pop("segments", [])]
    alignment = UtteranceAlignment(**data, segments=segments)
    return alignment
