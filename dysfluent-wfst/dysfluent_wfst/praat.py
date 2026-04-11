"""Praat/Parselmouth enrichment for saved WFST alignments."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np


def enrich_with_praat(
    alignment_path: str,
    output_path: Optional[str] = None,
    step_s: float = 0.005,
) -> dict:
    """Load a saved alignment and add Praat measurements to each segment."""
    import parselmouth

    with open(alignment_path, encoding="utf-8") as f:
        alignment = json.load(f)

    snd = parselmouth.Sound(alignment["audio_path"])
    pitch = snd.to_pitch()
    formants = snd.to_formant_burg()
    intensity = snd.to_intensity()

    for seg in alignment.get("segments", []):
        t0 = float(seg.get("start_time_s", 0.0))
        t1 = float(seg.get("end_time_s", 0.0))
        tmid = (t0 + t1) / 2 if t1 >= t0 else t0

        seg["duration_s"] = max(0.0, t1 - t0)

        if t1 > t0:
            times = np.arange(t0, t1, step_s)
            f0_vals = []
            for t in times:
                value = pitch.get_value_at_time(float(t))
                if value and value > 0:
                    f0_vals.append(float(value))
        else:
            f0_vals = []

        seg["f0_mean"] = float(np.mean(f0_vals)) if f0_vals else None
        seg["f0_range"] = float(np.ptp(f0_vals)) if f0_vals else None
        seg["f1"] = formants.get_value_at_time(1, tmid)
        seg["f2"] = formants.get_value_at_time(2, tmid)
        seg["f3"] = formants.get_value_at_time(3, tmid)
        seg["intensity_mean"] = intensity.get_average(t0, t1) if t1 > t0 else None

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alignment, f, indent=2, ensure_ascii=False)

    return alignment
