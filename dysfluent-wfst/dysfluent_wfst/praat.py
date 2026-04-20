"""Optional Praat/Parselmouth enrichment for saved WFST alignments.

This module is intentionally not part of the core runtime dependency set.
Install the optional ``praat`` extra to use it:

    pip install .[praat]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .acoustics import load_alignment_dict, save_alignment_dict


def enrich_with_praat(
    alignment_path: str,
    output_path: Optional[str] = None,
    step_s: float = 0.005,
) -> dict:
    """Load a saved alignment and add Praat measurements to each segment."""
    try:
        import parselmouth
    except ImportError as exc:
        raise RuntimeError(
            "Praat enrichment requires the optional 'praat-parselmouth' "
            "dependency. Install it with `pip install .[praat]`."
        ) from exc

    alignment = load_alignment_dict(alignment_path)

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

    return save_alignment_dict(alignment, output_path)
