"""A small, trainable baseline inspired by van Santen & Sproat (1999).

The implementation focuses on the acoustic boundary detector described in
sections 2.2-2.5 of the paper. It accepts a waveform and candidate phone
boundaries, and returns a refined boundary for each candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


BANDS = ((100.0, 300.0), (300.0, 800.0), (800.0, 2500.0),
         (2500.0, 3500.0), (3500.0, 5800.0))


@dataclass(frozen=True)
class GaborSpec:
    sigma_ms: float = 8.0
    cycles_per_ms: float = 0.12
    antisymmetric: bool = True
    sign: float = 1.0


@dataclass
class BroadDetector:
    """Five-band detector parameters and cross-band pooling weights."""

    specs: tuple[GaborSpec, ...] = field(
        default_factory=lambda: tuple(GaborSpec() for _ in BANDS))
    weights: np.ndarray = field(default_factory=lambda: np.ones(5))
    pooling_sigma_ms: float = 5.0
    search_radius_ms: float = 35.0
    frame_ms: float = 1.0

    def __post_init__(self) -> None:
        if len(self.specs) != 5 or len(self.weights) != 5:
            raise ValueError("exactly five band specifications and weights are required")


@dataclass
class NarrowDetector:
    """LDA projection and phone-side centroids for one narrow diphone class."""

    weights: np.ndarray
    centroid_left: np.ndarray
    centroid_right: np.ndarray
    sigma_ms: float = 8.0
    cycles_per_ms: float = 0.12
    search_radius_ms: float = 35.0

    def __post_init__(self) -> None:
        if self.weights.ndim != 1 or self.centroid_left.shape != self.weights.shape:
            raise ValueError("narrow detector vectors must have matching dimensions")
        if self.centroid_right.shape != self.weights.shape:
            raise ValueError("narrow detector centroids must have matching dimensions")


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("waveform must be a non-empty 1-D or 2-D array")
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    return x / peak if peak > 1.0 else x


def _resample(waveform: np.ndarray, sample_rate: int,
              target_rate: int = 12000) -> tuple[np.ndarray, int]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate == target_rate:
        return _to_mono(waveform), sample_rate
    x = _to_mono(waveform)
    old_t = np.arange(len(x), dtype=float) / sample_rate
    new_length = round(len(x) * target_rate / sample_rate)
    new_t = np.arange(new_length, dtype=float) / target_rate
    y = np.interp(new_t, old_t, x)
    return y, target_rate


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(4.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _fft_bandpass(x: np.ndarray, sample_rate: int,
                  low: float, high: float) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    frequencies = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    spectrum[(frequencies < low) | (frequencies > high)] = 0.0
    return np.fft.irfft(spectrum, n=len(x))


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_fft_features(waveform: np.ndarray, sample_rate: int,
                     bins: int = 55, frame_ms: float = 1.0) -> tuple[np.ndarray, int]:
    """Return 55-bin mel log spectra, sampled every frame_ms."""
    x, sr = _resample(waveform, sample_rate)
    step = max(1, round(sr * frame_ms / 1000.0))
    window = max(step * 25, round(sr * 0.025))
    nfft = 1 << (window - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(np.pad(x, (window // 2, window // 2)), n=nfft)) ** 2
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sr)
    mel_edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array([80.0]))[0],
                                       _hz_to_mel(np.array([5800.0]))[0], bins + 2))
    bank = np.zeros((bins, len(frequencies)))
    for i in range(bins):
        left, center, right = mel_edges[i:i + 3]
        bank[i] = np.maximum(0.0, np.minimum((frequencies - left) / (center - left),
                                               (right - frequencies) / (right - center)))
    frames = []
    for start in range(0, len(x), step):
        segment = np.zeros(window)
        chunk = x[start:start + window]
        segment[:len(chunk)] = chunk
        power = np.abs(np.fft.rfft(segment * np.hanning(window), n=nfft)) ** 2
        frames.append(10.0 * np.log10(np.maximum(bank @ power, 1e-12)))
    return np.asarray(frames), sr // step


def five_band_features(waveform: np.ndarray, sample_rate: int,
                       frame_ms: float = 1.0) -> tuple[np.ndarray, int]:
    """Return log-energy features shaped (frames, 5), sampled every frame_ms."""
    x, sr = _resample(waveform, sample_rate)
    frame_step = max(1, round(sr * frame_ms / 1000.0))
    bands = []
    for low, high in BANDS:
        filtered = _fft_bandpass(x, sr, low, high)
        power = filtered * filtered
        smoothed = _gaussian_smooth(power, max(0.5, sr * 0.001 / frame_step))
        samples = smoothed[::frame_step]
        bands.append(10.0 * np.log10(np.maximum(samples, 1e-12)))
    n = min(map(len, bands))
    return np.stack([b[:n] for b in bands], axis=1), sr // frame_step


def _gabor(spec: GaborSpec, frame_rate: int) -> np.ndarray:
    sigma = max(1.0, spec.sigma_ms * frame_rate / 1000.0)
    half_width = max(3, int(np.ceil(4.0 * sigma)))
    t = np.arange(-half_width, half_width + 1) / frame_rate * 1000.0
    envelope = np.exp(-0.5 * (t / spec.sigma_ms) ** 2)
    phase = 2.0 * np.pi * spec.cycles_per_ms * t
    kernel = envelope * (np.sin(phase) if spec.antisymmetric else np.cos(phase))
    if not spec.antisymmetric:
        kernel -= kernel.mean()
    kernel *= spec.sign
    norm = np.sum(np.abs(kernel))
    return kernel / norm if norm else kernel


def detector_outputs(features: np.ndarray, detector: BroadDetector,
                     frame_rate: int) -> np.ndarray:
    """Compute one edge response per band, preserving the feature length."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("features must have shape (frames, 5)")
    outputs = np.empty_like(x)
    for i, spec in enumerate(detector.specs):
        outputs[:, i] = np.convolve(x[:, i], _gabor(spec, frame_rate), mode="same")
    return outputs


def pooled_score(outputs: np.ndarray, detector: BroadDetector,
                 frame_rate: int) -> np.ndarray:
    """Pool imperfectly synchronized band peaks using a Gaussian window."""
    weighted = np.asarray(outputs) * np.asarray(detector.weights)[None, :]
    score = np.sum(np.maximum(weighted, 0.0), axis=1)
    sigma = max(0.5, detector.pooling_sigma_ms * frame_rate / 1000.0)
    return _gaussian_smooth(score, sigma)


def detect_boundaries(features: np.ndarray, frame_rate: int,
                      candidates_ms: Iterable[float],
                      detector: BroadDetector | None = None) -> np.ndarray:
    """Refine candidate boundaries, returning times in milliseconds."""
    detector = detector or BroadDetector()
    out = detector_outputs(features, detector, frame_rate)
    score = pooled_score(out, detector, frame_rate)
    radius = max(1, round(detector.search_radius_ms * frame_rate / 1000.0))
    result = []
    for time_ms in candidates_ms:
        center = round(time_ms * frame_rate / 1000.0)
        lo, hi = max(0, center - radius), min(len(score), center + radius + 1)
        if lo >= hi:
            raise ValueError(f"candidate {time_ms} ms is outside the feature sequence")
        result.append(1000.0 * (lo + int(np.argmax(score[lo:hi]))) / frame_rate)
    return np.asarray(result)


def train_narrow_detector(left_vectors: np.ndarray, right_vectors: np.ndarray,
                          regularization: float = 1e-3) -> NarrowDetector:
    """Train the paper's first-linear-discriminant-axis narrow detector.

    ``left_vectors`` are samples from the final part of the left phone and
    ``right_vectors`` from the initial part of the right phone.
    """
    left = np.asarray(left_vectors, dtype=float)
    right = np.asarray(right_vectors, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("left and right samples must be 2-D with equal feature dimensions")
    if len(left) < 2 or len(right) < 2:
        raise ValueError("at least two samples per phone are required")
    mu_left, mu_right = left.mean(axis=0), right.mean(axis=0)
    centered = np.vstack((left - mu_left, right - mu_right))
    covariance = centered.T @ centered / max(1, len(centered) - 2)
    scale = max(float(np.trace(covariance)) / covariance.shape[0], 1e-9)
    weights = np.linalg.pinv(covariance + regularization * scale * np.eye(covariance.shape[0])) @ (mu_right - mu_left)
    norm = np.linalg.norm(weights)
    if norm == 0:
        raise ValueError("left and right samples have identical means")
    weights /= norm
    return NarrowDetector(weights, mu_left, mu_right)


def narrow_curve(features: np.ndarray, detector: NarrowDetector) -> np.ndarray:
    """Compute the normalized distance curve from equation (3) in the paper."""
    x = np.asarray(features, dtype=float) @ detector.weights
    left = detector.centroid_left @ detector.weights
    right = detector.centroid_right @ detector.weights
    d_left, d_right = np.abs(x - left), np.abs(x - right)
    return (d_left - d_right) / np.maximum(d_left + d_right, 1e-12)


def detect_narrow_boundary(features: np.ndarray, frame_rate: int,
                           candidate_ms: float, detector: NarrowDetector) -> float:
    """Localize a narrow-class boundary at the steepest point of equation (3)."""
    curve = narrow_curve(features, detector)
    kernel = _gabor(GaborSpec(detector.sigma_ms, detector.cycles_per_ms, True, 1), frame_rate)
    response = np.abs(np.convolve(curve, kernel, mode="same"))
    center = round(candidate_ms * frame_rate / 1000.0)
    radius = max(1, round(detector.search_radius_ms * frame_rate / 1000.0))
    lo, hi = max(0, center - radius), min(len(response), center + radius + 1)
    if lo >= hi:
        raise ValueError(f"candidate {candidate_ms} ms is outside the feature sequence")
    return 1000.0 * (lo + int(np.argmax(response[lo:hi]))) / frame_rate


def enumerate_pronunciations(canonical: Sequence[str],
                             rules: Sequence[tuple[str, str, float]]) -> list[tuple[list[str], float]]:
    """Expand simple phone rewrite alternatives, retaining pronunciation costs."""
    paths = [(list(canonical), 0.0)]
    for index, phone in enumerate(canonical):
        alternatives = [(phone, 0.0)] + [(dst, cost) for src, dst, cost in rules if src == phone]
        paths = [(path[:index] + [replacement] + path[index + 1:], cost + added)
                 for path, cost in paths for replacement, added in alternatives]
    unique: dict[tuple[str, ...], float] = {}
    for path, cost in paths:
        key = tuple(path)
        unique[key] = min(cost, unique.get(key, np.inf))
    return [(list(path), cost) for path, cost in unique.items()]


def pronunciation_lattice(canonical: Sequence[str],
                          rules: Sequence[tuple[str, str, float]]):
    """Build a weighted Pynini acceptor; raises a helpful error if unavailable."""
    try:
        import pynini
    except ImportError as exc:
        raise ImportError("pronunciation_lattice requires optional dependency 'pynini'") from exc
    paths = enumerate_pronunciations(canonical, rules)
    return pynini.union(*[pynini.accep(" ".join(path), weight=cost, token_type="utf8")
                          for path, cost in paths]).optimize()


def train_detector(feature_sequences: Sequence[np.ndarray],
                   boundary_frames: Sequence[Sequence[int]],
                   frame_rate: int = 1000,
                   candidate_sigma_ms: Sequence[float] = (4, 6, 8, 10, 12)) -> BroadDetector:
    """Choose Gabor widths/signs per band by boundary localization error.

    This is the paper's fast grid-search training approximation. Each training
    item is a five-band feature matrix and a list of manually marked frames.
    """
    if len(feature_sequences) != len(boundary_frames) or not feature_sequences:
        raise ValueError("features and labels must be non-empty and have equal length")
    specs = []
    errors = []
    for band in range(5):
        best = None
        for sigma in candidate_sigma_ms:
            for antisymmetric in (True, False):
                for sign in (-1.0, 1.0):
                    spec = GaborSpec(sigma, 0.12, antisymmetric, sign)
                    trial = BroadDetector(tuple(spec if j == band else GaborSpec()
                                                for j in range(5)))
                    errs = []
                    for seq, marks in zip(feature_sequences, boundary_frames):
                        response = detector_outputs(seq, trial, frame_rate)[:, band]
                        for mark in marks:
                            lo, hi = max(0, mark - 50), min(len(response), mark + 51)
                            peak = lo + int(np.argmax(response[lo:hi]))
                            errs.append(abs(peak - mark))
                    error = float(np.mean(errs)) if errs else np.inf
                    if best is None or error < best[0]:
                        best = (error, spec)
        assert best is not None
        errors.append(best[0])
        specs.append(best[1])
    weights = 1.0 / np.maximum(np.asarray(errors), 1e-6)
    return BroadDetector(tuple(specs), weights / weights.mean())


def detect(waveform: np.ndarray, sample_rate: int,
           candidates_ms: Iterable[float],
           detector: BroadDetector | None = None) -> np.ndarray:
    """Convenience wrapper for waveform input."""
    features, frame_rate = five_band_features(waveform, sample_rate)
    return detect_boundaries(features, frame_rate, candidates_ms, detector)
