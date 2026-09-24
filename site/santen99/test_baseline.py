import numpy as np

from baseline import (BroadDetector, GaborSpec, detect_boundaries,
                      five_band_features, mel_fft_features,
                      train_narrow_detector, narrow_curve,
                      enumerate_pronunciations)


def test_features_have_five_bands_and_one_ms_hop():
    x = np.zeros(12000)
    x[6000:] = 0.5 * np.sin(2 * np.pi * 1200 * np.arange(6000) / 12000)
    features, frame_rate = five_band_features(x, 12000)
    assert features.shape[1] == 5
    assert frame_rate == 1000
    assert features.shape[0] == 1000


def test_detector_returns_boundary_near_candidate():
    frames = np.zeros((500, 5))
    frames[250:, 0] = 20.0
    detector = BroadDetector(
        specs=tuple(GaborSpec(sigma_ms=5, cycles_per_ms=0.02,
                              antisymmetric=True, sign=1) for _ in range(5)),
        search_radius_ms=40,
    )
    found = detect_boundaries(frames, 1000, [250], detector)
    assert 210 <= found[0] <= 290


def test_narrow_detector_and_pronunciation_expansion():
    rng = np.random.default_rng(7)
    left = rng.normal(0, 0.1, (20, 55))
    right = left + 1.0 + rng.normal(0, 0.1, (20, 55))
    detector = train_narrow_detector(left, right)
    curve = narrow_curve(np.vstack((left, right)), detector)
    assert curve.shape == (40,)
    paths = enumerate_pronunciations(["z", "a"], [("z", "s", 2.0)])
    assert sorted((tuple(p), c) for p, c in paths) == [(('s', 'a'), 2.0), (('z', 'a'), 0.0)]


def test_mel_features_have_expected_dimension():
    features, rate = mel_fft_features(np.zeros(12000), 12000)
    assert features.shape[1] == 55
    assert rate == 1000
