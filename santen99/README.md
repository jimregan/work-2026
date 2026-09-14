# Acoustic boundary detector baseline

This is a compact Python baseline inspired by Jan van Santen and Richard
Sproat, “High-Accuracy Automatic Segmentation” (EUROSPEECH 1999).

It implements the paper's broad-class acoustic path and the narrow-class
detector:

1. resample audio to 12 kHz;
2. compute five one-millisecond log-energy bands;
3. apply symmetric or antisymmetric Gabor edge detectors;
4. pool positive detector responses with a Gaussian window; and
5. refine supplied candidate phone boundaries.

For narrow diphones, it additionally computes 55-bin mel spectra, trains a
regularized Fisher LDA projection, and detects the steepest point of the
normalized distance curve from equation (3). `pronunciation_lattice` is an
optional Pynini integration for simple weighted rewrite alternatives.

The API deliberately takes candidate boundaries because the paper's full
system combines acoustic costs with a pronunciation lattice. That recognition
and rewrite-rule layer is outside this reproducible acoustic baseline.

```python
import numpy as np
from baseline import detect

sample_rate = 12000
audio = np.load("utterance.npy")
boundaries_ms = detect(audio, sample_rate, [120, 245, 380])
print(boundaries_ms)
```

Install the acoustic baseline with `python -m pip install numpy`; add
`pynini` to use `pronunciation_lattice`. Run
`python -m pytest -q`.
