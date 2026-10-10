# Power Quality Disturbance Classification

Classifies 9 types of voltage disturbance from raw 60 Hz waveforms, the job a power quality
meter or relay does on a feeder. It uses signal processing features (RMS tracking, FFT harmonics,
residual band energy) and a Random Forest, then stress-tests the model under field conditions
it never saw in training.

**Classes (IEEE 1159):** Normal, Sag, Swell, Interruption, Harmonics, Flicker, Oscillatory
transient (capacitor switching), Notch (converter commutation), Sag + harmonics.

## Results

Each test set has 200 waveforms per class.

| Configuration | 50 dB | 40 dB | 30 dB | 20 dB | **Stress test** |
|---|---|---|---|---|---|
| 1. Plain FFT (assumes 60.000 Hz) | 99.8% | 99.9% | 100.0% | 98.6% | 87.6% |
| 2. Synchronous resampling only | 99.6% | 99.7% | 99.8% | 98.3% | 93.0% |
| 3. Background harmonics in training only | 99.9% | 99.9% | 99.9% | 98.5% | 94.0% |
| **4. Both** | 99.7% | 99.8% | 99.9% | 98.3% | **99.6%** |

**Stress test conditions:** 30 dB noise, random phase, grid frequency drifting ±0.1 Hz, and 0.5–2%
background 3rd/5th harmonics. Frequency drift and random phase are never in the training data.

## The interesting part: why the clean-data accuracy was misleading

Most PQ classification papers stop at ~99% on clean synthetic signals. Here that number fell to
**87.6%** as soon as the test signals looked like a real feeder. There were two separate causes,
each with its own fix.

1. **Spectral leakage from frequency drift.** The FFT assumes exactly 10 cycles in the window.
   At 60.1 Hz that isn't true, so fundamental energy leaks into neighbouring bins and looks like
   distortion. Harmonics recall dropped to 53%, and Normal waveforms were called transients.
   **Fix:** measure the true frequency (Hann-windowed FFT with interpolation, accurate to
   1e-4 Hz), then resample to exactly 128 samples per cycle before the FFT. This is the
   synchronous window that IEC 61000-4-7 requires of PQ instruments. Harmonics recall went
   back to 100%.

2. **An unrealistic definition of "Normal".** In training, Normal was a perfect sine, so a real
   feeder's 1–3% background harmonics looked like a Notch. More than 60% of Normal waveforms were
   misclassified. **Fix:** include background distortion in the training waveforms. That's
   physically correct, since IEEE 519 allows 5–8% voltage THD. Normal recall went from 38% to 98%.

Neither fix alone is enough (93.0% and 94.0%). Together they reach 99.6%.

## Features (18)

- **RMS track:** one-cycle RMS sliding by half a cycle (IEC 61000-4-30 style). Min, max, std, and time below 0.9 / above 1.1 / below 0.1 pu. These separate sag, swell and interruption.
- **Harmonics:** THD plus the 3rd, 5th and 7th harmonics relative to the fundamental.
- **Residual:** the signal minus the fundamental and its harmonics. Peak, std, energy share in 250–1000 Hz (transients) and above 1 kHz (notches), kurtosis, and how concentrated it is in time (a localized transient vs repetitive notches).
- **Flicker:** envelope modulation in the 5–30 Hz band.
- **Max dV/dt.**

Most important: RMS min/max, 250–1000 Hz band energy, the 7th harmonic, and the energy above 1 kHz.

## Run

```bash
pip install -r requirements.txt
python pq.py
```

Takes about 30 s. A self-check runs first and asserts that a 1 pu sine reads 1.0 pu RMS, a 10%
3rd harmonic reads THD 0.10, the frequency estimate is within 0.001 Hz, and resampling removes
leakage. Full output is in `results.txt`.

## Limitations

- Waveforms are synthetic, from the standard IEEE 1159 parametric models. Next steps are field
  recordings (e.g. the IEEE PES PQ datasets) and a CNN on raw samples for comparison.
- It handles a single disturbance per record, plus one combined class (sag + harmonics).

## References

- IEEE Std 1159-2019, *Recommended Practice for Monitoring Electric Power Quality*
- IEC 61000-4-7, *Harmonics and interharmonics measurement*: synchronous window
- IEC 61000-4-30, *Power quality measurement methods*: RMS calculation
- IEEE Std 519-2022, *Harmonic Control in Electric Power Systems*
