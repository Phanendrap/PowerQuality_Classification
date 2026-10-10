"""Power quality disturbance classification with signal processing features + Random Forest.

Generates 60 Hz voltage waveforms for 9 disturbance types using the IEEE 1159 parametric
models used throughout the PQ literature, extracts RMS / FFT / residual features, trains a
Random Forest, and reports accuracy at different noise levels (SNR) plus a field-conditions
stress test (grid frequency drift, random phase, background harmonics).

    python pq.py               # generate, train, evaluate (~30 s)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

F0, FS, CYCLES = 60, 7680, 10           # 128 samples per cycle, 10 cycles per record
SPC = FS // F0
t = np.arange(CYCLES * SPC) / FS
CLASSES = ["Normal", "Sag", "Swell", "Interruption", "Harmonics", "Flicker",
           "Oscillatory transient", "Notch", "Sag + harmonics"]


def window(rng, min_cyc=1, max_cyc=9):
    """Random event window [t1, t2) lasting min_cyc..max_cyc cycles (IEEE 1159 short-duration range)."""
    dur = rng.uniform(min_cyc, max_cyc) / F0
    t1 = rng.uniform(0, t[-1] - dur)
    return ((t >= t1) & (t < t1 + dur)).astype(float), t1


def waveform(kind, rng, drift=False, background=False):
    """drift: random phase and +/-0.1 Hz grid frequency deviation.
    background: 0.5-2% 3rd/5th harmonics, present on any real feeder (well inside IEEE 519 limits)."""
    f, phi = (rng.uniform(59.9, 60.1), rng.uniform(0, 2 * np.pi)) if drift else (F0, 0.0)
    w = 2 * np.pi * f * t + phi
    base = np.sin(w)
    if background:
        base = base + rng.uniform(0.005, 0.02) * np.sin(3 * w) + rng.uniform(0.005, 0.02) * np.sin(5 * w)
    harmonics = lambda rng: sum(rng.uniform(0.05, 0.15) * np.sin(h * w) for h in (3, 5, 7))
    if kind == "Normal":
        return base
    if kind == "Sag":
        u, _ = window(rng)
        return (1 - rng.uniform(0.1, 0.9) * u) * base
    if kind == "Swell":
        u, _ = window(rng)
        return (1 + rng.uniform(0.1, 0.8) * u) * base
    if kind == "Interruption":
        u, _ = window(rng)
        return (1 - rng.uniform(0.9, 1.0) * u) * base
    if kind == "Harmonics":
        return base + harmonics(rng)
    if kind == "Flicker":
        return (1 + rng.uniform(0.08, 0.2) * np.sin(2 * np.pi * rng.uniform(8, 25) * t)) * base
    if kind == "Oscillatory transient":  # capacitor switching
        u, t1 = window(rng, 0.5, 3)
        fn, tau = rng.uniform(300, 900), rng.uniform(0.008, 0.04)
        return base + rng.uniform(0.1, 0.8) * np.exp(-(t - t1) / tau) * np.sin(2 * np.pi * fn * (t - t1)) * u
    if kind == "Notch":  # converter commutation, one notch per half cycle
        width = rng.uniform(0.02, 0.05) / F0
        start = rng.uniform(0, 0.5 / F0 - width)
        phase = (t - start) % (0.5 / F0)
        return base - np.sign(base) * rng.uniform(0.1, 0.4) * (phase < width)
    if kind == "Sag + harmonics":
        u, _ = window(rng)
        return (1 - rng.uniform(0.1, 0.9) * u) * (base + harmonics(rng))
    raise ValueError(kind)


def add_noise(x, snr_db, rng):
    return x + rng.normal(0, np.sqrt(np.mean(x**2) / 10 ** (snr_db / 10)), x.shape)


def rms_track(x):
    """One-cycle RMS sliding by half a cycle, normalized so a 1 pu sine reads 1.0 (IEC 61000-4-30 style)."""
    half = SPC // 2
    return np.array([np.sqrt(2 * np.mean(x[i:i + SPC] ** 2)) for i in range(0, len(x) - SPC + 1, half)])


def grid_frequency(x):
    """Fundamental frequency from a Hann-windowed FFT peak, interpolated between bins
    with the Hann-window formula (accurate to ~1e-4 Hz for +/-0.1 Hz drift)."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = CYCLES - 2 + np.argmax(X[CYCLES - 2:CYCLES + 3])  # search near 60 Hz
    a, b, c = X[k - 1:k + 2]
    f = (k + 2 * (c - a) / (a + 2 * b + c)) * FS / len(x)
    return np.clip(f, F0 - 0.5, F0 + 0.5)  # grid frequency stays within +/-0.5 Hz in normal operation


def synchronize(x):
    """Resample to exactly SPC samples per measured cycle over a whole number of cycles
    (IEC 61000-4-7 synchronous window), so harmonics land exactly on FFT bins."""
    f = grid_frequency(x)
    n_cyc = int(np.floor(len(x) / FS * f + 1e-9))
    return np.interp(np.arange(n_cyc * SPC) / (SPC * f), t, x), f


def features(x, sync=True):
    f = F0
    if sync:
        x, f = synchronize(x)
    r = rms_track(x)
    X = np.fft.rfft(x) / (len(x) / 2)
    mag = np.abs(X)
    k0 = len(x) // SPC                            # FFT bin of the fundamental (= number of cycles)
    h = mag[k0 * np.arange(2, 16)] / mag[k0]      # harmonics 2..15 relative to fundamental
    thd = np.sqrt(np.sum(h**2))
    # Residual: what is left after removing the fundamental and its harmonics
    keep = np.ones_like(mag, bool); keep[k0 * np.arange(1, 16)] = False; keep[0] = False
    resid = np.fft.irfft(np.where(keep, X, 0) * (len(x) / 2), n=len(x))
    freqs = np.fft.rfftfreq(len(x), 1 / (SPC * f))
    e_total = np.sum(mag[keep] ** 2) + 1e-12
    mid = np.sum(mag[keep & (freqs >= 250) & (freqs < 1000)] ** 2) / e_total
    high = np.sum(mag[keep & (freqs >= 1000)] ** 2) / e_total
    ra = np.abs(resid)
    peak_share = np.sort(ra)[-len(ra) // 10:].sum() / (ra.sum() + 1e-12)  # localized (transient) vs spread out
    kurt = np.mean(resid**4) / (np.mean(resid**2) ** 2 + 1e-12)
    # Envelope modulation in the flicker band (5-30 Hz)
    env = np.abs(x) - np.abs(x).mean()
    E = np.abs(np.fft.rfft(env))
    flicker = E[(freqs >= 5) & (freqs <= 30)].max() / (np.abs(x).mean() * len(x) / 2)
    return np.r_[r.min(), r.max(), r.std(), np.mean(r < 0.9), np.mean(r > 1.1),
                 np.mean(r < 0.1), thd, h[1], h[3], h[5], ra.max(), np.std(resid), mid, high,
                 peak_share, kurt, flicker, np.abs(np.diff(x)).max()]


FEATURES = ["rms_min", "rms_max", "rms_std", "frac_below_0.9", "frac_above_1.1", "frac_below_0.1",
            "THD", "h3", "h5", "h7", "resid_peak", "resid_std", "band_250_1000Hz", "band_above_1kHz",
            "resid_peak_share", "resid_kurtosis", "flicker_mod", "max_dv_dt"]


def signals(n_per_class, snr, rng, drift=False, background=False):
    x, y = [], []
    for c in CLASSES:
        for _ in range(n_per_class):
            s = snr if np.isscalar(snr) else rng.uniform(*snr)
            x.append(add_noise(waveform(c, rng, drift, background), s, rng))
            y.append(c)
    return x, np.array(y)


def feats(sigs, sync):
    return np.array([features(x, sync) for x in sigs])


def selfcheck():
    x = np.sin(2 * np.pi * F0 * t)
    f = dict(zip(FEATURES, features(x)))
    assert abs(f["rms_min"] - 1) < 1e-4 and abs(f["rms_max"] - 1) < 1e-4, "1 pu sine must read 1.0 pu RMS"
    assert f["THD"] < 1e-3, "pure sine has no harmonic distortion"
    x3 = x + 0.1 * np.sin(2 * np.pi * 3 * F0 * t)
    assert abs(features(x3)[FEATURES.index("THD")] - 0.1) < 1e-3, "10% 3rd harmonic -> THD 0.10"
    xd = np.sin(2 * np.pi * 60.1 * t + 1.0)
    assert abs(grid_frequency(xd) - 60.1) < 1e-3, "frequency estimate off"
    assert features(xd)[FEATURES.index("THD")] < 1e-3, "sync resampling should remove leakage"
    assert features(xd, sync=False)[FEATURES.index("THD")] > features(xd)[FEATURES.index("THD")]
    print("selfcheck ok")


def report(name, y, p):
    print()
    print(f"{name}: accuracy {np.mean(p == y):.1%}")
    print(classification_report(y, p, labels=CLASSES, digits=3))
    short = ["Norm", "Sag", "Swell", "Intr", "Harm", "Flick", "Trans", "Notch", "Sag+H"]
    print("Confusion matrix (rows = true, cols = predicted):")
    print("        " + " ".join(f"{s:>5}" for s in short))
    for s, row in zip(short, confusion_matrix(y, p, labels=CLASSES)):
        print(f"  {s:>5} " + " ".join(f"{v:5d}" for v in row))


def main():
    selfcheck()
    rng = np.random.default_rng(42)
    train = {False: signals(600, (20, 50), rng),                    # ideal sine as "Normal"
             True: signals(600, (20, 50), rng, background=True)}    # real-feeder background harmonics
    tests = {snr: signals(200, snr, rng) for snr in (50, 40, 30, 20)}
    st_x, yst = signals(200, 30, rng, drift=True, background=True)  # field conditions
    print(f"Training: {len(train[False][1])} waveforms, {len(CLASSES)} classes, SNR 20-50 dB, nominal 60 Hz")
    print("Stress test: SNR 30 dB + random phase, +/-0.1 Hz drift, 0.5-2% background harmonics")
    print("(frequency drift and phase are never seen in training)")
    print()

    configs = [("1. Plain FFT (assumes 60.000 Hz)", False, False),
               ("2. Synchronous resampling only", True, False),
               ("3. Background harmonics training only", False, True),
               ("4. Both", True, True)]
    print(f"{'Configuration':40s} " + " ".join(f"{s} dB" for s in tests) + "   stress test")
    preds = {}
    for name, sync, bg in configs:
        tr_x, ytr = train[bg]
        clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1).fit(feats(tr_x, sync), ytr)
        accs = [np.mean(clf.predict(feats(x, sync)) == y) for x, y in tests.values()]
        preds[name] = p = clf.predict(feats(st_x, sync))
        print(f"{name:40s} " + " ".join(f"{a:5.1%}" for a in accs) + f"   {np.mean(p == yst):6.1%}")
    for name, p in preds.items():
        report(f"Stress test, {name}", yst, p)
    imp = sorted(zip(clf.feature_importances_, FEATURES), reverse=True)[:5]
    print()
    print("Top features:",", ".join(f"{n} ({v:.2f})" for v, n in imp))


if __name__ == "__main__":
    main()
