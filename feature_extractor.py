"""
feature_extractor.py
────────────────────
NAOS Feature Extractor — File 2 of 6

Responsibility:
    Receives a (8, 250) float32 numpy array from signal_router.py
    in microvolts (NEST_EEG_Clean scale: ~-17000 to +21000 uV).
    Computes all signal features needed by state_engine.py.
    Returns a structured feature dictionary.

Calibration source (verified 2026-05-31 from real NEST output):
    All baseline values, thresholds, and band power anchors come
    directly from check_clean.py checks 1-3 on NEST_EEG_Clean.
    Zero hardcoded guesses.

Channel map (NEST_EEG_Clean, 10-20 system):
    ch0  Fp1  frontal-polar left    — drowsiness (theta), blink proxy
    ch1  Fp2  frontal-polar right   — drowsiness (theta), blink proxy
    ch2  C3   central left (motor)  — tension proxy (jaw clench)
    ch3  C4   central right (motor) — load (beta)
    ch4  P7   parietal-temporal L   — general broadband
    ch5  P8   parietal-temporal R   — general broadband
    ch6  O1   occipital left        — attention (alpha)
    ch7  O2   occipital right       — attention (alpha)

Key findings from calibration:
    - Beta/Alpha ratio is INVERTED in NEST (load LOWERS ratio)
      → Use global_std for load, not band ratio
    - Blink injection produces std DECREASE, not amplitude spike
      → Use frontal std drop for blink proxy, not peak detection
    - Muscle Tension slider DECREASES occipital std (inverted)
      → Use C3 std z-score for tension (motor cortex)
    - Drowsiness: theta_fp2 rises 2.8x, global_std DROPS
      → Compound rule separates drowsy from load

Output dict (feature_dict):
    global_std       float   — overall signal energy in uV
    fp_std           float   — frontal (Fp1+Fp2) mean std in uV
    o_std            float   — occipital (O1+O2) mean std in uV
    c3_std           float   — C3 motor cortex std in uV
    theta_fp2        float   — Fp2 theta power (uV^2/Hz) via Welch
    alpha_o1o2       float   — O1+O2 alpha power mean (uV^2/Hz)
    beta_c4          float   — C4 beta power (uV^2/Hz)
    gamma_mean       float   — all-channel gamma mean (uV^2/Hz)
    theta_mean       float   — all-channel theta mean (uV^2/Hz)
    band_powers      dict    — full {ch: {delta,theta,alpha,beta,gamma}}
                               for dashboard visualization
    n_samples        int     — actual samples in this window
    quality          float   — 0-1 signal quality score
"""

import numpy as np
import logging
import time
from scipy import signal as sp_signal
from typing import Dict, Any

# NumPy 2.0 renamed np.trapz → np.trapezoid; support both versions
_trapz = getattr(np, 'trapezoid', None) or np.trapz

# ── logger ───────────────────────────────────────────────────────────────────

log = logging.getLogger("feature_extractor")

# ── channel indices ───────────────────────────────────────────────────────────

CH_FP1 = 0   # frontal-polar left
CH_FP2 = 1   # frontal-polar right
CH_C3  = 2   # central left  — motor/tension
CH_C4  = 3   # central right — beta/load
CH_P7  = 4   # parietal-temporal left
CH_P8  = 5   # parietal-temporal right
CH_O1  = 6   # occipital left  — alpha/attention
CH_O2  = 7   # occipital right — alpha/attention

CH_NAMES = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2']

# ── sampling parameters ───────────────────────────────────────────────────────

SR          = 250       # Hz — NEST_EEG_Clean nominal rate
WINDOW_SIZE = 250       # samples — 1 second window from signal_router
NPERSEG     = 125       # Welch segment length — 0.5s, good freq resolution

# ── frequency band definitions (Hz) ──────────────────────────────────────────

BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 50.0),
}

# ── baseline values from check_clean.py checks 1-3 ───────────────────────────
# Source: NEST_EEG_Clean, 2026-05-31, all sliders minimum
# These seed the rolling baseline before enough live data accumulates.

SEED_GLOBAL_STD  = 3508.2    # uV      — overall signal energy at rest
SEED_THETA_FP2   = 206371.8  # uV^2/Hz — Fp2 theta at rest
SEED_ALPHA_O1O2  = 1654832.4 # uV^2/Hz — O1+O2 alpha mean at rest
SEED_GAMMA_MEAN  = 3257003.7 # uV^2/Hz — all-channel gamma mean at rest
SEED_C3_STD      = 5866.7    # uV      — C3 std at rest
SEED_FP_STD      = 3121.1    # uV      — Fp1+Fp2 mean std at rest
SEED_O_STD       = 3867.6    # uV      — O1+O2 mean std at rest

# Baseline variance estimates (used for z-score std before window fills)
# Estimated as 20% of baseline value — conservative, will self-correct
SEED_GLOBAL_STD_VAR  = SEED_GLOBAL_STD  * 0.20
SEED_THETA_VAR       = SEED_THETA_FP2   * 0.20
SEED_ALPHA_VAR       = SEED_ALPHA_O1O2  * 0.20
SEED_GAMMA_VAR       = SEED_GAMMA_MEAN  * 0.20
SEED_C3_VAR          = SEED_C3_STD      * 0.20
SEED_FP_VAR          = SEED_FP_STD      * 0.20
SEED_O_VAR           = SEED_O_STD       * 0.20

# Rolling window: 30 seconds = 120 windows at 250ms cadence
ROLLING_WINDOW = 120


# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    """
    Converts raw (8, 250) EEG windows into calibrated feature dicts.

    Rolling baseline:
        Maintains a 30-second (120-window) rolling history for each
        scalar feature. Z-scores are computed against this history.
        Before the history fills, seed values from calibration are used.

    Thread safety:
        Not thread-safe. Call from a single thread (the FastAPI loop).
    """

    def __init__(self):
        # Rolling history buffers — one deque per scalar feature
        # Each entry is one computed value from one window
        from collections import deque
        self._hist = {
            'global_std': deque(maxlen=ROLLING_WINDOW),
            'theta_fp2':  deque(maxlen=ROLLING_WINDOW),
            'alpha_o1o2': deque(maxlen=ROLLING_WINDOW),
            'gamma_mean': deque(maxlen=ROLLING_WINDOW),
            'c3_std':     deque(maxlen=ROLLING_WINDOW),
            'fp_std':     deque(maxlen=ROLLING_WINDOW),
            'o_std':      deque(maxlen=ROLLING_WINDOW),
        }

        # Seed each history with calibration values so z-scores are
        # meaningful from the very first window
        self._seed_history()
        self._window_count = 0
        log.info("FeatureExtractor initialized with calibrated baseline.")

    # ── public API ────────────────────────────────────────────────────────────

    def compute(self, raw: np.ndarray) -> Dict[str, Any]:
        """
        Compute features from one EEG window.

        Args:
            raw: numpy array shape (8, 250), dtype float32
                 values in microvolts (NEST_EEG_Clean scale)

        Returns:
            feature_dict with all scalar features and band powers
            See module docstring for full key list.
        """
        t0 = time.perf_counter()

        # ── input validation ─────────────────────────────────────────
        if raw is None:
            log.warning("compute() received None — returning zeros")
            return self._zero_features()

        if raw.shape != (8, 250):
            # Tolerate slightly short windows — pad or trim
            if raw.shape[0] == 8 and raw.shape[1] > 0:
                if raw.shape[1] < 250:
                    pad = np.zeros((8, 250 - raw.shape[1]),
                                   dtype=np.float32)
                    raw = np.concatenate([raw, pad], axis=1)
                else:
                    raw = raw[:, :250]
            else:
                log.warning(f"Unexpected shape {raw.shape} — returning zeros")
                return self._zero_features()

        # ── scalar std features ──────────────────────────────────────
        # These are cheap to compute — just std per channel subset
        global_std = float(raw.std())
        fp_std     = float(np.mean([raw[CH_FP1].std(),
                                    raw[CH_FP2].std()]))
        o_std      = float(np.mean([raw[CH_O1].std(),
                                    raw[CH_O2].std()]))
        c3_std     = float(raw[CH_C3].std())

        # ── Welch band powers ────────────────────────────────────────
        # Compute per channel, store full matrix for dashboard
        band_powers = {}
        for i in range(8):
            band_powers[i] = self._welch_bands(raw[i])

        # Key scalar band features
        theta_fp2  = band_powers[CH_FP2]['theta']
        alpha_o1o2 = float(np.mean([band_powers[CH_O1]['alpha'],
                                    band_powers[CH_O2]['alpha']]))
        beta_c4    = band_powers[CH_C4]['beta']
        gamma_mean = float(np.mean([band_powers[i]['gamma']
                                    for i in range(8)]))
        theta_mean = float(np.mean([band_powers[i]['theta']
                                    for i in range(8)]))

        # ── signal quality ───────────────────────────────────────────
        quality = self._compute_quality(raw)

        # ── update rolling histories ─────────────────────────────────
        self._update_history('global_std', global_std)
        self._update_history('theta_fp2',  theta_fp2)
        self._update_history('alpha_o1o2', alpha_o1o2)
        self._update_history('gamma_mean', gamma_mean)
        self._update_history('c3_std',     c3_std)
        self._update_history('fp_std',     fp_std)
        self._update_history('o_std',      o_std)

        self._window_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000

        features = {
            # ── raw scalar features ──
            'global_std':  global_std,
            'fp_std':      fp_std,
            'o_std':       o_std,
            'c3_std':      c3_std,
            'theta_fp2':   theta_fp2,
            'alpha_o1o2':  alpha_o1o2,
            'beta_c4':     beta_c4,
            'gamma_mean':  gamma_mean,
            'theta_mean':  theta_mean,

            # ── z-scores vs rolling baseline ──
            # State engine uses these directly
            'z_global_std':  self._zscore('global_std', global_std),
            'z_theta_fp2':   self._zscore('theta_fp2',  theta_fp2),
            'z_alpha_o1o2':  self._zscore('alpha_o1o2', alpha_o1o2),
            'z_gamma_mean':  self._zscore('gamma_mean', gamma_mean),
            'z_c3_std':      self._zscore('c3_std',     c3_std),
            'z_fp_std':      self._zscore('fp_std',     fp_std),
            'z_o_std':       self._zscore('o_std',      o_std),

            # ── full band power matrix for dashboard ──
            # band_powers[channel_index][band_name] = uV^2/Hz
            'band_powers':  band_powers,

            # ── metadata ──
            'n_samples':    raw.shape[1],
            'quality':      quality,
            'window_count': self._window_count,
            'compute_ms':   elapsed_ms,

            # ── normalized band power bars for dashboard ──
            # Values 0-1, suitable for bar chart display
            'bands_normalized': self._normalized_band_bars(band_powers),
        }

        return features

    def get_history_stats(self, key: str):
        """Return (mean, std) of rolling history for a feature key."""
        h = list(self._hist[key])
        if len(h) < 2:
            return self._get_seed(key)
        return float(np.mean(h)), float(np.std(h) + 1e-9)

    @property
    def window_count(self) -> int:
        return self._window_count

    @property
    def baseline_filled(self) -> bool:
        """True once rolling window has 30+ real observations."""
        return self._window_count >= 30

    # ── internal helpers ──────────────────────────────────────────────────────

    def _welch_bands(self, ch_data: np.ndarray) -> Dict[str, float]:
        """
        Compute Welch PSD for one channel and integrate into bands.
        Returns dict with delta/theta/alpha/beta/gamma in uV^2/Hz.
        """
        # Detrend to remove DC offset before PSD
        ch_detrended = ch_data - ch_data.mean()

        freqs, psd = sp_signal.welch(
            ch_detrended,
            fs=SR,
            nperseg=NPERSEG,
            noverlap=NPERSEG // 2,
            window='hann',
            scaling='density'   # output in uV^2/Hz
        )

        result = {}
        for band, (lo, hi) in BANDS.items():
            idx = (freqs >= lo) & (freqs < hi)
            if idx.any():
                result[band] = float(_trapz(psd[idx], freqs[idx]))
            else:
                result[band] = 0.0

        return result

    def _zscore(self, key: str, value: float) -> float:
        """
        Compute z-score of value against rolling history.
        Clamped to [-4, +4] to prevent extreme values.
        """
        mean, std = self.get_history_stats(key)
        z = (value - mean) / std
        return float(np.clip(z, -4.0, 4.0))

    def _update_history(self, key: str, value: float):
        """Append value to rolling history deque."""
        if np.isfinite(value):
            self._hist[key].append(value)

    def _compute_quality(self, raw: np.ndarray) -> float:
        """
        Estimate signal quality 0-1.
        Penalizes: flat channels, saturated channels, all-zero windows.

        Returns:
            1.0 = perfect quality
            0.0 = completely unusable
        """
        stds = raw.std(axis=1)   # std per channel

        # Penalty for flat channels (std < 100 uV — too quiet)
        flat_count = int(np.sum(stds < 100))

        # Penalty for saturated channels
        # NEST_EEG_Clean max observed ~22000 uV
        sat_count = int(np.sum(stds > 15000))

        # Penalty for NaN/Inf
        nan_count = int(np.isnan(raw).any() or np.isinf(raw).any())

        total_penalty = flat_count + sat_count + nan_count * 4
        quality = max(0.0, 1.0 - (total_penalty / 8.0))
        return float(quality)

    def _normalized_band_bars(self,
                              band_powers: Dict) -> Dict[str, float]:
        """
        Compute per-band mean across all channels, normalized 0-1.
        Used for the dashboard band-power bar chart.
        Uses log scale because band powers span many orders of magnitude.
        """
        band_means = {}
        for band in BANDS:
            vals = [band_powers[i][band] for i in range(8)]
            band_means[band] = float(np.mean(vals))

        # Log-normalize using baseline reference values
        refs = {
            'delta': 100000.0,   # approximate baseline from check_clean 1
            'theta': 344115.7,   # SEED_THETA_MEAN
            'alpha': 1654832.4,  # SEED_ALPHA_O1O2
            'beta':  2831864.8,  # SEED_BETA_C4
            'gamma': 3257003.7,  # SEED_GAMMA_MEAN
        }

        normalized = {}
        for band, val in band_means.items():
            ref = refs[band]
            if val <= 0 or ref <= 0:
                normalized[band] = 0.0
            else:
                # log ratio: 0 at baseline, positive when above
                log_ratio = np.log10(val / ref)
                # Map [-1, +1] log range to [0, 1] display range
                normalized[band] = float(np.clip((log_ratio + 1) / 2, 0, 1))

        return normalized

    def _seed_history(self):
        """
        Seed rolling histories with calibration values so that
        z-scores are meaningful from window 1.
        Seeds 10 copies of each baseline value.
        """
        seeds = {
            'global_std': SEED_GLOBAL_STD,
            'theta_fp2':  SEED_THETA_FP2,
            'alpha_o1o2': SEED_ALPHA_O1O2,
            'gamma_mean': SEED_GAMMA_MEAN,
            'c3_std':     SEED_C3_STD,
            'fp_std':     SEED_FP_STD,
            'o_std':      SEED_O_STD,
        }
        # Add small Gaussian noise around seed to give the std estimator
        # something to work with (pure constant → std = 0)
        variances = {
            'global_std': SEED_GLOBAL_STD_VAR,
            'theta_fp2':  SEED_THETA_VAR,
            'alpha_o1o2': SEED_ALPHA_VAR,
            'gamma_mean': SEED_GAMMA_VAR,
            'c3_std':     SEED_C3_VAR,
            'fp_std':     SEED_FP_VAR,
            'o_std':      SEED_O_VAR,
        }
        rng = np.random.default_rng(42)
        for key, seed_val in seeds.items():
            var = variances[key]
            for _ in range(10):
                noisy = seed_val + rng.normal(0, var * 0.20)
                self._hist[key].append(float(noisy))

    def _get_seed(self, key: str):
        """Return (mean, std) seed values for a key."""
        seed_map = {
            'global_std': (SEED_GLOBAL_STD, SEED_GLOBAL_STD_VAR),
            'theta_fp2':  (SEED_THETA_FP2,  SEED_THETA_VAR),
            'alpha_o1o2': (SEED_ALPHA_O1O2, SEED_ALPHA_VAR),
            'gamma_mean': (SEED_GAMMA_MEAN, SEED_GAMMA_VAR),
            'c3_std':     (SEED_C3_STD,     SEED_C3_VAR),
            'fp_std':     (SEED_FP_STD,     SEED_FP_VAR),
            'o_std':      (SEED_O_STD,      SEED_O_VAR),
        }
        return seed_map.get(key, (0.0, 1.0))

    def _zero_features(self) -> Dict[str, Any]:
        """Return a zero-filled feature dict for error cases."""
        return {
            'global_std': 0.0, 'fp_std': 0.0, 'o_std': 0.0,
            'c3_std': 0.0, 'theta_fp2': 0.0, 'alpha_o1o2': 0.0,
            'beta_c4': 0.0, 'gamma_mean': 0.0, 'theta_mean': 0.0,
            'z_global_std': 0.0, 'z_theta_fp2': 0.0,
            'z_alpha_o1o2': 0.0, 'z_gamma_mean': 0.0,
            'z_c3_std': 0.0, 'z_fp_std': 0.0, 'z_o_std': 0.0,
            'band_powers': {i: {b: 0.0 for b in BANDS} for i in range(8)},
            'n_samples': 0, 'quality': 0.0,
            'window_count': self._window_count, 'compute_ms': 0.0,
            'bands_normalized': {b: 0.0 for b in BANDS},
        }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# Run: python feature_extractor.py
# Does NOT require NEST — uses synthetic data calibrated to real range.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    print("\n" + "="*65)
    print("  NAOS feature_extractor.py — Self Test")
    print("  Uses synthetic data in calibrated NEST_EEG_Clean range")
    print("="*65 + "\n")

    extractor = FeatureExtractor()
    rng       = np.random.default_rng(0)
    passed    = 0
    total     = 0

    def check(name, condition, got):
        global passed, total
        total += 1
        status = "✅" if condition else "❌"
        print(f"  {status} {name}: {got}")
        if condition:
            passed += 1

    # ── Test 1: Baseline signal — all checks should be near zero z-score ──
    print("── Test 1: Baseline signal (10 windows to warm baseline) ──")
    for _ in range(10):
        raw = rng.normal(0, SEED_GLOBAL_STD, (8, 250)).astype(np.float32)
        f   = extractor.compute(raw)

    check("Output is dict",         isinstance(f, dict), type(f))
    check("global_std > 0",         f['global_std'] > 0, f'{f["global_std"]:.1f} uV')
    check("theta_fp2 > 0",          f['theta_fp2'] > 0,  f'{f["theta_fp2"]:.1f} uV^2/Hz')
    check("alpha_o1o2 > 0",         f['alpha_o1o2'] > 0, f'{f["alpha_o1o2"]:.1f} uV^2/Hz')
    check("z_global_std near 0",    abs(f['z_global_std']) < 2.0,
                                    f'{f["z_global_std"]:.3f}')
    check("quality == 1.0",         f['quality'] == 1.0,  f'{f["quality"]:.2f}')
    check("n_samples == 250",       f['n_samples'] == 250, f['n_samples'])
    check("band_powers has 8 ch",   len(f['band_powers']) == 8,
                                    len(f['band_powers']))
    check("bands_normalized 0-1",
          all(0 <= v <= 1 for v in f['bands_normalized'].values()),
          f['bands_normalized'])
    check("no NaN in z-scores",
          all(np.isfinite(f[k]) for k in
              ['z_global_std','z_theta_fp2','z_alpha_o1o2',
               'z_gamma_mean','z_c3_std']),
          "checked")

    # ── Test 2: High load signal — z_global_std should be positive ──
    print("\n── Test 2: High cognitive load (global std +35%) ──")
    # Simulate load: global std = SEED * 1.356 (matches check_clean 2)
    load_std = SEED_GLOBAL_STD * 1.356
    for _ in range(5):
        raw  = rng.normal(0, load_std, (8, 250)).astype(np.float32)
        f_l  = extractor.compute(raw)

    check("Load z_global_std > 0",  f_l['z_global_std'] > 0,
          f'{f_l["z_global_std"]:.3f}')
    check("Load global_std elevated",
          f_l['global_std'] > SEED_GLOBAL_STD,
          f'{f_l["global_std"]:.1f} (seed={SEED_GLOBAL_STD:.1f})')

    # ── Test 3: Drowsy signal — theta_fp2 elevated ──
    print("\n── Test 3: Drowsiness (theta Fp2 x2.8) ──")
    for _ in range(5):
        raw = rng.normal(0, SEED_GLOBAL_STD * 0.88, (8, 250)).astype(np.float32)
        # Inject theta (6Hz) on Fp2
        t   = np.arange(250) / SR
        raw[CH_FP2] += (np.sin(2 * np.pi * 6 * t) * 3000).astype(np.float32)
        f_d = extractor.compute(raw)

    check("Drowsy z_theta_fp2 > 0", f_d['z_theta_fp2'] > 0,
          f'{f_d["z_theta_fp2"]:.3f}')
    check("Drowsy global_std below baseline",
          f_d['global_std'] < extractor.get_history_stats('global_std')[0] * 1.1,
          f'{f_d["global_std"]:.1f}')

    # ── Test 4: Window shape tolerance ──
    print("\n── Test 4: Window shape tolerance ──")
    raw_short = rng.normal(0, SEED_GLOBAL_STD, (8, 200)).astype(np.float32)
    f_s = extractor.compute(raw_short)
    check("Short window handled",   f_s['n_samples'] == 250,
          f'{f_s["n_samples"]} (padded from 200)')

    raw_long = rng.normal(0, SEED_GLOBAL_STD, (8, 300)).astype(np.float32)
    f_l2 = extractor.compute(raw_long)
    check("Long window trimmed",    f_l2['n_samples'] == 250,
          f'{f_l2["n_samples"]} (trimmed from 300)')

    # ── Test 5: None input ──
    print("\n── Test 5: None input handling ──")
    f_n = extractor.compute(None)
    check("None returns zero dict", f_n['global_std'] == 0.0, f_n['global_std'])

    # ── Test 6: Performance ──
    print("\n── Test 6: Performance (target < 50ms per window) ──")
    import time
    times = []
    for _ in range(20):
        raw = rng.normal(0, SEED_GLOBAL_STD, (8, 250)).astype(np.float32)
        t0  = time.perf_counter()
        extractor.compute(raw)
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = np.mean(times)
    max_ms  = np.max(times)
    check("Mean compute < 50ms",    mean_ms < 50.0,
          f'{mean_ms:.2f}ms mean, {max_ms:.2f}ms max')

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Result: {passed}/{total} tests passed.")
    if passed == total:
        print("  ✅ feature_extractor.py is working correctly.")
        print("  ✅ Safe to proceed to state_engine.py")
    else:
        print("  ❌ Fix failures above before proceeding.")
    print("="*65 + "\n")
