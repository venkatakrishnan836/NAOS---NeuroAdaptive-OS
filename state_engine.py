"""
state_engine.py
───────────────
NAOS State Engine — File 3 of 6

Responsibility:
    Receives feature_dict from feature_extractor.py.
    Produces state_dict with five values in [0, 1]:
        attention, load, tension, drowsiness, confidence

Calibration basis (check_clean.py on NEST_EEG_Clean 2026-05-31):
    load      <- z_global_std    (+35.6% at max cognitive load)
    drowsiness<- z_theta_fp2 AND inverted z_global_std
                 (theta +178.7% drowsy, global_std -12% drowsy)
    attention <- z_o_std         (+68.2% at max attention/focus)
    tension   <- z_c3_std + z_gamma_mean  (+35% at jaw clench)

Formula: sigmoid(z, k) = 1 / (1 + exp(-k * z))
    z=0 (baseline) -> 0.500, z=+2 -> 0.880, z=-2 -> 0.119

Smoothing: EMA alpha=0.35 (~2s response time)
"""

import numpy as np
import time
import logging
from typing import Dict, Any

log = logging.getLogger("state_engine")

K_LOAD       = 1.5
K_DROWSY     = 1.0
K_ATTENTION  = 1.0
K_TENSION    = 1.2

W_DROWSY_THETA  = 0.6
W_DROWSY_QUIET  = 0.4
W_TENSION_C3    = 0.7
W_TENSION_GAMMA = 0.3

EMA_ALPHA      = 0.35
WARMUP_WINDOWS = 30


class StateEngine:
    def __init__(self):
        self._ema = {
            'attention':  0.5,
            'load':       0.5,
            'tension':    0.5,
            'drowsiness': 0.5,
        }
        self._window_count = 0
        log.info("StateEngine initialized.")

    def compute(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not features:
            return self._neutral_state()

        self._window_count += 1

        zg  = features.get('z_global_std',  0.0)
        zt  = features.get('z_theta_fp2',   0.0)
        zc3 = features.get('z_c3_std',      0.0)
        zo  = features.get('z_o_std',       0.0)
        zga = features.get('z_gamma_mean',  0.0)
        q   = features.get('quality',       1.0)

        raw_load      = self._sigmoid(zg, K_LOAD)
        drowsy_theta  = self._sigmoid(zt,  K_DROWSY)
        drowsy_quiet  = self._sigmoid(-zg, K_DROWSY)
        raw_drowsy    = (W_DROWSY_THETA * drowsy_theta +
                         W_DROWSY_QUIET * drowsy_quiet)
        raw_attention = self._sigmoid(zo, K_ATTENTION)
        raw_tension   = (W_TENSION_C3    * self._sigmoid(zc3, K_TENSION) +
                         W_TENSION_GAMMA * self._sigmoid(zga, K_TENSION))

        self._ema['load']       = self._ema_update('load',       raw_load)
        self._ema['drowsiness'] = self._ema_update('drowsiness', raw_drowsy)
        self._ema['attention']  = self._ema_update('attention',  raw_attention)
        self._ema['tension']    = self._ema_update('tension',    raw_tension)

        warmup     = min(self._window_count / WARMUP_WINDOWS, 1.0)
        confidence = float(q * warmup)

        return {
            'attention':   round(self._ema['attention'],  4),
            'load':        round(self._ema['load'],       4),
            'tension':     round(self._ema['tension'],    4),
            'drowsiness':  round(self._ema['drowsiness'], 4),
            'confidence':  round(confidence,              4),
            'raw': {
                'attention':  round(raw_attention, 4),
                'load':       round(raw_load,      4),
                'tension':    round(raw_tension,   4),
                'drowsiness': round(raw_drowsy,    4),
            },
            'window_count': self._window_count,
            'timestamp':    time.time(),
            'warmed_up':    self._window_count >= WARMUP_WINDOWS,
        }

    def reset_ema(self):
        for key in self._ema:
            self._ema[key] = 0.5
        log.info("StateEngine EMA reset to neutral.")

    @property
    def window_count(self) -> int:
        return self._window_count

    @staticmethod
    def _sigmoid(z: float, k: float = 1.0) -> float:
        z_clamped = float(np.clip(z, -10.0, 10.0))
        return float(1.0 / (1.0 + np.exp(-k * z_clamped)))

    def _ema_update(self, key: str, new_val: float) -> float:
        old = self._ema[key]
        new = EMA_ALPHA * new_val + (1.0 - EMA_ALPHA) * old
        return float(np.clip(new, 0.0, 1.0))

    def _neutral_state(self) -> Dict[str, Any]:
        return {
            'attention': 0.5, 'load': 0.5,
            'tension': 0.5,   'drowsiness': 0.5,
            'confidence': 0.0,
            'raw': {'attention': 0.5, 'load': 0.5,
                    'tension': 0.5,   'drowsiness': 0.5},
            'window_count': self._window_count,
            'timestamp': time.time(),
            'warmed_up': False,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    print("\n" + "="*65)
    print("  NAOS state_engine.py — Self Test")
    print("="*65 + "\n")

    engine = StateEngine()
    passed = 0; total = 0

    def check(name, condition, got=""):
        global passed, total
        total += 1
        ok = "✅" if condition else "❌"
        print(f"  {ok} {name}" + (f": {got}" if got else ""))
        if condition: passed += 1

    def F(zg=0.0, zt=0.0, zc3=0.0, zo=0.0, zga=0.0, q=1.0):
        return {'z_global_std': zg, 'z_theta_fp2': zt, 'z_c3_std': zc3,
                'z_o_std': zo, 'z_gamma_mean': zga, 'quality': q}

    # Test 1: Neutral baseline
    print("── Test 1: Neutral baseline ──")
    for _ in range(30): s = engine.compute(F())
    check("All scores 0.40-0.60",
          all(0.40 <= s[k] <= 0.60 for k in ['attention','load','tension','drowsiness']),
          {k: round(s[k],3) for k in ['attention','load','tension','drowsiness']})
    check("confidence == 1.0", s['confidence'] == 1.0, s['confidence'])
    check("warmed_up True", s['warmed_up'])
    check("all keys present",
          all(k in s for k in ['attention','load','tension','drowsiness',
                                'confidence','raw','window_count','timestamp','warmed_up']))

    # Test 2: High load
    print("\n── Test 2: High load (z_global=+2.0) ──")
    for _ in range(8): s_l = engine.compute(F(zg=2.0))
    check("Load > 0.65", s_l['load'] > 0.65, f"load={s_l['load']:.3f}")
    check("Attention not falsely elevated", s_l['attention'] < 0.70,
          f"attention={s_l['attention']:.3f}")

    # Test 3: Drowsiness — theta up, global_std down
    print("\n── Test 3: Drowsiness (z_theta=+2.5, z_global=-0.5) ──")
    engine.reset_ema()
    for _ in range(30): engine.compute(F())
    for _ in range(8): s_d = engine.compute(F(zt=2.5, zg=-0.5))
    check("Drowsiness > 0.65", s_d['drowsiness'] > 0.65,
          f"drowsiness={s_d['drowsiness']:.3f}")
    check("Load NOT elevated (global low)", s_d['load'] < 0.55,
          f"load={s_d['load']:.3f}")

    # Test 4: Tension
    print("\n── Test 4: Tension (z_c3=+2.0, z_gamma=+1.5) ──")
    engine.reset_ema()
    for _ in range(30): engine.compute(F())
    for _ in range(8): s_t = engine.compute(F(zc3=2.0, zga=1.5))
    check("Tension > 0.70", s_t['tension'] > 0.70,
          f"tension={s_t['tension']:.3f}")

    # Test 5: Load vs Drowsy disambiguation
    print("\n── Test 5: Load vs Drowsy disambiguation ──")
    engine.reset_ema()
    for _ in range(30): engine.compute(F())
    for _ in range(8): sl = engine.compute(F(zg=2.0, zt=2.0))
    engine.reset_ema()
    for _ in range(30): engine.compute(F())
    for _ in range(8): sd = engine.compute(F(zg=-0.5, zt=2.0))
    check("Load scenario: load > drowsiness",
          sl['load'] > sl['drowsiness'],
          f"load={sl['load']:.3f} drowsy={sl['drowsiness']:.3f}")
    check("Drowsy scenario: drowsiness > load",
          sd['drowsiness'] > sd['load'],
          f"drowsy={sd['drowsiness']:.3f} load={sd['load']:.3f}")

    # Test 6: EMA smoothing
    print("\n── Test 6: EMA smoothing ──")
    engine.reset_ema()
    for _ in range(30): engine.compute(F())
    s_spike = engine.compute(F(zg=4.0))
    check("Single spike load < 0.90", s_spike['load'] < 0.90,
          f"load={s_spike['load']:.3f}")
    for _ in range(7): s_sus = engine.compute(F(zg=4.0))
    check("Sustained signal load > 0.85", s_sus['load'] > 0.85,
          f"load={s_sus['load']:.3f}")

    # Test 7: Confidence with low quality
    print("\n── Test 7: Low quality signal ──")
    engine.reset_ema()
    for _ in range(30): engine.compute(F())
    s_lq = engine.compute(F(zg=2.0, q=0.3))
    check("Confidence < 0.40 when quality=0.3",
          s_lq['confidence'] < 0.40, f"confidence={s_lq['confidence']:.3f}")

    # Test 8: None / empty
    print("\n── Test 8: None / empty features ──")
    s_none = engine.compute(None)
    check("None returns neutral", s_none['load'] == 0.5 and
          s_none['confidence'] == 0.0, str(s_none['load']))
    s_empty = engine.compute({})
    check("Empty dict returns neutral", s_empty['load'] == 0.5)

    # Test 9: Range invariant
    print("\n── Test 9: All outputs always in [0,1] ──")
    engine.reset_ema()
    violations = 0
    for z in [-5, -3, -1, 0, 1, 3, 5]:
        for _ in range(3):
            s = engine.compute(F(zg=z, zt=z, zc3=z, zo=z, zga=z))
            for key in ['attention','load','tension','drowsiness','confidence']:
                if not (0.0 <= s[key] <= 1.0):
                    violations += 1
    check("Zero violations across extreme z-scores", violations == 0,
          f"{violations} violations")

    # Test 10: Performance
    print("\n── Test 10: Performance ──")
    import time as _t
    times = []
    for _ in range(100):
        t0 = _t.perf_counter()
        engine.compute(F(zg=1.0, zt=0.5))
        times.append((_t.perf_counter() - t0) * 1000)
    mean_ms = np.mean(times)
    check("Mean compute < 1ms", mean_ms < 1.0,
          f"{mean_ms:.3f}ms mean, {np.max(times):.3f}ms max")

    print(f"\n{'='*65}")
    print(f"  Result: {passed}/{total} tests passed.")
    if passed == total:
        print("  ✅ state_engine.py is working correctly.")
        print("  ✅ Safe to proceed to detectors.py")
    else:
        print("  ❌ Fix failures before proceeding.")
    print("="*65 + "\n")
