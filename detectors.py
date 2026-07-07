"""
detectors.py
────────────
NAOS Detectors — File 4 of 6

Responsibility:
    Three independent detectors running on every state_dict.
    Each detector returns a list of events (usually empty).
    Events are consumed by server.py and forwarded to the dashboard.

Detectors:
    DrowsinessDetector  — Feature #8
        Fires when drowsiness > 0.65 AND load < 0.55
        sustained for 1.5 seconds (6 windows).
        Compound rule separates drowsy from cognitive load.
        30s cooldown after firing.

    EmergencyDetector   — Feature #7
        Sequential pattern — step order matters:
        Step 1: tension > 0.80
        Step 2: attention < 0.25 within 10s of Step 1
        Step 3: behavioral silence > 3.0s after Step 2
        Resets if 10s window expires without completing.

    LLMTrigger          — Feature #12
        Fires when load > 0.70 sustained 3.0s (12 windows).
        30s cooldown. Single event per trigger.

    ThemeRule           — Feature #13
        Stateless. Returns theme_change event whenever
        load crosses 0.70 or 0.30 boundary.
        Tracks last theme to avoid redundant events.

Behavioral marker integration:
    update_marker(marker_string) must be called whenever
    NEST_Behavioral_Markers receives a sample.
    EmergencyDetector uses last_marker_time for silence detection.

Event format:
    {
        'type':      str,   # event type name
        'timestamp': float, # time.time()
        'data':      dict,  # event-specific payload
    }

Calibration basis (NEST_EEG_Clean 2026-05-31):
    All thresholds derived from check_clean + verify_controls data.
    Zero hardcoded guesses.
"""

import time
import logging
from typing import List, Dict, Any, Optional

log = logging.getLogger("detectors")

# ── thresholds ────────────────────────────────────────────────────────────────

# Drowsiness
DROWN_THRESH        = 0.65   # drowsiness score
DROWN_LOAD_MAX      = 0.55   # load must be BELOW this (disambiguate)
DROWN_SUSTAIN_WIN   = 6      # consecutive windows required
DROWN_COOLDOWN_S    = 30.0   # seconds before can fire again

# Emergency (sequential)
EMERG_TENSION_THRESH  = 0.80   # Step 1 trigger
EMERG_ATTN_THRESH     = 0.25   # Step 2 trigger
EMERG_WINDOW_S        = 10.0   # whole sequence must complete in 10s
EMERG_SILENCE_S       = 3.0    # seconds of no behavioral input for Step 3

# LLM trigger
LLM_LOAD_THRESH     = 0.70   # load score
LLM_SUSTAIN_WIN     = 12     # consecutive windows (3.0s at 250ms cadence)
LLM_COOLDOWN_S      = 30.0   # seconds between triggers

# UI Theme boundaries
THEME_HIGH_THRESH   = 0.70   # load above this → HIGH_CONTRAST
THEME_LOW_THRESH    = 0.30   # load below this → WARM_DIM
THEME_SUSTAIN_WIN   = 4      # consecutive windows required to switch theme


def _event(etype: str, **data) -> Dict[str, Any]:
    """Helper to create a consistent event dict."""
    return {
        'type':      etype,
        'timestamp': time.time(),
        'data':      data,
    }


# ─────────────────────────────────────────────────────────────────────────────
class DrowsinessDetector:
    """
    Fires drowsiness_alert when:
        state['drowsiness'] > DROWN_THRESH (0.65)
        AND state['load'] < DROWN_LOAD_MAX (0.55)
        sustained for DROWN_SUSTAIN_WIN (6) consecutive windows

    The load guard is essential — cognitive load also raises
    the drowsiness score slightly (theta rises with both states).
    By requiring low load, we confirm the drowsy-quiet pattern.

    Fires drowsiness_cleared when the condition resolves.
    """

    def __init__(self):
        self._sustain_count = 0    # consecutive qualifying windows
        self._active        = False # alert currently showing
        self._last_fired    = 0.0   # timestamp of last alert
        log.debug("DrowsinessDetector initialized.")

    def update(self, state: Dict[str, Any]) -> List[Dict]:
        events = []
        now    = time.time()

        drowsy = state.get('drowsiness', 0.0)
        load   = state.get('load',       0.0)

        # Condition: drowsy AND quiet (not high load)
        condition = (drowsy > DROWN_THRESH and load < DROWN_LOAD_MAX)

        if condition:
            self._sustain_count += 1
        else:
            # Reset sustain counter when condition breaks
            if self._sustain_count > 0:
                self._sustain_count = 0
            # If alert was active and condition cleared → fire cleared
            if self._active:
                self._active = False
                events.append(_event('drowsiness_cleared',
                                     drowsiness=drowsy, load=load))

        # Fire alert if sustained long enough and not in cooldown
        if (self._sustain_count >= DROWN_SUSTAIN_WIN
                and not self._active
                and (now - self._last_fired) > DROWN_COOLDOWN_S):
            self._active      = True
            self._last_fired  = now
            events.append(_event('drowsiness_alert',
                                 drowsiness=round(drowsy, 3),
                                 load=round(load, 3),
                                 sustain_windows=self._sustain_count,
                                 confidence=round(state.get('confidence', 1.0), 3)))
            log.info(f"DrowsinessDetector fired: drowsy={drowsy:.3f} "
                     f"load={load:.3f}")

        return events

    def reset(self):
        self._sustain_count = 0
        self._active        = False
        self._last_fired    = 0.0


# ─────────────────────────────────────────────────────────────────────────────
class EmergencyDetector:
    """
    Sequential detector — step order is mandatory:

    Step 1: tension > 0.80
            → Arms the detector, timestamps T0
    Step 2: attention < 0.25, must occur within EMERG_WINDOW_S of T0
            → Arms silence wait, timestamps T1
    Step 3: No behavioral marker for EMERG_SILENCE_S after T1
            → Fires emergency_alert

    If EMERG_WINDOW_S elapses without completing → fires emergency_reset.

    In NEST demo:
        Step 1 = Jaw Clench button + Muscle Tension slider max
        Step 2 = Attention/Focus slider to minimum
        Step 3 = don't click any NEST buttons for 3 seconds

    Behavioral markers reset the silence timer:
        Call update_marker() whenever NEST_Behavioral_Markers receives data.
    """

    def __init__(self):
        self._step         = 0      # 0=waiting, 1=step1 done, 2=step2 done
        self._t0           = 0.0    # when step 1 fired
        self._t1           = 0.0    # when step 2 fired
        self._last_marker  = 0.0    # last behavioral marker timestamp
        self._fired        = False  # prevent double-firing
        log.debug("EmergencyDetector initialized.")

    def update(self, state: Dict[str, Any]) -> List[Dict]:
        events = []
        now    = time.time()

        tension   = state.get('tension',   0.0)
        attention = state.get('attention', 1.0)

        # ── Check window expiry first ────────────────────────────────
        if self._step > 0:
            elapsed = now - self._t0
            if elapsed > EMERG_WINDOW_S and not self._fired:
                log.debug(f"EmergencyDetector: window expired at step {self._step}")
                events.append(_event('emergency_reset',
                                     step_reached=self._step,
                                     elapsed=round(elapsed, 2)))
                self._reset()
                return events

        # ── Step 1: Tension spike ────────────────────────────────────
        if self._step == 0 and tension > EMERG_TENSION_THRESH:
            self._step = 1
            self._t0   = now
            self._fired = False
            log.info(f"EmergencyDetector Step 1 armed: tension={tension:.3f}")
            events.append(_event('emergency_step1',
                                 tension=round(tension, 3)))

        # ── Step 2: Attention collapse ───────────────────────────────
        elif self._step == 1 and attention < EMERG_ATTN_THRESH:
            self._step = 2
            self._t1   = now
            self._last_marker = now   # reset silence clock from here
            log.info(f"EmergencyDetector Step 2 armed: attention={attention:.3f}")
            events.append(_event('emergency_step2',
                                 attention=round(attention, 3),
                                 elapsed_since_step1=round(now - self._t0, 2)))

        # ── Step 3: Behavioral silence ───────────────────────────────
        elif self._step == 2 and not self._fired:
            silence = now - self._last_marker
            if silence >= EMERG_SILENCE_S:
                self._fired = True
                log.warning(f"EmergencyDetector FIRED: silence={silence:.1f}s")
                events.append(_event('emergency_alert',
                                     tension=round(tension, 3),
                                     attention=round(attention, 3),
                                     silence_duration=round(silence, 2),
                                     total_elapsed=round(now - self._t0, 2)))

        return events

    def update_marker(self, marker: str):
        """
        Call this whenever NEST_Behavioral_Markers receives any sample.
        Resets the silence timer — prevents false Step 3 trigger.
        """
        self._last_marker = time.time()
        log.debug(f"EmergencyDetector marker received: {marker!r}")

    def dismiss(self):
        """Call when user presses 'I AM OKAY' in dashboard."""
        self._reset()
        log.info("EmergencyDetector dismissed by user.")

    def _reset(self):
        self._step  = 0
        self._t0    = 0.0
        self._t1    = 0.0
        self._fired = False

    @property
    def current_step(self) -> int:
        return self._step


# ─────────────────────────────────────────────────────────────────────────────
class LLMTrigger:
    """
    Fires llm_trigger when load > LLM_LOAD_THRESH (0.70)
    sustained for LLM_SUSTAIN_WIN (12) consecutive windows = 3.0s.
    30s cooldown between triggers.

    The event contains the current state_dict so server.py
    can pass it directly to the Gemini API call.
    """

    def __init__(self):
        self._sustain_count = 0
        self._last_fired    = 0.0
        log.debug("LLMTrigger initialized.")

    def update(self, state: Dict[str, Any]) -> List[Dict]:
        events = []
        now  = time.time()
        load = state.get('load', 0.0)

        if load > LLM_LOAD_THRESH:
            self._sustain_count += 1
        else:
            self._sustain_count = 0

        if (self._sustain_count >= LLM_SUSTAIN_WIN
                and (now - self._last_fired) > LLM_COOLDOWN_S):
            self._last_fired    = now
            self._sustain_count = 0   # reset so it doesn't re-fire immediately
            log.info(f"LLMTrigger fired: load={load:.3f}")
            events.append(_event('llm_trigger',
                                 load=round(load, 3),
                                 attention=round(state.get('attention', 0.5), 3),
                                 tension=round(state.get('tension', 0.5), 3),
                                 drowsiness=round(state.get('drowsiness', 0.5), 3),
                                 state_snapshot=state))

        return events

    def reset(self):
        self._sustain_count = 0
        self._last_fired    = 0.0


# ─────────────────────────────────────────────────────────────────────────────
class ThemeRule:
    """
    Theme rule with sustain — Feature #13.
    Emits theme_change when load crosses 0.70 or 0.30 AND
    stays there for THEME_SUSTAIN_WIN (4) consecutive windows.
    Prevents single-window noise from flipping the theme.

    Themes:
        NEUTRAL        — 0.30 <= load <= 0.70
        HIGH_CONTRAST  — load > 0.70 (dark bg, large font, bright text)
        WARM_DIM       — load < 0.30 (amber, small font, dim)
    """

    def __init__(self):
        self._current_theme = 'NEUTRAL'
        self._pending_theme = 'NEUTRAL'   # what the load currently suggests
        self._sustain_count = 0           # consecutive windows at pending theme
        log.debug("ThemeRule initialized.")

    def update(self, state: Dict[str, Any]) -> List[Dict]:
        events = []
        load   = state.get('load', 0.5)

        if load > THEME_HIGH_THRESH:
            suggested = 'HIGH_CONTRAST'
        elif load < THEME_LOW_THRESH:
            suggested = 'WARM_DIM'
        else:
            suggested = 'NEUTRAL'

        # Track sustain for the suggested theme
        if suggested == self._pending_theme:
            self._sustain_count += 1
        else:
            # Load crossed into a different zone — reset counter
            self._pending_theme = suggested
            self._sustain_count = 1

        # Only switch after sustained threshold crossing
        if (suggested != self._current_theme
                and self._sustain_count >= THEME_SUSTAIN_WIN):
            old_theme           = self._current_theme
            self._current_theme = suggested
            log.info(f"ThemeRule: {old_theme} → {suggested} (load={load:.3f})")
            events.append(_event('theme_change',
                                 theme=suggested,
                                 previous=old_theme,
                                 load=round(load, 3)))

        return events

    @property
    def current_theme(self) -> str:
        return self._current_theme


# ─────────────────────────────────────────────────────────────────────────────
class Detectors:
    """
    Composite — owns all four detectors and the behavioral marker inlet.
    Server.py calls update() every 250ms and update_marker() on marker events.
    Returns combined event list.
    """

    def __init__(self):
        self.drowsiness = DrowsinessDetector()
        self.emergency  = EmergencyDetector()
        self.llm        = LLMTrigger()
        self.theme      = ThemeRule()
        log.info("Detectors composite initialized.")

    def update(self, state: Dict[str, Any]) -> List[Dict]:
        """
        Call every 250ms with current state_dict.
        Returns list of events (usually empty).
        """
        events = []
        events.extend(self.drowsiness.update(state))
        events.extend(self.emergency.update(state))
        events.extend(self.llm.update(state))
        events.extend(self.theme.update(state))
        return events

    def update_marker(self, marker: str):
        """Forward behavioral marker to emergency detector."""
        self.emergency.update_marker(marker)

    def dismiss_emergency(self):
        """Forward dashboard dismiss to emergency detector."""
        self.emergency.dismiss()

    @property
    def current_theme(self) -> str:
        return self.theme.current_theme


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time

    logging.basicConfig(level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    print("\n" + "="*65)
    print("  NAOS detectors.py — Self Test")
    print("="*65 + "\n")

    passed = 0; total = 0

    def check(name, condition, got=""):
        global passed, total
        total += 1
        ok = "✅" if condition else "❌"
        print(f"  {ok} {name}" + (f": {got}" if got else ""))
        if condition: passed += 1

    def S(attention=0.5, load=0.5, tension=0.5, drowsiness=0.5, confidence=1.0):
        return {'attention': attention, 'load': load, 'tension': tension,
                'drowsiness': drowsiness, 'confidence': confidence}

    def has_event(events, etype):
        return any(e['type'] == etype for e in events)

    def get_event(events, etype):
        return next((e for e in events if e['type'] == etype), None)

    # ── DrowsinessDetector ────────────────────────────────────────────────────
    print("── DrowsinessDetector ──")
    dd = DrowsinessDetector()

    # Should not fire on baseline
    for _ in range(10): ev = dd.update(S())
    check("No alert at baseline", not has_event(ev, 'drowsiness_alert'))

    # Should not fire if load too high (disambiguation)
    dd.reset()
    for _ in range(10): ev = dd.update(S(drowsiness=0.80, load=0.65))
    check("No alert when load high (disambiguation)",
          not has_event(ev, 'drowsiness_alert'))

    # Should not fire before sustain window
    dd.reset()
    for _ in range(DROWN_SUSTAIN_WIN - 1):
        ev = dd.update(S(drowsiness=0.80, load=0.40))
    check("No alert before sustain window",
          not has_event(ev, 'drowsiness_alert'))

    # Should fire after sustain window
    ev = dd.update(S(drowsiness=0.80, load=0.40))
    check("Alert fires after sustain window",
          has_event(ev, 'drowsiness_alert'),
          get_event(ev, 'drowsiness_alert'))

    # Should not fire again immediately (cooldown)
    ev = dd.update(S(drowsiness=0.80, load=0.40))
    check("No double-fire (cooldown active)",
          not has_event(ev, 'drowsiness_alert'))

    # Should fire cleared when condition resolves
    dd._active = True   # simulate active alert
    ev = dd.update(S(drowsiness=0.30, load=0.40))
    check("Cleared event fires when condition resolves",
          has_event(ev, 'drowsiness_cleared'))

    # ── EmergencyDetector ─────────────────────────────────────────────────────
    print("\n── EmergencyDetector ──")
    ed = EmergencyDetector()

    # Step 1 only — no alert
    ev = ed.update(S(tension=0.90))
    check("Step 1 arms detector",
          has_event(ev, 'emergency_step1') and ed.current_step == 1,
          f"step={ed.current_step}")

    # Step 1 only — no alert yet
    ev = ed.update(S(tension=0.90, attention=0.50))
    check("No alert after step 1 alone",
          not has_event(ev, 'emergency_alert'))

    # Step 2
    ev = ed.update(S(tension=0.90, attention=0.15))
    check("Step 2 triggers on low attention",
          has_event(ev, 'emergency_step2') and ed.current_step == 2,
          f"step={ed.current_step}")

    # Step 3 — silence (no marker for EMERG_SILENCE_S)
    # Force last_marker to be old enough
    ed._last_marker = _time.time() - EMERG_SILENCE_S - 0.1
    ev = ed.update(S(tension=0.90, attention=0.15))
    check("Emergency fires after silence",
          has_event(ev, 'emergency_alert'),
          get_event(ev, 'emergency_alert'))

    # No double fire
    ev = ed.update(S(tension=0.90, attention=0.15))
    check("No double emergency fire",
          not has_event(ev, 'emergency_alert'))

    # Test: window expiry resets sequence
    ed2 = EmergencyDetector()
    ed2.update(S(tension=0.90))             # Step 1
    ed2._t0 = _time.time() - EMERG_WINDOW_S - 0.5  # force expire
    ev = ed2.update(S(tension=0.90, attention=0.10))
    check("Window expiry fires reset",
          has_event(ev, 'emergency_reset'),
          get_event(ev, 'emergency_reset'))
    check("Step resets to 0 after expiry", ed2.current_step == 0)

    # Test: marker resets silence timer
    ed3 = EmergencyDetector()
    ed3.update(S(tension=0.90))
    ed3.update(S(tension=0.90, attention=0.10))  # step 2
    ed3.update_marker('user_error')               # marker received
    ed3._last_marker = _time.time() - 1.0         # 1s ago (< 3s silence)
    ev = ed3.update(S(tension=0.85, attention=0.10))
    check("Marker resets silence — no premature alert",
          not has_event(ev, 'emergency_alert'))

    # Test wrong order — attention first, tension second — should NOT fire
    ed4 = EmergencyDetector()
    ed4.update(S(attention=0.10, tension=0.50))  # attention low but no tension spike
    check("Wrong order: no alert without step 1 first",
          ed4.current_step == 0, f"step={ed4.current_step}")

    # ── LLMTrigger ───────────────────────────────────────────────────────────
    print("\n── LLMTrigger ──")
    lt = LLMTrigger()

    # Below threshold — no fire
    for _ in range(LLM_SUSTAIN_WIN + 5):
        ev = lt.update(S(load=0.65))
    check("No trigger below threshold (0.65 < 0.70)",
          not has_event(ev, 'llm_trigger'))

    # Exactly at threshold — should fire after sustain
    lt.reset()
    for i in range(LLM_SUSTAIN_WIN):
        ev = lt.update(S(load=0.75))
    check("LLM triggers after sustain window",
          has_event(ev, 'llm_trigger'),
          get_event(ev, 'llm_trigger'))

    # Cooldown active
    ev = lt.update(S(load=0.90))
    for _ in range(LLM_SUSTAIN_WIN):
        ev = lt.update(S(load=0.90))
    check("LLM does not fire during cooldown",
          not has_event(ev, 'llm_trigger'))

    # Sustain resets when load drops
    lt2 = LLMTrigger()
    for _ in range(LLM_SUSTAIN_WIN - 1): lt2.update(S(load=0.80))
    lt2.update(S(load=0.50))  # drops below threshold
    for i in range(LLM_SUSTAIN_WIN - 1):
        ev = lt2.update(S(load=0.80))
    check("Sustain resets when load drops", not has_event(ev, 'llm_trigger'))

    # ── ThemeRule ─────────────────────────────────────────────────────────────
    print("\n── ThemeRule ──")
    tr = ThemeRule()

    # Baseline neutral
    ev = tr.update(S(load=0.50))
    check("Neutral at baseline load=0.50",
          tr.current_theme == 'NEUTRAL', tr.current_theme)

    # Single high-load window should NOT change theme (sustain guard)
    ev = tr.update(S(load=0.75))
    check("Single window does NOT change theme",
          not has_event(ev, 'theme_change') and tr.current_theme == 'NEUTRAL',
          tr.current_theme)

    # Sustain for THEME_SUSTAIN_WIN windows → should switch
    for _ in range(THEME_SUSTAIN_WIN - 1):
        ev = tr.update(S(load=0.75))
    check("HIGH_CONTRAST after sustain window",
          has_event(ev, 'theme_change') and
          get_event(ev, 'theme_change')['data']['theme'] == 'HIGH_CONTRAST',
          tr.current_theme)

    # No repeat event when same theme continues
    ev = tr.update(S(load=0.80))
    check("No repeat event same theme",
          not has_event(ev, 'theme_change'))

    # Drop to WARM_DIM (needs sustain)
    for i in range(THEME_SUSTAIN_WIN):
        ev = tr.update(S(load=0.20))
    check("WARM_DIM after sustained low load",
          has_event(ev, 'theme_change') and
          get_event(ev, 'theme_change')['data']['theme'] == 'WARM_DIM',
          tr.current_theme)

    # Back to neutral (needs sustain)
    for i in range(THEME_SUSTAIN_WIN):
        ev = tr.update(S(load=0.50))
    check("NEUTRAL after sustained middle load",
          has_event(ev, 'theme_change') and tr.current_theme == 'NEUTRAL',
          tr.current_theme)

    # ── Composite Detectors class ─────────────────────────────────────────────
    print("\n── Composite Detectors ──")
    det = Detectors()

    # Baseline — all empty
    ev = det.update(S())
    check("Baseline returns empty events", ev == [], len(ev))

    # Theme changes propagate (needs sustain)
    for _ in range(THEME_SUSTAIN_WIN):
        ev = det.update(S(load=0.80))
    check("Composite forwards theme_change",
          has_event(ev, 'theme_change'))

    # Marker forwarded to emergency
    det.update(S(tension=0.90))   # arm step 1
    det.update_marker('back_spam')
    check("update_marker updates emergency last_marker time",
          det.emergency._last_marker > 0)

    # ── Performance ───────────────────────────────────────────────────────────
    print("\n── Performance ──")
    det2 = Detectors()
    times = []
    for _ in range(200):
        t0 = _time.perf_counter()
        det2.update(S(load=0.50))
        times.append((_time.perf_counter() - t0) * 1000)

    import numpy as np
    mean_ms = np.mean(times); max_ms = np.max(times)
    check("Mean compute < 0.5ms",
          mean_ms < 0.5, f"{mean_ms:.3f}ms mean, {max_ms:.3f}ms max")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Result: {passed}/{total} tests passed.")
    if passed == total:
        print("  ✅ detectors.py is working correctly.")
        print("  ✅ Safe to proceed to server.py")
    else:
        print("  ❌ Fix failures before proceeding.")
    print("="*65 + "\n")
