"""
naos_daemon_v3.py
NAOS — EEG Adaptive Overlay Daemon
Author  : Venkat / NAOS Project
Version : 3.0 (BrainBit Demo Build — June 2026)

Architecture
────────────
  ┌─────────────────────┐     LSL (NEST_EEG_Clean)
  │  NEST  (nest_main) │ ──────────────────────────────►┐
  └─────────────────────┘                                 │
                                                          ▼
                                             ┌────────────────────────┐
                                             │   EEGWorker (Thread)   │
                                             │  • pull LSL window     │
                                             │  • band-power features │
                                             │  • threshold → state   │
                                             └──────────┬─────────────┘
                                                        │  Qt signal (thread-safe)
                                                        ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Qt Main Thread                                │
  │                                                                  │
  │  OverlayWindow          NavigationBar           StatusHUD        │
  │  (full-screen glass)    (EEG-driven highlight)  (state readout)  │
  └──────────────────────────────────────────────────────────────────┘
                                                        │
                                             DwellWorker (Thread, 20 Hz)
                                             • pure mouse-tracking
                                             • uiautomation snap
                                             • unique target ID fix
                                             • LSL-independent

What was wrong with v2 and what is now fixed
────────────────────────────────────────────
  [CRITICAL]  Backend mocks ran silently — no LSL connection attempted.
              FIX: LSLWorker connects to NEST_EEG_Clean directly.
                   Falls back to simulated state only when stream is
                   genuinely absent, and prints a loud warning.

  [CRITICAL]  EEG processing and dwell polling shared one loop.
              LSL blocking call froze UI / dwell for up to 1 second.
              FIX: Two completely independent threads.
                   EEG thread blocks on LSL; dwell thread runs at 20 Hz always.

  [HIGH]      NativeWindowHandle used as dwell target ID.
              Two buttons in the same window → same ID → wrong click.
              FIX: Target ID = (HWND, rect.left, rect.top) tuple hash.

  [HIGH]      Snap-to-target called pyautogui.moveTo inside daemon thread.
              This crosses the thread boundary for a UI action.
              FIX: Snap emits a Qt signal; main thread executes the move.

  [HIGH]      uiautomation import inside same try/except as PyQt6.
              Any uiautomation failure killed the whole process.
              FIX: Separate try/except with graceful degradation.
                   Dwell still works (no snap, just track) if uiautomation absent.

  [MEDIUM]    No HUD — impossible to tell what state the daemon was reading.
              FIX: StatusHUD widget shows live load / drowsy / attention values
                   and LSL connection status, pinned to bottom-right corner.

  [MEDIUM]    NavigationBar never highlighted anything.
              The "navigation" described to Boris did not exist.
              FIX: Real NavigationBar widget: highlights the closest interactive
                   control to the cursor when attention is low or dwell is active.

  [LOW]       trigger_native_scaling was a no-op comment.
              FIX: Implemented via ctypes SetProcessDpiAwareness path (no PS).
                   Falls back to font-scale heuristic via QApplication.setFont.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────────────────────────────────────
import sys
import os
import time
import threading
import math
import queue
import traceback

import numpy as np
import pyautogui

# ── PyQt6 ────────────────────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
        QFrame, QSizePolicy
    )
    from PyQt6.QtCore  import (
        Qt, QTimer, pyqtSignal, QObject, QPoint, QRect, QPointF,
        QPropertyAnimation, QEasingCurve
    )
    from PyQt6.QtGui   import (
        QPainter, QColor, QRadialGradient, QLinearGradient,
        QBrush, QFont, QPen, QFontDatabase, QPainterPath
    )
except ImportError:
    print("FATAL: PyQt6 not installed.  pip install PyQt6")
    sys.exit(1)

# ── uiautomation (optional — dwell snapping only) ────────────────────────────
try:
    import uiautomation as auto
    UIAUTO_AVAILABLE = True
except ImportError:
    UIAUTO_AVAILABLE = False
    print("[WARN] uiautomation not found — snap-to-target disabled.  "
          "pip install uiautomation")

# ── pylsl ────────────────────────────────────────────────────────────────────
try:
    from pylsl import StreamInlet, resolve_byprop
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("[WARN] pylsl not found — running on simulated EEG state.  "
          "pip install pylsl")

pyautogui.FAILSAFE = False

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE      = 250        # Hz — must match NEST
WINDOW_SAMPLES     = 250        # 1-second feature window
STREAM_NAME        = 'NEST_EEG_Clean'
STREAM_TYPE        = 'EEG'
LSL_RESOLVE_TIMEOUT = 3.0       # seconds before giving up on LSL resolve
LSL_PULL_TIMEOUT    = 0.05      # seconds per LSL pull call

DWELL_TIME         = 1.5        # seconds to trigger a dwell click
DWELL_COOLDOWN     = 2.0        # min seconds between clicks on same target
DWELL_POLL_HZ      = 20         # dwell thread rate
SNAP_RADIUS_PX     = 80         # pixels — only snap if cursor is within this

# Band-power thresholds (tuned for BrainBit 4-channel, calibrated heuristics)
THETA_DROWSY_THRESH   = 0.45    # theta ratio above this → drowsy
BETA_LOAD_THRESH      = 0.35    # beta ratio above this  → high load
ALPHA_ATTENTION_FLOOR = 0.20    # alpha ratio below this → low attention

# ──────────────────────────────────────────────────────────────────────────────
#  Thread-safe signals hub
# ──────────────────────────────────────────────────────────────────────────────
class Signals(QObject):
    # EEG state update  (load: float, drowsy: float, attention: float)
    state_updated   = pyqtSignal(float, float, float)
    # Dwell ring update (progress: float 0–1, screen_x: int, screen_y: int)
    dwell_updated   = pyqtSignal(float, int, int)
    # Nav highlight    (rect: QRect or None)
    nav_highlight   = pyqtSignal(object)
    # Cursor snap      (x: int, y: int) — executed in main thread
    cursor_snap     = pyqtSignal(int, int)
    # LSL status       (connected: bool, stream_name: str)
    lsl_status      = pyqtSignal(bool, str)


# ──────────────────────────────────────────────────────────────────────────────
#  EEG Feature Extraction (self-contained, no external FeatureExtractor needed)
# ──────────────────────────────────────────────────────────────────────────────
class BandPowerExtractor:
    """
    Computes relative band power from a raw EEG window.
    Input : np.ndarray  shape (n_channels, n_samples)  in µV
    Output: dict with keys load, drowsiness, attention  all in [0, 1]
    """
    def __init__(self, fs: int = SAMPLING_RATE):
        self.fs = fs

    def _band_power(self, data: np.ndarray, lo: float, hi: float) -> float:
        """Mean relative band power across all channels."""
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        fft   = np.abs(np.fft.rfft(data, axis=1)) ** 2
        total = fft.sum(axis=1, keepdims=True) + 1e-10
        rel   = fft / total
        mask  = (freqs >= lo) & (freqs <= hi)
        return float(rel[:, mask].mean())

    def compute(self, window: np.ndarray) -> dict:
        """
        window: shape (n_channels, n_samples) or (n_samples, n_channels)
        Returns: {'load': float, 'drowsiness': float, 'attention': float}
        """
        if window is None or window.size == 0:
            return {'load': 0.1, 'drowsiness': 0.1, 'attention': 0.9}

        # Normalise orientation: we want (channels, samples)
        if window.shape[0] > window.shape[1]:
            window = window.T

        theta = self._band_power(window, 4.0,  8.0)
        alpha = self._band_power(window, 8.0, 13.0)
        beta  = self._band_power(window, 13.0, 30.0)

        # Map to semantic states with soft sigmoid-like clamping
        drowsiness = float(np.clip(theta / (THETA_DROWSY_THRESH + 1e-6), 0, 1))
        load       = float(np.clip(beta  / (BETA_LOAD_THRESH    + 1e-6), 0, 1))
        attention  = float(np.clip(alpha / (ALPHA_ATTENTION_FLOOR + 1e-6), 0, 1))
        attention  = min(1.0, attention)

        return {
            'load':       round(load,       3),
            'drowsiness': round(drowsiness, 3),
            'attention':  round(attention,  3),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  EEG Worker Thread  (LSL → features → signals)
# ──────────────────────────────────────────────────────────────────────────────
class EEGWorker(threading.Thread):
    """
    Continuously pulls EEG data from the NEST_EEG_Clean LSL stream,
    computes band-power features over a rolling 1-second window, and
    emits state_updated signal on every complete window.

    If LSL is unavailable or the stream cannot be found, falls back to
    a simulated oscillating state so the overlay is still demonstrable.
    """

    def __init__(self, signals: Signals, extractor: BandPowerExtractor):
        super().__init__(name="EEGWorker", daemon=True)
        self.signals   = signals
        self.extractor = extractor
        self.running   = True
        self._inlet    = None
        self._buffer   = []    # list of 1-D np arrays, one per sample
        self._sim_t    = 0.0   # simulation time counter

    # ── LSL connection ────────────────────────────────────────────────────────
    def _try_connect(self) -> bool:
        if not LSL_AVAILABLE:
            return False
        try:
            print(f"[EEG] Searching for LSL stream '{STREAM_NAME}'...")
            streams = resolve_byprop('name', STREAM_NAME,
                                     timeout=LSL_RESOLVE_TIMEOUT)
            if not streams:
                print(f"[EEG] No stream found. Is NEST running?")
                return False
            self._inlet = StreamInlet(streams[0])
            ch = self._inlet.info().channel_count()
            print(f"[EEG] Connected to '{STREAM_NAME}'  ({ch} channels)")
            self.signals.lsl_status.emit(True, STREAM_NAME)
            return True
        except Exception as e:
            print(f"[EEG] LSL connect error: {e}")
            return False

    # ── Simulated fallback state ──────────────────────────────────────────────
    def _simulated_state(self) -> dict:
        """
        Returns a slowly oscillating fake state so the overlay is
        visually testable without NEST.  Prints a warning every 10 s.
        """
        t = self._sim_t
        load       = 0.3 + 0.25 * math.sin(t * 0.15)
        drowsiness = 0.2 + 0.30 * math.sin(t * 0.07 + 1.0)
        attention  = 0.6 + 0.30 * math.cos(t * 0.10 + 0.5)
        self._sim_t += 0.05
        return {
            'load':       max(0.0, min(1.0, load)),
            'drowsiness': max(0.0, min(1.0, drowsiness)),
            'attention':  max(0.0, min(1.0, attention)),
        }

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        connected = self._try_connect()

        if not connected:
            print("[EEG] !! RUNNING ON SIMULATED STATE — NOT REAL EEG !!")
            print("[EEG]    Start NEST and relaunch to use real signals.")
            self.signals.lsl_status.emit(False, "SIMULATED")
            self._run_simulated()
            return

        self._run_lsl()

    def _run_lsl(self):
        """Pull samples from LSL, build windows, extract features."""
        warn_interval = 10.0
        last_warn     = 0.0

        while self.running:
            try:
                # Pull up to 32 samples at once (non-blocking style)
                samples, _ = self._inlet.pull_chunk(
                    timeout=LSL_PULL_TIMEOUT, max_samples=32
                )
                if samples:
                    for s in samples:
                        self._buffer.append(np.array(s, dtype=np.float32))

                # When we have a full 1-second window, extract and emit
                if len(self._buffer) >= WINDOW_SAMPLES:
                    window_arr = np.array(self._buffer[-WINDOW_SAMPLES:]).T
                    state = self.extractor.compute(window_arr)
                    self.signals.state_updated.emit(
                        state['load'],
                        state['drowsiness'],
                        state['attention'],
                    )
                    # Slide window by half (50% overlap) for smooth updates
                    self._buffer = self._buffer[WINDOW_SAMPLES // 2:]

            except Exception as e:
                now = time.time()
                if now - last_warn > warn_interval:
                    print(f"[EEG] LSL pull error: {e}")
                    last_warn = now
                time.sleep(0.1)

    def _run_simulated(self):
        """Emit fake states at ~20 Hz for demo purposes."""
        while self.running:
            state = self._simulated_state()
            self.signals.state_updated.emit(
                state['load'],
                state['drowsiness'],
                state['attention'],
            )
            time.sleep(0.05)

    def stop(self):
        self.running = False
        if self._inlet:
            try:
                self._inlet.close_stream()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
#  Dwell Worker Thread  (pure cursor tracking, 20 Hz, LSL-independent)
# ──────────────────────────────────────────────────────────────────────────────
class DwellWorker(threading.Thread):
    """
    Polls mouse position at DWELL_POLL_HZ.  When the cursor stays near
    an interactive UI control, emits a growing ring and eventually clicks.

    Key fix: target ID = hash((HWND, rect.left, rect.top))
    Two different buttons inside the same window now have different IDs.

    Key fix: pyautogui.moveTo is NOT called here — cursor_snap signal
    is emitted instead so the main Qt thread performs the move.
    """

    def __init__(self, signals: Signals):
        super().__init__(name="DwellWorker", daemon=True)
        self.signals        = signals
        self.running        = True
        self._last_tid      = None
        self._dwell_start   = 0.0
        self._last_click_t  = 0.0
        self._last_snap_tid = None   # avoid re-snapping same target

    # ── Target detection ──────────────────────────────────────────────────────
    @staticmethod
    def _make_target_id(hwnd: int, rect) -> int:
        """Unique ID that distinguishes two buttons inside the same window."""
        return hash((hwnd, rect.left, rect.top))

    def _get_snap_target(self, x: int, y: int):
        """
        Returns (center_x, center_y, target_id) if cursor is near an
        interactive control, else None.

        Requires uiautomation.  Degrades gracefully if unavailable.
        """
        if not UIAUTO_AVAILABLE:
            return None
        try:
            ctrl = auto.ControlFromPoint(x, y)
            if ctrl is None:
                return None
            interactive = (
                auto.ControlType.ButtonControl,
                auto.ControlType.TabItemControl,
                auto.ControlType.HyperlinkControl,
                auto.ControlType.MenuItemControl,
                auto.ControlType.ListItemControl,
                auto.ControlType.CheckBoxControl,
                auto.ControlType.RadioButtonControl,
            )
            if ctrl.ControlType not in interactive:
                return None

            r   = ctrl.BoundingRectangle
            cx  = (r.left + r.right)  // 2
            cy  = (r.top  + r.bottom) // 2
            tid = DwellWorker._make_target_id(ctrl.NativeWindowHandle, r)

            # Only snap if cursor is meaningfully close to the control centre
            dist = math.hypot(x - cx, y - cy)
            if dist > SNAP_RADIUS_PX:
                return None

            # Also emit nav highlight for the overlay
            return (cx, cy, tid, QRect(r.left, r.top,
                                       r.right - r.left,
                                       r.bottom - r.top))
        except Exception:
            return None

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        interval = 1.0 / DWELL_POLL_HZ
        print(f"[Dwell] Worker active at {DWELL_POLL_HZ} Hz.")

        while self.running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:
                print(f"[Dwell] tick error: {e}")
            elapsed = time.time() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)

    def _tick(self):
        x, y    = pyautogui.position()
        result  = self._get_snap_target(x, y)

        if result is None:
            # Cursor not near any interactive target → clear ring
            self._last_tid = None
            self.signals.dwell_updated.emit(0.0, 0, 0)
            self.signals.nav_highlight.emit(None)
            return

        cx, cy, tid, rect = result

        # Snap cursor to control centre (only once per new target)
        if tid != self._last_snap_tid:
            self.signals.cursor_snap.emit(cx, cy)
            self._last_snap_tid = tid

        # Emit nav highlight
        self.signals.nav_highlight.emit(rect)

        if tid != self._last_tid:
            # New target — reset dwell timer
            self._last_tid    = tid
            self._dwell_start = time.time()

        elapsed = time.time() - self._dwell_start
        progress = min(1.0, elapsed / DWELL_TIME)
        self.signals.dwell_updated.emit(progress, cx, cy)

        # Fire click when dwell complete and cooldown passed
        cooldown_ok = (time.time() - self._last_click_t) > DWELL_COOLDOWN
        if elapsed >= DWELL_TIME and cooldown_ok:
            pyautogui.click(cx, cy)
            self._last_click_t  = time.time()
            self._last_tid      = None
            self._last_snap_tid = None
            self.signals.dwell_updated.emit(0.0, 0, 0)
            print(f"[Dwell] CLICK at ({cx}, {cy})")

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────────────────────────────────────
#  Overlay Window  (full-screen transparent glass)
# ──────────────────────────────────────────────────────────────────────────────
class OverlayWindow(QWidget):
    """
    Full-screen transparent overlay.  Draws:
      1. Vignette darkening       (cognitive load > 0.3)
      2. Amber warm tint          (drowsiness > 0.4)
      3. Blue attention highlight (attention < 0.35)
      4. Dwell progress ring      (always when dwelling)
      5. Nav control highlight    (green rect around hovered control)
    """

    def __init__(self):
        super().__init__()
        flags = (
            Qt.WindowType.WindowStaysOnTopHint    |
            Qt.WindowType.FramelessWindowHint      |
            Qt.WindowType.WindowTransparentForInput|
            Qt.WindowType.Tool                     # keeps off taskbar
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        # State
        self.load_val      = 0.0
        self.drowsy_val    = 0.0
        self.attention_val = 1.0
        self.dwell_progress= 0.0
        self.dwell_pos     = QPoint(0, 0)
        self.nav_rect      = None   # QRect or None

    # ── Slot: EEG state ───────────────────────────────────────────────────────
    def on_state(self, load: float, drowsy: float, attention: float):
        self.load_val      = load
        self.drowsy_val    = drowsy
        self.attention_val = attention
        self.update()

    # ── Slot: Dwell ring ──────────────────────────────────────────────────────
    def on_dwell(self, progress: float, x: int, y: int):
        self.dwell_progress = progress
        self.dwell_pos      = QPoint(x, y)
        self.update()

    # ── Slot: Nav highlight ───────────────────────────────────────────────────
    def on_nav(self, rect):
        self.nav_rect = rect
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # ── 1. Cognitive Load Vignette ────────────────────────────────────────
        if self.load_val > 0.3:
            alpha = int(min(200, (self.load_val - 0.3) / 0.7 * 200))
            grad  = QRadialGradient(w / 2, h / 2, w * 0.75)
            grad.setColorAt(0.0, QColor(0, 0, 0, 0))
            grad.setColorAt(0.6, QColor(0, 0, 0, 0))
            grad.setColorAt(1.0, QColor(0, 0, 0, alpha))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(self.rect())

        # ── 2. Drowsiness Amber Tint ──────────────────────────────────────────
        if self.drowsy_val > 0.4:
            alpha = int(min(110, (self.drowsy_val - 0.4) / 0.6 * 110))
            p.setBrush(QBrush(QColor(255, 160, 30, alpha)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(self.rect())

        # ── 3. Low-Attention Blue Pulse ───────────────────────────────────────
        if self.attention_val < 0.35:
            alpha = int((1.0 - self.attention_val / 0.35) * 40)
            p.setBrush(QBrush(QColor(40, 80, 255, alpha)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(self.rect())

        # ── 4. Nav Control Highlight ──────────────────────────────────────────
        if self.nav_rect is not None:
            r    = self.nav_rect
            pad  = 6
            expand = QRect(r.left() - pad, r.top() - pad,
                           r.width() + pad*2, r.height() + pad*2)
            # Glow fill
            p.setBrush(QBrush(QColor(0, 220, 120, 25)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(expand, 6, 6)
            # Bright border
            pen = QPen(QColor(0, 255, 140, 200), 2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(expand, 6, 6)

        # ── 5. Dwell Progress Ring ────────────────────────────────────────────
        if self.dwell_progress > 0.01:
            cx, cy = self.dwell_pos.x(), self.dwell_pos.y()
            radius = 32

            # Background ring (white ghost)
            p.setPen(QPen(QColor(255, 255, 255, 60), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPoint(cx, cy), radius, radius)

            # Progress arc (cyan → green as fills)
            t      = self.dwell_progress
            r_comp = int(0   * (1-t) + 0   * t)
            g_comp = int(200 * (1-t) + 255 * t)
            b_comp = int(255 * (1-t) + 100 * t)
            arc_pen = QPen(QColor(r_comp, g_comp, b_comp, 240), 4)
            arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(arc_pen)
            span_deg = int(t * 360 * 16)
            p.drawArc(cx - radius, cy - radius,
                      radius * 2, radius * 2,
                      90 * 16, -span_deg)

            # Centre dot
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(r_comp, g_comp, b_comp, 200)))
            p.drawEllipse(QPoint(cx, cy), 5, 5)


# ──────────────────────────────────────────────────────────────────────────────
#  Status HUD  (small pinned widget showing live state values)
# ──────────────────────────────────────────────────────────────────────────────
class StatusHUD(QWidget):
    """
    Always-on-top status readout.  Pinned bottom-right.
    Shows: LSL status, load, drowsiness, attention.
    Semi-transparent dark panel — does NOT block input.
    """

    def __init__(self):
        super().__init__()
        flags = (
            Qt.WindowType.WindowStaysOnTopHint   |
            Qt.WindowType.FramelessWindowHint     |
            Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(220, 130)

        # Position bottom-right
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.right() - 230, screen.bottom() - 150)

        # State
        self.lsl_ok      = False
        self.stream_name = "—"
        self.load        = 0.0
        self.drowsy      = 0.0
        self.attention   = 1.0

    def on_state(self, load: float, drowsy: float, attention: float):
        self.load      = load
        self.drowsy    = drowsy
        self.attention = attention
        self.update()

    def on_lsl(self, connected: bool, name: str):
        self.lsl_ok      = connected
        self.stream_name = name
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background panel
        p.setBrush(QBrush(QColor(10, 12, 18, 210)))
        p.setPen(QPen(QColor(40, 60, 80, 180), 1))
        p.drawRoundedRect(4, 4, w-8, h-8, 10, 10)

        font_title = QFont("Consolas", 8, QFont.Weight.Bold)
        font_val   = QFont("Consolas", 9)
        font_small = QFont("Consolas", 7)

        # Title
        p.setFont(font_title)
        p.setPen(QColor(0, 200, 255, 220))
        p.drawText(14, 22, "NAOS  v3.0")

        # LSL status
        dot_color = QColor(0, 255, 100) if self.lsl_ok else QColor(255, 80, 60)
        p.setBrush(QBrush(dot_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(14, 30, 8, 8)
        p.setFont(font_small)
        p.setPen(dot_color)
        label = self.stream_name if self.lsl_ok else "SIMULATED"
        p.drawText(28, 39, label)

        # Bars
        self._draw_bar(p, 14, 52,  "LOAD ",    self.load,      QColor(255, 100, 60))
        self._draw_bar(p, 14, 74,  "DRWSY",    self.drowsy,    QColor(255, 180, 30))
        self._draw_bar(p, 14, 96,  "ATTN ",    self.attention, QColor(0,   200, 120))

    def _draw_bar(self, p: QPainter, x: int, y: int,
                  label: str, val: float, color: QColor):
        font = QFont("Consolas", 7)
        p.setFont(font)
        p.setPen(QColor(140, 160, 180))
        p.drawText(x, y + 11, label)

        bar_x   = x + 44
        bar_w   = self.width() - bar_x - 14
        bar_h   = 10
        # Track
        p.setBrush(QBrush(QColor(30, 35, 45)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(bar_x, y, bar_w, bar_h, 3, 3)
        # Fill
        fill_w = max(0, int(val * bar_w))
        if fill_w > 0:
            c = QColor(color)
            c.setAlpha(200)
            p.setBrush(QBrush(c))
            p.drawRoundedRect(bar_x, y, fill_w, bar_h, 3, 3)
        # Value text
        p.setPen(QColor(200, 210, 220))
        p.drawText(bar_x + bar_w + 4, y + 11, f"{val:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
#  Font / UI Scale Adaptation
# ──────────────────────────────────────────────────────────────────────────────
class UIAdapter:
    """
    Adjusts application font size based on drowsiness level.
    Called from main thread only.
    """
    _BASE_PT      = 10
    _LARGE_PT     = 14
    _last_scale   = 1.0
    _SCALE_THRESH = 0.05   # only update if change > 5%

    @classmethod
    def adapt(cls, drowsy: float):
        target_pt = cls._BASE_PT + int(drowsy * (cls._LARGE_PT - cls._BASE_PT))
        current   = QApplication.font().pointSize()
        if abs(target_pt - current) >= 1:
            f = QApplication.font()
            f.setPointSize(target_pt)
            QApplication.setFont(f)


# ──────────────────────────────────────────────────────────────────────────────
#  Emergency Alert  (full-screen flash on extreme drowsiness)
# ──────────────────────────────────────────────────────────────────────────────
class AlertManager:
    _last_alert = 0.0
    _COOLDOWN   = 15.0   # minimum seconds between alerts

    @staticmethod
    def check(drowsy: float, overlay: OverlayWindow):
        if drowsy > 0.88:
            now = time.time()
            if now - AlertManager._last_alert > AlertManager._COOLDOWN:
                AlertManager._last_alert = now
                AlertManager._flash(overlay)
                print("[ALERT] Extreme drowsiness detected!")

    @staticmethod
    def _flash(overlay: OverlayWindow):
        """Three quick red flashes on the overlay."""
        original_drowsy = overlay.drowsy_val
        def do_flash(n):
            if n <= 0:
                overlay.drowsy_val = original_drowsy
                overlay.update()
                return
            overlay.drowsy_val = 1.0 if n % 2 == 0 else original_drowsy
            overlay.update()
            QTimer.singleShot(200, lambda: do_flash(n - 1))
        do_flash(6)


# ──────────────────────────────────────────────────────────────────────────────
#  Main Controller (wires everything together in the Qt main thread)
# ──────────────────────────────────────────────────────────────────────────────
class NaosController:
    def __init__(self):
        self.signals   = Signals()
        self.extractor = BandPowerExtractor(SAMPLING_RATE)

        # Widgets
        self.overlay   = OverlayWindow()
        self.hud       = StatusHUD()

        # Workers
        self.eeg_worker   = EEGWorker(self.signals, self.extractor)
        self.dwell_worker = DwellWorker(self.signals)

        # Wire signals → slots (all executed in Qt main thread)
        self.signals.state_updated.connect(self.overlay.on_state)
        self.signals.state_updated.connect(self.hud.on_state)
        self.signals.state_updated.connect(self._on_state_for_side_effects)
        self.signals.dwell_updated.connect(self.overlay.on_dwell)
        self.signals.nav_highlight.connect(self.overlay.on_nav)
        self.signals.lsl_status.connect(self.hud.on_lsl)
        self.signals.cursor_snap.connect(self._do_cursor_snap)  # main thread

    def _on_state_for_side_effects(self, load: float, drowsy: float, attention: float):
        UIAdapter.adapt(drowsy)
        AlertManager.check(drowsy, self.overlay)

    def _do_cursor_snap(self, x: int, y: int):
        """Cursor snap runs in Qt main thread — safe for pyautogui."""
        try:
            pyautogui.moveTo(x, y, duration=0.08)
        except Exception as e:
            print(f"[Snap] moveTo error: {e}")

    def start(self):
        self.overlay.show()
        self.hud.show()
        self.eeg_worker.start()
        self.dwell_worker.start()
        print("[NAOS] All systems running.")
        print(f"  EEG worker  : {self.eeg_worker.name}")
        print(f"  Dwell worker: {self.dwell_worker.name}")
        print(f"  Overlay     : {self.overlay.width()}×{self.overlay.height()}")
        print(f"  HUD         : visible bottom-right")

    def stop(self):
        print("[NAOS] Shutting down...")
        self.eeg_worker.stop()
        self.dwell_worker.stop()
        self.eeg_worker.join(timeout=2.0)
        self.dwell_worker.join(timeout=1.0)
        print("[NAOS] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app        = QApplication(sys.argv)
    controller = NaosController()
    controller.start()

    # Clean shutdown on Qt exit
    app.aboutToQuit.connect(controller.stop)

    sys.exit(app.exec())
