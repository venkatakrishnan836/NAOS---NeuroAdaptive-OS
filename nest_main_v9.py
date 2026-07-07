import sys, time as systime, numpy as np, random, os, struct, queue, threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QGridLayout, QCheckBox, QFrame)
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont
import pyqtgraph as pg
from scipy import signal
from pylsl import StreamInfo, StreamOutlet
from bci_models import HeadModel

# --- Import and check for ML libraries ---
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# --- Constants ---
HEADER_BYTE = 0xA0
FOOTER_BYTE = 0xC0
SCALE_FACTOR_uV_PER_COUNT = 0.02235
LATENT_DIM = 128

# Minimum pre-fill for the async GAN queue before the main loop starts reading
_GAN_QUEUE_PREFILL = 1000   # samples (~4 seconds headroom)
_GAN_BATCH_SIZE    = 500    # epochs per predict() call (~2 seconds)
_PINK_NOISE_CHUNK  = 30     # seconds of pink noise generated at a time


# =============================================================================
#  [FIX-10] Async GAN Worker Thread
# =============================================================================
class GANWorker(threading.Thread):
    """
    Runs GAN predict() in a daemon thread so the 250 Hz acquisition loop
    never blocks on inference.  The main loop calls get_sample() which pops
    from a pre-filled queue (O(1)).  When the queue drops below the low-water
    mark this thread automatically refills it.
    """
    def __init__(self, gan_generator, norm_params, num_channels, latent_dim=128):
        super().__init__(daemon=True)
        self.gan_generator = gan_generator
        self.norm_params   = norm_params
        self.num_channels  = num_channels
        self.latent_dim    = latent_dim
        self._q            = queue.Queue(maxsize=8000)   # ~32 s at 250 Hz
        self._stop_event   = threading.Event()
        self._low_water    = _GAN_QUEUE_PREFILL // 2

    def run(self):
        # Pre-fill before returning control to caller
        self._generate_batch()
        while not self._stop_event.is_set():
            if self._q.qsize() < self._low_water:
                self._generate_batch()
            else:
                systime.sleep(0.05)   # idle: check queue every 50 ms

    def _generate_batch(self):
        noise = np.random.normal(0, 1, (_GAN_BATCH_SIZE, self.latent_dim))
        generated = self.gan_generator.predict(noise, verbose=0)
        min_val, max_val = self.norm_params
        denorm = ((generated + 1) / 2) * (max_val - min_val) + min_val
        flat = denorm.reshape(-1, self.num_channels)
        for row in flat:
            try:
                self._q.put_nowait(row)
            except queue.Full:
                break   # queue is full; drop and retry next cycle

    def get_sample(self):
        """Called by the main thread 250×/s — non-blocking, returns zeros on underrun."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            print("!! GAN queue underrun — returning zeros")
            return np.zeros(self.num_channels)

    def stop(self):
        self._stop_event.set()


# =============================================================================
#  Signal Generation v9.0
# =============================================================================
class SignalGenerator:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.head_model    = HeadModel(sampling_rate)
        self.num_channels  = self.head_model.num_channels
        self.time          = 0.0
        self.components_enabled = {'rhythms': True, 'noise': True, 'environment': True}

        self.state = {
            'load':      0.0,
            'drowsy':    0.0,
            'tension':   0.0,
            'attention': 1.0,
            'gaze_x':    0.0,   # -1 = full left,  +1 = full right
            'gaze_y':    0.0,   # -1 = full down,  +1 = full up
        }

        # --- [FIX-4] Per-channel drift: independent random phases ---
        rng = np.random.default_rng(seed=42)
        self.channel_drift_phase = rng.uniform(0, 2 * np.pi, self.num_channels)
        self.channel_drift_freq  = rng.uniform(0.03, 0.08, self.num_channels)   # 0.03–0.08 Hz
        self.channel_drift_amp   = rng.uniform(2.0,  6.0,  self.num_channels)   # μV per electrode

        # --- [FIX-3] Pink noise: index + regeneration flag ---
        self.pink_noise_source = self._generate_pink_noise(sampling_rate * _PINK_NOISE_CHUNK)
        self.noise_idx         = 0

        # --- GAN setup ---
        self.use_gan    = False
        self._gan_worker = None
        if TENSORFLOW_AVAILABLE:
            print("TensorFlow found. Attempting to load GAN generator...")
            try:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                gan_model   = tf.keras.models.load_model('gan_models/generator.h5', compile=False)
                norm_params = np.load('als_eeg_norm_params.npy')
                # [FIX-10] Start background worker; pre-fills queue before returning
                self._gan_worker = GANWorker(gan_model, norm_params, self.num_channels, LATENT_DIM)
                self._gan_worker.start()
                self.use_gan = True
                print("--> GAN Generator loaded; async worker running.")
            except (IOError, ValueError) as e:
                print(f"!! Failed to load GAN model: {e}. Falling back to procedural.")
        else:
            print("!! TensorFlow not found. Using procedural generation.")

        try:
            self.head_model.templates['blink'] = np.load('template_library/real_blink_template.npy')
        except FileNotFoundError:
            pass

        self.artifacts = {'powerline': False, 'involuntary_blink': False}
        self.last_involuntary_blink_time    = systime.time()
        self.next_involuntary_blink_interval = random.uniform(3, 8)
        self.command_queue = []
        self.source_locs   = self.head_model.get_source_locations()
        self.falloff       = {name: self.head_model.get_spatial_falloff(pos)
                              for name, pos in self.source_locs.items()}

    # ── Pink noise helpers ─────────────────────────────────────────────────

    def _generate_pink_noise(self, points):
        """FFT-based pink (1/f) noise, normalised to [-1, 1]."""
        white   = np.random.normal(size=points)
        fft_w   = np.fft.fft(white)
        freqs   = np.fft.fftfreq(points, 1 / self.sampling_rate)
        pink    = fft_w / (np.sqrt(np.abs(freqs)) + 1e-6)
        pink[0] = 0
        result  = np.fft.ifft(pink).real
        return result / np.max(np.abs(result))

    def _get_pink_sample(self):
        """[FIX-3] Returns one pink noise scalar; regenerates buffer when exhausted."""
        if self.noise_idx >= len(self.pink_noise_source):
            self.pink_noise_source = self._generate_pink_noise(
                self.sampling_rate * _PINK_NOISE_CHUNK)
            self.noise_idx = 0
        val = self.pink_noise_source[self.noise_idx]
        self.noise_idx += 1
        return val

    # ── Core sample generation ─────────────────────────────────────────────

    def get_next_sample(self):
        """Assembles the final EEG sample from all active signal components."""
        dt = 1.0 / self.sampling_rate
        t  = self.time

        # ── [FIX-5/6] Burst envelopes + frequency jitter ──────────────────
        # Each rhythm modulates amplitude with a slow sinusoidal envelope
        # (0.15–0.25 Hz) so bursts appear and fade, and the carrier frequency
        # drifts by ±0.5 Hz per rhythm to avoid perfectly locked sinusoids.
        alpha_env  = 0.5 + 0.5 * np.sin(2 * np.pi * 0.18 * t)
        beta_env   = 0.5 + 0.5 * np.sin(2 * np.pi * 0.22 * t + 1.1)
        theta_env  = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t + 2.3)

        alpha_freq = 10.0 + 0.5 * np.sin(2 * np.pi * 0.07 * t)
        beta_freq  = 20.0 + 0.8 * np.sin(2 * np.pi * 0.11 * t + 0.5)
        theta_freq =  6.0 + 0.4 * np.sin(2 * np.pi * 0.09 * t + 1.7)

        if self.use_gan:
            # [FIX-10] Pull from async queue — never blocks
            channel_signals = self._gan_worker.get_sample().copy()
            if self.components_enabled['rhythms']:
                beta_noise      = (np.random.randn(self.num_channels) * 0.2) * self.state['load']
                channel_signals += beta_noise * (self.falloff['motor_left'] + self.falloff['motor_right'])

                attn_suppression = (1.0 - self.state['attention']) * 15.0 * np.random.randn(self.num_channels)
                channel_signals -= attn_suppression * self.falloff['occipital_visual']

                # [FIX-5] Theta drift now has burst envelope
                theta_drift = 10.0 * self.state['drowsy'] * theta_env * np.sin(2 * np.pi * theta_freq * t)
                channel_signals += theta_drift * self.falloff['frontal_eyes']
        else:
            channel_signals = np.zeros(self.num_channels)
            if self.components_enabled['rhythms']:
                # [FIX-5/6] Alpha with burst + jitter
                alpha_amp  = 30.0 * (1.0 - self.state['load']) * self.state['attention']
                alpha_wave = alpha_amp * alpha_env * np.sin(2 * np.pi * alpha_freq * t)
                channel_signals += alpha_wave * self.falloff['occipital_visual']

                # [FIX-5/6] Beta with burst + jitter
                beta_amp  = 18.0 * self.state['load']
                beta_wave = beta_amp * beta_env * np.sin(2 * np.pi * beta_freq * t)
                channel_signals += beta_wave * (self.falloff['motor_left'] + self.falloff['motor_right'])

                # [FIX-5/6] Theta with burst + jitter
                theta_amp  = 35.0 * self.state['drowsy']
                theta_wave = theta_amp * theta_env * np.sin(2 * np.pi * theta_freq * t)
                channel_signals += theta_wave * self.falloff['frontal_eyes']

        # ── CONTINUOUS EOG DIPOLE ──────────────────────────────────────────
        heog  = self.state['gaze_x'] * 100.0 + np.random.normal(0, 0.8)
        veog  = self.state['gaze_y'] *  75.0 + np.random.normal(0, 0.6)
        channel_signals[0] += (-heog + veog)   # Fp1
        channel_signals[1] += ( heog + veog)   # Fp2

        # ── [FIX-2] Shared + local noise ──────────────────────────────────
        if self.components_enabled['noise']:
            noise_amp    = 5.0 + self.state['tension'] * 20.0
            common_noise = self._get_pink_sample() * noise_amp
            local_noise  = np.random.normal(0, noise_amp * 0.3, self.num_channels)
            channel_signals += common_noise + local_noise

        # ── [FIX-4] Per-channel independent electrode drift ────────────────
        if self.components_enabled['environment']:
            for ch in range(self.num_channels):
                phase = self.channel_drift_phase[ch] + 2 * np.pi * self.channel_drift_freq[ch] * t
                channel_signals[ch] += self.channel_drift_amp[ch] * np.sin(phase)

        # ── POWERLINE ARTIFACT ─────────────────────────────────────────────
        if self.artifacts.get('powerline', False):
            channel_signals += 20.0 * np.sin(2 * np.pi * 50.0 * t)

        # ── INVOLUNTARY BLINKS ─────────────────────────────────────────────
        if self.artifacts.get('involuntary_blink', False):
            now = systime.time()
            if now - self.last_involuntary_blink_time > self.next_involuntary_blink_interval:
                self.command_queue.append({'type': 'blink', 'idx': 0})
                self.last_involuntary_blink_time         = now
                self.next_involuntary_blink_interval     = random.uniform(3, 8)

        # ── COMMAND QUEUE ──────────────────────────────────────────────────
        if self.command_queue:
            cmd   = self.command_queue[0]
            ctype = cmd['type']
            if ctype == 'silence':
                cmd['idx'] += 1
                if cmd['idx'] >= cmd.get('duration', 1):
                    self.command_queue.pop(0)
            elif ctype == 'pop':
                spike_amp = random.uniform(150, 400) * random.choice([-1, 1])
                ch        = random.randint(0, self.num_channels - 1)
                channel_signals[ch] += spike_amp
                self.command_queue.pop(0)
            else:
                template = self.head_model.templates.get(ctype)
                if template is not None and cmd['idx'] < len(template):
                    artifact_sample = template[cmd['idx']]
                    key = 'global_tension' if ctype == 'jaw_clench' else 'frontal_eyes'
                    channel_signals += artifact_sample * self.falloff[key]
                    cmd['idx'] += 1
                    if cmd['idx'] >= len(template):
                        self.command_queue.pop(0)
                else:
                    self.command_queue.pop(0)

        self.time += dt
        return channel_signals

    # ── Utility methods ────────────────────────────────────────────────────

    def create_openbci_packet(self, channel_data_uV, sample_index):
        packet    = bytearray(33)
        packet[0] = HEADER_BYTE
        packet[1] = sample_index % 256
        for i in range(self.num_channels):
            counts = int(channel_data_uV[i] / SCALE_FACTOR_uV_PER_COUNT)
            packet[2 + i*3: 5 + i*3] = counts.to_bytes(3, 'big', signed=True)
        packet[32] = FOOTER_BYTE
        return packet

    def set_state_value(self, state_name, value):
        self.state[state_name] = value / 100.0

    def set_gaze(self, gaze_x, gaze_y):
        """Set gaze directly in [-1, 1] range. Called by the GazeTrackpad."""
        self.state['gaze_x'] = float(gaze_x)
        self.state['gaze_y'] = float(gaze_y)

    def toggle_artifact(self, artifact_name, state):
        self.artifacts[artifact_name] = bool(state)

    def execute_command(self, command, args=None):
        if command == 'single_blink':
            self.command_queue.append({'type': 'blink', 'idx': 0})
        elif command == 'double_blink':
            self.command_queue.append({'type': 'blink', 'idx': 0})
            self.command_queue.append({'type': 'silence', 'idx': 0,
                                       'duration': int(0.15 * self.sampling_rate)})
            self.command_queue.append({'type': 'blink', 'idx': 0})
        elif command == 'jaw_clench':
            self.command_queue.append({'type': 'jaw_clench', 'idx': 0})
        elif command == 'electrode_pop':
            self.command_queue.append({'type': 'pop', 'idx': 0})

    def shutdown(self):
        """Clean up background worker on app exit."""
        if self._gan_worker is not None:
            self._gan_worker.stop()


# =============================================================================
#  Gaze Trackpad Widget (v9.0 — unchanged logic, EOG buttons removed in GUI)
# =============================================================================
class GazeTrackpad(QWidget):
    """
    2-D click-and-drag widget that drives continuous gaze_x / gaze_y in SignalGenerator.
    Gaze range: [-1, 1] on both axes.  Centre = (0, 0) = eyes straight ahead.
    On mouse release the dot springs back to centre with exponential decay.
    """
    def __init__(self, signal_gen, parent=None):
        super().__init__(parent)
        self.signal_gen = signal_gen
        self.setMinimumSize(180, 180)
        self.gaze_x    = 0.0
        self.gaze_y    = 0.0
        self._dragging = False
        self._spring   = QTimer(self)
        self._spring.setInterval(16)
        self._spring.timeout.connect(self._spring_back)
        self._spring.start()

    def _clamp(self, v):
        return max(-1.0, min(1.0, v))

    def _spring_back(self):
        if not self._dragging:
            self.gaze_x *= 0.85
            self.gaze_y *= 0.85
            if abs(self.gaze_x) < 0.01: self.gaze_x = 0.0
            if abs(self.gaze_y) < 0.01: self.gaze_y = 0.0
            self.signal_gen.set_gaze(self.gaze_x, self.gaze_y)
            self.update()

    def mousePressEvent(self, e):
        self._dragging = True
        self._update_gaze(e.position())

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._update_gaze(e.position())

    def mouseReleaseEvent(self, e):
        self._dragging = False

    def _update_gaze(self, pos):
        w, h = self.width(), self.height()
        self.gaze_x = self._clamp((pos.x() / w) * 2.0 - 1.0)
        self.gaze_y = self._clamp(-((pos.y() / h) * 2.0 - 1.0))
        self.signal_gen.set_gaze(self.gaze_x, self.gaze_y)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(30, 30, 40))
        pen = QPen(QColor(60, 60, 80)); pen.setWidth(1); p.setPen(pen)
        p.drawLine(w // 2, 0, w // 2, h)
        p.drawLine(0, h // 2, w, h // 2)
        pen.setColor(QColor(80, 80, 120)); pen.setWidth(2); p.setPen(pen)
        p.drawRect(1, 1, w - 2, h - 2)
        cx = int((self.gaze_x + 1.0) / 2.0 * w)
        cy = int((-self.gaze_y + 1.0) / 2.0 * h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 200, 255)))
        p.drawEllipse(cx - 8, cy - 8, 16, 16)
        p.setPen(QPen(QColor(120, 120, 160)))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(4, h - 4, "← HEOG-   HEOG+ →")


# =============================================================================
#  GUI v9.0 — Full Control Panel
# =============================================================================
class NEST_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 250
        self.head_model    = HeadModel(self.sampling_rate)
        self.num_channels  = self.head_model.num_channels
        self.sample_counter = 0

        self.setWindowTitle("NEST v9.0: The Complete Naos Control Panel")
        self.setGeometry(50, 50, 1850, 1000)

        main_widget  = QWidget(); self.setCentralWidget(main_widget)
        main_layout  = QHBoxLayout(main_widget)
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)

        left_panel_layout.addWidget(self._create_state_control_box())
        left_panel_layout.addWidget(self._create_command_box())    # [FIX-1] No EOG buttons
        left_panel_layout.addWidget(self._create_chaos_box())
        left_panel_layout.addWidget(self._create_diagnostic_box())

        self.spectral_plot = pg.PlotWidget()
        self.spectral_plot.setTitle("Real-Time Spectral Analysis (PSD)", color='w')
        self.spectral_plot.enableAutoRange(axis='y'); self.spectral_plot.setXRange(0, 60)
        self.psd_curve = self.spectral_plot.plot(pen=pg.mkPen('y', width=2))
        left_panel_layout.addWidget(self.spectral_plot)
        self.theta_label = QLabel("Theta: 0%")
        self.alpha_label = QLabel("Alpha: 0%")
        self.beta_label  = QLabel("Beta: 0%")
        self.gamma_label = QLabel("Gamma: 0%")

        self.dominant_label = QLabel("Dominant Band: Alpha")

        left_panel_layout.addWidget(self.theta_label)
        left_panel_layout.addWidget(self.alpha_label)
        left_panel_layout.addWidget(self.beta_label)
        left_panel_layout.addWidget(self.gamma_label)
        left_panel_layout.addWidget(self.dominant_label)
        right_panel_widget = self._create_oscilloscope_panel()
        main_layout.addWidget(left_panel_widget)
        main_layout.addWidget(right_panel_widget, stretch=3)

        # --- [FIX-8] Ring buffer ---
        self.data_buffer_size = self.sampling_rate * 3
        self.data_buffer      = np.zeros((self.data_buffer_size, self.num_channels))
        self.ring_idx         = 0   # circular write pointer

        self.plot_curves      = [plot.plot(pen=pg.mkPen('c', width=2))
                                 for plot in self.oscilloscope_plots]
        self.signal_generator = SignalGenerator(self.sampling_rate)

        # Gaze trackpad — added after signal_generator exists
        gaze_frame = QFrame(); gaze_frame.setFrameShape(QFrame.Shape.StyledPanel)
        gaze_layout = QVBoxLayout(gaze_frame)
        gaze_layout.addWidget(QLabel(
            "<b>Gaze Trackpad (EOG)</b><br>"
            "<small>Click + Drag to steer gaze. Release to spring back.</small>"))
        self.gaze_trackpad = GazeTrackpad(self.signal_generator)
        gaze_layout.addWidget(self.gaze_trackpad)
        left_panel_layout.addWidget(gaze_frame)

        # --- LSL outlets ---
        print("NEST v9.0: Configuring LSL outlets...")
        self.eeg_outlet = StreamOutlet(
            StreamInfo('NEST_Cyton_RAW', 'EEG', 33, self.sampling_rate, 'int8', 'nest_eeg_v9.0'))
        print("--> EEG outlet streaming hardware-accurate binary packets.")

        self.eeg_clean_outlet = StreamOutlet(
            StreamInfo('NEST_EEG_Clean', 'EEG', 8, self.sampling_rate, 'float32', 'nest_eeg_clean_v9.0'))
        print("--> Clean EEG outlet (float32, 8ch, μV) ready.")

        self.behavioral_outlet = StreamOutlet(
            StreamInfo('NEST_Behavioral_Markers', 'Markers', 1, 0, 'string', 'nest_behavioral_v9.0'))
        print("--> Behavioral marker outlet ready.")

        self._connect_signals()

        # Timers
        self.plot_timer = QTimer()
        self.plot_timer.setInterval(int(1000 / 60))
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start()

        self.signal_timer = QTimer()
        self.signal_timer.setInterval(16)
        self.signal_timer.timeout.connect(self.update_signal)
        self.signal_timer.start()

        self.unprocessed_samples = 0.0
        self.last_update_time    = systime.time()

        # LSL chunk accumulators (filled each tick, pushed once) [FIX-9]
        self._raw_chunk   = []
        self._clean_chunk = []

    # ── Widget builders ────────────────────────────────────────────────────

    def _create_state_control_box(self):
        g = QGridLayout()
        g.addWidget(QLabel("<b>State Controls</b>"), 0, 0, 1, 3)
        self.load_slider = QSlider(Qt.Orientation.Horizontal); self.load_slider.setRange(0, 100)
        self.drowsy_slider = QSlider(Qt.Orientation.Horizontal); self.drowsy_slider.setRange(0, 100)
        self.tension_slider = QSlider(Qt.Orientation.Horizontal); self.tension_slider.setRange(0, 100)
        self.attention_slider = QSlider(Qt.Orientation.Horizontal)
        self.attention_slider.setRange(0, 100); self.attention_slider.setValue(100)
        for row, (lbl, slider) in enumerate([
            ("Cognitive Load:", self.load_slider),
            ("Drowsiness:",     self.drowsy_slider),
            ("Muscle Tension:", self.tension_slider),
            ("Attention/Focus:", self.attention_slider),
        ], start=1):
            g.addWidget(QLabel(lbl), row, 0)
            g.addWidget(slider, row, 1, 1, 2)
        box = QFrame(); box.setFrameShape(QFrame.Shape.StyledPanel); box.setLayout(g)
        return box

    def _create_command_box(self):
        """[FIX-1] EOG directional buttons removed — GazeTrackpad is the only EOG input."""
        g = QGridLayout()
        g.addWidget(QLabel("<b>Patient Commands</b>"), 0, 0, 1, 2)
        self.single_blink_btn  = QPushButton("Execute Single Blink");  g.addWidget(self.single_blink_btn,  1, 0)
        self.double_blink_btn  = QPushButton("Execute Double-Blink");  g.addWidget(self.double_blink_btn,  1, 1)
        self.jaw_button        = QPushButton("Inject Jaw Clench");      g.addWidget(self.jaw_button,        2, 0)
        self.pop_button        = QPushButton("Inject Electrode Pop");   g.addWidget(self.pop_button,        2, 1)
        self.error_button      = QPushButton("Simulate User Error");    g.addWidget(self.error_button,      3, 0)
        self.back_spam_button  = QPushButton("Simulate 'Back' Spam");  g.addWidget(self.back_spam_button,  3, 1)
        box = QFrame(); box.setFrameShape(QFrame.Shape.StyledPanel); box.setLayout(g)
        return box

    def _create_chaos_box(self):
        g = QGridLayout()
        g.addWidget(QLabel("<b>Chaos Monkey (Environment)</b>"), 0, 0, 1, 2)
        self.powerline_checkbox = QCheckBox("Enable 50Hz Powerline Noise")
        self.involuntary_blink_checkbox = QCheckBox("Enable Involuntary Blinks")
        self.no_interaction_checkbox    = QCheckBox("Simulate No Interaction")
        g.addWidget(self.powerline_checkbox,          1, 0)
        g.addWidget(self.involuntary_blink_checkbox,  1, 1)
        g.addWidget(self.no_interaction_checkbox,     2, 0, 1, 2)
        box = QFrame(); box.setFrameShape(QFrame.Shape.StyledPanel); box.setLayout(g)
        return box

    def _create_diagnostic_box(self):
        g = QGridLayout()
        g.addWidget(QLabel("<b>Diagnostic Controls</b>"), 0, 0, 1, 2)
        self.rhythms_checkbox = QCheckBox("Enable Brain Rhythms"); self.rhythms_checkbox.setChecked(True)
        self.noise_checkbox   = QCheckBox("Enable Pink Noise");    self.noise_checkbox.setChecked(True)
        self.env_checkbox     = QCheckBox("Enable Drift/Environment"); self.env_checkbox.setChecked(True)
        g.addWidget(self.rhythms_checkbox, 1, 0); g.addWidget(self.noise_checkbox, 1, 1)
        g.addWidget(self.env_checkbox,     2, 0)
        box = QFrame(); box.setFrameShape(QFrame.Shape.StyledPanel); box.setLayout(g)
        return box

    def _create_oscilloscope_panel(self):
        layout = QVBoxLayout(); widget = QWidget(); widget.setLayout(layout)
        self.oscilloscope_plots = []
        for i in range(self.num_channels):
            plot = pg.PlotWidget()
            plot.getAxis('left').setLabel(self.head_model.channel_names[i], color='c')
            plot.getAxis('bottom').setVisible(False)
            plot.setYRange(-100, 100); plot.getAxis('left').setWidth(40)
            layout.addWidget(plot); self.oscilloscope_plots.append(plot)
        return widget

    # ── Signal connections ─────────────────────────────────────────────────

    def _send_behavioral_marker(self, marker_string):
        self.behavioral_outlet.push_sample([marker_string])
        print(f"--> Marker: '{marker_string}'")

    def _connect_signals(self):
        # State sliders
        self.load_slider.valueChanged.connect(
            lambda v: self.signal_generator.set_state_value('load', v))
        self.drowsy_slider.valueChanged.connect(
            lambda v: self.signal_generator.set_state_value('drowsy', v))
        self.tension_slider.valueChanged.connect(
            lambda v: self.signal_generator.set_state_value('tension', v))
        self.attention_slider.valueChanged.connect(
            lambda v: self.signal_generator.set_state_value('attention', v))

        # Artifact commands
        self.single_blink_btn.clicked.connect(
            lambda: self.signal_generator.execute_command('single_blink'))
        self.double_blink_btn.clicked.connect(lambda: [
            self.signal_generator.execute_command('double_blink'),
            self._send_behavioral_marker('double_blink')])
        self.jaw_button.clicked.connect(lambda: [
            self.signal_generator.execute_command('jaw_clench'),
            self._send_behavioral_marker('jaw_clench')])
        self.pop_button.clicked.connect(lambda: [
            self.signal_generator.execute_command('electrode_pop'),
            self._send_behavioral_marker('electrode_pop')])

        # Behavioral markers
        self.error_button.clicked.connect(
            lambda: self._send_behavioral_marker('user_error'))
        self.back_spam_button.clicked.connect(
            lambda: self._send_behavioral_marker('back_spam'))
        self.no_interaction_checkbox.stateChanged.connect(
            lambda s: self._send_behavioral_marker(
                'no_interaction_start' if s else 'no_interaction_stop'))

        # Chaos monkey
        self.powerline_checkbox.stateChanged.connect(
            lambda s: self.signal_generator.toggle_artifact('powerline', s))
        self.involuntary_blink_checkbox.stateChanged.connect(
            lambda s: self.signal_generator.toggle_artifact('involuntary_blink', s))

        # Diagnostic toggles
        self.rhythms_checkbox.stateChanged.connect(
            lambda s: self.signal_generator.components_enabled.update({'rhythms': bool(s)}))
        self.noise_checkbox.stateChanged.connect(
            lambda s: self.signal_generator.components_enabled.update({'noise': bool(s)}))
        self.env_checkbox.stateChanged.connect(
            lambda s: self.signal_generator.components_enabled.update({'environment': bool(s)}))

    # ── Main acquisition loop ──────────────────────────────────────────────

    def update_signal(self):
        now     = systime.time()
        elapsed = now - self.last_update_time
        self.last_update_time = now

        self.unprocessed_samples += elapsed * self.sampling_rate
        samples_to_generate = int(self.unprocessed_samples)
        self.unprocessed_samples -= samples_to_generate

        # [FIX-7] Late-timer warning
        if samples_to_generate > 8:
            print(f"!! Timer late: {samples_to_generate} samples in one tick "
                  f"(expected ≤4). Qt timer fired ~{elapsed*1000:.0f} ms late.")

        # Accumulators for chunked LSL push [FIX-9]
        raw_chunk   = []
        clean_chunk = []

        for _ in range(samples_to_generate):
            sample_uV  = self.signal_generator.get_next_sample()
            byte_packet = self.signal_generator.create_openbci_packet(sample_uV, self.sample_counter)
            raw_chunk.append(list(byte_packet))
            clean_chunk.append(sample_uV.tolist())
            self.sample_counter += 1

            # [FIX-8] Ring buffer — O(1) write, no array shift
            self.data_buffer[self.ring_idx] = sample_uV
            self.ring_idx = (self.ring_idx + 1) % self.data_buffer_size

        # [FIX-9] Single push_chunk() per tick instead of N push_sample() calls
        if raw_chunk:
            self.eeg_outlet.push_chunk(raw_chunk)
            self.eeg_clean_outlet.push_chunk(clean_chunk)

    def update_plots(self):
        """[FIX-8] Unroll the circular buffer before rendering so oldest→newest is left→right."""

        ordered = np.roll(self.data_buffer, -self.ring_idx, axis=0)

        for i in range(self.num_channels):
            self.plot_curves[i].setData(ordered[:, i])

        freqs, psd = signal.welch(
            ordered[:, 2],
            self.sampling_rate,
            nperseg=int(self.sampling_rate * 2)
        )

        # PSD display
        psd_log = 10.0 * np.log10(psd + 1e-12)
        self.psd_curve.setData(freqs, psd_log)

        # Band masks
        theta_mask = (freqs >= 4) & (freqs < 8)
        alpha_mask = (freqs >= 8) & (freqs < 13)
        beta_mask  = (freqs >= 13) & (freqs < 30)
        gamma_mask = (freqs >= 30) & (freqs < 50)

        # Band powers
        theta = np.sum(psd[theta_mask])
        alpha = np.sum(psd[alpha_mask])
        beta  = np.sum(psd[beta_mask])
        gamma = np.sum(psd[gamma_mask])

        total = theta + alpha + beta + gamma + 1e-12

        theta_pct = 100.0 * theta / total
        alpha_pct = 100.0 * alpha / total
        beta_pct  = 100.0 * beta  / total
        gamma_pct = 100.0 * gamma / total

        # Update labels
        self.theta_label.setText(f"Theta: {theta_pct:.1f}%")
        self.alpha_label.setText(f"Alpha: {alpha_pct:.1f}%")
        self.beta_label.setText(f"Beta: {beta_pct:.1f}%")
        self.gamma_label.setText(f"Gamma: {gamma_pct:.1f}%")

        # Dominant band
        bands = {
            "Theta": theta_pct,
            "Alpha": alpha_pct,
            "Beta": beta_pct,
            "Gamma": gamma_pct
        }

        dominant = max(bands, key=bands.get)

        self.dominant_label.setText(
            f"Dominant Band: {dominant}"
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = NEST_GUI()
    main_window.show()
    sys.exit(app.exec())