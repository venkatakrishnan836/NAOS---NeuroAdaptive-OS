"""
signal_router.py
────────────────
NAOS Signal Router — File 1 of 6

Responsibility:
    Single interface between any signal source and the rest of the
    NAOS pipeline. Currently connects to NEST via LSL.
    When BrainBit SDK arrives: change MODE = 'lsl' stays the same,
    only STREAM_NAME changes to whatever BrainBit publishes.

NEST stream facts (verified 2026-05-31):
    Stream name : NEST_EEG_Clean
    Type        : EEG
    Channels    : 8  (all EEG, float32 microvolts, no packet header)
    EEG indices : 0-7  (all 8 channels are clean EEG — no slicing needed)
    Sample rate : 250 Hz
    Value range : ~-17000 to +22000 uV (float32 microvolts)
    Chunk size  : irregular 22-31 samples per pull (NOT 250)
    Fix applied : ring buffer — accumulates until exactly WINDOW_SAMPLES
                  are available, then yields one clean (8, 250) window.

Coordinate map (10-20 system):
    Index 0 → Fp1  (frontal-polar left)
    Index 1 → Fp2  (frontal-polar right)
    Index 2 → C3   (central left)
    Index 3 → C4   (central right)
    Index 4 → P7   (parietal-temporal left)
    Index 5 → P8   (parietal-temporal right)
    Index 6 → O1   (occipital left)
    Index 7 → O2   (occipital right)

Output:
    get_window() → numpy ndarray shape (8, 250), dtype float32
                   values in raw Cyton units (-128 to 127)
                   caller (feature_extractor) handles μV conversion

Usage:
    router = SignalRouter()
    router.connect()
    while True:
        window = router.get_window()   # blocks until 250 samples ready
        if window is not None:
            # window.shape == (8, 250)
            process(window)
    router.disconnect()
"""

import time
import threading
import collections
import logging
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

STREAM_NAME     = "NEST_EEG_Clean"   # confirmed stream — 8ch float32 microvolts
STREAM_TYPE     = "EEG"
EEG_CHANNELS    = 8                   # indices 0-7 out of 33
TOTAL_CHANNELS  = 8                   # NEST_EEG_Clean — all 8 are EEG
SAMPLE_RATE     = 250                 # Hz — nominal, used for buffer sizing
WINDOW_SAMPLES  = 250                 # 1 second window
WINDOW_STEP     = 63                  # ~252ms per step — server.py owns actual 250ms cadence via asyncio.sleep
RESOLVE_TIMEOUT = 10.0                # seconds to wait for LSL stream
PULL_TIMEOUT    = 0.1                 # seconds per pull_chunk call
MAX_PULL        = 128                 # max samples per pull (> largest chunk)
MAX_FILL_S      = 3.0                 # max wall-clock seconds to fill a window
                                      # if fill takes longer, data is sparse
                                      # and Welch PSD at fs=250 is meaningless

# ── logger ───────────────────────────────────────────────────────────────────

log = logging.getLogger("signal_router")


# ─────────────────────────────────────────────────────────────────────────────
class SignalRouter:
    """
    Connects to NEST's LSL outlet and yields 1-second EEG windows.

    Thread safety:
        connect() starts an internal reader thread that continuously
        pulls chunks from LSL and appends them to a deque ring buffer.
        get_window() reads from that deque in the calling thread.
        A threading.Event signals when enough samples are available.
    """

    def __init__(self):
        # ring buffer — holds raw samples as they arrive from LSL
        # maxlen = 3 × WINDOW_SAMPLES to avoid memory growth
        self._buffer   = collections.deque(maxlen=WINDOW_SAMPLES * 3)
        self._lock     = threading.Lock()
        self._ready    = threading.Event()   # set when buffer >= WINDOW_SAMPLES
        self._running  = False
        self._thread   = None
        self._inlet    = None
        self._connected = False

        # sliding window pointer — how many samples to advance per get_window()
        # WINDOW_STEP = 63 → ~252ms cadence at 250Hz
        self._step = WINDOW_STEP

        # stale window detection — track when the buffer was last
        # drained below WINDOW_SAMPLES (i.e. needs refilling)
        self._fill_start_time = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Resolve the LSL stream and start the reader thread.
        Returns True on success, False if stream not found.

        Must be called before get_window().
        NEST must be running before calling connect().
        """
        try:
            from pylsl import StreamInlet, resolve_streams

            log.info(f"Resolving LSL stream '{STREAM_NAME}' ...")
            streams = [s for s in resolve_streams(wait_time=RESOLVE_TIMEOUT)
                              if s.name() == STREAM_NAME]

            if not streams:
                log.error(
                    f"Stream '{STREAM_NAME}' not found after {RESOLVE_TIMEOUT}s. "
                    "Is NEST running?"
                )
                return False

            # max_buflen=360 → 360 seconds of buffer in LSL (overkill but safe)
            self._inlet = StreamInlet(streams[0], max_buflen=360)
            info = self._inlet.info()
            log.info(
                f"Connected: '{info.name()}' | "
                f"ch={info.channel_count()} | "
                f"rate={info.nominal_srate()}Hz"
            )

            self._connected = True
            self._running   = True
            self._thread    = threading.Thread(
                target=self._reader_loop,
                name="lsl-reader",
                daemon=True          # dies with main process
            )
            self._thread.start()
            log.info("Reader thread started.")
            return True

        except Exception as exc:
            log.error(f"connect() failed: {exc}")
            return False

    def get_window(self, timeout: float = 5.0):
        """
        Block until WINDOW_SAMPLES (250) samples are in the buffer,
        then return a (8, 250) float32 numpy array.

        Advances the buffer by WINDOW_STEP (62 samples = 250ms) so that
        consecutive calls produce overlapping windows at 4Hz cadence.

        Returns:
            np.ndarray shape (8, 250) dtype float32, or
            None if timeout exceeded (stream stalled / NEST closed).
        """
        if not self._connected:
            log.warning("get_window() called before connect().")
            return None

        # wait until buffer has enough samples
        if not self._ready.wait(timeout=timeout):
            log.warning(f"get_window() timed out after {timeout}s — buffer not filling.")
            return None

        with self._lock:
            buf_list = list(self._buffer)

        n = len(buf_list)
        if n < WINDOW_SAMPLES:
            # edge case: event was set but samples drained between wait and lock
            self._ready.clear()
            return None

        # grab the most recent WINDOW_SAMPLES rows
        # buf_list is ordered oldest → newest
        window_rows = buf_list[-WINDOW_SAMPLES:]          # list of 250 rows
        window_np   = np.array(window_rows, dtype=np.float32)  # (250, 8)
        window_out  = window_np.T                         # (8, 250)

        # ── stale window check ───────────────────────────────────────
        # If the buffer took too long to fill, these 250 samples were
        # collected over many seconds, NOT over 1 second at 250Hz.
        # Welch PSD with fs=250 would produce garbage frequencies.
        fill_duration = time.time() - self._fill_start_time
        if self._fill_start_time > 0 and fill_duration > MAX_FILL_S:
            log.warning(
                f"Stale window discarded: took {fill_duration:.1f}s to fill "
                f"(max {MAX_FILL_S}s). NEST data rate too low."
            )
            # Drain the buffer so we start fresh next time
            with self._lock:
                self._buffer.clear()
                self._ready.clear()
                self._fill_start_time = time.time()
            return None

        # advance buffer by WINDOW_STEP to maintain sliding window
        with self._lock:
            for _ in range(self._step):
                if self._buffer:
                    self._buffer.popleft()
            # re-evaluate readiness after consuming
            if len(self._buffer) < WINDOW_SAMPLES:
                self._ready.clear()
                self._fill_start_time = time.time()

        return window_out   # shape (8, 250), float32

    def disconnect(self):
        """Stop reader thread and close LSL inlet cleanly."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._inlet:
            self._inlet.close_stream()
        self._connected = False
        log.info("SignalRouter disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── internal reader thread ────────────────────────────────────────────────

    def _reader_loop(self):
        """
        Runs in a background daemon thread.
        Continuously pulls chunks from the LSL inlet and appends
        each sample (first 8 channels only) to the ring buffer.

        Key details:
        - pull_chunk returns list-of-lists: [[ch0,ch1,...,ch32], ...]
        - we slice [:EEG_CHANNELS] to keep only indices 0-7
        - samples arrive in bursts of 22-31 (verified from NEST)
        - we append one row at a time to preserve order
        - _ready event is set once buffer >= WINDOW_SAMPLES
        """
        log.info("_reader_loop started.")

        while self._running:
            try:
                # pull_chunk returns (samples, timestamps)
                # samples: list of rows, each row is list of 33 floats
                samples, _ = self._inlet.pull_chunk(
                    timeout=PULL_TIMEOUT,
                    max_samples=MAX_PULL
                )

                if not samples:
                    # no data this tick — normal, just continue
                    continue

                with self._lock:
                    for row in samples:
                        # row is a list of 33 values
                        # we keep only the first 8 (EEG channels)
                        eeg_row = row[:EEG_CHANNELS]
                        self._buffer.append(eeg_row)

                    # signal get_window() if we have enough
                    if len(self._buffer) >= WINDOW_SAMPLES:
                        self._ready.set()

            except Exception as exc:
                log.error(f"_reader_loop error: {exc}")
                time.sleep(0.1)   # brief pause before retry

        log.info("_reader_loop stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# Run: python signal_router.py
# NEST must be running. Prints 5 windows and their stats.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    print("\n" + "="*60)
    print("  NAOS signal_router.py — Self Test")
    print("  NEST must be open and streaming.")
    print("="*60 + "\n")

    router = SignalRouter()

    if not router.connect():
        print("\n❌ FAIL: Could not connect to NEST LSL stream.")
        print("   Make sure NEST is running before this test.\n")
        exit(1)

    print("✅ Connected. Reading 5 windows...\n")

    passed = 0
    for i in range(5):
        t0  = time.time()
        win = router.get_window(timeout=5.0)
        dt  = (time.time() - t0) * 1000

        if win is None:
            print(f"  Window {i+1}: ❌ NONE returned (timeout)")
            continue

        # ── shape check ──
        shape_ok = (win.shape == (8, 250))

        # ── dtype check ──
        dtype_ok = (win.dtype == np.float32)

        # ── range check: values must be in Cyton 8-bit range ──
        range_ok = (win.min() >= -200.0 and win.max() <= 200.0)

        # ── not-all-zeros check ──
        nonzero_ok = (np.std(win) > 0.1)

        status = "✅" if all([shape_ok, dtype_ok, range_ok, nonzero_ok]) else "❌"
        if all([shape_ok, dtype_ok, range_ok, nonzero_ok]):
            passed += 1

        print(
            f"  Window {i+1}: {status} | "
            f"shape={win.shape} | "
            f"dtype={win.dtype} | "
            f"min={win.min():.1f} | "
            f"max={win.max():.1f} | "
            f"std={win.std():.2f} | "
            f"took {dt:.0f}ms"
        )

    router.disconnect()

    print(f"\n{'='*60}")
    print(f"  Result: {passed}/5 windows passed all checks.")
    if passed == 5:
        print("  ✅ signal_router.py is working correctly.")
        print("  ✅ Safe to proceed to feature_extractor.py")
    else:
        print("  ❌ Fix issues above before proceeding.")
    print("="*60 + "\n")
