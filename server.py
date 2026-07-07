"""
server.py
─────────
NAOS FastAPI Server — File 5 of 6

Responsibilities:
    - Connects to NEST_EEG_Clean via signal_router (EEG data)
    - Connects to NEST_Behavioral_Markers (event markers)
    - Runs signal loop every 250ms via asyncio
    - Feeds pipeline: signal_router → extractor → state → detectors
    - Broadcasts JSON payload over WebSocket to all connected dashboards
    - Calls Gemini API on llm_trigger events
    - Exposes REST endpoints for dashboard interaction

Endpoints:
    WS   /ws              — real-time state stream (250ms)
    POST /api/dismiss     — user dismissed emergency overlay
    POST /api/llm         — manual LLM trigger for demo
    GET  /api/status      — health check + current state summary

WebSocket payload schema:
    {
        "state":    {attention, load, tension, drowsiness, confidence},
        "features": {bands_normalized, global_std, fp_std, o_std, ...},
        "events":   [{type, timestamp, data}],
        "theme":    "NEUTRAL" | "HIGH_CONTRAST" | "WARM_DIM",
        "llm":      null | {message, timestamp}
    }

Environment variables:
    GEMINI_API_KEY   — required for LLM feature
    NEST_READY       — set to 0 to run in demo mode without NEST

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

    Or directly:
    python server.py
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Set

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── load .env file (no extra dependency needed) ──────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                os.environ.setdefault(key, val)

# ── local imports ─────────────────────────────────────────────────────────────
from signal_router   import SignalRouter
from feature_extractor import FeatureExtractor
from state_engine    import StateEngine
from detectors       import Detectors

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger("server")

# ── constants ─────────────────────────────────────────────────────────────────
LOOP_INTERVAL_S      = 0.250          # 250ms signal loop cadence
MARKER_POLL_S        = 0.100          # 100ms marker polling
GEMINI_MODEL         = "gemini-1.5-flash"
GEMINI_ENDPOINT      = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
MARKER_STREAM_NAME   = "NEST_Behavioral_Markers"
LLM_MAX_TOKENS       = 100

# ── shared state (module-level, accessed by all async tasks) ──────────────────
_router    : Optional[SignalRouter]   = None
_extractor : Optional[FeatureExtractor] = None
_engine    : Optional[StateEngine]   = None
_detectors : Optional[Detectors]     = None
_marker_inlet                        = None   # pylsl StreamInlet or None

_clients   : Set[WebSocket]          = set()  # connected WS clients
_last_state: dict                    = {}     # most recent state_dict
_last_features: dict                 = {}     # most recent feature_dict
_last_llm  : Optional[dict]         = None   # last LLM response {message, ts}
_loop_running: bool                  = False


# ── Gemini API call ───────────────────────────────────────────────────────────

async def call_gemini(state: dict) -> Optional[str]:
    """
    Call Gemini API with physiological evidence from state_dict.
    Returns one-sentence wellness suggestion, or None on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        log.warning("GEMINI_API_KEY not set — returning demo message")
        return (
            f"Your cognitive load is elevated at "
            f"{state.get('load', 0):.0%} — "
            "try taking three slow deep breaths to reset your focus."
        )

    prompt = (
        "You are a wellness assistant embedded in an adaptive operating "
        "system. You receive real physiological evidence from EEG sensors. "
        "Respond in exactly ONE sentence. Be specific and actionable. "
        "Do not use medical language. Do not diagnose.\n\n"
        f"Physiological evidence:\n"
        f"  Cognitive load:  {state.get('load', 0.5):.2f} / 1.0\n"
        f"  Attention level: {state.get('attention', 0.5):.2f} / 1.0\n"
        f"  Physical tension:{state.get('tension', 0.5):.2f} / 1.0\n"
        f"  Drowsiness:      {state.get('drowsiness', 0.5):.2f} / 1.0\n\n"
        "Suggest one simple, immediate wellness action the user can "
        "take right now based on this evidence."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": LLM_MAX_TOKENS,
            "temperature": 0.7,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{GEMINI_ENDPOINT}?key={api_key}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data["candidates"][0]["content"]["parts"][0]["text"]
                    .strip())
            log.info(f"Gemini response: {text[:80]}...")
            return text
    except Exception as exc:
        log.error(f"Gemini API error: {exc}")
        return (
            "Your sensors show elevated activity — "
            "pause for a moment and take a few slow breaths."
        )


# ── broadcast helper ──────────────────────────────────────────────────────────

async def broadcast(payload: dict):
    """Send JSON payload to all connected WebSocket clients."""
    global _clients
    if not _clients:
        return
    message = json.dumps(payload, default=str)
    dead    = set()
    for ws in list(_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _clients -= dead


# ── signal loop ───────────────────────────────────────────────────────────────

async def signal_loop():
    """
    Main 250ms loop.
    Pulls one EEG window → extract → state → detect → broadcast.
    Handles llm_trigger events by calling Gemini asynchronously.

    No synthetic fallback — only real NEST data is processed.
    If get_window() returns None, the pipeline stays idle.
    """
    global _last_state, _last_features, _last_llm, _loop_running

    log.info("Signal loop starting...")
    _loop_running = True
    _waiting_logged = False

    while _loop_running:
        t0 = time.perf_counter()

        try:
            # ── 1. Get EEG window (real data only) ───────────────────
            if _router is None:
                # No router — NEST not connected, wait and retry
                if not _waiting_logged:
                    log.warning("No NEST connection — waiting for signal.")
                    _waiting_logged = True
                await asyncio.sleep(1.0)
                continue

            # Run blocking get_window in thread executor so we
            # don't freeze the asyncio event loop.
            # Timeout 5s — NEST on Windows may deliver samples
            # slower than 250Hz due to QTimer resolution.
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _router.get_window(timeout=5.0)
            )
            if raw is None:
                if not _waiting_logged:
                    log.warning(
                        "No EEG data from NEST — pipeline idle. "
                        "Waiting for real signal."
                    )
                    _waiting_logged = True
                await asyncio.sleep(LOOP_INTERVAL_S)
                continue

            # Got real data — reset waiting flag
            if _waiting_logged:
                log.info("EEG data received — pipeline active.")
                _waiting_logged = False

            # ── 2. Feature extraction ────────────────────────────────
            features = _extractor.compute(raw)
            _last_features = features

            # ── 3. State engine ──────────────────────────────────────
            state = _engine.compute(features)
            _last_state = state

            # ── 4. Detectors ─────────────────────────────────────────
            events = _detectors.update(state)

            # ── 5. Handle LLM trigger ────────────────────────────────
            for ev in events:
                if ev['type'] == 'llm_trigger':
                    # Fire Gemini call in background — don't block loop
                    asyncio.create_task(_handle_llm(ev['data']))

            # ── 6. Build and broadcast payload ───────────────────────
            # Strip heavy fields from features before sending
            # band_powers is a dict of dicts — JSON-safe but large
            # We send bands_normalized (8 floats) instead
            payload = {
                "state":    state,
                "features": {
                    "bands_normalized": features.get("bands_normalized", {}),
                    "global_std":       features.get("global_std", 0),
                    "fp_std":           features.get("fp_std", 0),
                    "o_std":            features.get("o_std", 0),
                    "c3_std":           features.get("c3_std", 0),
                    "theta_fp2":        features.get("theta_fp2", 0),
                    "quality":          features.get("quality", 1),
                    "window_count":     features.get("window_count", 0),
                    "compute_ms":       features.get("compute_ms", 0),
                },
                "events":   events,
                "theme":    _detectors.current_theme,
                "llm":      _last_llm,
            }
            await broadcast(payload)

        except Exception as exc:
            log.error(f"Signal loop error: {exc}", exc_info=True)

        # ── 7. Maintain 250ms cadence ────────────────────────────────
        elapsed = time.perf_counter() - t0
        sleep   = max(0.0, LOOP_INTERVAL_S - elapsed)
        await asyncio.sleep(sleep)

    log.info("Signal loop stopped.")


async def _handle_llm(event_data: dict):
    """
    Background task: call Gemini, store result, broadcast immediately.
    Does not block the signal loop.
    """
    global _last_llm
    log.info("LLM trigger — calling Gemini...")
    message = await call_gemini(event_data)
    if message:
        _last_llm = {"message": message, "timestamp": time.time()}
        # Broadcast immediately so dashboard shows without waiting for next loop
        await broadcast({
            "state":    _last_state,
            "features": {},
            "events":   [{"type": "llm_response",
                          "timestamp": time.time(),
                          "data": _last_llm}],
            "theme":    _detectors.current_theme,
            "llm":      _last_llm,
        })


# ── marker listener ───────────────────────────────────────────────────────────

async def marker_listener():
    """
    Polls NEST_Behavioral_Markers every 100ms.
    Forwards any received markers to detectors.update_marker().
    Runs as a background asyncio task.
    """
    if _marker_inlet is None:
        log.info("Marker listener: no inlet (not connected)")
        return

    log.info("Marker listener started.")
    while _loop_running:
        try:
            # Non-blocking pull — timeout=0 returns immediately
            sample, ts = _marker_inlet.pull_sample(timeout=0.0)
            if sample:
                marker_str = str(sample[0]) if sample else ""
                log.debug(f"Marker received: {marker_str}")
                _detectors.update_marker(marker_str)
        except Exception as exc:
            log.warning(f"Marker listener error: {exc}")
        await asyncio.sleep(MARKER_POLL_S)


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect all components and launch background tasks.
    Shutdown: stop loop and disconnect cleanly.
    """
    global _router, _extractor, _engine, _detectors, _marker_inlet
    global _loop_running

    log.info("="*55)
    log.info("  NAOS Server starting up")
    log.info("="*55)

    # ── initialise pipeline components ──────────────────────────────
    _extractor = FeatureExtractor()
    _engine    = StateEngine()
    _detectors = Detectors()

    # ── connect EEG stream ───────────────────────────────────────
    _router = SignalRouter()
    connected = _router.connect()
    if connected:
        log.info("SignalRouter connected to NEST_EEG_Clean")
    else:
        log.error(
            "Could not connect to NEST_EEG_Clean. "
            "Is NEST open and streaming? "
            "Server will wait for data."
        )
        _router = None

    # ── connect marker stream ────────────────────────────────────
    try:
        from pylsl import StreamInlet, resolve_streams
        streams = resolve_streams(wait_time=2.0)
        marker_streams = [s for s in streams
                          if s.name() == MARKER_STREAM_NAME]
        if marker_streams:
            _marker_inlet = StreamInlet(marker_streams[0])
            log.info(f"Marker inlet connected: {MARKER_STREAM_NAME}")
        else:
            log.warning(f"Marker stream '{MARKER_STREAM_NAME}' not found")
    except Exception as exc:
        log.warning(f"Marker stream connection failed: {exc}")

    # ── start background tasks ───────────────────────────────────────
    loop_task   = asyncio.create_task(signal_loop(),   name="signal_loop")
    marker_task = asyncio.create_task(marker_listener(), name="marker_listener")

    log.info("Background tasks started. Server ready.")
    log.info(f"  WebSocket: ws://localhost:8000/ws")
    log.info(f"  Status:    http://localhost:8000/api/status")
    log.info(f"  Gemini:    {'enabled' if os.getenv('GEMINI_API_KEY') else 'GEMINI_API_KEY not set — using fallback'}")

    yield   # server is running

    # ── shutdown ─────────────────────────────────────────────────────
    log.info("Server shutting down...")
    _loop_running = False
    loop_task.cancel()
    marker_task.cancel()
    if _router:
        _router.disconnect()
    log.info("Shutdown complete.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NAOS BCI Server",
    description="Real-time EEG state detection and adaptive UI backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow dashboard HTML to connect from any origin (localhost dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Dashboard connects here. Receives JSON every 250ms.
    Stays connected until dashboard closes or network drops.
    """
    await ws.accept()
    _clients.add(ws)
    client_id = id(ws)
    log.info(f"WebSocket client connected: {client_id} "
             f"(total: {len(_clients)})")

    # Send current state immediately on connect (don't make dashboard wait)
    if _last_state:
        await ws.send_text(json.dumps({
            "state":    _last_state,
            "features": {},
            "events":   [],
            "theme":    _detectors.current_theme if _detectors else "NEUTRAL",
            "llm":      _last_llm,
        }, default=str))

    try:
        while True:
            # Keep connection alive — dashboard sends pings
            await ws.receive_text()
    except WebSocketDisconnect:
        log.info(f"WebSocket client disconnected: {client_id}")
    except Exception as exc:
        log.warning(f"WebSocket error {client_id}: {exc}")
    finally:
        _clients.discard(ws)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.post("/api/dismiss")
async def dismiss_emergency():
    """
    Dashboard calls this when user presses 'I AM OKAY'.
    Resets emergency detector sequence.
    """
    if _detectors:
        _detectors.dismiss_emergency()
        log.info("Emergency dismissed by user.")
    return {"status": "ok", "message": "Emergency dismissed"}


@app.post("/api/llm")
async def manual_llm_trigger():
    """
    Manual LLM trigger for demo — call this from dashboard
    'Fire LLM' button without waiting for load threshold.
    """
    if not _last_state:
        return JSONResponse(
            status_code=503,
            content={"error": "No state data yet — wait for signal loop to warm up"}
        )
    asyncio.create_task(_handle_llm(_last_state))
    return {"status": "ok", "message": "LLM call triggered"}


@app.get("/api/status")
async def status():
    """Health check — returns current system state summary."""
    return {
        "status":         "running",
        "router_connected": _router is not None and _router.is_connected
                            if _router else False,
        "marker_connected": _marker_inlet is not None,
        "clients":        len(_clients),
        "loop_running":   _loop_running,
        "window_count":   _last_state.get("window_count", 0),
        "warmed_up":      _last_state.get("warmed_up", False),
        "current_theme":  _detectors.current_theme if _detectors else "NEUTRAL",
        "last_state":     {
            k: _last_state.get(k, 0)
            for k in ["attention", "load", "tension", "drowsiness", "confidence"]
        } if _last_state else {},
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.get("/")
async def root():
    return {"message": "NAOS BCI Server", "ws": "ws://localhost:8000/ws",
            "status": "/api/status"}


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # reload=True causes issues with shared state
        log_level="info",
    )
