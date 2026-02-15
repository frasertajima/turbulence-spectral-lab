#!/usr/bin/env python3
"""
Real-time fluid simulation server with neural network error correction.

Runs a Stam (semi-Lagrangian) solver on GPU, applies FNO/CNN correction,
and streams results to a browser via WebSocket.

Usage:
    python server.py
    # Open http://localhost:8765/
"""

import asyncio
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
except ImportError:
    print("Install with: pip install fastapi uvicorn websockets")
    raise

from model_bridge import ModelBridge
from solver_bridge import SolverBridge

# --- Configuration ---
NX, NY = 128, 128
NU = 0.0
DT = 0.1
STEPS_PER_FRAME = 1
TARGET_FPS = 60

app = FastAPI(title="Turbulence Spectral Lab - Real-Time Demo")
WEB_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Shared state between simulation thread and WebSocket handlers
# ---------------------------------------------------------------------------
class SimState:
    def __init__(self):
        self.lock = threading.Lock()

        # Fields (updated by sim thread, read by WebSocket)
        z = lambda: np.zeros((NX, NY), dtype=np.float32)
        self.stam_u = z()
        self.stam_v = z()
        self.stam_rho = z()
        self.corrected_u = z()
        self.corrected_v = z()
        self.spectral_u = z()
        self.spectral_v = z()

        # Precomputed spectrum
        self.spectrum_k = np.zeros(NX // 2, dtype=np.float32)
        self.spectrum_stam = np.zeros(NX // 2, dtype=np.float32)
        self.spectrum_corrected = np.zeros(NX // 2, dtype=np.float32)
        self.spectrum_spectral = np.zeros(NX // 2, dtype=np.float32)

        # Timing / counters
        self.frame_number = 0
        self.total_steps = 0
        self.sim_ms = 0.0
        self.correction_ms = 0.0
        self.fps = 0.0

        # Control signals (written by WebSocket, read by sim thread)
        self.steps_per_frame = STEPS_PER_FRAME
        self.run_spectral = False
        self.ns_mode = True
        self.model_type = "fno"  # "fno", "cnn", "none"
        self.paused = False

        # Pending commands (consumed once by sim thread)
        self.pending_ic_seed = None
        self.pending_zero_ic = False
        self.pending_force = None  # (fx, fy, cx, cy)
        self.pending_mode_change = None  # bool
        self.pending_model_change = None  # str
        self.pending_alpha = None  # float
        self.pending_walls = None  # bool

        self.running = True


state = SimState()


# ---------------------------------------------------------------------------
# Energy spectrum (vectorized)
# ---------------------------------------------------------------------------
# Precompute wavenumber bin indices once
# rfft2 on (NX, NY) gives shape (NX, NY//2+1)
_NKX = NY // 2 + 1
_kx = np.arange(_NKX, dtype=np.float32)  # 0..64
_ky = np.fft.fftfreq(NX, d=1.0 / NX).astype(np.float32)  # 0..63, -64..-1
_KX, _KY = np.meshgrid(_kx, _ky)  # both shape (NX, NKX) = (128, 65)
_KMAG = np.sqrt(_KX**2 + _KY**2)
_KBIN = _KMAG.astype(int)
_KMAX = NX // 2
# Multiplier: interior rfft modes counted twice (Hermitian symmetry)
_MULT = np.ones_like(_KX)
_MULT[:, 1:-1] = 2.0


def energy_spectrum(field: np.ndarray) -> np.ndarray:
    """Compute 1D energy spectrum E(k) from a 2D real field."""
    fft = np.fft.rfft2(field)
    power = (np.abs(fft) ** 2) / (NX * NY) ** 2 * _MULT
    E_k = np.zeros(_KMAX, dtype=np.float32)
    np.add.at(E_k, _KBIN[_KBIN < _KMAX], power[_KBIN < _KMAX])
    return E_k


# ---------------------------------------------------------------------------
# Simulation thread
# ---------------------------------------------------------------------------
def simulation_loop():
    solver = SolverBridge(NX, NY, NU)
    models = ModelBridge()

    # Start in N-S mode with empty field and density tracer
    solver.set_ns_mode(True)
    solver.enable_density(diffusion=0.0001, decay=0.05)
    solver.zero_ic()
    models.load_model("fno_ns")

    frame_times: list[float] = []

    while state.running:
        t_frame = time.perf_counter()

        # --- Process pending commands ---
        with state.lock:
            if state.pending_zero_ic:
                solver.zero_ic()
                state.pending_zero_ic = False
                state.total_steps = 0

            if state.pending_ic_seed is not None:
                solver.new_ic(state.pending_ic_seed)
                state.pending_ic_seed = None
                state.total_steps = 0

            if state.pending_mode_change is not None:
                ns = state.pending_mode_change
                # Re-init solver for mode change
                solver.cleanup()
                solver = SolverBridge(NX, NY, NU)
                solver.set_ns_mode(ns)
                solver.enable_density(diffusion=0.0001, decay=0.05)
                solver.zero_ic()
                state.ns_mode = ns
                state.total_steps = 0
                name = ModelBridge.model_name_for(ns, state.model_type)
                models.load_model(name)
                state.pending_mode_change = None

            if state.pending_model_change is not None:
                state.model_type = state.pending_model_change
                name = ModelBridge.model_name_for(state.ns_mode, state.model_type)
                models.load_model(name)
                state.pending_model_change = None

            if state.pending_alpha is not None:
                models.alpha = state.pending_alpha
                state.pending_alpha = None

            if state.pending_walls is not None:
                solver.set_walls(state.pending_walls)
                state.pending_walls = None

            if state.pending_force is not None:
                fx, fy, cx, cy = state.pending_force
                solver.add_force(fx, fy, cx, cy, radius=5.0)
                state.pending_force = None

            steps = state.steps_per_frame
            run_spectral = state.run_spectral
            paused = state.paused

        if paused:
            time.sleep(1.0 / TARGET_FPS)
            continue

        # --- Step solver ---
        t0 = time.perf_counter()
        solver.step(DT, steps, run_spectral=run_spectral)
        sim_ms = (time.perf_counter() - t0) * 1000

        # --- Extract Stam fields ---
        stam_u = solver.get_stam_u()
        stam_v = solver.get_stam_v()
        stam_rho = solver.get_stam_rho()

        # --- Neural network correction (two components) ---
        t0 = time.perf_counter()
        corrected_u = models.correct(stam_u)
        corrected_v = models.correct(stam_v)
        correction_ms = (time.perf_counter() - t0) * 1000

        # --- Optional spectral reference ---
        if run_spectral:
            spectral_u = solver.get_spectral_u()
            spectral_v = solver.get_spectral_v()
        else:
            spectral_u = np.zeros_like(stam_u)
            spectral_v = np.zeros_like(stam_v)

        # --- Energy spectra ---
        k_vals = np.arange(_KMAX, dtype=np.float32)
        e_stam = energy_spectrum(stam_u)
        e_corr = energy_spectrum(corrected_u)
        e_spec = energy_spectrum(spectral_u) if run_spectral else np.zeros_like(e_stam)

        # --- FPS tracking ---
        frame_ms = (time.perf_counter() - t_frame) * 1000
        frame_times.append(frame_ms)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = 1000.0 / (sum(frame_times) / len(frame_times))

        # --- Update shared state ---
        with state.lock:
            state.stam_u = stam_u
            state.stam_v = stam_v
            state.stam_rho = stam_rho
            state.corrected_u = corrected_u
            state.corrected_v = corrected_v
            state.spectral_u = spectral_u
            state.spectral_v = spectral_v
            state.spectrum_k = k_vals
            state.spectrum_stam = e_stam
            state.spectrum_corrected = e_corr
            state.spectrum_spectral = e_spec
            state.frame_number += 1
            state.total_steps += steps
            state.sim_ms = sim_ms
            state.correction_ms = correction_ms
            state.fps = avg_fps

        # --- Frame rate limit ---
        elapsed = time.perf_counter() - t_frame
        target = 1.0 / TARGET_FPS
        if elapsed < target:
            time.sleep(target - elapsed)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
def handle_client_message(msg: dict):
    """Process a JSON command from the browser."""
    with state.lock:
        if "zero_ic" in msg:
            state.pending_zero_ic = True
        if "new_ic" in msg:
            state.pending_ic_seed = int(msg["new_ic"])
        if "force" in msg:
            f = msg["force"]
            state.pending_force = (f["fx"], f["fy"], f["cx"], f["cy"])
        if "ns_mode" in msg:
            state.pending_mode_change = bool(msg["ns_mode"])
        if "model" in msg:
            state.pending_model_change = str(msg["model"])
        if "steps_per_frame" in msg:
            state.steps_per_frame = max(1, min(50, int(msg["steps_per_frame"])))
        if "alpha" in msg:
            state.pending_alpha = max(0.0, min(1.0, float(msg["alpha"])))
        if "run_spectral" in msg:
            state.run_spectral = bool(msg["run_spectral"])
        if "walls" in msg:
            state.pending_walls = bool(msg["walls"])
        if "paused" in msg:
            state.paused = bool(msg["paused"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_frame = -1

    try:
        while True:
            # Non-blocking check for client messages
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.002)
                handle_client_message(json.loads(msg))
            except asyncio.TimeoutError:
                pass

            # Send new frame if available
            with state.lock:
                if state.frame_number <= last_frame:
                    continue

                has_spectral = state.run_spectral

                # Flags: bit 0 = has spectral, bit 1 = has density
                flags = (1 if has_spectral else 0) | 2  # density always on

                # Header: frame(i32) steps(i32) sim_ms(f32) corr_ms(f32) fps(f32) flags(i32)
                header = struct.pack(
                    "<2i3fi",
                    state.frame_number,
                    state.total_steps,
                    state.sim_ms,
                    state.correction_ms,
                    state.fps,
                    flags,
                )

                parts = [
                    header,
                    state.stam_u.tobytes(),
                    state.stam_v.tobytes(),
                    state.corrected_u.tobytes(),
                    state.corrected_v.tobytes(),
                    state.stam_rho.tobytes(),
                ]
                if has_spectral:
                    parts.append(state.spectral_u.tobytes())
                    parts.append(state.spectral_v.tobytes())

                # Spectrum data: k, E_stam, E_corrected [, E_spectral]
                parts.append(state.spectrum_k.tobytes())
                parts.append(state.spectrum_stam.tobytes())
                parts.append(state.spectrum_corrected.tobytes())
                if has_spectral:
                    parts.append(state.spectrum_spectral.tobytes())

                last_frame = state.frame_number

            await websocket.send_bytes(b"".join(parts))

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/fullscreen")
async def fullscreen():
    return FileResponse(WEB_DIR / "fullscreen.html")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    print("=" * 60)
    print("  Turbulence Spectral Lab - Real-Time Demo")
    print("=" * 60)
    print(f"  Grid: {NX}x{NY}, nu={NU}, dt={DT}")
    print(f"  Steps/frame: {STEPS_PER_FRAME}, Target FPS: {TARGET_FPS}")
    print()

    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()

    print()
    print(f"\033[92m{'=' * 60}\033[0m")
    print(f"\033[92m\033[1m  SERVER READY!\033[0m")
    print(f"\033[92m  Open browser to:\033[0m")
    print()
    print(f"\033[92m\033[1m    >>> http://localhost:8765/ <<<\033[0m")
    print(f"\033[92m    >>> http://localhost:8765/fullscreen <<<\033[0m")
    print()
    print(f"\033[92m{'=' * 60}\033[0m")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
