#!/usr/bin/env python3
"""test_realtime_correction.py - Real-time CNN correction of Stam solver.

Runs both solvers in lockstep. Applies CNN correction as POST-PROCESSING
(display correction) — the Stam solver continues uncorrected internally.

The CNN was trained on 500-step accumulated error. At shorter evolution
times, the error is smaller but structurally similar. We test whether
the CNN correction improves the Stam field at every snapshot, even though
the CNN only saw the 500-step case during training.

Two modes:
  Mode 1: "Snapshot" — Run Stam for N steps, correct once, measure
  Mode 2: "Online" — Correct for display at intervals, Stam runs uncorrected
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add parent for model imports
sys.path.insert(0, str(Path(__file__).parent))
from models.fno_2d import FNO2d, SimpleCNN

# --- Configuration ---
N = 128
NU = 0.01
DT = 0.001
TOTAL_STEPS = 500
CORRECT_EVERY = 10  # apply CNN correction every K Stam steps
SEED = 9999  # different from training seeds
FORTRAN_LIB = str(Path(__file__).parent / "../fortran/realtime_solvers.so")

# Parse --ns flag early so we can set RUNS_DIR
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--ns", action="store_true", help="Use N-S mode (experiment 005b)")
_args, _ = _parser.parse_known_args()
NS_MODE = _args.ns

if NS_MODE:
    RUNS_DIR = Path(__file__).parent / "../experiments/005b_ns_correction/runs"
else:
    RUNS_DIR = Path(__file__).parent / "../experiments/005_stam_correction/runs"

cnn_dirs = sorted(RUNS_DIR.glob("cnn_*"))
fno_dirs = sorted(RUNS_DIR.glob("fno_*"))


def load_model(model_type, model_dir):
    if model_type == "cnn":
        model = SimpleCNN(n_channels=32, n_layers=4)
    else:
        model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", weights_only=True))
    model.eval()
    return model


def run_test(model, model_name, lib, correct_every):
    """Run test: spectral vs stam vs stam+correction (post-processing only).

    The CNN correction is applied as display post-processing — the Stam
    solver continues running uncorrected. This avoids feedback instability.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Initialize both solvers with same IC
    lib.rt_init(ctypes.c_int(N), ctypes.c_int(N), ctypes.c_float(NU))
    if NS_MODE:
        lib.rt_set_ns_mode(ctypes.c_int(1))
        lib.rt_set_ic_from_seed_ns(ctypes.c_int(SEED))
    else:
        lib.rt_set_ic_from_seed(ctypes.c_int(SEED))

    # Buffers for field extraction
    spectral_buf = np.zeros((N, N), dtype=np.float32)
    stam_buf = np.zeros((N, N), dtype=np.float32)

    spectral_ptr = spectral_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    stam_ptr = stam_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    results = {
        "step": [],
        "mse_uncorrected": [],
        "mse_corrected": [],
        "spectral_max": [],
        "stam_max": [],
        "corrected_max": [],
        "correction_time_ms": [],
    }

    total_correction_time = 0.0
    n_corrections = 0

    for step in range(1, TOTAL_STEPS + 1):
        lib.rt_step_spectral(ctypes.c_float(DT), ctypes.c_int(1))
        lib.rt_step_stam(ctypes.c_float(DT), ctypes.c_int(1))

        if step % correct_every == 0:
            # Extract both fields
            lib.rt_get_spectral_u(spectral_ptr)
            lib.rt_get_stam_u(stam_ptr)

            # Uncorrected error
            mse_uncorrected = float(np.mean((spectral_buf - stam_buf) ** 2))

            # CNN correction (post-processing only — NOT fed back to solver)
            t0 = time.perf_counter()
            with torch.no_grad():
                stam_tensor = torch.from_numpy(stam_buf.copy()).unsqueeze(0).to(device)
                correction = model(stam_tensor)
                corrected = stam_tensor + correction
                corrected_np = corrected.squeeze(0).cpu().numpy()
            correction_ms = (time.perf_counter() - t0) * 1000
            total_correction_time += correction_ms
            n_corrections += 1

            # Corrected error
            mse_corrected = float(np.mean((spectral_buf - corrected_np) ** 2))

            results["step"].append(step)
            results["mse_uncorrected"].append(mse_uncorrected)
            results["mse_corrected"].append(mse_corrected)
            results["spectral_max"].append(float(np.max(np.abs(spectral_buf))))
            results["stam_max"].append(float(np.max(np.abs(stam_buf))))
            results["corrected_max"].append(float(np.max(np.abs(corrected_np))))
            results["correction_time_ms"].append(correction_ms)

    lib.rt_cleanup()

    avg_correction_ms = total_correction_time / max(n_corrections, 1)

    return results, avg_correction_ms, n_corrections


def main():
    # Load Fortran library
    lib = ctypes.CDLL(FORTRAN_LIB)

    lib.rt_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.rt_init.restype = None
    lib.rt_set_ic_from_seed.argtypes = [ctypes.c_int]
    lib.rt_set_ic_from_seed.restype = None
    lib.rt_step_spectral.argtypes = [ctypes.c_float, ctypes.c_int]
    lib.rt_step_spectral.restype = None
    lib.rt_step_stam.argtypes = [ctypes.c_float, ctypes.c_int]
    lib.rt_step_stam.restype = None
    lib.rt_get_spectral_u.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.rt_get_spectral_u.restype = None
    lib.rt_get_stam_u.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.rt_get_stam_u.restype = None
    lib.rt_set_stam_u.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.rt_set_stam_u.restype = None
    lib.rt_cleanup.argtypes = []
    lib.rt_cleanup.restype = None
    lib.rt_set_ns_mode.argtypes = [ctypes.c_int]
    lib.rt_set_ns_mode.restype = None
    lib.rt_set_ic_from_seed_ns.argtypes = [ctypes.c_int]
    lib.rt_set_ic_from_seed_ns.restype = None

    mode_str = "Navier-Stokes" if NS_MODE else "Burgers"
    print("=" * 70)
    print(f"  Real-Time CNN Correction Test: Stam + CNN vs Spectral ({mode_str})")
    print("=" * 70)
    print(f"  Grid: {N}x{N}, nu={NU}, dt={DT}")
    print(f"  Total steps: {TOTAL_STEPS}, Correct every: {CORRECT_EVERY} steps")
    print(f"  IC seed: {SEED} (unseen during training)")
    print()

    # Test with CNN
    if cnn_dirs:
        cnn_model = load_model("cnn", cnn_dirs[-1])
        print(f"  CNN model: {cnn_dirs[-1].name} ({cnn_model.count_params():,} params)")

        results_cnn, avg_ms_cnn, n_corr = run_test(cnn_model, "CNN", lib, CORRECT_EVERY)

        print(f"\n  --- CNN Results (correction every {CORRECT_EVERY} steps) ---")
        print(
            f"  {'Step':>6}  {'MSE uncorr':>12}  {'MSE corrected':>14}  "
            f"{'Improvement':>12}  {'max|spec|':>10}  {'max|stam|':>10}  {'max|corr|':>10}"
        )
        print(
            f"  {'-' * 6}  {'-' * 12}  {'-' * 14}  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}"
        )

        for i in range(len(results_cnn["step"])):
            s = results_cnn["step"][i]
            mu = results_cnn["mse_uncorrected"][i]
            mc = results_cnn["mse_corrected"][i]
            imp = (1.0 - mc / mu) * 100 if mu > 0 else 0
            print(
                f"  {s:6d}  {mu:12.6f}  {mc:14.6f}  {imp:11.1f}%  "
                f"{results_cnn['spectral_max'][i]:10.4f}  "
                f"{results_cnn['stam_max'][i]:10.4f}  "
                f"{results_cnn['corrected_max'][i]:10.4f}"
            )

        final_mu = results_cnn["mse_uncorrected"][-1]
        final_mc = results_cnn["mse_corrected"][-1]
        final_imp = (1.0 - final_mc / final_mu) * 100

        print(
            f"\n  CNN correction time: {avg_ms_cnn:.2f} ms/correction "
            f"({n_corr} corrections total)"
        )
        print(f"  Final improvement: {final_imp:.1f}%")
        print(
            f"  Stam step budget @ 60fps: 16.7ms -> "
            f"correction overhead: {avg_ms_cnn:.1f}ms ({avg_ms_cnn / 16.7 * 100:.0f}%)"
        )

    # Test with FNO too
    if fno_dirs:
        fno_model = load_model("fno", fno_dirs[-1])
        print(
            f"\n  FNO model: {fno_dirs[-1].name} ({fno_model.count_params():,} params)"
        )

        results_fno, avg_ms_fno, n_corr = run_test(fno_model, "FNO", lib, CORRECT_EVERY)

        print(f"\n  --- FNO Results (correction every {CORRECT_EVERY} steps) ---")
        print(
            f"  {'Step':>6}  {'MSE uncorr':>12}  {'MSE corrected':>14}  "
            f"{'Improvement':>12}  {'max|spec|':>10}  {'max|stam|':>10}  {'max|corr|':>10}"
        )
        print(
            f"  {'-' * 6}  {'-' * 12}  {'-' * 14}  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}"
        )

        for i in range(len(results_fno["step"])):
            s = results_fno["step"][i]
            mu = results_fno["mse_uncorrected"][i]
            mc = results_fno["mse_corrected"][i]
            imp = (1.0 - mc / mu) * 100 if mu > 0 else 0
            print(
                f"  {s:6d}  {mu:12.6f}  {mc:14.6f}  {imp:11.1f}%  "
                f"{results_fno['spectral_max'][i]:10.4f}  "
                f"{results_fno['stam_max'][i]:10.4f}  "
                f"{results_fno['corrected_max'][i]:10.4f}"
            )

        final_mu = results_fno["mse_uncorrected"][-1]
        final_mc = results_fno["mse_corrected"][-1]
        final_imp = (1.0 - final_mc / final_mu) * 100

        print(f"\n  FNO correction time: {avg_ms_fno:.2f} ms/correction")
        print(f"  Final improvement: {final_imp:.1f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
