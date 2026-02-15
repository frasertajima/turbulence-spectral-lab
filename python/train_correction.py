#!/usr/bin/env python3
"""train_correction.py - Train CNN to correct numerical dissipation error.

Experiment 005: Learn the systematic error between a fast semi-Lagrangian
solver (Stam) and an accurate pseudo-spectral solver (Burgers).

Unlike experiments 001-004 which tried to predict MISSING information
(filtered spectral content), this predicts SYSTEMATIC NUMERICAL ERROR
which is a deterministic function of the local field structure.

Usage:
    python train_correction.py --data-dir ../experiments/005_stam_correction/fields/
    python train_correction.py --model cnn --epochs 100
    python train_correction.py --model fno  # compare architectures

Based on train_fno.py with minimal changes for the new data format.
"""

import argparse
import json
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.fno_2d import FNO2d, SimpleCNN
from torch.utils.data import DataLoader, Dataset


class CorrectionDataset(Dataset):
    """Load (stam, spectral) field pairs for error correction training."""

    def __init__(self, data_dir, n_samples=None):
        self.data_dir = Path(data_dir)

        # Find all stam sample files
        stam_files = sorted(self.data_dir.glob("stam_sample_*.bin"))
        if n_samples:
            stam_files = stam_files[:n_samples]

        self.samples = []
        n_skipped = 0
        for stam_file in stam_files:
            sample_id = stam_file.stem.replace("stam_sample_", "")
            spectral_file = self.data_dir / f"spectral_sample_{sample_id}.bin"
            if spectral_file.exists():
                if self._file_is_clean(stam_file) and self._file_is_clean(
                    spectral_file
                ):
                    self.samples.append((stam_file, spectral_file))
                else:
                    n_skipped += 1

        msg = f"  Loaded {len(self.samples)} sample pairs from {data_dir}"
        if n_skipped:
            msg += f" (skipped {n_skipped} with NaN/Inf)"
        print(msg)

    @staticmethod
    def _file_is_clean(filepath):
        with open(filepath, "rb") as f:
            nx = struct.unpack("i", f.read(4))[0]
            ny = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32)
            return np.all(np.isfinite(data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stam_file, spectral_file = self.samples[idx]
        stam = self._load_field(stam_file)
        spectral = self._load_field(spectral_file)
        error = spectral - stam  # Model predicts this correction
        return torch.from_numpy(stam), torch.from_numpy(error)

    @staticmethod
    def _load_field(filepath):
        with open(filepath, "rb") as f:
            nx = struct.unpack("i", f.read(4))[0]
            ny = struct.unpack("i", f.read(4))[0]
            data = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32).copy()
            return data.reshape((ny, nx))


def compute_spectrum_numpy(field):
    """Compute 1D energy spectrum from 2D field."""
    ny, nx = field.shape
    fft = np.fft.rfft2(field)
    power = np.abs(fft) ** 2 / (nx * ny) ** 2

    nkx = nx // 2 + 1
    k_max = min(nx, ny) // 2
    E_k = np.zeros(k_max)
    counts = np.zeros(k_max)

    for j in range(ny):
        ky = j if j <= ny // 2 else j - ny
        for i in range(nkx):
            kx = i
            k_mag = np.sqrt(kx**2 + ky**2)
            bin_idx = int(k_mag)
            if 0 <= bin_idx < k_max:
                e = power[j, i]
                if i > 0 and i < nkx - 1:
                    e *= 2
                E_k[bin_idx] += e
                counts[bin_idx] += 1

    k_vals = np.arange(k_max, dtype=np.float32)
    return k_vals, E_k


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # Load data
    dataset = CorrectionDataset(args.data_dir, n_samples=args.n_samples)

    # Train/val split
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {n_train}, Val: {n_val}")

    # Model
    if args.model == "fno":
        model = FNO2d(
            modes1=args.modes,
            modes2=args.modes,
            width=args.width,
            n_layers=args.n_layers,
        ).to(device)
    else:
        model = SimpleCNN(n_channels=args.width, n_layers=args.n_layers).to(device)

    print(f"  Model: {args.model.upper()}, Parameters: {model.count_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for stam_field, error in train_loader:
            stam_field, error = stam_field.to(device), error.to(device)

            pred = model(stam_field)
            loss = nn.functional.mse_loss(pred, error)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for stam_field, error in val_loader:
                stam_field, error = stam_field.to(device), error.to(device)
                pred = model(stam_field)
                val_loss += nn.functional.mse_loss(pred, error).item()
        val_loss /= len(val_loader)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

        if epoch % args.print_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t_start
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best validation loss: {best_val_loss:.6f}")

    # Save training history
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f)

    # Evaluate on validation set
    model.load_state_dict(
        torch.load(args.output_dir / "best_model.pt", weights_only=True)
    )
    model.eval()

    mse_uncorrected_total = 0.0
    mse_corrected_total = 0.0
    n_eval = 0

    with torch.no_grad():
        for stam_field, error in val_loader:
            stam_field, error = stam_field.to(device), error.to(device)
            pred_error = model(stam_field)

            # MSE of uncorrected Stam vs spectral = MSE of the error itself
            mse_uncorrected_total += torch.mean(error**2).item() * stam_field.shape[0]
            # MSE of corrected Stam vs spectral = MSE of (error - predicted_error)
            mse_corrected_total += (
                torch.mean((error - pred_error) ** 2).item() * stam_field.shape[0]
            )
            n_eval += stam_field.shape[0]

    mse_uncorrected = mse_uncorrected_total / n_eval
    mse_corrected = mse_corrected_total / n_eval
    improvement = 1.0 - mse_corrected / mse_uncorrected

    # Spectral comparison on first sample
    with torch.no_grad():
        stam_field, error = dataset[0]
        stam_field = stam_field.unsqueeze(0).to(device)
        error = error.unsqueeze(0).to(device)
        pred_error = model(stam_field)

        spectral_true = (stam_field + error).squeeze().cpu().numpy()
        stam_corrected = (stam_field + pred_error).squeeze().cpu().numpy()
        stam_np = stam_field.squeeze().cpu().numpy()

    k_true, E_true = compute_spectrum_numpy(spectral_true)
    k_corrected, E_corrected = compute_spectrum_numpy(stam_corrected)
    k_stam, E_stam = compute_spectrum_numpy(stam_np)

    np.savetxt(
        args.output_dir / "spectrum_spectral.dat",
        np.column_stack([k_true, E_true]),
        header="k E(k)",
    )
    np.savetxt(
        args.output_dir / "spectrum_corrected.dat",
        np.column_stack([k_corrected, E_corrected]),
        header="k E(k)",
    )
    np.savetxt(
        args.output_dir / "spectrum_stam.dat",
        np.column_stack([k_stam, E_stam]),
        header="k E(k)",
    )

    # Summary
    summary = {
        "model": args.model,
        "parameters": model.count_params(),
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "mse_uncorrected": float(mse_uncorrected),
        "mse_corrected": float(mse_corrected),
        "improvement_pct": float(improvement * 100),
        "training_time_s": total_time,
        "n_train": n_train,
        "n_val": n_val,
        "timestamp": datetime.now().isoformat(),
    }

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results (evaluated on {n_eval} validation samples):")
    print(f"    MSE (Stam vs spectral):           {mse_uncorrected:.6f}")
    print(f"    MSE (Stam+CNN vs spectral):        {mse_corrected:.6f}")
    print(f"    Improvement:                       {improvement * 100:.1f}%")
    print(f"\n  Output saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN to correct numerical dissipation error"
    )
    parser.add_argument("--model", type=str, default="cnn", choices=["fno", "cnn"])
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=["005", "005b"],
        help="Experiment shorthand (sets data-dir and output-dir)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../experiments/005_stam_correction/fields/"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--modes", type=int, default=12, help="Fourier modes (FNO only)"
    )
    parser.add_argument("--width", type=int, default=32, help="Channel width")
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    # Resolve --experiment shorthand
    exp_dirs = {
        "005": "../experiments/005_stam_correction/",
        "005b": "../experiments/005b_ns_correction/",
    }
    if args.experiment is not None:
        base = exp_dirs[args.experiment]
        args.data_dir = base + "fields/"
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = Path(f"{base}runs/{args.model}_{timestamp}")

    # Default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(
            f"../experiments/005_stam_correction/runs/{args.model}_{timestamp}"
        )
    else:
        args.output_dir = Path(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir = Path(args.data_dir)

    # Save config
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("=" * 60)
    print("  Experiment 005: Numerical Error Correction Training")
    print("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
