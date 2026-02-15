#!/usr/bin/env python3
"""train_temporal.py - Train temporal models on Burgers time-series data.

Compares three approaches:
  1. SingleFrame CNN  - uses only last coarse frame (experiment 003 baseline)
  2. Temporal 2D+Cat  - stacks time frames as input channels
  3. Temporal 3D CNN  - true spatiotemporal convolutions

Usage:
    python train_temporal.py                       # Run all three models
    python train_temporal.py --model 3d            # Run only 3D CNN
    python train_temporal.py --model single        # Run only single-frame baseline
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
from models.temporal_cnn import SingleFrameCNN, TemporalCNN2D_Cat, TemporalCNN3D
from torch.utils.data import DataLoader, Dataset


class TemporalDataset(Dataset):
    """Load time-series samples from experiment 004 binary format.

    Each file contains: [nx, ny, n_snap, coarse_1..N, fine]

    The 'velocity' input mode computes frame-to-frame differences and
    appends them to the last frame, giving the model explicit information
    about how the field is changing (motion, steepening, growth).
    """

    def __init__(self, data_dir, n_samples=None, input_mode="raw"):
        """
        Args:
            input_mode: "raw" = raw coarse frames
                        "velocity" = last frame + frame-to-frame diffs
        """
        self.data_dir = Path(data_dir)
        self.input_mode = input_mode

        sample_files = sorted(self.data_dir.glob("sample_*.bin"))
        if n_samples:
            sample_files = sample_files[:n_samples]

        self.samples = []
        n_skipped = 0
        for f in sample_files:
            data = self._load_sample(f)
            if data is not None:
                self.samples.append(data)
            else:
                n_skipped += 1

        msg = f"  Loaded {len(self.samples)} samples from {data_dir}"
        if n_skipped:
            msg += f" (skipped {n_skipped} with NaN/Inf)"
        print(msg)

    def _load_sample(self, filepath):
        """Load and validate one sample. Returns None if corrupted."""
        with open(filepath, "rb") as f:
            nx = struct.unpack("i", f.read(4))[0]
            ny = struct.unpack("i", f.read(4))[0]
            n_snap = struct.unpack("i", f.read(4))[0]

            # Read coarse snapshots
            coarse_frames = []
            for _ in range(n_snap):
                data = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32).copy()
                coarse_frames.append(data.reshape((ny, nx)))

            # Read fine field
            fine = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32).copy()
            fine = fine.reshape((ny, nx))

        coarse = np.stack(coarse_frames, axis=0)  # (n_snap, H, W)

        # NaN check
        if not (np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine))):
            return None

        # Residual: fine minus last coarse frame
        residual = fine - coarse[-1]

        return (
            torch.from_numpy(coarse),
            torch.from_numpy(residual),
            torch.from_numpy(fine),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coarse, residual, _ = self.samples[idx]

        if self.input_mode == "velocity":
            # Channel 0: last coarse frame (the state)
            # Channels 1..N-1: frame-to-frame differences (the dynamics)
            last_frame = coarse[-1:]  # (1, H, W)
            diffs = coarse[1:] - coarse[:-1]  # (N-1, H, W)
            input_tensor = torch.cat([last_frame, diffs], dim=0)
            return input_tensor, residual

        return coarse, residual


def train_model(model, train_loader, val_loader, args, model_name):
    """Train one model and return results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for coarse, residual in train_loader:
            coarse, residual = coarse.to(device), residual.to(device)
            pred = model(coarse)
            loss = nn.functional.mse_loss(pred, residual)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for coarse, residual in val_loader:
                coarse, residual = coarse.to(device), residual.to(device)
                pred = model(coarse)
                val_loss += nn.functional.mse_loss(pred, residual).item()
        val_loss /= len(val_loader)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % args.print_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f"    Epoch {epoch:4d}/{args.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} | {elapsed:.1f}s"
            )

    total_time = time.time() - t_start

    # Evaluate: MSE with model vs MSE without (coarse only = zero residual)
    model.eval()
    mse_model = 0
    mse_baseline = 0
    n_eval = 0
    with torch.no_grad():
        for coarse, residual in val_loader:
            coarse, residual = coarse.to(device), residual.to(device)
            pred = model(coarse)
            mse_model += nn.functional.mse_loss(pred, residual, reduction="sum").item()
            mse_baseline += (residual**2).sum().item()
            n_eval += residual.numel()

    mse_model /= n_eval
    mse_baseline /= n_eval
    improvement = (1.0 - mse_model / mse_baseline) * 100 if mse_baseline > 0 else 0

    return {
        "model": model_name,
        "parameters": model.count_params(),
        "best_val_loss": best_val_loss,
        "mse_model": mse_model,
        "mse_baseline": mse_baseline,
        "improvement_pct": improvement,
        "training_time_s": total_time,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Temporal super-resolution training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../experiments/004_temporal/fields/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "single", "cat", "3d", "velocity"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    print("=" * 64)
    print("  Experiment 004: Temporal Super-Resolution Training")
    print("=" * 64)
    print()

    # Load data
    use_velocity = args.model == "velocity"
    input_mode = "velocity" if use_velocity else "raw"
    dataset = TemporalDataset(args.data_dir, input_mode=input_mode)
    n_snap = dataset.samples[0][0].shape[0]
    n_input_channels = n_snap if not use_velocity else n_snap  # same count either way
    print(f"  Snapshots per sample: {n_snap}")
    print(f"  Input mode: {input_mode}")
    print(f"  Field size: {dataset.samples[0][0].shape[1:]}")

    # Split
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"  Train: {n_train}, Val: {n_val}")

    # Models to test
    models_to_run = {}
    if args.model in ("all", "single"):
        models_to_run["SingleFrame"] = SingleFrameCNN(n_channels=32, n_layers=4)
    if args.model in ("all", "cat"):
        models_to_run["Temporal2D_Cat"] = TemporalCNN2D_Cat(
            n_channels=32, n_snap=n_snap
        )
    if args.model in ("all", "3d"):
        models_to_run["Temporal3D"] = TemporalCNN3D(n_channels=32, n_snap=n_snap)
    if args.model == "velocity":
        models_to_run["Velocity_Cat"] = TemporalCNN2D_Cat(
            n_channels=32, n_snap=n_input_channels
        )

    # Train each
    results = {}
    for name, model in models_to_run.items():
        print(f"\n  --- {name} ({model.count_params():,} params) ---\n")
        result = train_model(model, train_loader, val_loader, args, name)
        results[name] = result

    # Summary
    print("\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)
    print()
    print(
        f"  {'Model':<20s} {'Params':>8s} {'MSE(model)':>12s} "
        f"{'MSE(zero)':>12s} {'Improvement':>12s} {'Time':>8s}"
    )
    print("  " + "-" * 76)

    for name, r in results.items():
        print(
            f"  {name:<20s} {r['parameters']:>8,} {r['mse_model']:>12.6f} "
            f"{r['mse_baseline']:>12.6f} {r['improvement_pct']:>11.1f}% "
            f"{r['training_time_s']:>7.1f}s"
        )

    print()

    # Save results
    output_dir = Path("../experiments/004_temporal/runs/")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "history"}
        for k, v in results.items()
    }
    with open(output_dir / f"summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
