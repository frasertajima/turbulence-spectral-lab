#!/usr/bin/env python3
"""train_denoise.py - Denoising approach to turbulence super-resolution.

Inspired by cryo-EM: instead of predicting the signal directly,
learn to characterize what's WRONG with the coarse field.

Three approaches tested:

1. "artifact" mode: Train on (filtered_field, filter_artifact) pairs.
   The filter artifact = filtered - true is a specific pattern caused
   by spectral truncation. The model learns to recognize and reverse
   the damage done by filtering.

2. "self-supervised" mode: Add random noise to the coarse field,
   train the model to remove it. The idea is that a denoiser must
   learn the structure of "clean" coarse fields, and this structural
   knowledge generalizes to recognizing filter artifacts.

3. "noise2fine" mode: Frame it as noise characterization.
   Input = coarse (the "noisy" observation where filtering is the noise).
   Target = fine (the "clean" truth).
   This is just standard super-resolution but framed differently in
   the loss — we weight the loss by local gradient magnitude,
   focusing the model's attention on where filtering does damage
   (near steep features) rather than treating all pixels equally.

The hypothesis: if filtering artifacts have learnable spatial structure
(they do — they concentrate near sharp gradients), then a model trained
to detect those artifacts can partially reverse them.
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
from models.fno_2d import SimpleCNN
from torch.utils.data import DataLoader, Dataset


class BurgersDataset(Dataset):
    """Load (coarse, fine) pairs from experiment 003/004 data."""

    def __init__(self, data_dir, mode="artifact"):
        self.data_dir = Path(data_dir)
        self.mode = mode

        fine_files = sorted(self.data_dir.glob("fine_sample_*.bin"))
        self.samples = []
        n_skipped = 0

        for fine_file in fine_files:
            sample_id = fine_file.stem.replace("fine_sample_", "")
            coarse_file = self.data_dir / f"coarse_sample_{sample_id}.bin"
            if coarse_file.exists():
                coarse = self._load_field(coarse_file)
                fine = self._load_field(fine_file)
                if np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine)):
                    self.samples.append(
                        (
                            torch.from_numpy(coarse),
                            torch.from_numpy(fine),
                        )
                    )
                else:
                    n_skipped += 1

        msg = f"  Loaded {len(self.samples)} pairs ({mode} mode)"
        if n_skipped:
            msg += f", skipped {n_skipped}"
        print(msg)

    @staticmethod
    def _load_field(filepath):
        with open(filepath, "rb") as f:
            nx = struct.unpack("i", f.read(4))[0]
            ny = struct.unpack("i", f.read(4))[0]
            return (
                np.frombuffer(f.read(nx * ny * 4), dtype=np.float32)
                .copy()
                .reshape((ny, nx))
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coarse, fine = self.samples[idx]

        if self.mode == "artifact":
            # Target: the filtering artifact (what filtering got wrong)
            # artifact = coarse - fine (negative residual)
            # Model learns: given coarse, predict what was subtracted
            artifact = coarse - fine
            return coarse, artifact

        elif self.mode == "residual":
            # Standard: predict what's missing (fine - coarse)
            residual = fine - coarse
            return coarse, residual

        elif self.mode == "direct":
            # Direct: predict the fine field itself
            return coarse, fine


def gradient_weighted_loss(pred, target, input_field, alpha=2.0):
    """Loss that focuses on high-gradient regions.

    Near steep gradients, filtering causes the worst artifacts.
    This loss makes the model pay more attention there — learning
    WHERE filtering does damage, not just what the average error is.
    """
    # Compute gradient magnitude of input
    gx = input_field[:, :, 1:] - input_field[:, :, :-1]
    gy = input_field[:, 1:, :] - input_field[:, :-1, :]

    # Pad to original size
    gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))

    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

    # Weight: 1 everywhere + alpha at high-gradient locations
    weight = 1.0 + alpha * grad_mag / (grad_mag.max() + 1e-8)

    return torch.mean(weight * (pred - target) ** 2)


def train_and_evaluate(model, train_loader, val_loader, dataset, args, name):
    """Train one configuration and return results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            if args.gradient_loss:
                loss = gradient_weighted_loss(pred, y, x)
            else:
                loss = nn.functional.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += nn.functional.mse_loss(pred, y).item()
        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:4d}/{args.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {best_val:.6f} | {time.time() - t0:.1f}s"
            )

    elapsed = time.time() - t0

    # Evaluate reconstruction quality
    model.eval()
    mse_model = 0
    mse_coarse = 0
    n_eval = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            if dataset.mode == "artifact":
                # Reconstruct: fine_pred = coarse - predicted_artifact
                fine_pred = x - pred
            elif dataset.mode == "residual":
                fine_pred = x + pred
            elif dataset.mode == "direct":
                fine_pred = pred

            # Get true fine for comparison
            batch_size = x.shape[0]
            for i in range(batch_size):
                coarse_i = x[i]
                if dataset.mode == "artifact":
                    fine_true_i = x[i] - y[i]
                elif dataset.mode == "residual":
                    fine_true_i = x[i] + y[i]
                elif dataset.mode == "direct":
                    fine_true_i = y[i]

                mse_model += ((fine_pred[i] - fine_true_i) ** 2).sum().item()
                mse_coarse += ((coarse_i - fine_true_i) ** 2).sum().item()
                n_eval += fine_true_i.numel()

    mse_model /= n_eval
    mse_coarse /= n_eval
    improvement = (1.0 - mse_model / mse_coarse) * 100 if mse_coarse > 0 else 0

    return {
        "name": name,
        "mode": dataset.mode,
        "params": model.count_params(),
        "best_val": best_val,
        "mse_model": mse_model,
        "mse_coarse": mse_coarse,
        "improvement": improvement,
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="../experiments/003_burgers/fields/"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--gradient-loss", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("  Denoising Approach to Turbulence Super-Resolution")
    print("=" * 64)
    print()

    modes = ["artifact", "residual", "direct"]
    results = []

    for mode in modes:
        print(f"\n  --- Mode: {mode} ---\n")

        dataset = BurgersDataset(args.data_dir, mode=mode)
        n_val = max(1, len(dataset) // 5)
        n_train = len(dataset) - n_val

        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        # CNN with 1 input channel, 1 output channel
        model = SimpleCNN(n_channels=32, n_layers=4)

        r = train_and_evaluate(model, train_loader, val_loader, dataset, args, mode)
        results.append(r)

    # Also test gradient-weighted loss on artifact mode
    print(f"\n  --- Mode: artifact + gradient loss ---\n")
    args_gl = argparse.Namespace(**vars(args))
    args_gl.gradient_loss = True

    dataset = BurgersDataset(args.data_dir, mode="artifact")
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(n_channels=32, n_layers=4)
    r = train_and_evaluate(
        model, train_loader, val_loader, dataset, args_gl, "artifact+grad_loss"
    )
    results.append(r)

    # Summary
    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print()
    print(
        f"  {'Mode':<25s} {'MSE(model)':>12s} {'MSE(coarse)':>12s} {'Improvement':>12s}"
    )
    print("  " + "-" * 65)
    for r in results:
        print(
            f"  {r['name']:<25s} {r['mse_model']:>12.6f} "
            f"{r['mse_coarse']:>12.6f} {r['improvement']:>11.1f}%"
        )

    # Save
    output_dir = Path("../experiments/003_burgers/runs/denoise/")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_dir}")


if __name__ == "__main__":
    main()
