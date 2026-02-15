"""temporal_cnn.py - 3D CNN for spatiotemporal super-resolution.

The key insight from experiment 003: fine-scale PDE structure can't be
predicted from a single coarse snapshot. It depends on the evolution history.

This module provides models that take a TIME SERIES of coarse snapshots
and predict the fine-scale structure of the final frame.

The 3D convolution kernels slide across (x, y, t), detecting:
  - Motion (features moving between frames)
  - Steepening (gradients increasing -> imminent shocks)
  - Growth (instabilities developing over time)

Architecture comparison:
  - CNN2D:     single frame  (batch, 1, H, W)     -> (batch, 1, H, W)
  - CNN3D:     time stack    (batch, 1, T, H, W)   -> (batch, 1, H, W)
  - CNN2D_cat: time stack as channels (batch, T, H, W) -> (batch, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN3D(nn.Module):
    """3D CNN: convolutions across (time, height, width) simultaneously.

    This is the 'true' spatiotemporal approach. Each kernel sees a local
    volume in space AND time, learning joint patterns.

    The temporal dimension is progressively collapsed by the kernels
    (kernel_t=3 with no temporal padding) until the output is a single frame.
    """

    def __init__(self, n_channels=32, n_snap=8):
        super().__init__()
        self.n_snap = n_snap

        # 3D conv layers: (1, T, H, W) -> (C, T', H, W) -> ... -> (C, 1, H, W)
        # Each layer with kernel (3, 3, 3) and no temporal padding
        # reduces temporal dimension by 2
        layers = []

        # First layer: 1 -> n_channels
        layers.append(
            nn.Conv3d(1, n_channels, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        )  # no temporal padding
        layers.append(nn.GELU())

        # Middle layers: keep collapsing temporal dimension
        t_remaining = n_snap - 2  # after first layer
        while t_remaining > 1:
            kt = min(3, t_remaining)
            layers.append(
                nn.Conv3d(
                    n_channels, n_channels, kernel_size=(kt, 3, 3), padding=(0, 1, 1)
                )
            )
            layers.append(nn.GELU())
            t_remaining = t_remaining - (kt - 1)

        self.encoder = nn.Sequential(*layers)

        # Final projection: (C, 1, H, W) -> (1, H, W)
        self.proj = nn.Conv2d(n_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: (batch, n_snap, H, W) - coarse time series
        Returns:
            residual: (batch, H, W) - predicted fine-scale residual
        """
        # Add channel dim: (B, T, H, W) -> (B, 1, T, H, W)
        x = x.unsqueeze(1)

        # 3D convolutions collapse temporal dimension
        x = self.encoder(x)  # (B, C, 1, H, W)

        # Squeeze temporal dim, project to output
        x = x.squeeze(2)  # (B, C, H, W)
        x = self.proj(x)  # (B, 1, H, W)
        return x.squeeze(1)  # (B, H, W)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalCNN2D_Cat(nn.Module):
    """2D CNN with time frames stacked as input channels.

    Simpler alternative: treat each time snapshot as a separate channel.
    The first 2D conv layer effectively does temporal mixing via its
    channel weights. Subsequent layers are purely spatial.

    This is computationally cheaper but can't learn local spatiotemporal
    patterns (the temporal mixing is global across the image).
    """

    def __init__(self, n_channels=32, n_snap=8, n_layers=4):
        super().__init__()
        self.n_snap = n_snap

        layers = [nn.Conv2d(n_snap, n_channels, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Conv2d(n_channels, n_channels, 3, padding=1), nn.GELU()])
        layers.append(nn.Conv2d(n_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, n_snap, H, W) - coarse time series
        Returns:
            residual: (batch, H, W) - predicted fine-scale residual
        """
        # x is already (B, T, H, W) which maps to (B, C_in, H, W)
        out = self.net(x)  # (B, 1, H, W)
        return out.squeeze(1)  # (B, H, W)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SingleFrameCNN(nn.Module):
    """Baseline: 2D CNN using only the LAST coarse frame.

    This is the experiment 003 baseline. It ignores all temporal
    information and tries to predict fine structure from one frame.
    If the temporal models beat this, the time series helps.
    """

    def __init__(self, n_channels=32, n_layers=4):
        super().__init__()
        layers = [nn.Conv2d(1, n_channels, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Conv2d(n_channels, n_channels, 3, padding=1), nn.GELU()])
        layers.append(nn.Conv2d(n_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, n_snap, H, W) - takes full series but uses only last frame
        Returns:
            residual: (batch, H, W)
        """
        last_frame = x[:, -1:, :, :]  # (B, 1, H, W)
        out = self.net(last_frame)
        return out.squeeze(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    n_snap = 8
    x = torch.randn(4, n_snap, 128, 128).to(device)

    models = {
        "SingleFrame CNN": SingleFrameCNN(n_channels=32, n_layers=4),
        "Temporal 2D+Cat": TemporalCNN2D_Cat(n_channels=32, n_snap=n_snap),
        "Temporal 3D CNN": TemporalCNN3D(n_channels=32, n_snap=n_snap),
    }

    for name, model in models.items():
        model = model.to(device)
        y = model(x)
        print(
            f"{name:20s}: params={model.count_params():>8,}, "
            f"in={list(x.shape)}, out={list(y.shape)}"
        )

    print("\nSanity check passed.")
