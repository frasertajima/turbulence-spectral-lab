"""fno_2d.py - Minimal 2D Fourier Neural Operator for spectral super-resolution.

This is deliberately kept small and readable. The goal is not production quality
but rapid experimentation: can an FNO learn to recover filtered spectral content?

Architecture:
    Input:  coarse field (nx, ny) - low-pass filtered turbulence
    Output: residual field (nx, ny) - predicted high-frequency content

    The model predicts the RESIDUAL, so final = coarse + model(coarse).
    This is easier to learn than the full fine field.

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (2021)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Core FNO layer: convolution in Fourier space.

    Instead of spatial convolution, we:
    1. FFT the input
    2. Multiply by learnable complex weights (for low modes only)
    3. IFFT back

    This gives global receptive field in one layer.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep (x)
        self.modes2 = modes2  # Number of Fourier modes to keep (y)

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication: (batch, in, x, y) * (in, out, x, y) -> (batch, out, x, y)"""
        weights_complex = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, weights_complex)

    def forward(self, x):
        batch_size = x.shape[0]

        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """Minimal 2D Fourier Neural Operator.

    Architecture:
        lift -> [spectral_conv + pointwise_conv + activation] x N -> project

    Default: 4 layers, 32 channels, 12 Fourier modes.
    Tiny enough to train in seconds on a single GPU.
    """

    def __init__(self, modes1=12, modes2=12, width=32, n_layers=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        # Lift: 1 channel (scalar field) -> width channels
        self.fc0 = nn.Linear(1, self.width)

        # Spectral convolution layers
        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
                for _ in range(n_layers)
            ]
        )

        # Pointwise convolution layers (1x1 convs)
        self.pointwise_convs = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(n_layers)]
        )

        # Project: width channels -> 1 channel (predicted residual)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, nx, ny) - coarse input field
        Returns:
            residual: (batch, nx, ny) - predicted high-frequency residual
        """
        # Add channel dim: (batch, nx, ny) -> (batch, nx, ny, 1)
        x = x.unsqueeze(-1)

        # Lift
        x = self.fc0(x)  # (batch, nx, ny, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, nx, ny)

        # FNO layers
        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.pointwise_convs[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # Project
        x = x.permute(0, 2, 3, 1)  # (batch, nx, ny, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # (batch, nx, ny, 1)

        return x.squeeze(-1)  # (batch, nx, ny)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNN(nn.Module):
    """Baseline CNN for comparison. Standard U-Net-like architecture.

    If this beats the FNO, spectral structure isn't helping.
    If FNO beats this, the Fourier structure matters.
    """

    def __init__(self, n_channels=32, n_layers=4):
        super().__init__()
        layers = [nn.Conv2d(1, n_channels, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Conv2d(n_channels, n_channels, 3, padding=1), nn.GELU()])
        layers.append(nn.Conv2d(n_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Same interface as FNO2d."""
        x = x.unsqueeze(1)  # (batch, 1, nx, ny)
        out = self.net(x)
        return out.squeeze(1)  # (batch, nx, ny)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fno = FNO2d(modes1=12, modes2=12, width=32, n_layers=4).to(device)
    cnn = SimpleCNN(n_channels=32, n_layers=4).to(device)

    print(f"FNO parameters: {fno.count_params():,}")
    print(f"CNN parameters: {cnn.count_params():,}")

    # Test forward pass
    x = torch.randn(4, 64, 64).to(device)
    y_fno = fno(x)
    y_cnn = cnn(x)

    print(f"Input shape:  {x.shape}")
    print(f"FNO output:   {y_fno.shape}")
    print(f"CNN output:   {y_cnn.shape}")
    print("Sanity check passed.")
