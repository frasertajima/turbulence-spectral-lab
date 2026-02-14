# Experiment Log

Running record of experiments, observations, and findings.

---

## 001 - Synthetic 2D Spectral Filtering and Recovery

**Date**: 2026-02-14
**Status**: Ready to run

**Question**: Can a small FNO recover high-frequency spectral content
removed by low-pass filtering from synthetic Kolmogorov turbulence?

**Setup**:
- 256x256 2D fields with -5/3 power law spectrum
- Low-pass filter at k=32 (removes everything a 64x64 grid would miss)
- 100 training pairs, each with different random phases
- FNO: 4 layers, 32 channels, 12 Fourier modes (~33K parameters)
- CNN baseline: 4 layers, 32 channels (~33K parameters)

**What to look for**:
- Does the FNO recover any energy above k=32?
- Does it outperform the CNN baseline?
- How does reconstruction quality vary across wavenumber?
- Is the spectral-weighted loss better than plain MSE?

**Results**: _pending_

**Observations**: _pending_

---
