# Experiment Log

Running record of experiments, observations, and findings.

---

## 001 - Synthetic 2D Spectral Filtering and Recovery

**Date**: 2026-02-14
**Status**: Complete

**Question**: Can a small FNO recover high-frequency spectral content
removed by low-pass filtering from synthetic Kolmogorov turbulence?

**Setup**:
- 256x256 2D fields with -5/3 power law spectrum
- Low-pass filter at k=32 (removes everything a 64x64 grid would miss)
- 100 training pairs, each with different random phases
- FNO: 4 layers, 32 channels, 12 Fourier modes (2.4M parameters)
- CNN baseline: 4 layers, 32 channels (19K parameters)
- Data generation: 311ms for 100 pairs on RTX 4060

**Results**:

| Model | Parameters | Training Time | Val Loss | MSE Improvement |
|-------|-----------|---------------|----------|-----------------|
| FNO   | 2,367,937 | 112s          | 0.0547   | 0.0%            |
| CNN   | 19,105    | 34s           | 0.0545   | 0.3%            |

**Both models achieved essentially 0% improvement.**

**Analysis - Why This Is The Correct Result**:

The synthetic Kolmogorov fields have **independent random phases** at each
wavenumber. When we low-pass filter, the removed high-frequency modes have
phases that are completely uncorrelated with the retained low-frequency modes.

Therefore: there is **no information in the coarse field that predicts the
fine-scale content**. The optimal prediction is zero residual (output nothing),
which is exactly what both models learn.

This is not a failure — it's a **sanity check**. It tells us:

1. The pipeline works correctly (data generation, training, evaluation)
2. The models aren't hallucinating false structure
3. **Random-phase synthetic turbulence is the wrong test case**

**Key Insight**: For super-resolution to work, the fine-scale content must
be **statistically dependent** on the coarse-scale content. This happens in
real turbulence because:

- The Navier-Stokes equations create **cross-scale coupling** (energy cascade)
- Coherent structures (vortices, shear layers) span multiple scales
- Local flow features constrain what small scales can exist

Pure Gaussian random fields with prescribed spectrum lack this coupling entirely.

**Next Steps (Experiment 002)**:

To create learnable structure, we need fields where fine scales depend on
coarse scales. Options:

1. **Navier-Stokes generated turbulence** — solve the equations, then filter
2. **Deterministic sub-grid model** — add fine structure that's a function
   of local gradients (e.g., strain-rate dependent texture)
3. **Vortex-coupled fields** — large vortices induce correlated fine structure
4. **Use actual LES data** — where cross-scale coupling is physical

Option 3 is the fastest to test: generate fields where fine-scale features
are explicitly correlated with coarse-scale gradients.

---
