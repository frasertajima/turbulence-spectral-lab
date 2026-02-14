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

## 002 - Gradient-Coupled Deterministic Fine Structure

**Date**: 2026-02-14
**Status**: Complete

**Question**: If fine-scale structure is a deterministic function of
coarse-scale gradients, can neural networks learn to predict it?

**Key Design Change from 001**: The fine-scale texture is computed from
nonlinear combinations of the coarse field:
- Cross-gradient: `dC/dx * dC/dy` (doubles wavenumber)
- Strain: `(dC/dx)^2 - (dC/dy)^2`
- Field-Laplacian product: `C * nabla^2(C)`
- Gradient magnitude coupling: `|grad C|^2 * C`

These mimic the advective nonlinearity `u.grad(u)` that generates
small scales from large scales in real turbulence.

**First attempt (random coupling)**: Failed. Gradient magnitude modulated
random noise — but the noise values are still unpredictable. Models
correctly learned to output zero. Same lesson as experiment 001:
random = unlearnable.

**Second attempt (deterministic coupling)**: Succeeded!

**Setup**:
- 256x256, k_cutoff=32, coupling_strength=1.0
- 200 training pairs, 5ms/sample generation on RTX 4060
- Residual RMS scales linearly with coupling (0.009 to 0.093)

**Results (coupling = 1.0)**:

| Model           | Parameters | Training Time | Val Loss | Improvement |
|-----------------|-----------|---------------|----------|-------------|
| CNN             | 19,105    | 67s           | 0.0039   | **12.8%**   |
| FNO             | 2,367,937 | 220s          | 0.0117   | 7.7%        |
| FNO + spec loss | 2,367,937 | 222s          | 0.0116   | 3.4%        |

**Surprises**:

1. **CNN beats FNO** — the 19K-parameter CNN outperforms the 2.4M-parameter
   FNO. The fine structure is local (gradients, Laplacian) so a CNN with 3x3
   kernels is the natural architecture. The FNO's global Fourier modes are
   unnecessary for this task.

2. **Spectral loss hurts** — the spectral-weighted loss caused overfitting
   (val MSE increased after epoch 10 while train loss kept dropping).
   Plain MSE was more stable.

3. **12.8% is real but modest** — the CNN captures some of the deterministic
   coupling but not all. This makes sense: the texture involves products
   of gradients, which are quadratic operations. A deeper or wider CNN
   might capture more.

**Analysis - What This Means**:

- Neural networks CAN learn cross-scale coupling when it exists
- The coupling needs to be **deterministic** (computable from input)
- CNN > FNO for local operations (gradients, Laplacian)
- FNO would shine if the coupling were truly global/spectral

**Lessons for Real Turbulence**:

Real Navier-Stokes turbulence has both:
- **Local coupling** (strain producing small eddies) — CNN-friendly
- **Nonlocal coupling** (pressure, long-range interactions) — FNO-friendly

A hybrid approach might be needed. But first, we should test on data with
actual physical cross-scale coupling (PDE-generated), not synthetic.

**Next Steps (Experiment 003)**:

Options in order of increasing realism:
1. **Burgers equation** — simplest PDE with nonlinear cross-scale coupling,
   cheap to solve in Fortran, deterministic ground truth
2. **2D Navier-Stokes** — realistic but more expensive
3. **LES data** — most realistic but requires preprocessing the 150GB dataset

---
