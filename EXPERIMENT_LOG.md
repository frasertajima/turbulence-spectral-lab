# Experiment Log

Running record of experiments, observations, and findings.

---

## 005 - Numerical Error Correction: Stam vs Spectral Solver

**Date**: 2026-02-14
**Status**: Complete

**Question**: Can a neural network learn to correct the systematic numerical
error of a fast semi-Lagrangian solver by comparing it to an accurate
pseudo-spectral solver?

This is a fundamentally different framing from experiments 001-004b. Instead
of predicting MISSING information (spectral content lost by filtering), we
predict SYSTEMATIC NUMERICAL ERROR — the difference between a cheap solver
and an accurate one, run from identical initial conditions.

### Motivation

Experiments 001-004b established that super-resolution fails because fine
scales depend on evolution history, not the current coarse state. The only
experiment that worked (002, 12.8%) had a **deterministic local mapping**.

The key insight: numerical dissipation error IS a deterministic local mapping.
The semi-Lagrangian scheme smears gradients in proportion to local velocity
and curvature — exactly the kind of coupling CNN excelled at in experiment 002.

### Implementation

**Stam Solver** (`stam_solver.cuf`) — new semi-Lagrangian Burgers solver:
- Same equation as `burgers_solver.cuf` (2D viscous Burgers)
- Semi-Lagrangian advection with bilinear interpolation (deliberately dissipative)
- Implicit Jacobi diffusion (20 iterations)
- Periodic boundary conditions (matching spectral solver)
- Single precision (matching spectral lab convention)

**Data Generation** (`experiment_005.cuf`):
- 128x128 grid, nu=0.01, dt=0.001, 500 steps (t=0.5)
- Both solvers run from identical ICs (low-k Kolmogorov, max|u|=1)
- 400 samples, ~250ms/pair, no blowups
- Output: paired `(stam_sample, spectral_sample)` binary fields

### Results

**Part 1 — Single run diagnostic:**
- Spectral max|u| = 0.877 (retains structure)
- Stam max|u| = 0.371 (heavily over-dissipated, ~42% of true amplitude)
- Error RMS = 0.162, Error max = 0.565
- The Stam solver loses about 58% of the field amplitude over 500 steps

**Part 2 — Training (400 samples, 80/20 split):**

| Model | Parameters | Training Time | Val Loss | Improvement |
|-------|-----------|---------------|----------|-------------|
| CNN   | 19,105    | 28s           | 0.01670  | **40.5%**   |
| FNO   | 2,367,937 | 95s           | 0.00973  | **65.3%**   |

### Analysis — Why This Works So Well

**40.5% CNN, 65.3% FNO** — dramatically better than any previous experiment.
The progression tells the complete story:

| Exp | Target | Deterministic? | Best | Why |
|-----|--------|---------------|------|-----|
| 001 | Random phases | No | 0% | Nothing to learn |
| 002 | Gradient texture | Yes (local) | 12.8% CNN | CNN matches local coupling |
| 003 | PDE fine scales | No | 5.0% | History-dependent |
| 004 | Temporal context | No | ~1% | Too subtle |
| 004b | Fixed IC | No | 1.3% | No fixed structure |
| **005** | **Numerical error** | **Yes (local+global)** | **65.3% FNO** | **Systematic scheme bias** |

Three factors explain the jump:

1. **Deterministic mapping**: Given the same Stam field, the error is always
   the same. No hidden variables, no history dependence. The error is a
   function of the current field structure, not the path that got there.

2. **Large, structured signal**: Error RMS ~0.16 (vs residual RMS ~0.05 in
   exp 003). The Stam solver introduces massive dissipation — there's a lot
   of learnable signal. The error is spatially structured (concentrated near
   gradients and steep features), not random noise.

3. **FNO beats CNN for the first time**: In experiment 002, the coupling was
   purely local (gradients, Laplacian) so CNN with 3x3 kernels was optimal.
   But numerical dissipation error has **both local AND global** components:
   - Local: bilinear interpolation smears gradients (CNN-friendly)
   - Global: accumulated phase errors shift features (FNO-friendly)
   
   The FNO's spectral convolutions can capture the global phase error that
   the CNN's local receptive field cannot reach.

**FNO overfitting note**: Train loss (0.0058) << Val loss (0.0097), suggesting
the FNO is memorizing some patterns. More data or regularization could
improve generalization. The CNN shows much less overfitting (0.0168 vs 0.0167).

### Implications

This result validates the core thesis: **ML-corrected fast solvers** are a
viable approach to getting spectral-quality results at real-time speeds.

The practical application is clear:
1. Run Stam solver at 60fps (16ms/frame on GPU)
2. Apply CNN correction in one forward pass (~1ms for 19K params at 128x128)
3. Get ~40% closer to spectral accuracy at ~59fps

With the FNO (65% correction but more expensive inference), the tradeoff
shifts — but even a single-pass CNN delivers 40% error reduction.

### Phase 2: Multi-Time Training and Real-Time Correction

**Problem**: The Phase 1 models were trained only on 500-step accumulated
error. When applied at intermediate times (e.g. step 10), the error is tiny
but the model applies a 500-step-sized correction → catastrophic overshoot.

Feeding the correction back into the solver caused exponential blowup
(CNN: max|u| grew from 4.9 at step 10 to infinity by step 430; FNO:
stabilized around max 140 but never recovered).

**Solution**: Multi-time training. Instead of sampling only at step 500,
sample at steps {50, 100, 150, 200, 250, 300, 350, 400, 450, 500} per IC.
This gives 10x more training pairs (4000 total from 400 ICs) and — crucially —
teaches the model that **correction magnitude scales with error magnitude**.

The model learns this implicitly: at step 50 the Stam field is close to
spectral, so the correction should be small. At step 500 the error is large,
so the correction should be large. No explicit time input needed — the
error magnitude is encoded in the Stam field structure itself.

**Multi-time training results** (4000 pairs, 800 val, 100 epochs):

| Model | Parameters | Training Time | Val Loss | Improvement |
|-------|-----------|---------------|----------|-------------|
| CNN   | 19,105    | 298s          | 0.01282  | **39.7%**   |
| FNO   | 2,367,937 | 1048s         | 0.00404  | **81.0%**   |

The FNO jumps from 65.3% → **81.0%** with multi-time data. The CNN holds
steady at ~40%. Both models show much less overfitting with 10x more data.

### Real-Time Post-Processing Test

**Setup**: Run both solvers from an unseen IC (seed=9999), apply CNN
correction as **display post-processing** every 10 steps. The Stam solver
runs uncorrected internally — corrections are for display only.

**CNN real-time trajectory** (2.8ms per correction, 17% of 60fps budget):

| Step | MSE uncorr | MSE corrected | Improvement |
|------|-----------|--------------|-------------|
| 10   | 0.00175   | 0.04283      | -2352% (overcorrects — error too small) |
| 40   | 0.00893   | 0.01056      | **-18%** (nearly breakeven) |
| 50   | 0.01075   | 0.00974      | **+9%** (helping!) |
| 100  | 0.01760   | 0.01164      | **+34%** |
| 200  | 0.02649   | 0.01612      | **+39%** |
| 300  | 0.03201   | 0.01848      | **+42%** (peak) |
| 400  | 0.03565   | 0.02074      | **+42%** |
| 500  | 0.03827   | 0.02350      | **+39%** |

**FNO real-time trajectory** (4.3ms per correction, 25% of 60fps budget):

| Step | MSE uncorr | MSE corrected | Improvement |
|------|-----------|--------------|-------------|
| 10   | 0.00175   | 0.02371      | -1257% (overcorrects, but 30x less than Phase 1) |
| 30   | 0.00684   | 0.00431      | **+37%** (already helping at step 30!) |
| 50   | 0.01075   | 0.00391      | **+64%** (peak!) |
| 100  | 0.01760   | 0.00708      | **+60%** |
| 200  | 0.02649   | 0.01155      | **+56%** |
| 300  | 0.03201   | 0.01444      | **+55%** |
| 400  | 0.03565   | 0.01592      | **+55%** |
| 500  | 0.03827   | 0.01751      | **+54%** |

**Stability is remarkable**: once the FNO starts helping (step 30), it
delivers a steady 54-64% improvement for the remaining 470 steps with no
drift, no oscillation, no blowup. The corrected max|u| tracks spectral
max|u| closely (0.86-0.88 vs 0.84-0.95).

### Comparison: Phase 1 vs Phase 2

| Model | Phase 1 (500-step only) | Phase 2 (multi-time) |
|-------|------------------------|---------------------|
| CNN real-time breakeven | Step 250 | **Step 50** |
| CNN peak improvement | 42% (step 300+) | **42% (step 300+)** |
| FNO real-time breakeven | Step 330 | **Step 30** |
| FNO peak improvement | 62% (step 500) | **64% (step 50!)** |
| FNO sustained improvement | N/A (grew with time) | **54-64% entire trajectory** |

Multi-time training moved the breakeven point 5-10x earlier without
sacrificing peak performance. The model learned to scale its corrections.

### What Makes This Work: Three Key Properties

1. **Deterministic error**: Given a Stam field, the error vs spectral is
   always the same. No hidden variables. This is why exp 002 (deterministic
   gradient coupling) worked but exp 003 (PDE history) failed.

2. **Error magnitude is self-evident**: The model doesn't need a time input.
   A heavily dissipated field (step 500, max|u|=0.30) looks different from a
   slightly dissipated one (step 50, max|u|=0.86). The error magnitude is
   encoded in the field structure. Multi-time training lets the model see this
   full range.

3. **Stable post-processing**: Because corrections aren't fed back, there's
   no feedback instability. The Stam solver runs its own dynamics; the CNN
   just improves the display output. This is inherently stable.

### Practical Implications

At 128x128 with an RTX 4060:
- Stam step: ~0.3ms (can run ~50 steps per frame at 60fps)
- CNN correction: 2.8ms (one per frame)
- FNO correction: 4.3ms (one per frame)
- **Total budget used**: CNN 17%, FNO 25% of 16.7ms frame budget

A real-time application would:
1. Step Stam solver N times per frame (~0.3ms × N)
2. Extract field to PyTorch tensor
3. Apply CNN/FNO correction (2.8-4.3ms)
4. Display corrected field
5. Continue Stam from uncorrected state next frame

This gives spectral-quality *display* at real-time speed, with the Stam
solver providing the underlying dynamics.

### Updated Experiment Summary

| Exp | Target | Deterministic? | Best | Why |
|-----|--------|---------------|------|-----|
| 001 | Random phases | No | 0% | Nothing to learn |
| 002 | Gradient texture | Yes (local) | 12.8% CNN | CNN matches local coupling |
| 003 | PDE fine scales | No | 5.0% | History-dependent |
| 004 | Temporal context | No | ~1% | Too subtle |
| 004b | Fixed IC | No | 1.3% | No fixed structure |
| **005** | **Numerical error** | **Yes** | **81% FNO, 40% CNN** | **Systematic scheme bias** |
| **005-RT** | **Real-time correction** | **Yes** | **54-64% FNO sustained** | **Multi-time training** |

### Next Steps

- Feed correction back into solver with damping (α × correction, α<1)
- Deeper CNN to close gap with FNO (try 8 layers, 64 channels)
- Test at 256x256 (does correction generalize to higher resolution?)
- Full Navier-Stokes: extend from Burgers to incompressible flow with pressure
- Integrate with the existing 60fps fluid_dynamics.so for interactive demo

---

## 003 - Burgers Equation: Physics-Generated Cross-Scale Coupling

**Date**: 2026-02-14
**Status**: Complete

**Question**: Can neural networks learn to predict fine-scale structure
generated by actual PDE dynamics (the 2D viscous Burgers equation)?

This is the key experiment that bridges synthetic coupling (experiment 002)
and real turbulence. The Burgers equation `du/dt + u*du/dx = nu*laplacian(u)`
generates fine structure through nonlinear advection steepening — the
simplest model of the turbulent energy cascade.

### Implementation

**Burgers Solver** (`burgers_solver.cuf`):
- 2D viscous Burgers equation (u and v components)
- Pseudo-spectral method: spatial derivatives via cuFFT
- RK4 time integration with explicit GPU kernels
- 2/3 dealiasing rule for nonlinear term
- Runs entirely on GPU (RTX 4060)

**Data Generation** (`experiment_003.cuf`):
- Random Kolmogorov ICs, low-pass filtered to k<16, scaled to max|u|=1
- Evolve 1000 steps at dt=0.001 (t_final=1.0) with nu=0.01
- Low-pass filter evolved field -> coarse input
- Fine - coarse = residual (training target)
- 200 samples, ~385ms/sample, 77s total

**Stability Issues Encountered**:

1. **`state%nu` in CUF kernel directive** — accessing derived type members
   from `!$cuf kernel do` directives reads garbage GPU memory. Fixed by
   extracting to a local scalar `nu_local` before the kernel.

2. **Shock blowup** — original IC with max|u|~3.2 and nu=0.001 caused
   nonlinear steepening to produce shocks that exceeded grid resolution
   around step 232. Exponential blowup: 3.0 -> 20.4 -> Inf in 12 steps.
   Fixed by normalizing IC to max|u|=1.0 and using nu=0.01.

3. **5 out of 200 samples still blew up** — certain random ICs produced
   slightly different dynamics that caused late instability. Handled by
   NaN-filtering in the Python data loader.

### Results

Tested at two filter cutoffs:

**k_cutoff=16** (removes only nonlinearly-generated fine structure):

| Model | Parameters | Training Time | Val Loss | Improvement |
|-------|-----------|---------------|----------|-------------|
| FNO   | 2,367,937 | 109s          | 0.000174 | **0.0%**    |
| CNN   | 19,105    | 32s           | 0.000167 | **2.9%**    |

**k_cutoff=8** (removes IC information too):

| Model | Parameters | Training Time | Val Loss | Improvement |
|-------|-----------|---------------|----------|-------------|
| FNO   | 2,367,937 | 109s          | 0.001696 | **0.0%**    |
| CNN   | 19,105    | 33s           | 0.001613 | **3.1%**    |

**k_cutoff=32** (first attempt — residual was negligible):
- Residual RMS ~ 0.002 (0.2% of field amplitude)
- Both models correctly learned to output zero
- The Burgers evolution at nu=0.01 barely populates modes above k=32

### Analysis — Why This Is The Expected Result

**The fundamental issue**: Unlike experiment 002's deterministic coupling,
PDE-generated fine structure from **random initial conditions** is not
predictable from a single coarse snapshot.

Here's why:

1. **IC information is lost by filtering**: The IC has energy at k=1-16.
   After evolving, the nonlinear term `u*du/dx` transfers energy to k>16.
   But this transferred energy depends on the *specific* k=1-16 modes.
   Filtering at k=16 removes the modes that *caused* the fine structure.

2. **Even at k=16 cutoff** (preserving the IC): the coarse field contains
   the IC, but the fine-scale residual depends on the *nonlinear interaction
   history* — 1000 timesteps of quadratic coupling. A single forward pass
   through an FNO cannot simulate this evolution.

3. **CNN's 3% improvement** comes from local gradient patterns: near steep
   features (proto-shocks), the CNN learns that high gradients predict
   nearby fine structure. But it can't reconstruct the specific fine-scale
   pattern.

**This is analogous to the weather prediction problem**: you can't predict
small-scale weather features from a single coarse observation without
running the dynamics. A single snapshot lacks the temporal information
needed to reconstruct the fine-scale state.

### Key Takeaway

**Super-resolution on PDE outputs requires temporal context.**

For a network to predict fine structure from coarse data, it needs either:

1. **Multiple time snapshots** — so it can infer the dynamics
2. **The initial condition** — so it can (implicitly) re-simulate
3. **Statistical conditioning** — predict the *distribution* of possible
   fine structures, not a specific realization (generative approach)
4. **Physical constraints** — e.g., predict fields consistent with the
   PDE (physics-informed approach)

This validates why PINN-style approaches (which embed the PDE) and
temporal super-resolution (which use time sequences) outperform
single-snapshot spatial super-resolution in turbulence.

### Comparison Across Experiments

| Experiment | Coupling Type | Best Improvement | Learnable? |
|-----------|---------------|-----------------|------------|
| 001 | Random phases (none) | 0% | No |
| 002 | Deterministic gradients | 12.8% (CNN) | Yes |
| 003 | PDE dynamics (random IC) | 3.1% (CNN) | Barely |

**The progression tells a clear story**:
- No coupling -> unlearnable (001)
- Deterministic local coupling -> learnable by CNN (002)
- PDE coupling with random IC -> mostly unlearnable from single snapshot (003)

### Next Steps

To make PDE super-resolution work, the experiment design needs to change:

1. **Time-series input**: Give the model N consecutive snapshots instead
   of one, so it can infer the dynamics
2. **Conditional generation**: Train a generative model (VAE/diffusion)
   to sample plausible fine-scale fields given coarse input
3. **Physics-informed loss**: Add Burgers residual as a regularizer
4. **Fixed IC family**: Use a parameterized IC (e.g., single vortex
   with varying strength) so the coarse field uniquely determines the
   fine field — analogous to experiment 002 but physics-based

Option 4 is the most direct next experiment: if the IC is fully determined
by the coarse field (because it only has k<16 modes and we filter at k=16),
then super-resolution becomes deterministic.

---

## 004 - Temporal Context and the Cryo-EM Hypothesis

**Date**: 2026-02-14
**Status**: Complete

**Question**: Two parallel questions motivated by experiment 003's key finding
that single coarse snapshots lack information to reconstruct fine scales:

1. **Temporal context (004)**: Can providing multiple time snapshots give the
   network enough dynamical information to infer fine-scale structure?
2. **Fixed IC family (004b)**: If we hold the initial condition fixed and only
   vary evolution time (like cryo-EM's fixed protein with different noise),
   does the consistent underlying structure make the mapping learnable?

### Experiment 004: Temporal Super-Resolution

**Setup**:
- 128x128 grid, nu=0.005, dt=0.0005, k_cutoff=8
- 8 coarse snapshots at regular intervals + 1 fine target at final time
- 400 samples from random ICs

**Models**:
- **TemporalCNN3D** (75K params): True 3D convolutions over (time, H, W)
- **TemporalCNN2D_Cat** (21K params): Stack 8 frames as input channels to 2D CNN
- **SingleFrameCNN** (19K params): Uses only the last frame (baseline)

**Results**:

| Model | Parameters | Improvement |
|-------|-----------|-------------|
| TemporalCNN3D | 75,329 | 0.0% |
| TemporalCNN2D_Cat | 21,345 | 0.0% |
| SingleFrameCNN | 19,105 | ~1% |

**Analysis**: Temporal models converged to predicting zero residual. The
temporal signal is too subtle for these architectures to extract — the
frame-to-frame changes at this cadence are small relative to the spatial
structure, and 8 snapshots of a chaotically evolving field don't contain
enough dynamical information to reconstruct fine scales. The SingleFrame
baseline matched experiment 003 performance (~1-3%).

### Experiment 004 — Denoising Approach (Cryo-EM Inspired)

**Motivation**: In cryo-EM, focusing on noise/artifacts rather than signal
led to better results. Can we reframe turbulence super-resolution the same
way? Instead of predicting what's missing (residual = fine - coarse), predict
what filtering got wrong (artifact = coarse - fine).

**Modes tested** on experiment 003 data (k=8, random ICs):

| Mode | Description | Improvement |
|------|-------------|-------------|
| Artifact | Predict coarse - fine | 4.7% |
| Residual | Predict fine - coarse | 4.7% |
| Direct | Predict fine directly | 3.0% |
| Artifact + gradient loss | Weighted by local gradient magnitude | **5.0%** |

**Key finding**: Artifact and residual modes are mathematically equivalent
(one is the negative of the other). The gradient-weighted loss gave the best
result (5.0%) because filtering artifacts concentrate near steep gradients,
and the loss function directs model attention there. However, the improvement
over standard residual training is marginal.

### Experiment 004b: Fixed IC Family (Cryo-EM Analogy)

**Setup**:
- 256x256 grid, nu=0.01, dt=0.001, k_cutoff=8
- 5 initial conditions x 60 time samples each = 300 (coarse, fine) pairs
- Each IC sampled every 10 steps (dt_save = 0.01, total t = 0.6 per IC)
- All ICs stable throughout evolution (max|u| ~ 0.85-0.96)
- Residual RMS ~ 0.05-0.08 (meaningful signal)

**Hypothesis**: Like cryo-EM where every image shows the SAME protein (just
with different noise/orientation), using few ICs but many time samples should
create consistent coarse→fine relationships that a network can learn.

**Results**:

| Configuration | Improvement |
|---|---|
| CNN residual (all 300 samples) | -0.2% |
| Artifact mode | 1.0% |
| Residual mode | 0.9% |
| Direct mode | 0.4% |
| Artifact + gradient loss | 1.2% |
| Single IC (train+test same IC, 48/12 split) | 1.3% |
| Cross IC (train IC1-4, test IC5) | **-0.6%** |

### Why the Cryo-EM Analogy Breaks Down

The fixed-IC approach performed **worse** than random-IC experiment 003
(1.2% vs 5.0% best). The single-IC vs cross-IC comparison is definitive:

1. **Within one IC** (1.3%): The model memorizes tiny patterns specific to
   that IC's evolution trajectory. Very slight improvement, not generalizable.

2. **Across ICs** (-0.6%): What the model learns from IC1-4 actively hurts
   performance on IC5. The "consistent structure" hypothesis is wrong.

**Why cryo-EM works but turbulence doesn't**:

| Property | Cryo-EM | Turbulence Filtering |
|----------|---------|---------------------|
| Signal | Fixed 3D protein | Evolving flow field |
| Noise | Additive, independent of signal | Deterministic function of signal |
| Structure | Same object every image | Different state every snapshot |
| Recovery | Average out noise | No averaging possible |

In cryo-EM, the protein is a **fixed unknown** that doesn't change between
images. The noise is additive electron shot noise — statistically independent
of the protein structure. Averaging/denoising across many images isolates the
consistent signal.

In turbulence, there is no fixed unknown. The coarse→fine mapping is
deterministic but **changes with time** because the field itself evolves.
At time t₁ the coarse pattern maps to one fine pattern; at t₂ the same IC
has evolved so the mapping is completely different. The "noise" (filtering
residual) is not independent — it's a nonlinear function of the coarse field
through the PDE history.

### Updated Comparison Across All Experiments

| Exp | Coupling Type | Best Improvement | Key Finding |
|-----|---------------|-----------------|-------------|
| 001 | Random phases (none) | 0% | Sanity check: random = unlearnable |
| 002 | Deterministic gradients | 12.8% (CNN) | Local coupling is learnable |
| 003 | PDE + random IC | 5.0% (artifact+grad) | Mostly unpredictable from snapshot |
| 004 temporal | PDE + 8 time frames | ~1% | Temporal context insufficient |
| 004 denoise | Artifact/residual framing | 5.0% | Gradient loss helps marginally |
| 004b fixed IC | Same IC, diff times | 1.3% (single IC) | Cryo-EM analogy fails |

### Implications for Turbulence Super-Resolution

The progression from 001→004b tells a coherent story:

1. **Cross-scale coupling must exist** for any learning (001 vs 002)
2. **Coupling must be deterministic from input** (002 works, 003 mostly doesn't)
3. **PDE dynamics make coupling non-deterministic** from single snapshots —
   the fine scales depend on evolution history, not just current coarse state
4. **Temporal context helps in principle** but extracting the signal is hard
   with simple architectures and limited data
5. **Fixed structure doesn't help** because PDE evolution means the
   "structure" itself changes (unlike cryo-EM's fixed protein)

**What WOULD work**: The only configuration that showed substantial learning
(12.8%) was experiment 002, where fine = f(coarse) was a deterministic local
function. This suggests that for turbulence super-resolution to succeed, we
need either:

- **Physics-informed networks** (PINNs) that can implicitly solve the PDE
- **Generative models** that predict distributions, not point estimates
- **Much longer time series** with architectures designed for sequence modeling
- **Data from statistically stationary turbulence** where the coarse→fine
  relationship has a stable statistical structure (unlike transient Burgers)

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
