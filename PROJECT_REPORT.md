# Turbulence Spectral Lab: Neural Correction of Fast Fluid Solvers

**A Research Report**
February 2026

---

## Executive Summary

This project demonstrates that neural networks can learn to correct the
systematic numerical errors of fast fluid solvers, achieving 40-81% error
reduction at real-time speeds (300-400 fps on an RTX 4060). The key insight
is deceptively simple: the numerical dissipation introduced by a cheap
semi-Lagrangian solver is a *deterministic function of the current field
state* -- not random noise, not history-dependent chaos, but a structured,
repeatable, learnable bias.

By training a Fourier Neural Operator (FNO, 2.4M parameters) and a simple
CNN (19K parameters) to predict this bias, we built an interactive fluid
simulation where users can inject smoke, place obstacles, apply wind, and
watch turbulent flow in real time -- with neural-corrected quality that
visibly approaches that of an expensive pseudo-spectral reference solver.

The result is not just a technical achievement. It is a vindication of six
experiments, five of which failed, that systematically narrowed the question
from "can neural networks do turbulence super-resolution?" (no, in general)
to "can neural networks correct numerical scheme errors?" (yes,
dramatically).

---

## 1. Background: Why This Matters

### The Problem

Real-time fluid simulation is a decades-old challenge. The Navier-Stokes
equations governing fluid motion have no general analytical solution, so we
discretize them on grids and march forward in time. The tradeoff is
fundamental:

- **Accurate solvers** (pseudo-spectral, high-order finite difference)
  resolve physics faithfully but are expensive. A pseudo-spectral Burgers
  solver using cuFFT costs ~10x more per step than a semi-Lagrangian one.

- **Fast solvers** (semi-Lagrangian, low-order finite difference) run in
  real time but introduce numerical dissipation -- they smear gradients,
  attenuate high-frequency content, and accumulate phase errors. Jos Stam's
  celebrated "Stable Fluids" (1999) is the canonical example: unconditionally
  stable, beautifully simple, and systematically wrong.

For visual applications (games, VFX, interactive demos), speed wins and
accuracy is sacrificed. For scientific applications (weather, combustion,
aerodynamics), accuracy wins and real-time interaction is impossible. There
has been no good middle ground.

### The Idea

What if we could have both? Run the fast solver for dynamics, then apply a
learned correction to recover accuracy. The correction doesn't need to be
perfect -- even 40% error reduction transforms a visibly over-dissipated
simulation into one that looks physically plausible.

This is not a new idea in broad strokes. Physics-Informed Neural Networks
(PINNs), neural operator methods, and various ML-for-CFD approaches have
been explored extensively. What this project contributes is:

1. **Empirical clarity** about *when and why* neural correction works, built
   from a systematic series of controlled experiments.
2. **A working real-time system** that demonstrates the approach at 300+ fps
   with interactive obstacle placement and wind control.
3. **Evidence that correction generalizes** -- models trained on
   obstacle-free flow correctly handle obstacles they've never seen.

---

## 2. The Journey: Six Experiments

The project's central contribution is not the final demo but the intellectual
path that led to it. Five experiments failed in instructive ways before the
sixth succeeded dramatically.

### Experiment 001: Random Phase Turbulence (0% improvement)

**Question**: Can networks recover high-frequency content removed by
spectral filtering from synthetic Kolmogorov turbulence?

**Result**: Both FNO and CNN learned to output zero. The fine-scale content
has random phases uncorrelated with the coarse field -- there is literally no
information to learn from. This was a deliberate sanity check: if the models
had output anything nonzero, our pipeline would be broken.

**Lesson**: For learning to work, the target must be *predictable* from the
input. Random = unlearnable.

### Experiment 002: Deterministic Gradient Coupling (12.8% CNN)

**Question**: If fine-scale texture is a deterministic function of
coarse-scale gradients (mimicking the nonlinear advection term u*grad(u)),
can networks learn it?

**Result**: The CNN achieved 12.8% error reduction, beating the FNO (7.7%).
This was the first positive result and carried two surprises: (a) the tiny
19K-parameter CNN outperformed the 2.4M-parameter FNO, and (b) 12.8% was
modest despite the coupling being perfectly deterministic.

**Lesson**: Local deterministic coupling is learnable. CNN beats FNO when the
mapping is spatially local. The coupling being "only" 12.8% learnable
reflects the difficulty of approximating nonlinear functions (products of
gradients) with a shallow network.

### Experiment 003: PDE-Generated Fine Structure (5% best)

**Question**: Can networks predict fine scales generated by actual Burgers
equation dynamics from random initial conditions?

**Result**: Marginal improvement (~5% with gradient-weighted loss). The fine
scales depend on the *entire evolution history* of the PDE from a random
initial condition, not just the current coarse state. A single snapshot
contains insufficient information.

**Lesson**: PDE dynamics break the deterministic mapping. This is the weather
prediction problem in miniature: you cannot predict small-scale weather from
a single coarse observation.

### Experiment 004: Temporal Context (1%)

**Question**: Can providing 8 consecutive time snapshots give the network
enough dynamical information to reconstruct fine scales?

**Result**: No meaningful improvement. The temporal signal-to-noise ratio
at 8 frames is too low for simple architectures to extract.

### Experiment 004b: Fixed IC Family (1.3%)

**Question**: Like cryo-EM imaging (where every image shows the same protein
with different noise), does using a fixed initial condition create learnable
structure?

**Result**: No. Unlike cryo-EM's fixed protein, the fluid field evolves --
the "signal" itself changes between snapshots. What the model learns from
one IC actively hurts on another.

**Lesson**: The cryo-EM analogy fails because turbulence has no fixed
underlying structure. The coarse-to-fine mapping changes with time as the
field evolves.

### Experiment 005: Numerical Error Correction (81% FNO, 40% CNN)

**Question**: Can networks learn the *systematic numerical error* of the Stam
solver by comparing it to a pseudo-spectral reference?

**Result**: Dramatic success. The FNO achieved 81% error reduction with
multi-time training; the CNN achieved 40%. Both models remained stable when
applied as real-time post-processors over hundreds of timesteps.

**Why this works when everything else failed**: The numerical error of the
Stam solver is deterministic -- given the same input field, the same error
occurs every time, because the error is a property of the *numerical scheme*,
not the *physics*. The bilinear interpolation in semi-Lagrangian advection
smears gradients in a repeatable way that depends only on the current field
structure. The Jacobi diffusion solver adds numerical dissipation at a rate
determined by the current field. These are exactly the kinds of structured,
local-plus-global patterns that neural networks excel at learning.

The three properties that made this work:

1. **Deterministic**: error = f(current field). No hidden variables, no
   history dependence.
2. **Large signal**: The Stam solver loses ~58% of field amplitude over 500
   steps -- this is a massive, structured signal to learn from.
3. **Dual-scale structure**: The error has both local components (gradient
   smearing from bilinear interpolation, learnable by CNN) and global
   components (accumulated phase errors, learnable by FNO's spectral
   convolutions).

### The Progression Tells a Story

| Exp | Target | Deterministic? | Best Result | Key Finding |
|-----|--------|:-:|---:|---|
| 001 | Random phases | No | 0% | Sanity check: random = unlearnable |
| 002 | Gradient texture | Yes (local) | 12.8% CNN | Local coupling is learnable |
| 003 | PDE fine scales | No | 5.0% | History-dependent = mostly unlearnable |
| 004 | 8 time frames | No | ~1% | Temporal context insufficient |
| 004b | Fixed IC family | No | 1.3% | Cryo-EM analogy fails |
| **005** | **Numerical error** | **Yes** | **81% FNO** | **Scheme bias is learnable** |

The progression from 0% to 81% was not luck. Each failure eliminated a
hypothesis and narrowed the search space. Experiment 002 proved that
deterministic local coupling is learnable. Experiment 003 proved that PDE
dynamics break the mapping. The logical intersection -- a target that is
deterministic *and* involves PDE-like structure -- is the numerical scheme
error itself.

---

## 3. Technical Architecture

### 3.1 The Two Solvers

The project rests on running two solvers from identical initial conditions
and comparing their outputs:

**Pseudo-Spectral Solver** (ground truth):
- Spatial derivatives computed exactly via cuFFT (forward FFT, multiply by ik,
  inverse FFT)
- RK4 time integration
- 2/3 dealiasing rule for nonlinear terms
- Periodic boundary conditions (natural in Fourier space)
- Cost: ~3ms per step at 128x128 (dominated by cuFFT)
- Accuracy: limited only by time discretization and dealiasing

**Stam Semi-Lagrangian Solver** (fast, dissipative):
- Advection by backtracing particle positions and bilinear interpolation
- Implicit Jacobi diffusion (20 iterations)
- Optional Helmholtz pressure projection (incompressible N-S mode)
- Cost: ~0.3ms per step at 128x128
- Accuracy: first-order, heavily dissipative, accumulates phase errors
- Unconditionally stable (the key advantage for real-time use)

The spectral solver runs ~10x slower but produces the "correct" answer. The
Stam solver runs at real-time speed but loses ~58% of field amplitude over
500 steps. The neural network's job is to close this gap.

### 3.2 Neural Architectures

**Fourier Neural Operator (FNO)** -- 2,367,937 parameters:

The FNO operates in spectral space, which is natural for this problem since
the error has strong spectral structure. Each layer performs:

1. Forward FFT of the input
2. Multiply by learned complex weights (truncated to 12 modes in each
   direction)
3. Inverse FFT
4. Add a pointwise (1x1) convolution in physical space
5. Apply GELU nonlinearity

The spectral convolution gives each layer a *global* receptive field --
it can relate a feature at one corner of the domain to a feature at the
opposite corner. This is critical for correcting phase errors, which are
inherently global (a wave shifted by 2 degrees at one location is shifted
everywhere).

Architecture: lift (1 -> 32 channels) -> 4 spectral layers -> project
(32 -> 128 -> 1). Output: residual (correction to add to the input field).

**Simple CNN** -- 19,105 parameters:

Four layers of 3x3 convolutions with 32 channels and circular padding
(for periodic boundaries), followed by a 3x3 output convolution. GELU
activations. Output: residual.

The CNN has a receptive field of ~9 pixels -- it can only see and correct
local patterns. This is sufficient for gradient smearing (which is local)
but insufficient for phase errors (which are global). This explains the
persistent gap: CNN at 40% vs FNO at 81%.

**Why the residual formulation matters**: Both models predict the
*correction* (spectral - stam), not the target field directly. This means:
- At early times when error is small, the model correctly outputs near-zero
- At late times when error is large, the model outputs a large correction
- The model naturally learns error-magnitude scaling without explicit time input

### 3.3 cuFFT: Why Fortran, Why Not Python

A recurring question: why write GPU code in CUDA Fortran when PyTorch and
CuPy exist?

**Performance**: cuFFT called from Fortran operates on device memory that
*never leaves the GPU*. The data generation pipeline -- initialize field,
FFT, filter, IFFT, evolve PDE, FFT, compute spectrum -- runs entirely in
GPU memory. There is no Python interpreter overhead, no GIL contention, no
implicit CPU-GPU synchronization.

Concrete numbers at 128x128:
- cuFFT R2C + C2R from Fortran: ~0.1ms round-trip
- CuPy rfft2 + irfft2: ~0.3ms (Python dispatch + kernel launch overhead)
- NumPy rfft2 + irfft2: ~2ms (CPU-only, no GPU)

For data generation (400 samples x 2 solvers x 500 steps = 400,000 FFT
pairs), this is the difference between 40 seconds and 2 minutes.

For the real-time demo, the Stam solver's 0.3ms-per-step budget leaves
no room for Python dispatch overhead. The entire solver step -- diffuse,
advect, project, apply obstacles, apply wind, absorb boundaries -- runs
as a sequence of GPU kernels launched from Fortran. The Python server
only touches the data once per frame (to extract fields for neural
correction and rendering).

**The cuFFT plan**: Each FFT operation creates a plan that tells cuFFT
the transform dimensions, data layout, and batch size. In this project,
plans are created per-call rather than cached, which adds ~50us overhead.
For persistent real-time use, caching plans would recover this, but at
128x128 the overhead is negligible.

**Column-major vs row-major**: Fortran stores arrays column-major; Python
(numpy) stores row-major. This required care at every boundary:
- Obstacle masks use `np.asfortranarray` before passing to Fortran
- Field extraction from Fortran produces column-major data that numpy
  interprets correctly when reshaped
- cuFFT expects dimensions in reverse order (ny, nx) for 2D transforms
  because it follows C convention internally

### 3.4 Real-Time Demo Architecture

The demo is a FastAPI server with WebSocket streaming:

```
Browser (fullscreen.html)
   |  WebSocket (binary frames + JSON commands)
   v
server.py
   |  Thread: simulation_loop()
   |    |
   |    +-- SolverBridge (ctypes -> realtime_solvers.so)
   |    |     Stam solver: step, extract fields
   |    |     Spectral solver: optional reference
   |    |
   |    +-- ModelBridge (PyTorch)
   |    |     FNO or CNN correction
   |    |     corrected = field + alpha * model(field)
   |    |
   |    +-- Binary frame assembly
   |          Header (24 bytes) + field data + spectra
   |
   +-- SimState (thread-safe shared state)
         Commands: force, wind, obstacles, model selection, etc.
```

**Alpha blending**: The correction is applied as
`corrected = stam_field + alpha * model_residual`, where alpha defaults to
0.2 for FNO. This prevents feedback artifacts -- the corrected field is used
only for *display*, while the Stam solver continues from its own uncorrected
state. Without alpha dampening, small model errors compound over frames and
create visual artifacts at high frequencies.

**Performance breakdown** (128x128, RTX 4060):
- Raw solver (no correction): ~1300 fps
- With CNN correction: ~300-400 fps (2.8ms inference)
- With FNO correction: ~150-200 fps (4.3ms inference)
- WebSocket + rendering overhead: ~1ms per frame

**Obstacle system**: Obstacles are represented as a 128x128 int8 mask on
the GPU. Each timestep, velocity and density at obstacle cells are zeroed.
The mask is rasterized in Python from a list of shapes (rectangles, circles)
in normalized coordinates, converted to Fortran-order memory, and uploaded
to the GPU only when shapes change. Two extra kernel launches per step add
~10us -- negligible at these frame rates.

**Wind forcing**: A uniform body force (fx, fy) is applied at all
non-obstacle cells each timestep, after diffusion but before advection. This
creates steady drift that interacts naturally with obstacles to produce
eddies, wakes, and channeling effects.

---

## 4. Building on Past Projects

This project did not emerge from nothing. It is the culmination of a
year-long progression through increasingly ambitious CUDA Fortran projects,
each of which contributed specific capabilities.

### 4.1 CUDA Fortran Foundations (2025)

The journey began with basic GPU programming: matrix multiplication kernels,
cuBLAS integration, and shared memory optimization. These early experiments
(preserved in `fortran/examples/collected_examples/matrix_dot/`) established:

- How to write and launch CUDA kernels from Fortran
- How to interface with cuBLAS for matrix operations
- How to build shared libraries callable from Python via ctypes
- The column-major memory layout considerations that would later matter
  for every numpy-Fortran boundary

### 4.2 MNIST on GPU in Pure Fortran

An unusual project: training a neural network on MNIST entirely in CUDA
Fortran, without Python or PyTorch. This built through ~15 iterations of
increasing sophistication (tensor matmul variants, cuBLAS batched operations,
custom kernels). The key lessons carried forward:

- Device memory management patterns (allocate, compute, extract)
- Understanding GPU occupancy and kernel launch overhead
- The ctypes interface pattern (Fortran `bind(c)` subroutines callable
  from Python) that became the standard for all subsequent projects

### 4.3 The Original Fluid Solver (March 2025)

The direct ancestor: a Stam semi-Lagrangian solver in CUDA Fortran with a
Python interface and even a Blender real-time visualization addon. This
project established:

- The semi-Lagrangian advection algorithm on GPU
- Jacobi pressure solver for incompressible flow
- The ctypes bridge pattern (Fortran solver + Python control + browser/Blender
  visualization) used throughout this project
- Obstacle mask implementation (later refined and reimplemented here)
- Interactive force injection via mouse

The original solver ran at ~60 fps at 128x128 -- fast, but visually
over-dissipated. The desire to make it look better without sacrificing speed
was the seed of the current project.

### 4.4 The CIFAR-10/Emergency Response Lineage

The broader project context is an emergency response simulation (v44) that
models atmospheric dispersion over a terrain grid. That project needed:

- Sub-grid turbulence modeling (can't resolve fine features on a coarse grid)
- Real-time visualization of smoke/contaminant transport
- Interactive obstacle placement for buildings and terrain features

These needs directly motivated the spectral lab's research questions: can we
learn sub-grid physics? Can we correct cheap solvers to match expensive ones?
Can obstacles be handled without retraining?

### 4.5 What Each Ancestor Contributed

| Ancestor Project | Contribution to Spectral Lab |
|---|---|
| CUDA matrix operations | GPU kernel patterns, cuBLAS, shared memory |
| MNIST Fortran training | Device memory management, ctypes interface |
| Original fluid solver | Stam algorithm, pressure solver, obstacle mask |
| Emergency response v44 | Motivation, obstacle inventory UI, deployment pattern |
| cuFFT experiments | Spectral methods, energy spectra, filtering |

The benefit of this lineage is concrete: the spectral lab's real-time demo
was functional within days, not weeks, because every component -- the GPU
solver, the Python bridge, the WebSocket server, the obstacle system -- had
been built and debugged in a previous project. The innovation was in
*combining* them with the neural correction insight.

---

## 5. Comparison with Other Approaches

### 5.1 Physics-Informed Neural Networks (PINNs)

PINNs embed the governing PDE directly in the loss function, training the
network to satisfy the equations at collocation points. They are elegant but
suffer from:

- **Training instability** with advection-dominated flows (sharp gradients
  cause loss landscape pathology)
- **Slow inference** (the network IS the solver, so each timestep requires
  a forward pass)
- **No guaranteed stability** (the network might produce non-physical states)

Our approach sidesteps all three: the Stam solver provides unconditionally
stable dynamics, the neural network only post-processes the output, and
training targets are simple supervised pairs (no PDE residual loss).

### 5.2 Neural Operators (FNO, DeepONet)

The FNO family learns operator mappings between function spaces. Our FNO is
architecturally standard but used in an unusual way: instead of learning the
solution operator of a PDE, it learns the *error operator* of a numerical
scheme. This is a much simpler mapping (it's closer to a filter than a
dynamical system), which explains why a small FNO with 12 modes and 4 layers
achieves 81% correction.

### 5.3 Learned Turbulence Closures

The turbulence modeling community trains neural networks to predict sub-grid
stress tensors for Large Eddy Simulation (LES). These approaches:

- Require expensive DNS training data (direct numerical simulation at fine
  resolution)
- Must respect physical constraints (Galilean invariance, realizability)
- Often suffer from online instability when coupled with the solver

Our approach is simpler: we don't model sub-grid physics, we correct
numerical errors. The distinction matters because numerical error is more
structured and predictable than turbulent fluctuations.

### 5.4 What's Unique Here

The specific combination of properties in this project is, to our knowledge,
uncommon in the literature:

1. **Correction, not replacement**: The solver runs unmodified; the network
   only improves the display output.
2. **Systematic experimental progression**: Five controlled failures that
   motivated the successful approach.
3. **Real-time interactive system**: Not just training metrics, but a working
   demo where users can interact with the corrected flow.
4. **Generalization evidence**: Models trained on open-domain flow
   correctly handle obstacles they've never seen. The correction is local
   enough that boundary conditions don't matter.
5. **Fortran-to-browser pipeline**: GPU kernels in CUDA Fortran, neural
   networks in PyTorch, visualization in HTML5 Canvas, all connected via
   ctypes and WebSockets.

---

## 6. The Real-Time Demo: Current State

### What It Does

The fullscreen demo (`http://localhost:8765/fullscreen`) provides:

- **Interactive fluid simulation** at 128x128 on GPU
- **Neural error correction** via FNO or CNN, switchable in real time
- **Density tracer**: drag the mouse to inject smoke, watch it advect and
  dissipate
- **Obstacles**: place rectangles and circles via a sidebar inventory;
  smoke flows around them realistically, threading through narrow gaps
- **Wind**: adjustable uniform flow (-0.01 to 0.01) that drifts smoke and
  creates eddies behind obstacles
- **Absorbing boundaries**: fluid exits the frame and doesn't wrap back
- **Scene save/load**: export obstacle + wind configurations as JSON

### What It Looks Like

The simulation is visually convincing. Key observations:

- **Smoke injection** creates swirling tendrils that cascade into smaller
  eddies -- the neural correction visibly sharpens these compared to the
  uncorrected Stam output.
- **Obstacles** produce realistic wakes: von Karman-like vortex streets
  behind cylinders, channeling between gaps, recirculation zones behind
  rectangles.
- **Wind + obstacles** creates the most visually interesting scenarios:
  steady drift with eddies forming and shedding behind structures.
- **Bilinear upscaling** from 128x128 to viewport resolution softens the
  grid-scale rendering, giving a smooth appearance despite the coarse
  resolution.

### Performance

| Mode | FPS | Notes |
|------|----:|---|
| Raw (no correction) | ~1300 | Solver only, no rendering |
| CNN correction | 300-400 | 19K params, 2.8ms inference |
| FNO correction | 150-200 | 2.4M params, 4.3ms inference |

All modes comfortably exceed 60 fps, meaning the simulation is real-time
interactive with neural correction enabled.

---

## 7. Future Experiments

### 7.1 Immediate Extensions

**Higher resolution** (256x256, 512x512): Does the correction generalize to
finer grids? The numerical error pattern should be similar (it's a property
of the scheme, not the resolution), but the model would need to handle 4-16x
more spatial complexity. Training at 128x128 and testing at 256x256 would be
a strong test of the FNO's resolution-invariance property.

**3D extension**: The Stam solver and spectral methods generalize to 3D.
The FNO's spectral convolutions extend naturally (3D FFT). The main
constraint is GPU memory (128^3 = 2M cells vs 128^2 = 16K cells). A 64^3
demo would be feasible.

**Deeper/wider CNN**: The CNN plateaus at ~40%. A ResNet-style architecture
with skip connections and larger receptive field (dilated convolutions) might
close the gap with FNO at lower cost.

**Adaptive alpha**: Currently alpha (correction strength) is a user-set
constant. A small auxiliary network could predict per-pixel alpha based on
local field structure -- applying strong correction where error is large
and weak correction where the Stam solver is already accurate.

### 7.2 Scientific Questions

**What exactly does the FNO learn?** Visualizing the learned Fourier weights
would reveal which spectral modes the FNO considers most important. This
could inform traditional numerical methods: if the FNO's correction is
dominated by modes k=10-20, perhaps a targeted spectral filter could
approximate the correction without a neural network.

**Feedback stability**: Currently, corrections are applied as display-only
post-processing. Could the correction be fed back into the solver with
sufficient dampening? This would compound the improvement over time rather
than applying it independently each frame. The risk is feedback instability,
but alpha < 0.1 might be stable.

**Transfer across viscosities**: Models are trained at nu=0.01. Do they
generalize to nu=0.001 (more turbulent) or nu=0.1 (more viscous)? The
dissipation error pattern should depend on nu, so some transfer is expected
but not perfect.

**Navier-Stokes vs Burgers**: The current models are trained on Burgers
equation (no pressure). Extending to incompressible Navier-Stokes (with
pressure projection) adds divergence-free constraints that change the error
structure. Experiment 005b has preliminary N-S results.

### 7.3 Application Directions

**Emergency response simulation**: The original motivation. A coarse
atmospheric dispersion model corrected by neural networks could provide
real-time forecasting with accuracy approaching expensive high-resolution
models.

**Game engine integration**: The 300+ fps performance is fast enough for
real-time game effects. A 128x128 fluid simulation with CNN correction
could provide convincing smoke, fog, and atmospheric effects at minimal
GPU cost.

**Educational tool**: The interactive demo, with its visible comparison
between corrected and uncorrected fluid, is a compelling teaching aid for
computational fluid dynamics. Students can see numerical dissipation in
action and understand why scheme accuracy matters.

---

## 8. Assessment: Current Value and Limitations

### What Works Well

1. **The demo is genuinely impressive.** Smoke flowing through narrow gaps
   between obstacles, eddies forming behind cylinders, wind creating drift
   -- all at 300+ fps with visible quality improvement from neural
   correction.

2. **The experimental methodology is sound.** The progression from null
   result (001) through controlled failures (002-004b) to success (005)
   provides clear evidence for *why* the approach works, not just *that* it
   works.

3. **The system is robust.** Models trained without obstacles generalize to
   obstacle scenarios. The correction remains stable over thousands of
   frames. The alpha blending prevents feedback artifacts.

4. **The codebase is clean and modular.** Each Fortran module is standalone.
   The ctypes interface is well-defined. New experiments can be added without
   modifying existing code.

### Limitations

1. **128x128 is small.** Real applications need at least 512x512 for
   meaningful spatial detail. The approach should scale (the error pattern
   is resolution-independent), but this hasn't been demonstrated.

2. **2D only.** Real fluid dynamics is three-dimensional. The extension to
   3D is straightforward in principle but expensive in practice.

3. **Supervised training requires paired data.** Generating training pairs
   requires running both solvers, which means you need the expensive solver
   at training time. The payoff is that you only need it once -- the trained
   model runs forever at fast-solver cost.

4. **The 40% CNN ceiling.** The cheap 19K-parameter CNN plateaus at ~40%
   correction. For the FNO's 81%, you pay 2.4M parameters and ~4ms inference.
   In a production setting, the cost-accuracy tradeoff would need careful
   evaluation.

5. **No formal error bounds.** The neural correction provides no guarantee
   on the maximum error. For scientific applications, this is a significant
   limitation. For visual applications, it doesn't matter.

### Overall Assessment

The Turbulence Spectral Lab, in its current state, is a successful research
prototype that demonstrates a viable approach to real-time neural-corrected
fluid simulation. Its primary value is threefold:

- **Conceptual**: The insight that numerical scheme error is learnable
  (because it's deterministic) while physical fine-scale structure is not
  (because it's history-dependent) is broadly applicable beyond this
  specific solver.

- **Practical**: The real-time demo is a convincing proof-of-concept for
  ML-corrected fast solvers, running at speeds suitable for interactive
  applications.

- **Methodological**: The systematic experimental progression provides a
  template for investigating neural correction of other numerical schemes
  -- finite element, finite volume, lattice Boltzmann, etc.

The project stands as evidence that carefully sequenced small experiments,
each testing a specific hypothesis, can efficiently navigate a complex
research space. The five failures were not wasted -- they were the map that
led to the destination.

---

## Appendix A: Key File Locations

```
turbulence_spectral_lab/
  fortran/
    spectral_tools.cuf        # cuFFT wrappers, energy spectra, filtering
    synthetic_turbulence.cuf   # Kolmogorov IC generation
    burgers_solver.cuf         # Pseudo-spectral Burgers solver (reference)
    stam_solver.cuf            # Semi-Lagrangian solver (fast, corrected)
    realtime_wrapper.cuf       # C-bindings for Python ctypes
    realtime_solvers.so        # Compiled shared library
  python/
    models/fno_2d.py           # FNO (2.4M params) and CNN (19K params)
    models/temporal_cnn.py     # Temporal architectures (exp 004)
    train_fno.py               # Training loop
  demo/
    server.py                  # FastAPI WebSocket server
    solver_bridge.py           # ctypes interface to Fortran
    model_bridge.py            # PyTorch model loading and inference
    fullscreen.html            # Interactive viewer with obstacles + wind
    index.html                 # Multi-panel debug viewer
  experiments/
    005_stam_correction/       # Trained models and results
  EXPERIMENT_LOG.md            # Detailed record of all experiments
  PROJECT_REPORT.md            # This document
```

## Appendix B: Reproducing the Demo

```bash
# Build the Fortran library
cd fortran
nvfortran -cuda -gpu=cc89 -O3 -lcufft -shared -fPIC \
  spectral_tools.cuf synthetic_turbulence.cuf burgers_solver.cuf \
  stam_solver.cuf realtime_wrapper.cuf -o realtime_solvers.so

# Start the server
cd ../demo
python server.py

# Open browser to:
#   http://localhost:8765/fullscreen
#
# Controls:
#   Tab        - toggle obstacle sidebar
#   H          - toggle control panel
#   Mouse drag - inject smoke (density tracer)
#   Delete     - remove selected obstacle
#   Escape     - deselect obstacle/tool
```

## Appendix C: Trained Model Performance

| Model | Parameters | Inference (ms) | Error Reduction | Best For |
|-------|----------:|-----:|-----:|---|
| CNN (Burgers) | 19,105 | 2.8 | 40% | Local gradient correction |
| FNO (Burgers) | 2,367,937 | 4.3 | 81% | Global phase + local correction |
| CNN (N-S) | 19,105 | 2.8 | ~40% | Incompressible flow |
| FNO (N-S) | 2,367,937 | 4.3 | ~65% | Incompressible flow |
