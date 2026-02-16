# Turbulence Spectral Lab

An experimental workbench for exploring spectral methods and neural operators
applied to turbulence super-resolution.

Videos:
- [Video 1](https://youtu.be/U0Njc96IzXU?si=XsNl9W7aDg3lmzya)
- [Video 2](https://youtu.be/VyYG-6TMB0k?si=rz2VlkK4MLedPcgI)
- [Video 3](https://youtu.be/g0YfFeLQcqc?si=VtEL0LoegzLm1s-W)

## Motivation

Our emergency response simulation (v44) uses a 2m-cell finite difference solver
that cannot resolve sub-grid features: narrow gaps, turbulent mixing, and fine-scale
transport. Previous attempts at a full PINN approach failed due to
advection-dominated sharp gradients.

This project takes a different angle: instead of learning the PDE directly, learn
to **recover high-frequency spectral content** from coarse fields. The approach:

1. **Spectral decomposition** — Understand what information coarse grids actually lose
2. **Fourier Neural Operators** — Learn mappings in spectral space (where turbulence
   energy cascades naturally live)
3. **Fast iteration** — GPU-accelerated Fortran tools (cuFFT) for rapid experiments

This is a research workbench, not a product. The value is in what we learn.

## Structure

```
turbulence_spectral_lab/
├── fortran/                    # GPU-accelerated tools (cuFFT)
│   ├── spectral_tools.cuf      # Core: FFT, filtering, spectrum computation
│   ├── synthetic_turbulence.cuf # Generate fields with known spectra
│   ├── experiment_001.cuf      # First experiment program
│   └── Makefile
├── python/
│   ├── models/
│   │   └── fno_2d.py           # Fourier Neural Operator + CNN baseline
│   ├── analyze_spectrum.py     # Spectral plotting and comparison
│   ├── train_fno.py            # Training loop
│   └── requirements.txt
├── experiments/
│   └── 001_synthetic_2d/       # Each experiment gets a numbered directory
│       ├── config.yaml         # What we're testing
│       ├── spectra/            # Spectrum data files
│       ├── fields/             # Binary field data
│       └── runs/               # Training runs with results
├── notebooks/                  # Exploration
├── data/
│   ├── synthetic/              # Generated test data
│   └── les/                    # Links to LES simulation data
├── EXPERIMENT_LOG.md           # Running record of what we tried
└── README.md
```

## Quick Start

### 1. Build Fortran tools

```bash
cd fortran
make experiment_001
```

### 2. Generate data and baseline spectra

```bash
./experiment_001
```

This generates 100 training pairs (coarse, fine) at 256x256 and computes
energy spectra. Takes seconds on GPU.

### 3. Analyze spectra

```bash
cd ../python
python analyze_spectrum.py --fields
```

### 4. Train FNO

```bash
python train_fno.py --model fno --epochs 100
python train_fno.py --model cnn --epochs 100   # baseline comparison
```

### 5. Compare results

Check `experiments/001_synthetic_2d/runs/` for training metrics and
reconstructed spectra.

## Requirements

- **Fortran**: nvfortran (NVIDIA HPC SDK) with CUDA support
- **Libraries**: cuFFT (included with CUDA toolkit)
- **Python**: PyTorch, NumPy, Matplotlib, SciPy
- **GPU**: NVIDIA GPU with compute capability >= 8.0

## Design Principles

1. **Tiny first** — Start with 2D 64x64, not 3D 256^3
2. **Fortran does heavy lifting** — Spectral transforms, data generation
3. **Python does learning** — Models kept deliberately simple
4. **Every experiment logged** — Config in, metrics out, compare across runs
5. **Baselines before complexity** — Know what simple approaches achieve
6. **Failure is data** — Document what doesn't work and why

## Modular Construction

Each component is independent:

- `spectral_tools.cuf` can be used standalone for any spectral analysis
- `synthetic_turbulence.cuf` generates test data without needing models
- `fno_2d.py` can be imported into any PyTorch project
- New experiments just add a new `.cuf` file and `experiments/NNN_*/` directory

## Connection to Emergency Response Simulation

If spectral super-resolution works, it could feed back into v44 as:

- **Sub-grid diffusion coefficients** learned from fine-scale turbulence
- **Post-processing upscaling** of coarse simulation output
- **Effective permeability** through gaps smaller than grid cells
