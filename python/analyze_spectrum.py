#!/usr/bin/env python3
"""analyze_spectrum.py - Load and plot energy spectra from Fortran output.

Usage:
    python analyze_spectrum.py                          # Plot experiment 001 spectra
    python analyze_spectrum.py --dir ../experiments/001_synthetic_2d/
    python analyze_spectrum.py --compare spectrum1.dat spectrum2.dat
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct


def load_spectrum(filepath):
    """Load energy spectrum from Fortran .dat file."""
    k_vals = []
    E_vals = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) >= 2:
                k_vals.append(float(parts[0]))
                E_vals.append(float(parts[1]))
    return np.array(k_vals), np.array(E_vals)


def load_field_2d(filepath):
    """Load 2D binary field written by Fortran write_field_2d."""
    with open(filepath, 'rb') as f:
        nx = struct.unpack('i', f.read(4))[0]
        ny = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32)
        return data.reshape((ny, nx))  # Fortran column-major -> numpy row-major


def plot_spectra(spectra_dict, title="Energy Spectrum", output_file=None,
                 reference_slopes=None):
    """Plot multiple energy spectra on log-log axes.

    Args:
        spectra_dict: {label: (k, E)} pairs
        reference_slopes: list of (slope, label) to draw reference lines
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(spectra_dict), 3)))

    for idx, (label, (k, E)) in enumerate(spectra_dict.items()):
        mask = (k > 0) & (E > 0)
        ax.loglog(k[mask], E[mask], '-o', label=label, color=colors[idx],
                  markersize=3, linewidth=1.5)

    # Reference slopes
    if reference_slopes is None:
        reference_slopes = [(-5/3, "k^{-5/3}"), (-3, "k^{-3}")]

    for slope, slope_label in reference_slopes:
        k_ref = np.logspace(0.3, 1.8, 50)
        # Anchor to middle of plot
        E_ref = k_ref ** slope
        E_ref *= 1e-2 / E_ref[len(E_ref)//2]  # normalize
        ax.loglog(k_ref, E_ref, '--', color='gray', alpha=0.5, linewidth=1)
        ax.text(k_ref[-1]*1.1, E_ref[-1], f"${slope_label}$",
                fontsize=10, color='gray')

    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('E(k)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {output_file}")
    plt.show()


def plot_fields(fine, coarse, residual=None, output_file=None):
    """Side-by-side visualization of fine, coarse, and residual fields."""
    n_plots = 3 if residual is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    vmax = max(abs(fine.min()), abs(fine.max()))

    axes[0].imshow(fine, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Fine (original)')
    axes[0].axis('off')

    axes[1].imshow(coarse, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Coarse (filtered)')
    axes[1].axis('off')

    if residual is not None:
        vmax_r = max(abs(residual.min()), abs(residual.max()))
        axes[2].imshow(residual, cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r)
        axes[2].set_title('Residual (fine - coarse)')
        axes[2].axis('off')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {output_file}")
    plt.show()


def compute_spectral_error(k_true, E_true, k_pred, E_pred):
    """Compute spectral reconstruction error metrics.

    Returns dict with:
        - mse_log: MSE in log-space (emphasizes relative error)
        - max_k_recovered: highest wavenumber where error < 50%
        - energy_fraction: fraction of total energy recovered
    """
    # Interpolate pred onto true k values
    from scipy.interpolate import interp1d

    mask_true = (k_true > 0) & (E_true > 0)
    mask_pred = (k_pred > 0) & (E_pred > 0)

    if mask_pred.sum() < 2:
        return {'mse_log': float('inf'), 'max_k_recovered': 0, 'energy_fraction': 0}

    interp = interp1d(k_pred[mask_pred], np.log10(E_pred[mask_pred]),
                      bounds_error=False, fill_value=-20)

    log_E_pred_interp = interp(k_true[mask_true])
    log_E_true = np.log10(E_true[mask_true])

    mse_log = np.mean((log_E_true - log_E_pred_interp)**2)

    # Max wavenumber where relative error < 50%
    rel_error = np.abs(10**log_E_pred_interp - E_true[mask_true]) / E_true[mask_true]
    good = rel_error < 0.5
    if good.any():
        max_k = k_true[mask_true][good].max()
    else:
        max_k = 0

    energy_fraction = np.sum(10**log_E_pred_interp) / np.sum(E_true[mask_true])

    return {
        'mse_log': mse_log,
        'max_k_recovered': max_k,
        'energy_fraction': energy_fraction
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze turbulence spectra')
    parser.add_argument('--dir', type=str,
                        default='../experiments/001_synthetic_2d/',
                        help='Experiment directory')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare specific spectrum files')
    parser.add_argument('--fields', action='store_true',
                        help='Also plot field visualizations')
    parser.add_argument('--no-show', action='store_true',
                        help='Save plots but do not display')
    args = parser.parse_args()

    exp_dir = Path(args.dir)
    spectra_dir = exp_dir / 'spectra'
    fields_dir = exp_dir / 'fields'

    if args.compare:
        # Compare specific files
        spectra = {}
        for f in args.compare:
            k, E = load_spectrum(f)
            spectra[Path(f).stem] = (k, E)
        plot_spectra(spectra, title="Spectrum Comparison")
        return

    # Default: analyze experiment 001
    if not spectra_dir.exists():
        print(f"No spectra directory found at {spectra_dir}")
        print("Run experiment_001 first to generate data.")
        return

    # Load and plot main spectra
    spectra = {}
    for dat_file in sorted(spectra_dir.glob('spectrum_*.dat')):
        label = dat_file.stem.replace('spectrum_', '')
        k, E = load_spectrum(dat_file)
        spectra[label] = (k, E)

    if spectra:
        # Separate filtering analysis from exponent comparison
        filter_spectra = {k: v for k, v in spectra.items()
                         if k in ['fine', 'coarse', 'residual']}
        exponent_spectra = {k: v for k, v in spectra.items()
                          if k not in ['fine', 'coarse', 'residual']}

        if filter_spectra:
            plot_spectra(filter_spectra,
                        title="Spectral Filtering: Fine vs Coarse vs Residual",
                        output_file=str(exp_dir / 'filter_analysis.png'))

        if exponent_spectra:
            plot_spectra(exponent_spectra,
                        title="Different Spectral Exponents",
                        output_file=str(exp_dir / 'exponent_comparison.png'))

    # Field visualization
    if args.fields and fields_dir.exists():
        fine_file = fields_dir / 'fine_sample_000.bin'
        coarse_file = fields_dir / 'coarse_sample_000.bin'
        residual_file = fields_dir / 'residual_sample_000.bin'

        if fine_file.exists() and coarse_file.exists():
            fine = load_field_2d(fine_file)
            coarse = load_field_2d(coarse_file)
            residual = load_field_2d(residual_file) if residual_file.exists() else None
            plot_fields(fine, coarse, residual,
                       output_file=str(exp_dir / 'field_visualization.png'))

    # Print quantitative summary
    if 'fine' in spectra and 'coarse' in spectra:
        k_fine, E_fine = spectra['fine']
        k_coarse, E_coarse = spectra['coarse']
        metrics = compute_spectral_error(k_fine, E_fine, k_coarse, E_coarse)
        print("\n  Filtering metrics:")
        print(f"    Log-space MSE:         {metrics['mse_log']:.4f}")
        print(f"    Max k recovered (<50%): {metrics['max_k_recovered']:.1f}")
        print(f"    Energy fraction:        {metrics['energy_fraction']:.4f}")


if __name__ == '__main__':
    main()
