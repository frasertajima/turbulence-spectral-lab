# Makefile for Turbulence Spectral Lab - Fortran/CUDA tools
#
# Usage:
#   make experiment_001   # Build first experiment
#   make all              # Build everything
#   make clean            # Remove binaries
#   make run_001          # Build and run experiment 001

# Compiler
FC = nvfortran

# GPU architectures (Ada Lovelace, Ampere, etc.)
GPU_ARCH = -gpu=cc80,cc86,cc89,cc90

# Flags
FFLAGS = -cuda $(GPU_ARCH) -O3 -fast
LIBS = -lcufft

# Module objects (order matters - dependencies first)
MODULES = spectral_tools.o synthetic_turbulence.o

# Targets
.PHONY: all clean run_001

all: experiment_001

# Module compilation
spectral_tools.o: spectral_tools.cuf
	$(FC) $(FFLAGS) -c $< -o $@

synthetic_turbulence.o: synthetic_turbulence.cuf spectral_tools.o
	$(FC) $(FFLAGS) -c $< -o $@

# Experiments
experiment_001: $(MODULES) experiment_001.cuf
	$(FC) $(FFLAGS) $(MODULES) experiment_001.cuf -o $@ $(LIBS)

# Convenience targets
run_001: experiment_001
	./experiment_001

clean:
	rm -f *.o *.mod experiment_001
	rm -f ../experiments/001_synthetic_2d/spectra/*.dat
	rm -f ../experiments/001_synthetic_2d/fields/*.bin
