"""solver_bridge.py - ctypes wrapper for realtime_solvers.so"""

import ctypes
from pathlib import Path

import numpy as np

_F = ctypes.c_float
_I = ctypes.c_int
_FP = ctypes.POINTER(ctypes.c_float)


class SolverBridge:
    """Wrapper around the CUDA Fortran realtime_solvers.so library."""

    def __init__(self, nx=128, ny=128, nu=0.01, lib_path=None):
        if lib_path is None:
            lib_path = Path(__file__).parent.parent / "fortran" / "realtime_solvers.so"
        self.nx = nx
        self.ny = ny
        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_argtypes()
        self.lib.rt_init(_I(nx), _I(ny), _F(nu))
        self._ns_mode = False

    def _setup_argtypes(self):
        self.lib.rt_init.argtypes = [_I, _I, _F]
        self.lib.rt_init.restype = None

        self.lib.rt_set_ic_from_seed.argtypes = [_I]
        self.lib.rt_set_ic_from_seed.restype = None

        self.lib.rt_set_ic_from_seed_ns.argtypes = [_I]
        self.lib.rt_set_ic_from_seed_ns.restype = None

        self.lib.rt_set_ns_mode.argtypes = [_I]
        self.lib.rt_set_ns_mode.restype = None

        for name in ("rt_step_spectral", "rt_step_stam"):
            fn = getattr(self.lib, name)
            fn.argtypes = [_F, _I]
            fn.restype = None

        for name in (
            "rt_get_spectral_u",
            "rt_get_stam_u",
            "rt_get_spectral_v",
            "rt_get_stam_v",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [_FP]
            fn.restype = None

        for name in ("rt_set_stam_u", "rt_set_stam_v"):
            fn = getattr(self.lib, name)
            fn.argtypes = [_FP]
            fn.restype = None

        self.lib.rt_add_force.argtypes = [_F, _F, _F, _F, _F]
        self.lib.rt_add_force.restype = None

        self.lib.rt_enable_density.argtypes = [_F, _F]
        self.lib.rt_enable_density.restype = None

        self.lib.rt_set_zero_ic.argtypes = []
        self.lib.rt_set_zero_ic.restype = None

        self.lib.rt_get_stam_rho.argtypes = [_FP]
        self.lib.rt_get_stam_rho.restype = None

        self.lib.rt_add_density.argtypes = [_F, _F, _F, _F]
        self.lib.rt_add_density.restype = None

        self.lib.rt_set_walls.argtypes = [_I]
        self.lib.rt_set_walls.restype = None

        self.lib.rt_cleanup.argtypes = []
        self.lib.rt_cleanup.restype = None

    # --- Mode & IC ---

    def set_ns_mode(self, enabled: bool):
        self._ns_mode = enabled
        self.lib.rt_set_ns_mode(_I(1 if enabled else 0))

    def new_ic(self, seed: int):
        if self._ns_mode:
            self.lib.rt_set_ic_from_seed_ns(_I(seed))
        else:
            self.lib.rt_set_ic_from_seed(_I(seed))

    def zero_ic(self):
        self.lib.rt_set_zero_ic()

    def enable_density(self, diffusion: float = 0.0001, decay: float = 0.5):
        self.lib.rt_enable_density(_F(diffusion), _F(decay))

    def set_walls(self, enabled: bool):
        self.lib.rt_set_walls(_I(1 if enabled else 0))

    # --- Stepping ---

    def step(self, dt: float, n_steps: int, run_spectral: bool = True):
        self.lib.rt_step_stam(_F(dt), _I(n_steps))
        if run_spectral:
            self.lib.rt_step_spectral(_F(dt), _I(n_steps))

    # --- Field extraction ---

    def _get_field(self, fn_name) -> np.ndarray:
        buf = np.zeros((self.nx, self.ny), dtype=np.float32)
        ptr = buf.ctypes.data_as(_FP)
        getattr(self.lib, fn_name)(ptr)
        return buf

    def get_stam_u(self):
        return self._get_field("rt_get_stam_u")

    def get_stam_v(self):
        return self._get_field("rt_get_stam_v")

    def get_stam_rho(self):
        return self._get_field("rt_get_stam_rho")

    def get_spectral_u(self):
        return self._get_field("rt_get_spectral_u")

    def get_spectral_v(self):
        return self._get_field("rt_get_spectral_v")

    # --- Density injection ---

    def add_density(
        self, cx: float, cy: float, amount: float = 1.0, radius: float = 5.0
    ):
        self.lib.rt_add_density(_F(cx), _F(cy), _F(amount), _F(radius))

    # --- Force injection ---

    def add_force(
        self, fx: float, fy: float, cx: float, cy: float, radius: float = 5.0
    ):
        self.lib.rt_add_force(_F(fx), _F(fy), _F(cx), _F(cy), _F(radius))

    # --- Cleanup ---

    def cleanup(self):
        self.lib.rt_cleanup()
