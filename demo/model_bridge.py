"""model_bridge.py - Load and run FNO/CNN correction models."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add python dir for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from models.fno_2d import FNO2d, SimpleCNN

_PROJECT = Path(__file__).parent.parent

# Model registry: name -> (class, kwargs, relative path to best_model.pt)
MODELS = {
    "fno_burgers": {
        "cls": FNO2d,
        "kwargs": {"modes1": 12, "modes2": 12, "width": 32, "n_layers": 4},
        "path": "experiments/005_stam_correction/runs/fno_20260214_210901/best_model.pt",
    },
    "cnn_burgers": {
        "cls": SimpleCNN,
        "kwargs": {"n_channels": 32, "n_layers": 4},
        "path": "experiments/005_stam_correction/runs/cnn_20260214_205347/best_model.pt",
    },
    "fno_ns": {
        "cls": FNO2d,
        "kwargs": {"modes1": 12, "modes2": 12, "width": 32, "n_layers": 4},
        "path": "experiments/005b_ns_correction/runs/fno_20260214_215255/best_model.pt",
    },
    "cnn_ns": {
        "cls": SimpleCNN,
        "kwargs": {"n_channels": 32, "n_layers": 4},
        "path": "experiments/005b_ns_correction/runs/cnn_20260214_215254/best_model.pt",
    },
}


class ModelBridge:
    """Manages neural network correction models."""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else _PROJECT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache: dict[str, torch.nn.Module] = {}
        self.active_model = None
        self.active_name = "none"
        self.alpha = 1.0  # correction strength: 0.0 = no correction, 1.0 = full
        self.last_inference_ms = 0.0

    def load_model(self, name: str):
        """Load a model by name. Caches after first load."""
        if name == "none":
            self.active_model = None
            self.active_name = "none"
            return

        if name not in self._cache:
            cfg = MODELS[name]
            model = cfg["cls"](**cfg["kwargs"])
            path = self.project_root / cfg["path"]
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval().to(self.device)
            self._cache[name] = model
            print(
                f"  Loaded model: {name} ({sum(p.numel() for p in model.parameters()):,} params)"
            )

        self.active_model = self._cache[name]
        self.active_name = name

    def correct(self, field: np.ndarray) -> np.ndarray:
        """Apply correction: corrected = field + model(field).

        Args:
            field: (nx, ny) float32 numpy array
        Returns:
            corrected: (nx, ny) float32 numpy array
        """
        if self.active_model is None:
            self.last_inference_ms = 0.0
            return field

        t0 = time.perf_counter()
        with torch.no_grad():
            tensor = torch.from_numpy(field).unsqueeze(0).to(self.device)
            residual = self.active_model(tensor)
            corrected = tensor + self.alpha * residual
            result = corrected.squeeze(0).cpu().numpy()
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        return result

    @staticmethod
    def model_name_for(ns_mode: bool, model_type: str) -> str:
        """Get registry key for current physics mode + model type."""
        if model_type == "none":
            return "none"
        suffix = "ns" if ns_mode else "burgers"
        return f"{model_type}_{suffix}"
