from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


@dataclass
class SpotiflowConfig:
    pretrained_model: str = "general"
    prob_thresh: float = 0.5
    min_distance: int = 2
    use_gpu: Optional[bool] = None
    subpix: bool = True
    peak_mode: str = "fast"
    normalizer: Optional[str] = "auto"


class SpotiflowBackend:
    def __init__(self, cfg: SpotiflowConfig):
        self.cfg = cfg
        self._model = None
        self._resolved_gpu = None

    @staticmethod
    def _auto_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_model(self):
        if self._model is not None:
            return self._model

        if self.cfg.use_gpu is None:
            use_gpu = self._auto_gpu()
        else:
            use_gpu = bool(self.cfg.use_gpu)

        self._resolved_gpu = use_gpu

        from spotiflow.model import Spotiflow

        self._model = Spotiflow.from_pretrained(self.cfg.pretrained_model)

        if use_gpu:
            self._model = self._model.cuda()

        return self._model

    def detect(self, img_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Detect spots in a 2D image.

        Parameters
        ----------
        img_2d : (H, W) ndarray

        Returns
        -------
        spots_rc : (N, 2) float32 — (row, col)
        scores   : (N,) float32 — per-spot probability/intensity
        meta     : dict with detection metadata
        """
        if img_2d.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape={img_2d.shape}")

        model = self._get_model()
        img_f = img_2d.astype(np.float32, copy=False)

        spots, details = model.predict(
            img_f,
            prob_thresh=self.cfg.prob_thresh,
            min_distance=self.cfg.min_distance,
            subpix=self.cfg.subpix,
            peak_mode=self.cfg.peak_mode,
            normalizer=self.cfg.normalizer,
        )

        # spots from Spotiflow: (N, 2) in (y, x) = (row, col)
        spots_rc = np.array(spots, dtype=np.float32)
        if spots_rc.ndim == 1:
            spots_rc = spots_rc.reshape(-1, 2)

        # Per-spot scores (probability / intensity)
        if hasattr(details, "intens") and details.intens is not None:
            scores = np.array(details.intens, dtype=np.float32).ravel()
        elif hasattr(details, "prob") and details.prob is not None:
            scores = np.array(details.prob, dtype=np.float32).ravel()
        else:
            scores = np.ones(len(spots_rc), dtype=np.float32)

        meta = {
            "model": self.cfg.pretrained_model,
            "prob_thresh": self.cfg.prob_thresh,
            "min_distance": self.cfg.min_distance,
            "subpix": self.cfg.subpix,
            "use_gpu": self._resolved_gpu,
            "n_spots": len(spots_rc),
            "img_shape": tuple(img_2d.shape),
        }

        return spots_rc, scores, meta
