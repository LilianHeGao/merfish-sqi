from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


@dataclass
class CellProximalConfig:
    """
    Defines cell-proximal region.
    """
    cell_proximal_px: int = 12

from scipy import ndimage as ndi
import numpy as np

def build_cell_proximal_and_distal_masks(
    nuclei_labels: np.ndarray,
    valid_mask: np.ndarray,
    cfg: CellProximalConfig,
):
    """
    Returns:
      cell_proximal_mask
      cell_distal_mask
    """
    if nuclei_labels.ndim != 2:
        raise ValueError("nuclei_labels must be 2D")

    nuclei_bin = nuclei_labels > 0

    # Cell-proximal = dilated nuclei
    cell_proximal = ndi.binary_dilation(
        nuclei_bin,
        structure=np.ones((2 * cfg.cell_proximal_px + 1,
                            2 * cfg.cell_proximal_px + 1)),
    )

    # Restrict to tissue
    cell_proximal &= valid_mask

    # Cell-distal = tissue but not cell-proximal
    cell_distal = valid_mask & (~cell_proximal)

    stats = {
        "n_nuclei": int(nuclei_labels.max()),
        "cell_proximal_px": int(cell_proximal.sum()),
        "cell_distal_px": int(cell_distal.sum()),
    }

    return cell_proximal, cell_distal, stats



def per_cell_fg_bg(
    nuclei_labels: np.ndarray,
    fg_union: np.ndarray,
    bg_union: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign FG/BG pixels to nearest nucleus label (Voronoi by EDT indices).
    This keeps per-cell accounting possible even when BG is global.
    """
    labels = nuclei_labels.astype(np.int32, copy=False)
    nuclei_bin = labels > 0

    _, (iy, ix) = ndi.distance_transform_edt(~nuclei_bin, return_indices=True)
    nearest_label = labels[iy, ix]

    fg_label_map = np.where(fg_union, nearest_label, 0).astype(np.int32, copy=False)
    bg_label_map = np.where(bg_union, nearest_label, 0).astype(np.int32, copy=False)

    return fg_label_map, bg_label_map
