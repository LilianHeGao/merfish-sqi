# sqi/qc/valid_mask_mosaic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import os
import hashlib
import json
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk, remove_small_objects
try:
    from skimage.morphology import closing
except ImportError:
    from skimage.morphology import binary_closing as closing


try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


@dataclass
class MosaicValidMaskConfig:
    """
    All spatial parameters are in FULL-RESOLUTION pixels.
    They are auto-scaled when downsample > 1.
    """
    closing_radius: int = 30          # full-res pixels
    min_object_size: int = 50_000     # full-res pixels
    fill_holes: bool = True
    bg_percentile: float = 1.0
    hi_percentile: float = 99.8
    use_otsu_on_nonzero: bool = True
    downsample: int = 4


def compute_global_valid_mask_from_mosaic(
    mosaic_img: np.ndarray,
    cfg: MosaicValidMaskConfig,
) -> np.ndarray:
    if mosaic_img.ndim != 2:
        raise ValueError("mosaic_img must be 2D")

    m = mosaic_img.astype(np.float32, copy=False)
    full_shape = m.shape

    # --- downsample ---
    ds = max(1, int(cfg.downsample))
    if ds > 1:
        m_small = m[::ds, ::ds]
    else:
        m_small = m

    # Scale spatial parameters to downsampled space
    closing_r = max(1, cfg.closing_radius // ds)
    min_obj = max(1, cfg.min_object_size // (ds * ds))

    # --- normalize nonzero pixels ---
    nz = m_small[m_small > 0]
    if nz.size == 0:
        return np.zeros(full_shape, dtype=bool)

    lo = np.percentile(nz, cfg.bg_percentile)
    hi = np.percentile(nz, cfg.hi_percentile)
    denom = (hi - lo) if (hi > lo) else 1.0
    mn = np.clip((m_small - lo) / (denom + 1e-6), 0, 1)

    # --- threshold ---
    if cfg.use_otsu_on_nonzero:
        t = threshold_otsu(mn[m_small > 0])
    else:
        t = threshold_otsu(mn)

    valid = mn > t

    # --- morphological cleanup (in downsampled space) ---
    if closing_r > 0:
        valid = closing(valid, footprint=disk(closing_r))

    if min_obj > 0:
        valid = remove_small_objects(valid, min_size=min_obj)

    if cfg.fill_holes:
        valid = ndi.binary_fill_holes(valid)

    # --- upsample back to full resolution ---
    if ds > 1:
        valid = ndi.zoom(valid.astype(np.uint8), ds, order=0).astype(bool)
        valid = valid[: full_shape[0], : full_shape[1]]

    return valid


def _cfg_hash(cfg: MosaicValidMaskConfig) -> str:
    """Short hash of config so cache invalidates when parameters change."""
    d = {
        "closing_radius": cfg.closing_radius,
        "min_object_size": cfg.min_object_size,
        "fill_holes": cfg.fill_holes,
        "bg_percentile": cfg.bg_percentile,
        "hi_percentile": cfg.hi_percentile,
        "use_otsu_on_nonzero": cfg.use_otsu_on_nonzero,
        "downsample": cfg.downsample,
    }
    h = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
    return h


def _mask_cache_path(cache_root: str, mosaic_tif_path: str, cfg: MosaicValidMaskConfig) -> str:
    os.makedirs(cache_root, exist_ok=True)
    base = os.path.basename(mosaic_tif_path).replace(".tif", "").replace(".tiff", "")
    return os.path.join(cache_root, f"{base}_valid_mask_{_cfg_hash(cfg)}.tiff")


def load_or_compute_global_valid_mask(
    mosaic_tif_path: str,
    cache_root: str,
    cfg: MosaicValidMaskConfig,
    *,
    force: bool = False,
) -> np.ndarray:
    """
    Loads cached valid mask if present; otherwise computes from mosaic TIFF
    and caches it in cache_root (NOT alongside the data).
    Cache key includes config parameters.
    """
    cache_path = _mask_cache_path(cache_root, mosaic_tif_path, cfg)

    if (not force) and os.path.exists(cache_path):
        return tifffile.imread(cache_path).astype(bool)

    mosaic = tifffile.imread(mosaic_tif_path).astype(np.float32, copy=False)
    valid = compute_global_valid_mask_from_mosaic(mosaic, cfg)

    tifffile.imwrite(cache_path, valid.astype(np.uint8))
    return valid


def save_mask_tiff(path: str, mask: np.ndarray) -> None:
    if tifffile is None:
        raise ImportError("tifffile is required (pip install tifffile).")
    tifffile.imwrite(path, mask.astype(np.uint8))


def crop_valid_mask_for_fov(
    global_valid_mask: np.ndarray,
    fov_anchor_xy: Tuple[float, float],
    fov_shape_hw: Tuple[int, int],
    *,
    mosaic_resc: int = 1,
    anchor_is_upper_left: bool = True,
    round_anchor: bool = True,
) -> np.ndarray:
    """
    Crop mosaic-level valid mask to the FOV region.

    Parameters
    ----------
    global_valid_mask:
        (H_mosaic, W_mosaic) boolean mask, in mosaic pixel space.
    fov_anchor_xy:
        (dim0, dim1) anchor in mosaic pixel coordinates from compose_mosaic.
        compose_mosaic convention: x → array dim 0, y → array dim 1.
    fov_shape_hw:
        (H_fov, W_fov) in **full-resolution** pixels.
    mosaic_resc:
        The rescale factor used to build the mosaic (MosaicBuildConfig.resc).
        Anchor and mask are at mosaic scale; fov_shape_hw is full-res.
    anchor_is_upper_left:
        True  -> anchor is upper-left corner of FOV in mosaic.
        False -> anchor is center of FOV (compose_mosaic default).

    Returns
    -------
    valid_mask_fov:
        (H_fov, W_fov) boolean at full resolution.  Out-of-bounds = False.
    """
    Hm, Wm = global_valid_mask.shape       # dim0, dim1 of mosaic
    hf_full, wf_full = map(int, fov_shape_hw)

    # FOV size in mosaic pixel space
    hf = hf_full // mosaic_resc             # extent along dim 0
    wf = wf_full // mosaic_resc             # extent along dim 1

    # anchor: (dim0_pos, dim1_pos) — following compose_mosaic convention
    a0, a1 = fov_anchor_xy
    if round_anchor:
        a0 = int(round(a0))
        a1 = int(round(a1))

    if anchor_is_upper_left:
        r0, c0 = a0, a1
    else:
        r0 = int(round(a0 - hf / 2))       # dim0 center → upper-left
        c0 = int(round(a1 - wf / 2))       # dim1 center → upper-left

    r1, c1 = r0 + hf, c0 + wf

    crop = np.zeros((hf, wf), dtype=bool)

    # Clip to mosaic bounds
    mr0 = max(0, r0);  mc0 = max(0, c0)
    mr1 = min(Hm, r1); mc1 = min(Wm, c1)

    if mr1 <= mr0 or mc1 <= mc0:
        return np.zeros((hf_full, wf_full), dtype=bool)

    # Offset in output crop
    or0 = mr0 - r0
    oc0 = mc0 - c0

    crop[or0 : or0 + (mr1 - mr0), oc0 : oc0 + (mc1 - mc0)] = \
        global_valid_mask[mr0:mr1, mc0:mc1]

    # Upsample to full resolution
    if mosaic_resc > 1:
        out = ndi.zoom(crop.astype(np.uint8), mosaic_resc, order=0).astype(bool)
        out = out[:hf_full, :wf_full]
    else:
        out = crop

    return out


def overlay_bbox_on_mosaic(
    mosaic_img: np.ndarray,
    fov_anchor_xy: Tuple[float, float],
    fov_shape_hw: Tuple[int, int],
    *,
    mosaic_resc: int = 1,
    anchor_is_upper_left: bool = True,
) -> "tuple[object, object]":
    """Debug: show FOV bbox on mosaic. Uses matplotlib (col, row) convention."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    hf = int(fov_shape_hw[0]) // mosaic_resc   # dim0 extent
    wf = int(fov_shape_hw[1]) // mosaic_resc   # dim1 extent
    a0, a1 = fov_anchor_xy                     # (dim0, dim1)
    if anchor_is_upper_left:
        r0, c0 = a0, a1
    else:
        r0 = a0 - hf / 2
        c0 = a1 - wf / 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mosaic_img, cmap="gray")
    # matplotlib Rectangle takes (col, row), width, height
    ax.add_patch(Rectangle((c0, r0), wf, hf, fill=False, linewidth=2, edgecolor="red"))
    ax.set_title("FOV bbox on mosaic")
    ax.set_axis_off()
    return fig, ax
