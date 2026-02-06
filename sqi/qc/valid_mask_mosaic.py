# sqi/qc/valid_mask_mosaic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, Union

import os
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects


try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


ArrayLike = Union[np.ndarray]

@dataclass
class MosaicValidMaskConfig:
    closing_radius: int = 30
    min_object_size: int = 50_000
    fill_holes: bool = True
    bg_percentile: float = 1.0
    hi_percentile: float = 99.8
    use_otsu_on_nonzero: bool = True

    # speed knob
    downsample: int = 4   # compute mask on downsampled mosaic, then upsample


def compute_global_valid_mask_from_mosaic(mosaic_img: np.ndarray, cfg: MosaicValidMaskConfig) -> np.ndarray:
    if mosaic_img.ndim != 2:
        raise ValueError("mosaic_img must be 2D")

    m = mosaic_img.astype(np.float32, copy=False)

    # optional downsample for speed
    ds = int(cfg.downsample) if cfg.downsample else 1
    print(f"[valid_mask] downsample = {cfg.downsample}")
    if ds > 1:
        m_small = m[::ds, ::ds]
    else:
        m_small = m

    nz = m_small[m_small > 0]
    if nz.size == 0:
        return np.zeros_like(m, dtype=bool)

    lo = np.percentile(nz, cfg.bg_percentile)
    hi = np.percentile(nz, cfg.hi_percentile)
    denom = (hi - lo) if (hi > lo) else 1.0
    mn = np.clip((m_small - lo) / (denom + 1e-6), 0, 1)

    if cfg.use_otsu_on_nonzero:
        t = threshold_otsu(mn[m_small > 0])
    else:
        t = threshold_otsu(mn)

    valid = mn > t

    if cfg.closing_radius and cfg.closing_radius > 0:
        valid = binary_closing(valid, footprint=disk(int(cfg.closing_radius)))

    if cfg.min_object_size and cfg.min_object_size > 0:
        valid = remove_small_objects(valid, min_size=int(cfg.min_object_size))

    if cfg.fill_holes:
        valid = ndi.binary_fill_holes(valid)

    # upsample back to full resolution if downsampled
    if ds > 1:
        valid = np.repeat(np.repeat(valid, ds, axis=0), ds, axis=1)
        valid = valid[: m.shape[0], : m.shape[1]]

    return valid.astype(bool)


def _mask_cache_path(cache_root: str, mosaic_tif_path: str) -> str:
    os.makedirs(cache_root, exist_ok=True)
    base = os.path.basename(mosaic_tif_path).replace(".tif", "").replace(".tiff", "")
    return os.path.join(cache_root, base + "_valid_mask.tiff")

def load_mosaic_tiff(path: str) -> np.ndarray:
    """
    Load a mosaic TIFF as float32 array.
    """
    if tifffile is None:
        raise ImportError("tifffile is required to read TIFF mosaics (pip install tifffile).")
    img = tifffile.imread(path)
    return img.astype(np.float32, copy=False)

	
def load_or_compute_global_valid_mask(
    mosaic_tif_path: str,
    cache_root: str,
    cfg: MosaicValidMaskConfig,
    *,
    force: bool = False,
) -> np.ndarray:
    """
    Loads cached valid mask if present; otherwise computes from mosaic TIFF and caches it
    in cache_root (NOT alongside the data).
    """
    cache_path = _mask_cache_path(cache_root, mosaic_tif_path)

    if (not force) and os.path.exists(cache_path):
        return tifffile.imread(cache_path).astype(bool)

    mosaic = tifffile.imread(mosaic_tif_path).astype(np.float32, copy=False)
    valid = compute_global_valid_mask_from_mosaic(mosaic, cfg)

    tifffile.imwrite(cache_path, valid.astype(np.uint8))
    return valid



def save_mask_tiff(path: str, mask: np.ndarray) -> None:
    """
    Save boolean mask as uint8 TIFF (0/1).
    """
    if tifffile is None:
        raise ImportError("tifffile is required to write TIFF masks (pip install tifffile).")
    tifffile.imwrite(path, mask.astype(np.uint8))


def fov_id_from_zarr_path(zarr_path: str) -> str:
    """
    Extract fov id from ...Conv_zscan1_074.zarr -> "074"
    (matches your existing naming scheme)
    """
    base = zarr_path.replace("\\", "/").split("/")[-1]
    # Conv_zscan1_074.zarr -> "074"
    return base.split("_")[-1].split(".")[0]


def build_fov_anchor_index(
    fov_paths: Sequence[str],
    xs: Sequence[float],
    ys: Sequence[float],
) -> Dict[str, Tuple[float, float]]:
    """
    Build mapping: fov_id -> (x_anchor, y_anchor) in *mosaic coordinate system*,
    using the outputs you already get from compose_mosaic(return_coords=True).

    IMPORTANT:
      This function is agnostic to whether xs/ys are px or already scaled;
      it just stores whatever compose_mosaic returned. Use crop_valid_mask_for_fov()
      with the same coordinate conventions.
    """
    if not (len(fov_paths) == len(xs) == len(ys)):
        raise ValueError("fov_paths, xs, ys must have the same length")

    idx: Dict[str, Tuple[float, float]] = {}
    for p, x, y in zip(fov_paths, xs, ys):
        idx[fov_id_from_zarr_path(p)] = (float(x), float(y))
    return idx


def crop_valid_mask_for_fov(
    global_valid_mask: np.ndarray,
    fov_anchor_xy: Tuple[float, float],
    fov_shape_hw: Tuple[int, int],
    *,
    anchor_is_upper_left: bool = True,
    round_anchor: bool = True,
) -> np.ndarray:
    """
    Crop mosaic-level valid mask to the FOV region.

    Parameters
    ----------
    global_valid_mask:
        (H_mosaic, W_mosaic) boolean mask.
    fov_anchor_xy:
        (x, y) anchor location in mosaic coordinates, matching compose_mosaic return_coords.
        If anchor_is_upper_left=True, (x,y) is interpreted as the upper-left corner of the FOV in mosaic space.
        If anchor_is_upper_left=False, (x,y) is interpreted as the center of the FOV.
    fov_shape_hw:
        (H_fov, W_fov) in pixels (same orientation as the mask you want to apply to your FOV arrays).
    anchor_is_upper_left:
        Set True if compose_mosaic returns top-left placement coordinates (common).
        If False, we'll treat anchor as center.
    round_anchor:
        If True, cast anchor to int via rounding.

    Returns
    -------
    valid_mask_fov:
        (H_fov, W_fov) boolean. Out-of-bounds regions are filled with False.
    """
    Hm, Wm = global_valid_mask.shape
    hf, wf = map(int, fov_shape_hw)
    x, y = fov_anchor_xy

    if round_anchor:
        x = int(round(x))
        y = int(round(y))

    if anchor_is_upper_left:
        x0, y0 = x, y
    else:
        x0 = int(round(x - wf / 2))
        y0 = int(round(y - hf / 2))

    # Compute crop with bounds checking
    x1, y1 = x0 + wf, y0 + hf

    # Create output initialized to False
    out = np.zeros((hf, wf), dtype=bool)

    # Intersection in mosaic coordinates
    mx0 = max(0, x0)
    my0 = max(0, y0)
    mx1 = min(Wm, x1)
    my1 = min(Hm, y1)

    if mx1 <= mx0 or my1 <= my0:
        return out  # entirely out of bounds

    # Corresponding region in output
    ox0 = mx0 - x0
    oy0 = my0 - y0
    ox1 = ox0 + (mx1 - mx0)
    oy1 = oy0 + (my1 - my0)

    out[oy0:oy1, ox0:ox1] = global_valid_mask[my0:my1, mx0:mx1]
    return out


# --- Optional quick visualization helper (debug only) ---
def overlay_bbox_on_mosaic(
    mosaic_img: np.ndarray,
    fov_anchor_xy: Tuple[float, float],
    fov_shape_hw: Tuple[int, int],
    *,
    anchor_is_upper_left: bool = True,
) -> "tuple[object, object]":
    """
    Debug helper to visually confirm the FOV crop location in the mosaic.
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    hf, wf = map(int, fov_shape_hw)
    x, y = fov_anchor_xy
    if anchor_is_upper_left:
        x0, y0 = x, y
    else:
        x0, y0 = x - wf / 2, y - hf / 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mosaic_img, cmap="gray")
    ax.add_patch(Rectangle((x0, y0), wf, hf, fill=False, linewidth=2))
    ax.set_title("FOV bbox on mosaic")
    ax.set_axis_off()
    return fig, ax
