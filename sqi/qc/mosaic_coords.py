# sqi/qc/mosaic_coords.py
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

import tifffile

from sqi.io.image_io import read_multichannel_from_conv_zarr

@dataclass
class MosaicBuildConfig:
    resc: int = 4
    icol: int = 1
    frame: int = 20          # or 'all'
    rescz: int = 2
    rot_k: int = 2           # np.rot90 k value (0-3), applied before T[::-1,::-1]
    um_per_pix_native: float = 0.1083333
    force: bool = False

def compose_mosaic(ims, xs_um, ys_um, ims_c=None, um_per_pix=0.108333, rot=0, return_coords=False):
    """
    Compose a mosaic from tiles.

    Coordinate convention:
    - xs_um (stage_x) -> COLUMNS in the mosaic
    - ys_um (stage_y) -> ROWS in the mosaic
    - NumPy indexing: im[row, col]

    If return_coords: returns (mosaic, xs_center, ys_center) where
      xs_center = column centers, ys_center = row centers.
    """
    dtype = np.float32
    im_ = ims[0]
    szs = im_.shape
    tile_h, tile_w = szs[-2], szs[-1]

    # Apply rotation to coordinates
    theta = -np.deg2rad(rot)
    xs_um_ = np.array(xs_um) * np.cos(theta) - np.array(ys_um) * np.sin(theta)
    ys_um_ = np.array(ys_um) * np.cos(theta) + np.array(xs_um) * np.sin(theta)

    # x (stage) -> column index,  y (stage) -> row index
    cols_pix = np.array(xs_um_ / um_per_pix, dtype=float)
    cols_pix = np.array(cols_pix - np.min(cols_pix), dtype=int)

    rows_pix = np.array(ys_um_ / um_per_pix, dtype=float)
    rows_pix = np.array(rows_pix - np.min(rows_pix), dtype=int)

    canvas_h = np.max(rows_pix) + tile_h + 1
    canvas_w = np.max(cols_pix) + tile_w + 1

    if len(szs) == 3:
        dim = [szs[0], canvas_h, canvas_w]
    else:
        dim = [canvas_h, canvas_w]

    if ims_c is None:
        if len(ims) > 25:
            try:
                ims_c = linear_flat_correction(ims, fl=None, reshape=False, resample=1,
                                               vec=[0.1, 0.15, 0.25, 0.5, 0.65, 0.75, 0.9])
            except:
                ims_c = np.median(ims, axis=0)
        else:
            ims_c = np.median(ims, axis=0)

    im_big = np.zeros(dim, dtype=dtype)

    for i, (im_, r0, c0) in enumerate(zip(ims, rows_pix, cols_pix)):
        if ims_c is not None:
            if len(ims_c) == 2:
                im_coef, im_inters = np.array(ims_c, dtype='float32')
                im__ = (np.array(im_, dtype='float32') - im_inters) / im_coef
            else:
                ims_c_ = np.array(ims_c, dtype='float32')
                im__ = np.array(im_, dtype='float32') / ims_c_ * np.median(ims_c_)
        else:
            im__ = np.array(im_, dtype='float32')
        im__ = np.array(im__, dtype=dtype)
        im_big[..., r0:r0 + tile_h, c0:c0 + tile_w] = im__

    if return_coords:
        xs_center = cols_pix + tile_w / 2   # x (stage) -> column center
        ys_center = rows_pix + tile_h / 2   # y (stage) -> row center
        return im_big, xs_center, ys_center
    return im_big

def mosaic_cache_paths(
    data_fld: str,
    cfg: MosaicBuildConfig,
    cache_root: str,
) -> Tuple[str, str]:
    """
    Returns:
      mosaic_tif_path, coords_npz_path
    Saved under cache_root/_mosaic/, NOT alongside data.
    """
    mosaic_dir = os.path.join(cache_root, "_mosaic")
    os.makedirs(mosaic_dir, exist_ok=True)

    # _v2 suffix: coordinate fix (row/col swap), invalidates old caches
    base = f"{os.path.basename(data_fld)}_frame{cfg.frame}_col{cfg.icol}_resc{cfg.resc}_rescz{cfg.rescz}_v2"
    if cfg.rot_k != 2:
        base += f"_rotk{cfg.rot_k}"
    mosaic_tif = os.path.join(mosaic_dir, base + ".tiff")
    coords_npz = os.path.join(mosaic_dir, base + "_coords.npz")
    return mosaic_tif, coords_npz


def build_mosaic_and_coords(
    data_fld: str,
    cfg: MosaicBuildConfig,
    *,
    cache_root: str,
    cache: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

    """
    Build (or load cached) mosaic and FOV placement coordinates.

    Returns:
      mosaic_img: 2D float32
      fls_: list of .zarr paths (aligned with xs, ys)
      xs, ys: arrays of placement coordinates returned by compose_mosaic(return_coords=True)
    """
    mosaic_tif, coords_npz = mosaic_cache_paths(data_fld, cfg, cache_root)

    # If cached and not forcing rebuild, load coords + mosaic
    if cache and (not cfg.force) and os.path.exists(mosaic_tif) and os.path.exists(coords_npz):
        coords = np.load(coords_npz, allow_pickle=True)
        fls_ = coords["fls_"].tolist()
        xs = coords["xs"].astype(np.float32, copy=False)
        ys = coords["ys"].astype(np.float32, copy=False)
        return None, fls_, xs, ys

    # Otherwise, compute
    fls_ = np.sort(glob.glob(os.path.join(data_fld, "*.zarr"))).tolist()
    ims: List[np.ndarray] = []
    xs_um: List[float] = []
    ys_um: List[float] = []

    for fl in tqdm(fls_, desc="Reading FOVs for mosaic"):
        try:
            im, x, y = read_multichannel_from_conv_zarr(fl, return_pos=True)

            if str(cfg.frame).lower() == "all":
                # (Z,Y,X) after max over z
                tile = np.array(np.max(im[cfg.icol][::cfg.rescz, ::cfg.resc, ::cfg.resc][:, ::-1], axis=0),
                                dtype=np.float32)
            else:
                tile = np.array(im[cfg.icol][cfg.frame, ::cfg.resc, ::cfg.resc][:, ::-1],
                                dtype=np.float32)

            ims.append(tile)
            xs_um.append(x)
            ys_um.append(y)
        except Exception as e:
            print(f"[mosaic] Error processing {fl}: {e}")
            continue

    if not ims:
        raise RuntimeError("No images were successfully processed for mosaic.")

    rotated_ims = [np.rot90(im_, k=cfg.rot_k) for im_ in ims]
    tiles = [im_.T[::-1, ::-1] for im_ in rotated_ims]

    um_per_pix = cfg.um_per_pix_native * cfg.resc

    mosaic_img, xs_center_pix, ys_center_pix = compose_mosaic(
		tiles,
		xs_um,
		ys_um,
		ims_c=None,
		um_per_pix=um_per_pix,
		rot=0,
		return_coords=True,
	)

    mosaic_img = mosaic_img.astype(np.float32, copy=False)
    xs = np.array(xs_center_pix, dtype=np.float32)
    ys = np.array(ys_center_pix, dtype=np.float32)

    # Cache
    if cache:
        tifffile.imwrite(mosaic_tif, mosaic_img)
        np.savez(coords_npz, fls_=np.array(fls_, dtype=object), xs=xs, ys=ys)
        print("[mosaic] Saved:", mosaic_tif)
        print("[mosaic] Saved:", coords_npz)

    return mosaic_img, fls_, xs, ys
	
def fov_id_from_zarr_path(zarr_path: str) -> str:
    base = os.path.basename(zarr_path)
    return base.split("_")[-1].split(".")[0]


def _fov_id_variants(fov_id: str) -> List[str]:
    """Return both '040' and '40' style variants for a FOV id."""
    stripped = fov_id.lstrip("0") or "0"
    padded = fov_id.zfill(3)
    seen = []
    for v in [fov_id, stripped, padded]:
        if v not in seen:
            seen.append(v)
    return seen


def lookup_fov_anchor(fov_index: Dict[str, Tuple[float, float]], fov_id: str) -> Tuple[float, float]:
    """Look up FOV anchor, trying both '040' and '40' variants."""
    for v in _fov_id_variants(fov_id):
        if v in fov_index:
            return fov_index[v]
    raise KeyError(
        f"FOV '{fov_id}' not found in anchor index. "
        f"Tried: {_fov_id_variants(fov_id)}. "
        f"Available: {list(fov_index.keys())[:10]}..."
    )


def build_fov_anchor_index(fls_: List[str], xs: np.ndarray, ys: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Build mapping from FOV id to (row_center, col_center) in mosaic pixels.

    xs from compose_mosaic = column centers, ys = row centers.
    We store as (row, col) to match NumPy / crop_valid_mask_for_fov convention.
    """
    if len(fls_) != len(xs) or len(fls_) != len(ys):
        raise ValueError("fls_, xs, ys length mismatch")
    out: Dict[str, Tuple[float, float]] = {}
    for p, x, y in zip(fls_, xs, ys):
        out[fov_id_from_zarr_path(p)] = (float(y), float(x))  # (row, col)
    return out
