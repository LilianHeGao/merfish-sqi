from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit


@dataclass
class SpotFeatureConfig:
    peak_radius: int = 1
    bg_inner: int = 3
    bg_outer: int = 5
    noise_inner: int = 1
    noise_outer: int = 3
    context_radius: int = 10


def _annulus_offsets(inner: int, outer: int):
    """Pre-compute (dr, dc) offsets for a square annulus."""
    offsets = []
    for dr in range(-outer, outer + 1):
        for dc in range(-outer, outer + 1):
            d = max(abs(dr), abs(dc))  # Chebyshev distance
            if inner <= d <= outer:
                offsets.append((dr, dc))
    return np.array(offsets, dtype=np.int32)


def _patch_offsets(radius: int):
    """Pre-compute (dr, dc) offsets for a square patch."""
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            offsets.append((dr, dc))
    return np.array(offsets, dtype=np.int32)


def _gauss1d(x, a, mu, sigma, bg):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg


def _fit_sigma_1d(profile: np.ndarray) -> float:
    """Fit a 1D Gaussian to a line profile, return sigma. NaN on failure."""
    n = len(profile)
    x = np.arange(n, dtype=np.float64)
    try:
        p0 = [profile.max() - profile.min(), n / 2, 1.0, profile.min()]
        popt, _ = curve_fit(_gauss1d, x, profile.astype(np.float64),
                            p0=p0, maxfev=200)
        return abs(popt[2])
    except Exception:
        return np.nan


def compute_spot_features(
    img_2d: np.ndarray,
    spots_rc: np.ndarray,
    scores: np.ndarray,
    nuclei_labels: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    cfg: SpotFeatureConfig = SpotFeatureConfig(),
) -> pd.DataFrame:
    """
    Compute per-spot features.

    Parameters
    ----------
    img_2d : (H, W) float32
    spots_rc : (N, 2) row, col
    scores : (N,) from Spotiflow
    nuclei_labels : (H, W) int32
    fg_mask, bg_mask : (H, W) bool
    cfg : SpotFeatureConfig

    Returns
    -------
    DataFrame with columns:
        row, col, score, I_peak, bg_local, noise_local, snr,
        sigma_x, sigma_y, ellipticity,
        in_nuclei, in_fg, in_bg, cell_id,
        n_neighbors, nearest_dist
    """
    H, W = img_2d.shape
    N = len(spots_rc)

    if N == 0:
        return pd.DataFrame(columns=[
            "row", "col", "score", "I_peak", "bg_local", "noise_local", "snr",
            "sigma_x", "sigma_y", "ellipticity",
            "in_nuclei", "in_fg", "in_bg", "cell_id",
            "n_neighbors", "nearest_dist",
        ])

    img = img_2d.astype(np.float32, copy=False)

    # Integer spot coords, clipped to image bounds
    ri = np.clip(np.round(spots_rc[:, 0]).astype(np.int32), 0, H - 1)
    ci = np.clip(np.round(spots_rc[:, 1]).astype(np.int32), 0, W - 1)

    # Pad image for safe indexing at borders
    pad = cfg.bg_outer + 1
    img_pad = np.pad(img, pad, mode="reflect")

    ri_p = ri + pad
    ci_p = ci + pad

    # Pre-compute offset arrays
    peak_off = _patch_offsets(cfg.peak_radius)
    bg_off = _annulus_offsets(cfg.bg_inner, cfg.bg_outer)
    noise_off = _annulus_offsets(cfg.noise_inner, cfg.noise_outer)

    # --- Intensity features (vectorized via gather) ---
    I_peak = np.empty(N, dtype=np.float32)
    bg_local = np.empty(N, dtype=np.float32)
    noise_local = np.empty(N, dtype=np.float32)

    # Peak: max in patch
    peak_rows = ri_p[:, None] + peak_off[:, 0]
    peak_cols = ci_p[:, None] + peak_off[:, 1]
    peak_vals = img_pad[peak_rows, peak_cols]  # (N, n_offsets)
    I_peak = peak_vals.max(axis=1)

    # Background: median of annulus
    bg_rows = ri_p[:, None] + bg_off[:, 0]
    bg_cols = ci_p[:, None] + bg_off[:, 1]
    bg_vals = img_pad[bg_rows, bg_cols]
    bg_local = np.median(bg_vals, axis=1).astype(np.float32)

    # Noise: std of annulus
    noise_rows = ri_p[:, None] + noise_off[:, 0]
    noise_cols = ci_p[:, None] + noise_off[:, 1]
    noise_vals = img_pad[noise_rows, noise_cols]
    noise_local = np.std(noise_vals, axis=1).astype(np.float32)

    # SNR
    snr = (I_peak - bg_local) / np.maximum(noise_local, 1e-6)

    # --- Morphology: sigma_x, sigma_y via 1D Gaussian fits ---
    fit_half = cfg.peak_radius + 2  # half-width for profile extraction
    sigma_x = np.full(N, np.nan, dtype=np.float32)
    sigma_y = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        r, c = int(ri_p[i]), int(ci_p[i])
        # Row profile (along columns = x direction)
        c_lo = max(0, c - fit_half)
        c_hi = min(img_pad.shape[1], c + fit_half + 1)
        profile_x = img_pad[r, c_lo:c_hi]
        sigma_x[i] = _fit_sigma_1d(profile_x)

        # Col profile (along rows = y direction)
        r_lo = max(0, r - fit_half)
        r_hi = min(img_pad.shape[0], r + fit_half + 1)
        profile_y = img_pad[r_lo:r_hi, c]
        sigma_y[i] = _fit_sigma_1d(profile_y)

    # Ellipticity: |1 - sigma_min / sigma_max|, 0 = perfectly round
    s_min = np.minimum(sigma_x, sigma_y)
    s_max = np.maximum(sigma_x, sigma_y)
    ellipticity = np.where(
        np.isfinite(s_min) & np.isfinite(s_max) & (s_max > 0),
        np.abs(1.0 - s_min / s_max),
        np.nan,
    ).astype(np.float32)

    # --- Context: mask lookups ---
    in_nuclei = nuclei_labels[ri, ci] > 0
    in_fg = fg_mask[ri, ci]
    in_bg = bg_mask[ri, ci]
    cell_id = nuclei_labels[ri, ci].astype(np.int32)

    # --- Spatial neighbors ---
    n_neighbors = np.zeros(N, dtype=np.int32)
    nearest_dist = np.full(N, np.inf, dtype=np.float32)

    if N >= 2:
        tree = cKDTree(spots_rc)
        # n_neighbors within context_radius
        counts = tree.query_ball_point(spots_rc, r=cfg.context_radius,
                                       return_length=True)
        n_neighbors = np.array(counts, dtype=np.int32) - 1  # exclude self

        # nearest neighbor distance
        dd, _ = tree.query(spots_rc, k=2)
        nearest_dist = dd[:, 1].astype(np.float32)

    return pd.DataFrame({
        "row": spots_rc[:, 0],
        "col": spots_rc[:, 1],
        "score": scores,
        "I_peak": I_peak,
        "bg_local": bg_local,
        "noise_local": noise_local,
        "snr": snr,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "ellipticity": ellipticity,
        "in_nuclei": in_nuclei,
        "in_fg": in_fg,
        "in_bg": in_bg,
        "cell_id": cell_id,
        "n_neighbors": n_neighbors,
        "nearest_dist": nearest_dist,
    })
