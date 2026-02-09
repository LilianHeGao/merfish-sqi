"""
Spot detection + feature extraction + quality scoring for a single FOV.

Example
-------
python scripts/run_spots.py \
  --fov_zarr  //server/data/Conv_zscan1_074.zarr \
  --labels    /cache/074/nuclei_labels.tif \
  --fg_mask   /cache/074/fg_mask.tif \
  --bg_mask   /cache/074/bg_mask.tif \
  --out_dir   /output/074
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_multichannel_from_conv_zarr
from sqi.io.spots_io import write_spots_parquet, write_spots_meta
from sqi.spot_calling.spotiflow_backend import SpotiflowBackend, SpotiflowConfig
from sqi.spot_features.features import compute_spot_features, SpotFeatureConfig
from sqi.spot_features.quality import compute_quality_scores, QualityGateConfig


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "spots.parquet"
    meta_path = out_dir / "spots_meta.json"

    if parquet_path.exists() and meta_path.exists() and not args.force:
        print(f"[CACHE HIT] {parquet_path}")
        return

    # --------------------------------------------------
    # 1. Load image channels (non-DAPI), z-project
    # --------------------------------------------------
    print("[1/4] Loading image channels ...")
    im = read_multichannel_from_conv_zarr(args.fov_zarr)  # (C, Z, Y, X) dask
    n_channels = im.shape[0]

    # DAPI is last channel; process all others
    spot_channel_indices = list(range(n_channels - 1))
    print(f"       {len(spot_channel_indices)} spot channels, DAPI at index {n_channels - 1}")

    # --------------------------------------------------
    # 2. Detect spots per channel
    # --------------------------------------------------
    print("[2/4] Running Spotiflow per channel ...")
    sf_cfg = SpotiflowConfig(
        pretrained_model=args.model,
        prob_thresh=args.prob_thresh,
    )
    backend = SpotiflowBackend(sf_cfg)

    all_spots = []
    all_scores = []
    all_channels = []
    all_metas = []

    for ch_idx in spot_channel_indices:
        ch_img = im[ch_idx]
        # Z-project (max)
        if ch_img.ndim == 3:
            ch_2d = np.array(ch_img.max(axis=0), dtype=np.float32)
        else:
            ch_2d = np.array(ch_img, dtype=np.float32)

        spots_rc, scores, meta = backend.detect(ch_2d)
        print(f"       channel {ch_idx}: {len(spots_rc)} spots")

        all_spots.append(spots_rc)
        all_scores.append(scores)
        all_channels.append(np.full(len(spots_rc), ch_idx, dtype=np.int32))
        all_metas.append(meta)

    if all_spots:
        spots_rc = np.vstack(all_spots).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)
        channels = np.concatenate(all_channels)
    else:
        spots_rc = np.zeros((0, 2), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        channels = np.zeros(0, dtype=np.int32)

    print(f"       total spots: {len(spots_rc)}")

    # --------------------------------------------------
    # 3. Compute per-spot features
    # --------------------------------------------------
    print("[3/4] Computing spot features ...")
    labels = tiff.imread(args.labels).astype(np.int32)
    fg_mask = tiff.imread(args.fg_mask).astype(bool)
    bg_mask = tiff.imread(args.bg_mask).astype(bool)

    # Use max-projected composite of all spot channels for intensity features
    composite = np.zeros(labels.shape, dtype=np.float32)
    for ch_idx in spot_channel_indices:
        ch_img = im[ch_idx]
        if ch_img.ndim == 3:
            ch_2d = np.array(ch_img.max(axis=0), dtype=np.float32)
        else:
            ch_2d = np.array(ch_img, dtype=np.float32)
        composite = np.maximum(composite, ch_2d)

    feat_cfg = SpotFeatureConfig()
    df = compute_spot_features(
        composite, spots_rc, scores,
        labels, fg_mask, bg_mask, feat_cfg,
    )
    df.insert(2, "channel", channels)

    # --------------------------------------------------
    # 4. Quality scoring
    # --------------------------------------------------
    print("[4/4] Computing quality scores ...")
    q_cfg = QualityGateConfig()
    df = compute_quality_scores(df, q_cfg)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    write_spots_parquet(df, str(parquet_path))

    combined_meta = {
        "fov_zarr": args.fov_zarr,
        "n_channels": len(spot_channel_indices),
        "spotiflow_config": sf_cfg.__dict__,
        "feature_config": feat_cfg.__dict__,
        "quality_config": q_cfg.__dict__,
        "n_spots_total": len(df),
        "n_pass_permissive": int(df["pass_permissive"].sum()),
        "n_pass_conservative": int(df["pass_conservative"].sum()),
        "per_channel": [m for m in all_metas],
    }
    write_spots_meta(combined_meta, str(meta_path))

    print("=" * 50)
    print(f"[DONE] {len(df)} spots")
    print(f"  pass_permissive    : {combined_meta['n_pass_permissive']}")
    print(f"  pass_conservative  : {combined_meta['n_pass_conservative']}")
    print(f"  output             : {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spot detection + features + quality for a single FOV",
    )
    parser.add_argument("--fov_zarr", required=True,
                        help="Path to Conv_zscanX_NNN.zarr")
    parser.add_argument("--labels", required=True,
                        help="Nuclei labels TIFF (int32)")
    parser.add_argument("--fg_mask", required=True,
                        help="FG (cell-proximal) mask TIFF")
    parser.add_argument("--bg_mask", required=True,
                        help="BG (cell-distal) mask TIFF")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for spots.parquet + spots_meta.json")
    parser.add_argument("--model", default="general",
                        help="Spotiflow pretrained model (default: general)")
    parser.add_argument("--prob_thresh", type=float, default=0.5,
                        help="Spotiflow probability threshold (default: 0.5)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cached")
    main(parser.parse_args())
