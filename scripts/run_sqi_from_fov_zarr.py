import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile as tiff

from sqi.io.image_io import load_fov_from_zarr, get_fov_anchor_xy
from segmentation.cellpose_backend import segment_nuclei
from sqi.qc.valid_mask_mosaic import crop_valid_mask_for_fov
from sqi.qc.rings import CellProximalConfig, build_cell_proximal_and_distal_masks
from sqi.qc.metrics import compute_sqi_from_label_maps
from sqi.qc.plots import plot_sqi_distribution
from sqi.spots.detect import detect_spots_from_zarr   # refactored


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def main(args):
    fov_zarr = Path(args.fov_zarr)
    fov_id = fov_zarr.stem.replace(".zarr", "")

    cache_dir = Path(args.cache_root) / fov_id
    out_dir = Path(args.out_root) / fov_id

    ensure_dir(cache_dir)
    ensure_dir(out_dir)

    # -------------------------
    # Load image
    # -------------------------
    img = load_fov_from_zarr(fov_zarr, channel="DAPI")

    # -------------------------
    # Nuclei (cached)
    # -------------------------
    labels_path = cache_dir / "nuclei_labels.tif"
    if labels_path.exists():
        labels = tiff.imread(labels_path)
    else:
        labels = segment_nuclei(img, use_gpu=True)
        tiff.imwrite(labels_path, labels)

    # -------------------------
    # Spots (cached)
    # -------------------------
    spots_path = cache_dir / "spots_rc.npy"
    if spots_path.exists():
        spots_rc = np.load(spots_path)
    else:
        spots_rc = detect_spots_from_zarr(fov_zarr)
        np.save(spots_path, spots_rc)

    # -------------------------
    # Valid mask (cached)
    # -------------------------
    valid_mask_path = cache_dir / "valid_mask.tif"
    if valid_mask_path.exists():
        valid_mask = tiff.imread(valid_mask_path).astype(bool)
    else:
        global_valid = tiff.imread(args.mosaic_valid_mask).astype(bool)
        anchor_xy = get_fov_anchor_xy(fov_zarr)

        valid_mask = crop_valid_mask_for_fov(
            global_valid_mask=global_valid,
            fov_anchor_xy=anchor_xy,
            fov_shape_hw=labels.shape,
            mosaic_resc=1,  # LOCKED
            anchor_is_upper_left=False,
        )

        tiff.imwrite(valid_mask_path, valid_mask.astype(np.uint8))

    # -------------------------
    # FG / BG (cached)
    # -------------------------
    fg_path = cache_dir / "fg_mask.tif"
    bg_path = cache_dir / "bg_mask.tif"

    if fg_path.exists() and bg_path.exists():
        fg_mask = tiff.imread(fg_path).astype(bool)
        bg_mask = tiff.imread(bg_path).astype(bool)
    else:
        cfg = CellProximalConfig(cell_proximal_px=args.cell_proximal_px)
        fg_mask, bg_mask, _ = build_cell_proximal_and_distal_masks(
            labels, valid_mask, cfg
        )
        tiff.imwrite(fg_path, fg_mask.astype(np.uint8))
        tiff.imwrite(bg_path, bg_mask.astype(np.uint8))

    # -------------------------
    # SQI-1
    # -------------------------
    sqi_df, summary = compute_sqi_from_label_maps(
        labels, fg_mask, bg_mask, spots_rc
    )

    sqi_df.to_csv(out_dir / "sqi_cell.csv", index=False)
    json.dump(summary, open(out_dir / "sqi_summary.json", "w"), indent=2)

    plot_sqi_distribution(sqi_df, out_dir / "qc")

    print(f"[DONE] SQI-1 completed for {fov_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fov_zarr", required=True)
    parser.add_argument("--mosaic_valid_mask", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--cell_proximal_px", type=int, default=24)

    main(parser.parse_args())
