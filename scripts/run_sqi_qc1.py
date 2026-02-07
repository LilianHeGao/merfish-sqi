import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff

from sqi.qc.rings import CellProximalConfig, build_cell_proximal_and_distal_masks, per_cell_fg_bg
from sqi.qc.metrics import compute_sqi_from_label_maps
from sqi.qc.qc_plots import plot_sqi_distribution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="nuclei_labels.tif")
    ap.add_argument("--spots", required=True, help="spots_rc.npy (N,2)")
    ap.add_argument("--valid_mask", required=True, help="valid_mask.tif (mosaic-level, cropped to FOV)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cell_proximal_px", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = tiff.imread(args.labels).astype(np.int32, copy=False)
    spots_rc = np.load(args.spots).astype(np.int32)
    valid_mask = tiff.imread(args.valid_mask).astype(bool)

    # Build FG / BG
    cfg = CellProximalConfig(cell_proximal_px=args.cell_proximal_px)
    fg_mask, bg_mask, region_stats = build_cell_proximal_and_distal_masks(labels, valid_mask, cfg)

    # Per-cell label maps
    fg_label_map, bg_label_map = per_cell_fg_bg(labels, fg_mask, bg_mask)

    # Compute SQI
    sqi, _, _ = compute_sqi_from_label_maps(fg_label_map, bg_label_map, spots_rc)

    # Plot
    ax = plot_sqi_distribution(sqi, title="SQI distribution (cell-proximal / cell-distal)")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_dir / "sqi_distribution.png", dpi=200)

    # Save summary
    vals = np.array([v for v in sqi.values() if np.isfinite(v)])
    summary = {
        **region_stats,
        "n_cells_with_sqi": len(vals),
        "median_sqi": float(np.median(vals)) if len(vals) else None,
        "mean_log10_sqi": float(np.mean(np.log10(vals))) if len(vals) else None,
    }
    (out_dir / "sqi_summary.json").write_text(json.dumps(summary, indent=2))

    # Debug masks
    tiff.imwrite(str(out_dir / "fg_mask.tif"), fg_mask.astype(np.uint8), compression="zlib")
    tiff.imwrite(str(out_dir / "bg_mask.tif"), bg_mask.astype(np.uint8), compression="zlib")

    print("Saved:", out_dir / "sqi_distribution.png")
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
