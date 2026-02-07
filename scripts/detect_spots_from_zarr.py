import argparse
import numpy as np
from skimage.feature import blob_log

from sqi.io.image_io import read_multichannel_from_conv_zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    im = read_multichannel_from_conv_zarr(args.zarr)

    # DAPI is last channel
    spot_channels = im[:-1]

    spots = []

    for c in range(spot_channels.shape[0]):
        img = np.max(spot_channels[c], axis=0) if spot_channels[c].ndim == 3 else spot_channels[c]

        blobs = blob_log(
            img,
            min_sigma=1.0,
            max_sigma=2.5,
            num_sigma=5,
            threshold=0.02,
        )

        # blobs: (row, col, sigma)
        if blobs.size > 0:
            spots.append(blobs[:, :2])

    if spots:
        spots_rc = np.vstack(spots).astype(np.float32)
    else:
        spots_rc = np.zeros((0, 2), dtype=np.float32)

    np.save(args.out, spots_rc)
    print("Saved spots:", spots_rc.shape, "->", args.out)


if __name__ == "__main__":
    main()
