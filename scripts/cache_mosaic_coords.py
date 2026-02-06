import os
from sqi.qc.mosaic_coords import MosaicBuildConfig, build_mosaic_and_coords

DATA_FLD = r"M:\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1"
CACHE_ROOT = r"\\192.168.0.73\Papaya13\Lilian\merfish_sqi_cache"

cfg = MosaicBuildConfig(resc=4, icol=1, frame=20, rescz=2, force=False)

print("Building / loading cached mosaic coords...")
mosaic_img, fls_, xs, ys = build_mosaic_and_coords(
    DATA_FLD,
    cfg,
    cache_root=CACHE_ROOT,
    cache=True,
)

print("Done.")
print("n_fovs:", len(fls_))
print("mosaic shape:", mosaic_img.shape)
print("xs/ys:", xs.shape, ys.shape)
