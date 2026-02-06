import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk


def compute_valid_mask_from_dapi(
    dapi: np.ndarray,
    *,
    min_object_size: int = 5000,
    closing_radius: int = 10,
) -> np.ndarray:
    """
    Compute a conservative tissue / valid imaging mask from DAPI.

    Design:
    - Threshold DAPI to find nuclear signal
    - Morphologically close to fill gaps between nuclei
    - Remove small objects (dust, noise)
    - Fill holes

    Returns:
        valid_mask : boolean array, same shape as dapi
    """

    if dapi.ndim != 2:
        raise ValueError("DAPI must be 2D (use max-projection beforehand)")

    # --- normalize (robust) ---
    d = dapi.astype(np.float32)
    d = d - np.percentile(d, 1)
    d = d / (np.percentile(d, 99.8) + 1e-6)
    d = np.clip(d, 0, 1)

    # --- threshold ---
    t = threshold_otsu(d)
    tissue = d > t

    # --- close gaps between nuclei ---
    tissue = binary_closing(tissue, footprint=disk(closing_radius))

    # --- remove tiny islands ---
    tissue = remove_small_objects(tissue, min_size=min_object_size)

    # --- fill holes ---
    tissue = ndi.binary_fill_holes(tissue)

    return tissue.astype(bool)
