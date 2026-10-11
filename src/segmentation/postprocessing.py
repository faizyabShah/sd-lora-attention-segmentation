"""Small binary-mask postprocessing operations used by segmentation ablations."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def postprocess_mask(mask: np.ndarray, method: str) -> np.ndarray:
    """Apply one named morphology/component operation to a boolean mask."""
    if method == "none":
        return mask
    if method == "largest_component":
        labels, count = ndimage.label(mask)
        if count == 0:
            return mask
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == sizes.argmax()
    if method.startswith("remove_fraction_"):
        fraction = float(method.rsplit("_", 1)[-1])
        labels, count = ndimage.label(mask)
        if count == 0:
            return mask
        sizes = np.bincount(labels.ravel())
        keep = sizes >= int(np.ceil(mask.size * fraction))
        keep[0] = False
        return keep[labels]
    if method == "fill_holes":
        return ndimage.binary_fill_holes(mask)
    if method.startswith("closing_fill_r"):
        radius = int(method.rsplit("r", 1)[-1])
        structure = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 1), radius)
        return ndimage.binary_fill_holes(ndimage.binary_closing(mask, structure=structure))
    if method.startswith("opening_closing_r"):
        radius = int(method.rsplit("r", 1)[-1])
        structure = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 1), radius)
        opened = ndimage.binary_opening(mask, structure=structure)
        return ndimage.binary_closing(opened, structure=structure)
    operation, radius_text = method.split("_r")
    radius = int(radius_text)
    structure = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 1), radius)
    if operation == "opening":
        return ndimage.binary_opening(mask, structure=structure)
    if operation == "closing":
        return ndimage.binary_closing(mask, structure=structure)
    raise ValueError(method)
