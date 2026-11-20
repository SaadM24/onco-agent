"""Non-diagnostic quality indicators on image arrays."""
import numpy as np


def quality_indicators(rgb):
    image = np.asarray(rgb, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected H x W x 3 RGB")
    grayscale = image.mean(axis=2)
    gx = np.diff(grayscale, axis=1)
    gy = np.diff(grayscale, axis=0)
    return {"mean_brightness": float(np.mean(image) / 255),
            "edge_variance": float((gx.var() + gy.var()) / 2),
            "saturation_fraction": float(np.mean((image <= 2) | (image >= 253)))}
