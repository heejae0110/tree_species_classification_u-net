#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vegetation indices (7) and simple GLCM texture (7) helpers for 8-band stacks.
Bands: MS1(Blue), MS2(Green), MS3(Red), MS4(Red Edge), MS5(NIR), B9, B11, B12
"""
import numpy as np

__all__ = [
    'calculate_vegetation_indices_8bands',
    'compute_glcm_features'
]

def calculate_vegetation_indices_8bands(image_data: np.ndarray) -> np.ndarray:
    """Return stack of [NDVI, GNDVI, RVI, NDRE, CIre, MCARI, SAVI] from 8 bands.
    image_data: (8, H, W) float32/float64
    """
    blue = image_data[0]; green = image_data[1]; red = image_data[2]
    red_edge = image_data[3]; nir = image_data[4]
    # water_vapor = image_data[5]
    swir1 = image_data[6]; swir2 = image_data[7]

    def _safe(a): 
        x = a.astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    blue, green, red, red_edge, nir, swir1, swir2 = map(_safe, (blue, green, red, red_edge, nir, swir1, swir2))

    ndvi = np.where(nir + red != 0, (nir - red) / (nir + red), 0)
    gndvi = np.where(nir + green != 0, (nir - green) / (nir + green), 0)
    rvi = np.where((nir != 0) & (red != 0), nir / red, 0)
    ndre = np.where(nir + red_edge != 0, (nir - red_edge) / (nir + red_edge), 0)
    cire = np.where(red_edge != 0, (nir / red_edge) - 1, 0)
    mcari = np.where(red != 0, ((red_edge - red) - 0.2 * (red_edge - green)) * (red_edge / red), 0)
    L = 0.5
    savi = np.where(nir + red + L != 0, ((nir - red) / (nir + red + L)) * (1 + L), 0)

    vis = [ndvi, gndvi, rvi, ndre, cire, mcari, savi]
    vis = [np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in vis]
    return np.stack(vis, axis=0)

def compute_glcm_features(image: np.ndarray, levels: int = 16, window_size: int = 3) -> np.ndarray:
    """Compute basic GLCM textures for a single-band image.
    Returns 7-channel array: [Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM, Entropy]
    NOTE: This is a compact, CPU-friendly version (no numba dependency).
    """
    import numpy as np
    from skimage.feature import greycomatrix, greycoprops

    img = image.astype(np.float32, copy=False)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.max() > 0:
        img = (img / img.max() * (levels - 1)).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)

    # Use 1-pixel distance, 4 directions
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    props = {
        'contrast': greycoprops(glcm, 'contrast').mean(),
        'dissimilarity': greycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': greycoprops(glcm, 'homogeneity').mean(),
        'energy': greycoprops(glcm, 'energy').mean(),
        'correlation': greycoprops(glcm, 'correlation').mean(),
        'ASM': greycoprops(glcm, 'ASM').mean(),
    }
    # Entropy
    p = glcm / (glcm.sum() + 1e-12)
    entropy = -(p * np.log2(p + 1e-12)).sum()

    H, W = image.shape
    out = np.zeros((7, H, W), dtype=np.float32)
    for i, v in enumerate([props['contrast'], props['dissimilarity'], props['homogeneity'],
                           props['energy'], props['correlation'], props['ASM'], entropy]):
        out[i, :, :] = v
    return out
