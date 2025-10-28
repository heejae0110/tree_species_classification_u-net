# tree_species_classification_u-net

# Algorithm_9sp_core — Minimal Reproducible Core

This repository provides a minimal, clean subset of the author's research code for
tree species mapping with multi-season Sentinel-2 inputs and U-Net segmentation.
It is designed to support peer-review reproducibility without exposing full data
pipelines, internal paths, or licensed datasets.

## What's included
- `GEE/download_sentinel2.txt` — code for download sentinel-2 imagery in google earth engine.
- `dataset/add_image_metadata.py` — build per-image patch metadata aligned with augmentation.
- `dataset/calculate_forest_area_by_images.py` — compute class-wise area coverage for random subsets.
- `model/unet_model.py` — compact U-Net (66-channel input, N-class output).
- `model/dataset_class.py` — memory-mapped patch dataset and independent-test reader.
- `feature/vegetation_indices_glcm.py` — 7 vegetation indices and GLCM calculater.
