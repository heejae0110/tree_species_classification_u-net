#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create per-image metadata for patch indexing and augmentation bookkeeping.

This script is a cleaned, self-contained version derived from the author's
research pipeline. It reads a *_shapes.json (with train_X_shape) and a folder
of imsang raster labels, and writes an *_image_metadata.json file that maps
images to their (augmented) patch index ranges.

Inputs:
  --dataset-prefix   Path prefix for the dataset (without extension), which has
                     a JSON like <prefix>_shapes.json
  --imsang-folder    Folder that contains *.tif / *.TIF label rasters

Output:
  <dataset-prefix>_image_metadata.json
"""

import json
import glob
import os
import argparse

def create_image_metadata(dataset_prefix: str, imsang_folder: str, patches_per_image: int = 4, aug_factor: int = 5) -> str:
    imsang_files = glob.glob(os.path.join(imsang_folder, "*.tif"))
    if not imsang_files:
        imsang_files = glob.glob(os.path.join(imsang_folder, "*.TIF"))
    imsang_files = sorted(imsang_files)

    shapes_file = f"{dataset_prefix}_shapes.json"
    with open(shapes_file, 'r', encoding='utf-8') as f:
        shape_info = json.load(f)

    total_augmented = int(shape_info['train_X_shape'][0])
    total_original_patches = total_augmented // aug_factor

    image_metadata = []
    for i, imsang_path in enumerate(imsang_files):
        image_name = os.path.basename(imsang_path).rsplit('.', 1)[0]

        # original (pre-augmentation) indices
        patch_start_idx = i * patches_per_image
        patch_end_idx = (i + 1) * patches_per_image

        # post-augmentation indices
        augmented_start_idx = patch_start_idx * aug_factor
        augmented_end_idx = patch_end_idx * aug_factor

        image_metadata.append({
            'image_id': i,
            'image_name': image_name,
            'image_path': imsang_path,
            'num_patches': patches_per_image,
            'patch_start_idx': patch_start_idx,
            'patch_end_idx': patch_end_idx,
            'augmented_patch_start_idx': augmented_start_idx,
            'augmented_patch_end_idx': augmented_end_idx,
            'augmented_num_patches': patches_per_image * aug_factor
        })

    metadata = {
        'total_images': len(imsang_files),
        'total_original_patches': total_original_patches,
        'total_augmented_patches': total_augmented,
        'patches_per_image': patches_per_image,
        'augmentation_factor': aug_factor,
        'augmentation_info': 'original + vflip + hflip + rot90 + rot-90',
        'image_metadata': image_metadata,
        'class_info': {'classes': list(range(1, 10)), 'excluded_classes': [0]}
    }

    metadata_file = f"{dataset_prefix}_image_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-prefix', required=True, help='Path prefix for *_shapes.json')
    ap.add_argument('--imsang-folder', required=True, help='Folder containing imsang label rasters')
    ap.add_argument('--patches-per-image', type=int, default=4)
    ap.add_argument('--aug-factor', type=int, default=5)
    args = ap.parse_args()

    out = create_image_metadata(args.dataset_prefix, args.imsang_folder,
                                patches_per_image=args.patches_per_image,
                                aug_factor=args.aug_factor)
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()
