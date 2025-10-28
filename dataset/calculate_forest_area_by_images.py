#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute forest area (ha) per class for random subsets of label images,
using 9-class scheme and per-image pixel area from GeoTIFF metadata.

Outputs a CSV summarizing area coverage for experimental image counts.
"""
import json
import numpy as np
import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm

def compute_per_image_areas(label_files):
    import rasterio
    rows = []
    for label_path in tqdm(label_files, desc='per-image areas'):
        with rasterio.open(label_path) as src:
            label = src.read(1)
            pixel_size = src.res[0] * src.res[1]
            info = {
                'image_name': os.path.basename(label_path),
                'pixel_size_m2': pixel_size
            }
            total_px = 0
            for cls in range(1, 10):
                cnt = int((label == cls).sum())
                info[f'class_{cls}_pixels'] = cnt
                total_px += cnt
            info['total_9species_pixels'] = total_px
            info['total_9species_area_ha'] = (total_px * pixel_size) / 10000.0
            for cls in range(1, 10):
                info[f'class_{cls}_area_ha'] = (info[f'class_{cls}_pixels'] * pixel_size) / 10000.0
            rows.append(info)
    return pd.DataFrame(rows)

def summarize_by_image_counts(df, total_images, counts):
    results = []
    rng = np.random.default_rng(42)
    for n in counts:
        n = min(n, total_images)
        idx = rng.choice(len(df), n, replace=False)
        sdf = df.iloc[idx]
        rec = {
            'num_images': n,
            'num_patches': n * 4 * 5,  # consistent with augmentation
            'total_9species_area_ha': float(sdf['total_9species_area_ha'].sum())
        }
        for cls in range(1, 10):
            rec[f'class_{cls}_area_ha'] = float(sdf[f'class_{cls}_area_ha'].sum())
        results.append(rec)
    return pd.DataFrame(results)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=True, help='Path to *_image_metadata.json')
    ap.add_argument('--labels-folder', required=True, help='Folder with label rasters (*.tif)')
    ap.add_argument('--out-csv', default='forest_area_by_images.csv')
    args = ap.parse_args()

    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    total_images = metadata.get('total_images', 0)

    label_files = glob.glob(os.path.join(args.labels_folder, '*.tif'))
    label_files.extend(glob.glob(os.path.join(args.labels_folder, '*.TIF')))
    label_files = sorted(label_files)

    df = compute_per_image_areas(label_files)

    # choose experimental counts
    counts = [n for n in [1,2,3,5,10,20,40,60,80,100,120,140,160,180] if n <= total_images]
    x = 200
    while x <= total_images:
        counts.append(x)
        x += 50
    if total_images not in counts:
        counts.append(total_images)

    out = summarize_by_image_counts(df, total_images, counts)
    out.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f'Saved: {args.out_csv}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
