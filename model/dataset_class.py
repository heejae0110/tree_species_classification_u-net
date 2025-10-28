#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal dataset classes for memory-mapped patches and independent test tiles."""
import json, os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

class ImageBasedMemoryMapDataset(Dataset):
    """Patch dataset backed by NumPy memmap arrays.
    Assumes *_image_metadata.json and *_shapes.json next to memmap files.
    """
    def __init__(self, data_dat, label_dat, metadata_json, num_images,
                 train_ratio=0.8, mode='train', random_seed=73, normalization_stats_path=None):
        self.mode = mode
        self.random_seed = int(random_seed)

        with open(metadata_json, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        shapes_path = metadata_json.replace('_image_metadata.json', '_shapes.json')
        with open(shapes_path, 'r', encoding='utf-8') as f:
            shape_info = json.load(f)

        if '_train_X.dat' in data_dat:
            self.data_shape = tuple(shape_info['train_X_shape'])
            self.label_shape = tuple(shape_info['train_y_shape'])
        else:
            self.data_shape = tuple(shape_info['test_X_shape'])
            self.label_shape = tuple(shape_info['test_y_shape'])

        self.data = np.memmap(data_dat, dtype='float32', mode='r', shape=self.data_shape)
        self.labels = np.memmap(label_dat, dtype='float32', mode='r', shape=self.label_shape)

        self.patch_indices = self._select_patch_indices(num_images, train_ratio)

        self.normalizer = None
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            with open(normalization_stats_path, 'r', encoding='utf-8') as f:
                self.normalizer = json.load(f).get('stats', None)

    def _select_patch_indices(self, num_images, train_ratio):
        rng = np.random.default_rng(self.random_seed)
        total_images = len(self.metadata['image_metadata'])
        sel = rng.choice(total_images, min(num_images, total_images), replace=False)

        idxs = []
        for img_idx in sel:
            info = self.metadata['image_metadata'][img_idx]
            idxs.extend(range(info['augmented_patch_start_idx'], info['augmented_patch_end_idx']))

        rng.shuffle(idxs)
        split = int(len(idxs) * train_ratio)
        return idxs[:split] if self.mode == 'train' else idxs[split:]

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, i):
        j = self.patch_indices[i]
        x = self.data[j].copy()
        y = self.labels[j].copy().astype(np.int32)
        x = np.nan_to_num(x, nan=0.0); x[x == -9999] = 0
        y = np.nan_to_num(y, nan=-100.0); y[(y == 0) | (y == -9999)] = -100

        xt = torch.from_numpy(x).float()
        yt = torch.from_numpy(y).long()

        if self.normalizer:
            for c in range(xt.shape[0]):
                m = (xt[c] != 0) & (~torch.isnan(xt[c]))
                key = str(c)
                if m.any() and key in self.normalizer:
                    p2, p98 = self.normalizer[key]['p2'], self.normalizer[key]['p98']
                    if p98 > p2:
                        cl = torch.clamp(xt[c], p2, p98)
                        xt[c] = torch.where(m, (cl - p2) / (p98 - p2), torch.tensor(0.0))

        return xt, yt

class IndependentTestDataset(Dataset):
    """Reads independent test patches saved as GeoTIFFs in eval_folder/patches."""
    def __init__(self, eval_folder, normalization_stats_path=None):
        self.samples = []
        pdir = os.path.join(eval_folder, 'patches')
        ims = sorted(glob.glob(os.path.join(pdir, '*_image.tif')))
        for ip in ims:
            lp = ip.replace('_image.tif', '_label.tif')
            if os.path.exists(lp): self.samples.append((ip, lp))
        self.normalizer = None
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            with open(normalization_stats_path, 'r', encoding='utf-8') as f:
                self.normalizer = json.load(f).get('stats', None)
        if not self.samples:
            raise FileNotFoundError('No *_image.tif / *_label.tif pairs found in eval_folder/patches')

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ip, lp = self.samples[i]
        with rasterio.open(ip) as src: x = src.read().astype(np.float32)
        with rasterio.open(lp) as src: y = src.read(1).astype(np.int32)
        x = np.nan_to_num(x, nan=0.0); x[x == -9999] = 0
        y[y == 0] = -100
        xt = torch.from_numpy(x).float(); yt = torch.from_numpy(y).long()
        if self.normalizer:
            for c in range(xt.shape[0]):
                m = (xt[c] != 0) & (~torch.isnan(xt[c]))
                key = str(c)
                if m.any() and key in self.normalizer:
                    p2, p98 = self.normalizer[key]['p2'], self.normalizer[key]['p98']
                    if p98 > p2:
                        cl = torch.clamp(xt[c], p2, p98)
                        xt[c] = torch.where(m, (cl - p2) / (p98 - p2), torch.tensor(0.0))
        return xt, yt
