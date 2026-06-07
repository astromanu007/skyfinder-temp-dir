"""
datasets.py  —  SkyFinder temperature prediction dataset
Adapted from imdb-wiki-dir/datasets.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.stats import norm, laplace

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── LDS helpers (identical logic to original DIR paper) ─────────────────────

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = norm.pdf(np.arange(0, ks), (ks - 1) / 2, sigma)
        kernel_window /= max(kernel_window)
    elif kernel == 'triang':
        kernel_window = np.array(
            [1 - abs(x - half_ks) / (half_ks + 1) for x in range(ks)]
        )
    else:  # laplace
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = laplace.pdf(np.arange(0, ks), (ks - 1) / 2, sigma)
        kernel_window /= max(kernel_window)
    return kernel_window


def prepare_weights(labels, reweight, max_target=50, lds=False,
                    lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    """
    Compute per-sample loss weights.
    Temperatures are binned into 1°C buckets from -40 to +55.
    """
    assert reweight in {'none', 'inverse', 'sqrt_inv'}

    # --- bin temperatures into integer degrees -----------------------
    bin_index_per_label = [int(np.round(l)) + 40 for l in labels]   # shift so -40 → 0
    num_bins = 96   # -40..55  →  96 buckets

    Nb = torch.zeros(num_bins)
    for b in bin_index_per_label:
        if 0 <= b < num_bins:
            Nb[b] += 1

    num_per_label = np.array([Nb[b].item() if 0 <= b < num_bins else 0
                              for b in bin_index_per_label], dtype=float)

    if reweight == 'none':
        weights = np.ones_like(num_per_label)
    elif reweight == 'inverse':
        weights = 1.0 / (num_per_label + 1e-5)
    else:  # sqrt_inv
        weights = 1.0 / np.sqrt(num_per_label + 1e-5)

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        # Smooth the empirical label density
        smoothed_value = convolve1d(Nb.numpy(), weights=lds_kernel_window, mode='reflect')
        # Map smoothed density back to each sample
        num_per_label_smoothed = np.array(
            [smoothed_value[b] if 0 <= b < num_bins else 1.0
             for b in bin_index_per_label], dtype=float
        )
        if reweight == 'none':
            weights = np.ones_like(num_per_label_smoothed)
        elif reweight == 'inverse':
            weights = 1.0 / (num_per_label_smoothed + 1e-5)
        else:
            weights = 1.0 / np.sqrt(num_per_label_smoothed + 1e-5)

    # Normalise so mean weight = 1
    weights = weights / weights.mean()
    return weights


# ── Dataset ──────────────────────────────────────────────────────────────────

class SkyFinder(Dataset):
    """
    SkyFinder dataset for temperature regression.

    CSV must have columns: img_path, temperature, split
    """

    def __init__(self, df, img_size=224, split='train',
                 reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df    = df.reset_index(drop=True)
        self.split = split

        # Compute sample weights (only needed for training)
        if split == 'train' and reweight != 'none':
            self.weights = prepare_weights(
                self.df['temperature'].values, reweight,
                lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma
            )
        elif split == 'train' and lds:
            self.weights = prepare_weights(
                self.df['temperature'].values, 'sqrt_inv',
                lds=True, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma
            )
        else:
            self.weights = np.ones(len(self.df))

        # Transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row['img_path']
        temp = float(row['temperature'])

        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Return a blank image on read error so training doesn't crash
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        img    = self.transform(img)
        target = torch.tensor(temp, dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)

        return img, target, weight
