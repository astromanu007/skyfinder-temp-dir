"""
fds.py  —  Feature Distribution Smoothing (FDS)
Directly adapted from the original DIR paper implementation.
Reference: https://github.com/YyzHarry/imbalanced-regression
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import convolve1d
from scipy.stats import norm, laplace


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        kernel_window = norm.pdf(np.arange(0, ks), (ks - 1) / 2, sigma)
        kernel_window /= max(kernel_window)
    elif kernel == 'triang':
        kernel_window = np.array(
            [1 - abs(x - half_ks) / (half_ks + 1) for x in range(ks)]
        )
    else:  # laplace
        kernel_window = laplace.pdf(np.arange(0, ks), (ks - 1) / 2, sigma)
        kernel_window /= max(kernel_window)
    return kernel_window


class FDS(nn.Module):
    """
    Feature Distribution Smoothing.

    Maintains running per-bucket mean & variance of features,
    then smooths them across adjacent label buckets so that
    rare-label samples borrow statistical strength from nearby
    common-label samples.

    Buckets: integer degrees from -40°C to +55°C (96 buckets).
    """

    def __init__(self,
                 feature_dim,
                 bucket_num=96,
                 bucket_start=0,
                 start_update=0,
                 start_smooth=1,
                 kernel='gaussian',
                 ks=5,
                 sigma=1,
                 momentum=0.9):
        super().__init__()

        self.feature_dim  = feature_dim
        self.bucket_num   = bucket_num
        self.bucket_start = bucket_start
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.momentum     = momentum
        self.kernel       = kernel

        self.lds_kernel_window = torch.tensor(
            get_lds_kernel_window(kernel, ks, sigma),
            dtype=torch.float32
        )

        # Running statistics (not parameters — no gradient)
        self.register_buffer('running_mean',      torch.zeros(bucket_num, feature_dim))
        self.register_buffer('running_var',       torch.ones(bucket_num, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num, feature_dim))
        self.register_buffer('running_var_last_epoch',  torch.ones(bucket_num, feature_dim))
        self.register_buffer('smoothed_mean',     torch.zeros(bucket_num, feature_dim))
        self.register_buffer('smoothed_var',      torch.ones(bucket_num, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num))

    def _label_to_bucket(self, labels):
        """Convert temperature (°C) to bucket index."""
        return (labels.round().long() + 40).clamp(self.bucket_start, self.bucket_num - 1)

    @torch.no_grad()
    def update_last_epoch_stats(self, epoch):
        self.running_mean_last_epoch = self.running_mean.clone()
        self.running_var_last_epoch  = self.running_var.clone()

    @torch.no_grad()
    def update_running_stats(self, features, labels, epoch):
        if epoch < self.start_update:
            return

        bucket_ids = self._label_to_bucket(labels)

        for b in bucket_ids.unique():
            b = b.item()
            mask = (bucket_ids == b)
            feat_b = features[mask]                       # (N_b, D)

            # EMA update of mean
            mean_b = feat_b.mean(dim=0)
            self.running_mean[b] = (
                self.momentum * self.running_mean_last_epoch[b]
                + (1 - self.momentum) * mean_b
            )

            # EMA update of variance (use std dev across samples)
            if feat_b.size(0) > 1:
                var_b = feat_b.var(dim=0, unbiased=True)
            else:
                var_b = torch.zeros(self.feature_dim, device=features.device)
            self.running_var[b] = (
                self.momentum * self.running_var_last_epoch[b]
                + (1 - self.momentum) * var_b
            )

            self.num_samples_tracked[b] += mask.sum().item()

        # After collecting, smooth across buckets
        self._smooth_stats()

    def _smooth_stats(self):
        """Convolve running stats across adjacent buckets."""
        kernel = self.lds_kernel_window.numpy()

        mean_np = self.running_mean.cpu().numpy()   # (B, D)
        var_np  = self.running_var.cpu().numpy()

        smoothed_mean = np.stack(
            [convolve1d(mean_np[:, d], weights=kernel, mode='reflect')
             for d in range(self.feature_dim)], axis=1
        )
        smoothed_var = np.stack(
            [convolve1d(var_np[:, d], weights=kernel, mode='reflect')
             for d in range(self.feature_dim)], axis=1
        )

        self.smoothed_mean = torch.tensor(smoothed_mean, dtype=torch.float32,
                                          device=self.running_mean.device)
        self.smoothed_var  = torch.tensor(smoothed_var,  dtype=torch.float32,
                                          device=self.running_var.device)

    def smooth(self, features, labels, epoch):
        """
        Re-calibrate features using smoothed statistics.
        z = (x - running_mean) / sqrt(running_var)  →  z * sqrt(smoothed_var) + smoothed_mean
        """
        if epoch < self.start_smooth:
            return features

        bucket_ids = self._label_to_bucket(labels)

        smoothed = features.clone()
        for b in bucket_ids.unique():
            b_val = b.item()
            mask  = (bucket_ids == b)

            mean_b = self.running_mean[b_val].to(features.device)
            var_b  = self.running_var[b_val].to(features.device)
            s_mean = self.smoothed_mean[b_val].to(features.device)
            s_var  = self.smoothed_var[b_val].to(features.device)

            # Normalise then rescale
            feat_b = features[mask]
            feat_b = (feat_b - mean_b) / (var_b.sqrt() + 1e-8)
            feat_b = feat_b * (s_var.sqrt() + 1e-8) + s_mean
            smoothed[mask] = feat_b

        return smoothed
