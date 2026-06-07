"""
loss.py  —  Weighted loss functions for imbalanced regression.
Mirrors the original DIR paper's loss.py.
"""

import torch
import torch.nn.functional as F


def weighted_mse_loss(pred, target, weight):
    loss = F.mse_loss(pred, target, reduction='none')   # (B,)
    return (loss * weight).mean()


def weighted_l1_loss(pred, target, weight):
    loss = F.l1_loss(pred, target, reduction='none')    # (B,)
    return (loss * weight).mean()


def weighted_huber_loss(pred, target, weight, delta=1.0):
    loss = F.huber_loss(pred, target, reduction='none', delta=delta)
    return (loss * weight).mean()


def weighted_focal_l1_loss(pred, target, weight, gamma=1.0):
    """Focal variant that up-weights hard (high-error) samples."""
    err  = torch.abs(pred - target)
    loss = err * (1 + err) ** gamma
    return (loss * weight).mean()


def weighted_focal_mse_loss(pred, target, weight, gamma=1.0):
    err  = (pred - target) ** 2
    loss = err * (1 + err) ** gamma
    return (loss * weight).mean()
