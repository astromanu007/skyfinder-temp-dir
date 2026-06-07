"""
model.py  —  ResNet-18 regression model with optional FDS support.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Regressor(nn.Module):
    def __init__(self, pretrained=True, fds_module=None):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_dim = backbone.fc.in_features   # 512
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(self.feature_dim, 1)
        self.fds = fds_module

    def forward(self, x, targets=None, epoch=None):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)   # (B, 512)

        if self.fds is not None and self.training and targets is not None and epoch is not None:
            feat_for_output = self.fds.smooth(feat, targets, epoch)
        else:
            feat_for_output = feat

        out = self.fc(feat_for_output).squeeze(1)   # (B,)

        if self.fds is not None and self.training:
            return out, feat   # exactly 2 values always

        return out
