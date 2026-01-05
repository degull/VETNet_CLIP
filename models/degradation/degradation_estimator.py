# E:/VETNet_CLIP/models/degradation/degradation_estimator.py
# ============================================================
# Degradation Estimator D(x) for Restoration-aware Controller
# - Lightweight CNN that predicts:
#   (1) Spatial degradation map M: [B, K, H/8, W/8]
#   (2) Global severity vector  v: [B, K]
#
# K (recommended): 5
#   [noise, blur, haze, raindrop, snow]
#
# Notes
# - Designed for Phase-3 end-to-end training (no explicit labels required).
# - Works as a "restoration-friendly" signal (where/what/rough level).
# - Keep it fast: depthwise separable convs + strided downsampling.
# - Optional: return intermediate feature for spatial gating / tokenization.
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Blocks
# ============================================================

class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


class DWConv(nn.Module):
    """Depthwise conv (groups=in_channels)."""
    def __init__(self, c: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c, c, kernel_size=k, stride=s, padding=p, groups=c, bias=False)

    def forward(self, x):
        return self.conv(x)


class PWConv(nn.Module):
    """Pointwise conv (1x1)."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)


class DSConvBlock(nn.Module):
    """
    Depthwise-separable conv block:
      x -> DWConv -> PWConv -> GN -> SiLU
    Optional stride on DWConv for downsampling.
    """
    def __init__(self, c_in: int, c_out: int, stride: int = 1, gn_groups: int = 8):
        super().__init__()
        self.dw = DWConv(c_in, k=3, s=stride, p=1)
        self.pw = PWConv(c_in, c_out)
        # GroupNorm is stable for small batch sizes
        g = min(gn_groups, c_out)
        # ensure divisible
        while c_out % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(num_groups=g, num_channels=c_out)
        self.act = SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class ResidualDSBlock(nn.Module):
    """
    Residual block using DSConvBlock * 2
    If channel mismatch, use 1x1 skip.
    """
    def __init__(self, c_in: int, c_out: int, gn_groups: int = 8):
        super().__init__()
        self.b1 = DSConvBlock(c_in, c_out, stride=1, gn_groups=gn_groups)
        self.b2 = DSConvBlock(c_out, c_out, stride=1, gn_groups=gn_groups)
        self.skip = None
        if c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        s = x if self.skip is None else self.skip(x)
        x = self.b1(x)
        x = self.b2(x)
        return x + s


# ============================================================
# Config
# ============================================================

@dataclass
class DegradationEstimatorConfig:
    num_degradations: int = 5         # K
    base_channels: int = 48           # light but expressive
    gn_groups: int = 8
    downsample_factor: int = 8        # output H/8, W/8
    map_act: str = "sigmoid"          # "sigmoid" (recommended) or "softplus"
    # If you want harder sparsity for spatial maps, you can try softplus.
    # In most cases, sigmoid is easiest for stable gating.

    # Optional regularizers (used in training loop, not inside module):
    # - entropy / sparsity / smoothness on M
    # - consistency losses, etc.


# ============================================================
# Main Module
# ============================================================

class DegradationEstimator(nn.Module):
    """
    D(x) -> (M, v)
      M: spatial degradation map [B, K, H/8, W/8]
      v: global severity        [B, K]
    """
    def __init__(self, cfg: Optional[DegradationEstimatorConfig] = None):
        super().__init__()
        self.cfg = cfg or DegradationEstimatorConfig()

        K = self.cfg.num_degradations
        C0 = self.cfg.base_channels
        G  = self.cfg.gn_groups

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, C0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=self._valid_gn_groups(C0, G), num_channels=C0),
            SiLU(),
        )

        # Downsample to H/2
        self.ds1 = nn.Sequential(
            DSConvBlock(C0, C0, stride=2, gn_groups=G),
            ResidualDSBlock(C0, C0, gn_groups=G),
        )

        # Downsample to H/4
        self.ds2 = nn.Sequential(
            DSConvBlock(C0, C0 * 2, stride=2, gn_groups=G),
            ResidualDSBlock(C0 * 2, C0 * 2, gn_groups=G),
        )

        # Downsample to H/8
        self.ds3 = nn.Sequential(
            DSConvBlock(C0 * 2, C0 * 3, stride=2, gn_groups=G),
            ResidualDSBlock(C0 * 3, C0 * 3, gn_groups=G),
        )

        # Spatial map head: [B, K, H/8, W/8]
        self.map_head = nn.Sequential(
            nn.Conv2d(C0 * 3, C0 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=self._valid_gn_groups(C0 * 2, G), num_channels=C0 * 2),
            SiLU(),
            nn.Conv2d(C0 * 2, K, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # Global severity head:
        # Use pooled feature rather than pooling M (more stable early training).
        self.sev_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sev_head = nn.Sequential(
            nn.Conv2d(C0 * 3, C0 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=self._valid_gn_groups(C0 * 2, G), num_channels=C0 * 2),
            SiLU(),
            nn.Conv2d(C0 * 2, K, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # (Optional) return features for spatial gate/tokenize
        self._out_feature_channels = C0 * 3

    @staticmethod
    def _valid_gn_groups(ch: int, g: int) -> int:
        g = min(g, ch)
        while ch % g != 0 and g > 1:
            g -= 1
        return max(1, g)

    def _apply_map_activation(self, m: torch.Tensor) -> torch.Tensor:
        if self.cfg.map_act.lower() == "sigmoid":
            return torch.sigmoid(m)
        if self.cfg.map_act.lower() == "softplus":
            # positive unbounded; you can normalize outside if needed
            return F.softplus(m)
        raise ValueError(f"Unknown map_act: {self.cfg.map_act}")

    def forward(
        self,
        x: torch.Tensor,
        return_feat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B,3,H,W] in [0,1] (recommended) or [-1,1] (also works)
          return_feat: if True, also returns feat [B,C,H/8,W/8]

        Returns:
          M: [B,K,H/8,W/8]
          v: [B,K]
          (optional) feat: [B,C,H/8,W/8]
        """
        # Feature trunk
        h = self.stem(x)
        h = self.ds1(h)
        h = self.ds2(h)
        feat = self.ds3(h)  # [B, C, H/8, W/8]

        # Map
        m_logits = self.map_head(feat)
        M = self._apply_map_activation(m_logits)

        # Severity
        sev_logits = self.sev_head(self.sev_pool(feat))  # [B,K,1,1]
        v = torch.sigmoid(sev_logits.flatten(1))         # [B,K] in [0,1]

        if return_feat:
            return M, v, feat
        return M, v

    @property
    def out_feature_channels(self) -> int:
        return self._out_feature_channels


# ============================================================
# Quick Self-test
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DegradationEstimatorConfig(num_degradations=5, base_channels=48, downsample_factor=8)
    net = DegradationEstimator(cfg).to(device).eval()

    # Dummy input [B,3,H,W]
    x = torch.rand(2, 3, 256, 256, device=device)
    with torch.no_grad():
        M, v, feat = net(x, return_feat=True)

    print("[DegradationEstimator] M:", tuple(M.shape), "range:", (float(M.min()), float(M.max())))
    print("[DegradationEstimator] v:", tuple(v.shape), "range:", (float(v.min()), float(v.max())))
    print("[DegradationEstimator] feat:", tuple(feat.shape))
