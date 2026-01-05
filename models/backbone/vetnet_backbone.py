# E:\VETNet_CLIP\models\backbone\vetnet_backbone.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.blocks import VETBlock


# ============================================================
# Basic Ops
# ============================================================

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(
            in_channels, in_channels * 2,
            kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# ============================================================
# Encoder / Decoder Stages
# ============================================================

class EncoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads,
                 volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(
                dim=dim,
                num_heads=num_heads,
                volterra_rank=volterra_rank,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias
            )
            for _ in range(depth)
        ])

    def forward(self, x, gamma=None, beta=None):
        for blk in self.blocks:
            x = blk(x, gamma=gamma, beta=beta)
        return x


class DecoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads,
                 volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(
                dim=dim,
                num_heads=num_heads,
                volterra_rank=volterra_rank,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias
            )
            for _ in range(depth)
        ])

    def forward(self, x, gamma=None, beta=None):
        for blk in self.blocks:
            x = blk(x, gamma=gamma, beta=beta)
        return x


# ============================================================
# VETNet Backbone (FiLM + Stage Gate)
# ============================================================

class VETNetBackbone(nn.Module):
    """
    Macro stages (S=8):
      0 encoder1
      1 encoder2
      2 encoder3
      3 latent
      4 decoder3
      5 decoder2
      6 decoder1
      7 refinement
    """
    NUM_MACRO_STAGES = 8

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=2,
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=3, stride=1, padding=1
        )

        # ---------------- Encoder ----------------
        self.encoder1 = EncoderStage(dim, num_blocks[0], heads[0],
                                     volterra_rank, ffn_expansion_factor, bias)
        self.down1 = Downsample(dim)

        self.encoder2 = EncoderStage(dim * 2, num_blocks[1], heads[1],
                                     volterra_rank, ffn_expansion_factor, bias)
        self.down2 = Downsample(dim * 2)

        self.encoder3 = EncoderStage(dim * 4, num_blocks[2], heads[2],
                                     volterra_rank, ffn_expansion_factor, bias)
        self.down3 = Downsample(dim * 4)

        # ---------------- Bottleneck ----------------
        self.latent = EncoderStage(dim * 8, num_blocks[3], heads[3],
                                   volterra_rank, ffn_expansion_factor, bias)

        # ---------------- Decoder ----------------
        self.up3 = Upsample(dim * 8, dim * 4)
        self.decoder3 = DecoderStage(dim * 4, num_blocks[2], heads[2],
                                     volterra_rank, ffn_expansion_factor, bias)

        self.up2 = Upsample(dim * 4, dim * 2)
        self.decoder2 = DecoderStage(dim * 2, num_blocks[1], heads[1],
                                     volterra_rank, ffn_expansion_factor, bias)

        self.up1 = Upsample(dim * 2, dim)
        self.decoder1 = DecoderStage(dim, num_blocks[0], heads[0],
                                     volterra_rank, ffn_expansion_factor, bias)

        # ---------------- Refinement ----------------
        self.refinement = EncoderStage(dim, num_blocks[0], heads[0],
                                       volterra_rank, ffn_expansion_factor, bias)

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _pad_and_add(x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return x + skip

    @staticmethod
    def _apply_stage_gate(F, g):
        """
        F : [B,C,H,W]
        g : [B,1,1,1]
        """
        return F * g

    def _normalize_g_stage(self, g_stage, B, device, dtype):
        if g_stage is None:
            return None
        if isinstance(g_stage, (list, tuple)):
            g_stage = torch.tensor(g_stage, device=device, dtype=dtype)
        if g_stage.dim() == 1:
            g_stage = g_stage.unsqueeze(0).repeat(B, 1)
        assert g_stage.shape == (B, 8)
        return g_stage

    def _normalize_film(self, film, B, device, dtype):
        if film is None:
            return None
        gammas = film["gammas"]
        betas = film["betas"]
        out_g, out_b = [], []
        for g, b in zip(gammas, betas):
            if g.size(0) == 1:
                g = g.repeat(B, 1, 1, 1)
            if b.size(0) == 1:
                b = b.repeat(B, 1, 1, 1)
            out_g.append(g.to(device=device, dtype=dtype))
            out_b.append(b.to(device=device, dtype=dtype))
        return {"gammas": out_g, "betas": out_b}

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x, g_stage=None, film=None):
        """
        x       : [B,3,H,W]
        g_stage : [B,8] or None
        film    : {"gammas":[8], "betas":[8]} or None
        """
        B = x.size(0)
        device, dtype = x.device, x.dtype

        g_stage = self._normalize_g_stage(g_stage, B, device, dtype)
        film = self._normalize_film(film, B, device, dtype)

        gammas = film["gammas"] if film is not None else [None] * 8
        betas  = film["betas"]  if film is not None else [None] * 8

        x0 = self.patch_embed(x)

        # ---------------- Encoder ----------------
        e1 = self.encoder1(x0, gammas[0], betas[0])
        if g_stage is not None:
            e1 = self._apply_stage_gate(e1, g_stage[:, 0].view(B, 1, 1, 1))

        e2_in = self.down1(e1)
        e2 = self.encoder2(e2_in, gammas[1], betas[1])
        if g_stage is not None:
            e2 = self._apply_stage_gate(e2, g_stage[:, 1].view(B, 1, 1, 1))

        e3_in = self.down2(e2)
        e3 = self.encoder3(e3_in, gammas[2], betas[2])
        if g_stage is not None:
            e3 = self._apply_stage_gate(e3, g_stage[:, 2].view(B, 1, 1, 1))

        # ---------------- Latent ----------------
        b_in = self.down3(e3)
        b = self.latent(b_in, gammas[3], betas[3])
        if g_stage is not None:
            b = self._apply_stage_gate(b, g_stage[:, 3].view(B, 1, 1, 1))

        # ---------------- Decoder ----------------
        d3_in = self._pad_and_add(self.up3(b), e3)
        d3 = self.decoder3(d3_in, gammas[4], betas[4])
        if g_stage is not None:
            d3 = self._apply_stage_gate(d3, g_stage[:, 4].view(B, 1, 1, 1))

        d2_in = self._pad_and_add(self.up2(d3), e2)
        d2 = self.decoder2(d2_in, gammas[5], betas[5])
        if g_stage is not None:
            d2 = self._apply_stage_gate(d2, g_stage[:, 5].view(B, 1, 1, 1))

        d1_in = self._pad_and_add(self.up1(d2), e1)
        d1 = self.decoder1(d1_in, gammas[6], betas[6])
        if g_stage is not None:
            d1 = self._apply_stage_gate(d1, g_stage[:, 6].view(B, 1, 1, 1))

        # ---------------- Refinement ----------------
        r = self.refinement(d1, gammas[7], betas[7])
        if g_stage is not None:
            r = self._apply_stage_gate(r, g_stage[:, 7].view(B, 1, 1, 1))

        out = self.output(r + x0)
        return out


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 256, 256).to(device)

    model = VETNetBackbone(dim=64).to(device).eval()

    g = torch.full((2, 8), 0.9, device=device)
    film = {
        "gammas": [torch.ones(1, c, 1, 1, device=device)
                   for c in [64, 128, 256, 512, 256, 128, 64, 64]],
        "betas": [torch.zeros(1, c, 1, 1, device=device)
                  for c in [64, 128, 256, 512, 256, 128, 64, 64]],
    }

    with torch.no_grad():
        y0 = model(x)
        y1 = model(x, g_stage=g, film=film)

    print("Output:", y0.shape, y1.shape,
          "Mean|diff|:", (y0 - y1).abs().mean().item())
