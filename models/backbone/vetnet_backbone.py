# E:\VETNet_CLIP\models\backbone\vetnet_backbone.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.blocks import VETBlock


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2,
                              kernel_size=3, stride=2, padding=1)

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


class EncoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(dim=dim,
                     num_heads=num_heads,
                     volterra_rank=volterra_rank,
                     ffn_expansion_factor=ffn_expansion_factor,
                     bias=bias)
            for _ in range(depth)
        ])

    def forward(self, x, gamma=None, beta=None):
        for blk in self.blocks:
            x = blk(x, gamma=gamma, beta=beta)
        return x


class DecoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, volterra_rank, ffn_expansion_factor, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            VETBlock(dim=dim,
                     num_heads=num_heads,
                     volterra_rank=volterra_rank,
                     ffn_expansion_factor=ffn_expansion_factor,
                     bias=bias)
            for _ in range(depth)
        ])

    def forward(self, x, gamma=None, beta=None):
        for blk in self.blocks:
            x = blk(x, gamma=gamma, beta=beta)
        return x


class VETNetBackbone(nn.Module):
    NUM_MACRO_STAGES = 8

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        heads=(1, 2, 4, 8),
        volterra_rank=2,
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)

        # Encoder
        self.encoder1 = EncoderStage(dim=dim, depth=num_blocks[0], num_heads=heads[0],
                                     volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.down1 = Downsample(dim)

        self.encoder2 = EncoderStage(dim=dim * 2, depth=num_blocks[1], num_heads=heads[1],
                                     volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.down2 = Downsample(dim * 2)

        self.encoder3 = EncoderStage(dim=dim * 4, depth=num_blocks[2], num_heads=heads[2],
                                     volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.down3 = Downsample(dim * 4)

        # Bottleneck
        self.latent = EncoderStage(dim=dim * 8, depth=num_blocks[3], num_heads=heads[3],
                                   volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        # Decoder
        self.up3 = Upsample(dim * 8, dim * 4)
        self.decoder3 = DecoderStage(dim=dim * 4, depth=num_blocks[2], num_heads=heads[2],
                                     volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.up2 = Upsample(dim * 4, dim * 2)
        self.decoder2 = DecoderStage(dim=dim * 2, depth=num_blocks[1], num_heads=heads[1],
                                     volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.up1 = Upsample(dim * 2, dim)
        self.decoder1 = DecoderStage(dim=dim, depth=num_blocks[0], num_heads=heads[0],
                                     volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        # Refinement
        self.refinement = EncoderStage(dim=dim, depth=num_blocks[0], num_heads=heads[0],
                                       volterra_rank=volterra_rank, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _pad_and_add(up_tensor, skip_tensor):
        if up_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:],
                                      mode="bilinear", align_corners=False)
        return up_tensor + skip_tensor

    @staticmethod
    def _apply_gate(x_in: torch.Tensor, x_out: torch.Tensor, g: torch.Tensor):
        return x_in + g * (x_out - x_in)

    def _normalize_gates(self, g_stage, batch_size, device, dtype):
        if g_stage is None:
            return None
        if isinstance(g_stage, (list, tuple)):
            g_stage = torch.tensor(g_stage, device=device, dtype=dtype)
        if not torch.is_tensor(g_stage):
            raise TypeError(f"g_stage must be Tensor/list/tuple/None, got {type(g_stage)}")
        if g_stage.dim() == 1:
            g_stage = g_stage.view(1, -1).repeat(batch_size, 1)
        elif g_stage.dim() != 2:
            raise ValueError(f"g_stage must be [S] or [B,S], got shape={tuple(g_stage.shape)}")
        if g_stage.size(1) != self.NUM_MACRO_STAGES:
            raise ValueError(f"Expected g_stage with S={self.NUM_MACRO_STAGES}, got {g_stage.size(1)}")
        return g_stage.to(device=device, dtype=dtype)

    def _normalize_film(self, film, B, device, dtype):
        """
        film: None or dict {"gammas": list(8), "betas": list(8)}
        each gamma/beta should be [B,C,1,1] or [1,C,1,1] (broadcastable)
        """
        if film is None:
            return None

        if not isinstance(film, dict):
            raise TypeError("film must be dict or None")

        gammas = film.get("gammas", None)
        betas = film.get("betas", None)

        if (gammas is None) or (betas is None):
            raise ValueError("film dict must have keys 'gammas' and 'betas'")

        if (len(gammas) != 8) or (len(betas) != 8):
            raise ValueError("film['gammas'] and film['betas'] must be length 8")

        out_g, out_b = [], []
        for g, b in zip(gammas, betas):
            if g.dim() == 4 and g.size(0) == 1:
                g = g.repeat(B, 1, 1, 1)
            if b.dim() == 4 and b.size(0) == 1:
                b = b.repeat(B, 1, 1, 1)
            out_g.append(g.to(device=device, dtype=dtype))
            out_b.append(b.to(device=device, dtype=dtype))

        return {"gammas": out_g, "betas": out_b}

    def forward(self, x, g_stage=None, film=None):
        """
        Args:
            x: [B,3,H,W]
            g_stage: [B,8] or [8] (optional)
            film: dict {"gammas":[...8], "betas":[...8]} (optional)
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        g_stage = self._normalize_gates(g_stage, B, device, dtype)
        film = self._normalize_film(film, B, device, dtype)

        gammas = film["gammas"] if film is not None else [None] * 8
        betas = film["betas"] if film is not None else [None] * 8

        x_embed = self.patch_embed(x)

        # stage 0: encoder1
        x_in = x_embed
        e1 = self.encoder1(x_in, gamma=gammas[0], beta=betas[0])
        if g_stage is not None:
            e1 = self._apply_gate(x_in, e1, g_stage[:, 0].view(B, 1, 1, 1))

        # stage 1: encoder2
        x_in = self.down1(e1)
        e2 = self.encoder2(x_in, gamma=gammas[1], beta=betas[1])
        if g_stage is not None:
            e2 = self._apply_gate(x_in, e2, g_stage[:, 1].view(B, 1, 1, 1))

        # stage 2: encoder3
        x_in = self.down2(e2)
        e3 = self.encoder3(x_in, gamma=gammas[2], beta=betas[2])
        if g_stage is not None:
            e3 = self._apply_gate(x_in, e3, g_stage[:, 2].view(B, 1, 1, 1))

        # stage 3: latent
        x_in = self.down3(e3)
        b = self.latent(x_in, gamma=gammas[3], beta=betas[3])
        if g_stage is not None:
            b = self._apply_gate(x_in, b, g_stage[:, 3].view(B, 1, 1, 1))

        # stage 4: decoder3
        d3_in = self._pad_and_add(self.up3(b), e3)
        d3 = self.decoder3(d3_in, gamma=gammas[4], beta=betas[4])
        if g_stage is not None:
            d3 = self._apply_gate(d3_in, d3, g_stage[:, 4].view(B, 1, 1, 1))

        # stage 5: decoder2
        d2_in = self._pad_and_add(self.up2(d3), e2)
        d2 = self.decoder2(d2_in, gamma=gammas[5], beta=betas[5])
        if g_stage is not None:
            d2 = self._apply_gate(d2_in, d2, g_stage[:, 5].view(B, 1, 1, 1))

        # stage 6: decoder1
        d1_in = self._pad_and_add(self.up1(d2), e1)
        d1 = self.decoder1(d1_in, gamma=gammas[6], beta=betas[6])
        if g_stage is not None:
            d1 = self._apply_gate(d1_in, d1, g_stage[:, 6].view(B, 1, 1, 1))

        # stage 7: refinement
        r_in = d1
        r = self.refinement(r_in, gamma=gammas[7], beta=betas[7])
        if g_stage is not None:
            r = self._apply_gate(r_in, r, g_stage[:, 7].view(B, 1, 1, 1))

        out = self.output(r + x_embed)
        return out


if __name__ == "__main__":
    # quick smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 256, 256).to(device)

    model = VETNetBackbone(in_channels=3, out_channels=3, dim=64).to(device).eval()

    # dummy gates + dummy FiLM identity
    g_stage = torch.full((2, 8), 0.9, device=device)
    film = {
        "gammas": [torch.ones(2, c, 1, 1, device=device) for c in [64,128,256,512,256,128,64,64]],
        "betas":  [torch.zeros(2, c, 1, 1, device=device) for c in [64,128,256,512,256,128,64,64]],
    }

    with torch.no_grad():
        y0 = model(x)
        y1 = model(x, g_stage=g_stage, film=film)

    print("Shapes:", y0.shape, y1.shape, "Mean|diff|:", (y0-y1).abs().mean().item())
