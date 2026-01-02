# E:\VETNet_CLIP\models\backbone\blocks.py
# phase -1 (vetnet backbone) + (phase-3) FiLM injection
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

from models.backbone.mdta_strategy import MDTA
from models.backbone.gdfn_volterra import GDFN
from models.backbone.volterra_layer import VolterraLayer2D


class VETBlock(nn.Module):
    def __init__(self, dim, num_heads=8, volterra_rank=4, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim=dim, num_heads=num_heads, bias=bias)
        self.volterra = VolterraLayer2D(in_channels=dim, out_channels=dim,
                                        kernel_size=3, rank=volterra_rank, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(dim=dim, expansion_factor=ffn_expansion_factor, bias=bias)

    @staticmethod
    def _apply_film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        # h: [B,C,H,W], gamma/beta: [B,C,1,1]
        return gamma * h + beta

    def forward(self, x, gamma=None, beta=None):
        """
        Args:
            x: [B,C,H,W]
            gamma/beta: optional FiLM params [B,C,1,1]
        """
        b, c, h, w = x.shape

        # ---- Global branch: LN -> (FiLM) -> MDTA -> Volterra ----
        x_ln = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_ln = self.norm1(x_ln)
        x_ln = x_ln.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B,C,H,W]

        if (gamma is not None) and (beta is not None):
            x_ln = self._apply_film(x_ln, gamma, beta)

        g = self.attn(x_ln)
        g = self.volterra(g)
        x = x + g

        # ---- Local branch: LN -> (FiLM) -> GDFN ----
        x_ln2 = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_ln2 = self.norm2(x_ln2)
        x_ln2 = x_ln2.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B,C,H,W]

        if (gamma is not None) and (beta is not None):
            x_ln2 = self._apply_film(x_ln2, gamma, beta)

        x = x + self.gdfn(x_ln2)
        return x


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    x = torch.randn(2, 48, 64, 64)
    block = VETBlock(dim=48, num_heads=8, volterra_rank=4)

    # identity film
    gamma = torch.ones(2, 48, 1, 1)
    beta = torch.zeros(2, 48, 1, 1)

    y0 = block(x)
    y1 = block(x, gamma=gamma, beta=beta)

    print("=== VETBlock FiLM Test ===")
    print("Input:", x.shape, "Out0:", y0.shape, "Out1:", y1.shape)
    print("Mean |y0-y1|:", (y0 - y1).abs().mean().item())
