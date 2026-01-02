# Condition Translator (MLP):
# E:\VETNet_CLIP\models\controller\condition_translator.py
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionTranslator8Stage(nn.Module):
    """
    Translate CLIP image embedding e_img -> FiLM + stage gates.

    Outputs:
      - gammas: list length 8, each [B, C_s, 1, 1]
      - betas : list length 8, each [B, C_s, 1, 1]
      - g_stage: [B, 8] in (0,1) via sigmoid
    """

    def __init__(
        self,
        clip_dim: int = 768,   # ViT-L/14 -> 768
        base_dim: int = 64,    # your VETNetBackbone dim (cfg.dim)
        hidden: int = 1024,
        gamma_scale: float = 0.1,
        beta_scale: float = 0.1,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.base_dim = base_dim
        self.hidden = hidden
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale

        # Macro-stage channel sizes aligned with your VETNetBackbone:
        # [enc1, enc2, enc3, latent, dec3, dec2, dec1, refine]
        self.stage_channels = [
            base_dim,
            base_dim * 2,
            base_dim * 4,
            base_dim * 8,
            base_dim * 4,
            base_dim * 2,
            base_dim,
            base_dim,
        ]

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # per-stage heads: output (2*C + 1) -> gamma, beta, gate
        self.heads = nn.ModuleList([
            nn.Linear(hidden, 2 * c + 1) for c in self.stage_channels
        ])

        # init stable near-identity:
        # gamma ~ 1, beta ~ 0, gate ~ 1 (or ~0.9)
        self._init_stable()

    def _init_stable(self):
        for head, c in zip(self.heads, self.stage_channels):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            # gate bias -> sigmoid(bias) ~ 0.9
            # last dim is gate
            with torch.no_grad():
                head.bias[-1] = 2.2  # sigmoid(2.2) ~ 0.900

    def forward(self, e_img: torch.Tensor) -> Dict[str, object]:
        """
        Args:
            e_img: [B, D] (normalized CLIP embedding)
        Returns:
            dict with keys: gammas, betas, g_stage
        """
        B, D = e_img.shape
        h = self.trunk(e_img)  # [B, hidden]

        gammas: List[torch.Tensor] = []
        betas: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for head, c in zip(self.heads, self.stage_channels):
            out = head(h)  # [B, 2*C+1]
            g_raw = out[:, -1]                 # [B]
            gb = out[:, :-1]                   # [B, 2C]
            gamma_raw, beta_raw = gb.split(c, dim=-1)

            # stable FiLM: gamma = 1 + s*tanh(), beta = s*tanh()
            gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_raw)
            beta = self.beta_scale * torch.tanh(beta_raw)

            gamma = gamma.view(B, c, 1, 1)
            beta = beta.view(B, c, 1, 1)

            g = torch.sigmoid(g_raw).view(B, 1)  # [B,1]

            gammas.append(gamma)
            betas.append(beta)
            gates.append(g)

        g_stage = torch.cat(gates, dim=1)  # [B,8]

        return {
            "gammas": gammas,
            "betas": betas,
            "g_stage": g_stage,
        }
