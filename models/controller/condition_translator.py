# E:/VETNet_CLIP/models/controller/condition_translator.py
# ============================================================
# Condition Translator T( e_clip ⊕ v_deg ) -> {g_stage, (optional) FiLM}
#
# Practical design for your pipeline:
# - Input:
#   e_clip : [B, D]          (CLIP image embedding; frozen encoder output)
#   v_deg  : [B, K]          (global severity from DegradationEstimator)
#
# - Output (Phase-2 compatible):
#   g_stage: [B, 8]          stage-wise gates in [0,1]
#     (encoder1, encoder2, encoder3, latent, decoder3, decoder2, decoder1, refinement)
#
# - Output (Phase-3 / ablation):
#   Optional FiLM:
#     gamma_s, beta_s per stage (channel-wise) OR compressed + expanded later
#
# IMPORTANT:
# - In Phase-2: ONLY g_stage is used (no FiLM)
# - In Phase-3: you can enable FiLM heads and inject into backbone blocks
#
# This module is "real code" (no placeholders), minimal dependencies.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class ConditionTranslatorConfig:
    # Input dims
    clip_dim: int = 768          # openai/clip-vit-large-patch14 -> 768
    deg_dim: int = 5             # K (noise, blur, haze, raindrop, snow)

    # Stages
    num_stages: int = 8

    # MLP trunk
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.0
    use_layernorm: bool = True

    # Gates
    gate_min: float = 0.0
    gate_max: float = 1.0
    gate_init_bias: float = 2.0  # sigmoid(2)=0.88 : start near "mostly ON"

    # FiLM (optional)
    enable_film: bool = False

    # If enable_film=True, you can choose:
    #  (A) "stage_scalar": gamma/beta are [B, num_stages] then expanded by stage modules later
    #  (B) "stage_channel": gamma/beta are per-stage channel-wise [B, sum(C_s)] (requires channel table)
    film_mode: str = "stage_scalar"  # "stage_scalar" or "stage_channel"

    # For "stage_channel", provide stage_channels in constructor call (list of length num_stages).
    # film_scale: optionally keep gamma close to 1 at init.
    film_init_scale: float = 0.0     # 0.0 => gamma starts ~1, beta starts ~0

    # Clamp FiLM outputs to avoid extreme modulation (practical stability)
    film_gamma_min: float = 0.5
    film_gamma_max: float = 1.5
    film_beta_min: float = -0.5
    film_beta_max: float = 0.5


# ============================================================
# Utilities
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int,
                 dropout: float = 0.0, use_layernorm: bool = True):
        super().__init__()
        assert num_layers >= 1
        layers: List[nn.Module] = []

        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim

        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _clamp01(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    if vmin == 0.0 and vmax == 1.0:
        return x
    return x.clamp(vmin, vmax)


# ============================================================
# Condition Translator
# ============================================================

class ConditionTranslator(nn.Module):
    """
    T( e_clip, v_deg ) -> dict:
      {
        "g_stage": [B, num_stages] in [gate_min, gate_max],
        (optional)
        "film": {
           "gamma": ...,
           "beta" : ...
        }
      }

    - Phase-2: use only "g_stage"
    - Phase-3: enable_film=True and consume "film" too
    """
    def __init__(
        self,
        cfg: Optional[ConditionTranslatorConfig] = None,
        stage_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.cfg = cfg or ConditionTranslatorConfig()

        self.in_dim = int(self.cfg.clip_dim + self.cfg.deg_dim)
        self.num_stages = int(self.cfg.num_stages)

        # Trunk: produces shared latent
        self.trunk = MLP(
            in_dim=self.in_dim,
            hidden_dim=self.cfg.hidden_dim,
            out_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            use_layernorm=self.cfg.use_layernorm,
        )

        # Gate head
        self.gate_head = nn.Linear(self.cfg.hidden_dim, self.num_stages)
        # Initialize gates near "ON"
        nn.init.constant_(self.gate_head.bias, self.cfg.gate_init_bias)
        nn.init.zeros_(self.gate_head.weight)

        # Optional FiLM
        self.enable_film = bool(self.cfg.enable_film)
        self.film_mode = str(self.cfg.film_mode).lower()

        self.stage_channels = stage_channels
        self.total_channels = None

        if self.enable_film:
            if self.film_mode == "stage_scalar":
                # gamma/beta each: [B, num_stages]
                self.film_gamma = nn.Linear(self.cfg.hidden_dim, self.num_stages)
                self.film_beta  = nn.Linear(self.cfg.hidden_dim, self.num_stages)
                self._init_film_heads_stage_scalar()

            elif self.film_mode == "stage_channel":
                if stage_channels is None:
                    raise ValueError("stage_channels must be provided when film_mode='stage_channel'")
                if len(stage_channels) != self.num_stages:
                    raise ValueError(f"stage_channels length must be {self.num_stages}, got {len(stage_channels)}")
                self.total_channels = int(sum(stage_channels))
                # gamma/beta each: [B, sum(C_s)]
                self.film_gamma = nn.Linear(self.cfg.hidden_dim, self.total_channels)
                self.film_beta  = nn.Linear(self.cfg.hidden_dim, self.total_channels)
                self._init_film_heads_stage_channel()

            else:
                raise ValueError(f"Unknown film_mode: {self.film_mode}")

    # -------------------------
    # init helpers
    # -------------------------
    def _init_film_heads_stage_scalar(self):
        # gamma starts near 1, beta near 0
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        nn.init.zeros_(self.film_gamma.weight)
        # gamma = 1 + film_init_scale * tanh(...)
        # we keep bias at 0 so gamma starts exactly 1
        nn.init.zeros_(self.film_gamma.bias)

    def _init_film_heads_stage_channel(self):
        # same idea for channel-wise
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)

    # -------------------------
    # forward
    # -------------------------
    def forward(
        self,
        e_clip: torch.Tensor,
        v_deg: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
          e_clip: [B, D]
          v_deg : [B, K]
        """
        if e_clip.dim() != 2:
            raise ValueError(f"e_clip must be [B,D], got {tuple(e_clip.shape)}")
        if v_deg.dim() != 2:
            raise ValueError(f"v_deg must be [B,K], got {tuple(v_deg.shape)}")
        if e_clip.size(0) != v_deg.size(0):
            raise ValueError(f"Batch mismatch: e_clip B={e_clip.size(0)} vs v_deg B={v_deg.size(0)}")
        if e_clip.size(1) != self.cfg.clip_dim:
            raise ValueError(f"clip_dim mismatch: expected {self.cfg.clip_dim}, got {e_clip.size(1)}")
        if v_deg.size(1) != self.cfg.deg_dim:
            raise ValueError(f"deg_dim mismatch: expected {self.cfg.deg_dim}, got {v_deg.size(1)}")

        x = torch.cat([e_clip, v_deg], dim=1)  # ⊕ concat
        h = self.trunk(x)

        # gates
        g_logits = self.gate_head(h)
        g = torch.sigmoid(g_logits)  # [0,1]
        g = _clamp01(g, self.cfg.gate_min, self.cfg.gate_max)

        out: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {"g_stage": g}

        # FiLM (optional)
        if self.enable_film:
            # gamma near 1, beta near 0 (stable)
            # gamma_raw -> tanh -> scaled, add 1
            gamma_raw = self.film_gamma(h)
            beta_raw  = self.film_beta(h)

            if self.cfg.film_init_scale > 0:
                gamma = 1.0 + self.cfg.film_init_scale * torch.tanh(gamma_raw)
                beta  = self.cfg.film_init_scale * torch.tanh(beta_raw)
            else:
                # still keep bounded but allow learning
                gamma = 1.0 + torch.tanh(gamma_raw) * 0.1
                beta  = torch.tanh(beta_raw) * 0.1

            # clamp for safety
            gamma = gamma.clamp(self.cfg.film_gamma_min, self.cfg.film_gamma_max)
            beta  = beta.clamp(self.cfg.film_beta_min,  self.cfg.film_beta_max)

            if self.film_mode == "stage_scalar":
                # [B, num_stages] each
                out["film"] = {"gamma": gamma, "beta": beta}

            elif self.film_mode == "stage_channel":
                # [B, sum(C_s)] each
                out["film"] = {"gamma": gamma, "beta": beta}

            else:
                raise RuntimeError("invalid film_mode internal state")

        return out

    # -------------------------
    # helpers for film split
    # -------------------------
    def split_film_by_stage(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Only used when film_mode == "stage_channel".
        Converts:
          gamma,beta: [B, sum(C_s)]
        to list of per-stage:
          [(gamma_s [B,C_s,1,1], beta_s [B,C_s,1,1]), ...]
        """
        if not self.enable_film or self.film_mode != "stage_channel":
            raise RuntimeError("split_film_by_stage is only valid for enable_film=True & film_mode='stage_channel'")
        if self.stage_channels is None:
            raise RuntimeError("stage_channels missing")

        B = gamma.size(0)
        out = []
        offset = 0
        for c in self.stage_channels:
            g_s = gamma[:, offset:offset + c].view(B, c, 1, 1)
            b_s = beta[:,  offset:offset + c].view(B, c, 1, 1)
            out.append((g_s, b_s))
            offset += c
        return out


# ============================================================
# Quick Self-test
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example: phase-2 usage (gates only)
    cfg = ConditionTranslatorConfig(
        clip_dim=768, deg_dim=5, num_stages=8,
        hidden_dim=512, num_layers=3,
        enable_film=False
    )
    T = ConditionTranslator(cfg).to(device).eval()
    e = torch.randn(2, 768, device=device)
    v = torch.rand(2, 5, device=device)

    with torch.no_grad():
        out = T(e, v)
    print("g_stage:", out["g_stage"].shape, out["g_stage"].min().item(), out["g_stage"].max().item())

    # Example: phase-3 usage (stage-scalar film)
    cfg2 = ConditionTranslatorConfig(
        clip_dim=768, deg_dim=5, num_stages=8,
        hidden_dim=512, num_layers=3,
        enable_film=True, film_mode="stage_scalar"
    )
    T2 = ConditionTranslator(cfg2).to(device).eval()
    with torch.no_grad():
        out2 = T2(e, v)
    print("g_stage:", out2["g_stage"].shape)
    print("film.gamma:", out2["film"]["gamma"].shape)
    print("film.beta :", out2["film"]["beta"].shape)
