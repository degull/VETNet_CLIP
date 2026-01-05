# E:\VETNet_CLIP\models\controller\gate_controller.py
import torch
import torch.nn as nn


class GateController(nn.Module):
    """
    Phase-2 Gate-only Controller

    Input:
        e_img : CLIP image embedding [B, D]

    Output:
        g     : stage-wise gates [B, S], each in [gate_min, gate_max]

    Design:
        g = gate_min + (gate_max - gate_min) * sigmoid(MLP(e_img))
    """

    def __init__(
        self,
        in_dim: int,
        num_stages: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"
        assert 0.0 <= gate_min < gate_max <= 1.0, "Require 0 ≤ gate_min < gate_max ≤ 1"

        self.in_dim = in_dim
        self.num_stages = num_stages
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)

        layers = []

        # 1-layer linear controller (minimal)
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, num_stages))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Linear(hidden_dim, num_stages))

        self.mlp = nn.Sequential(*layers)

    def forward(self, e_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_img: [B, D] CLIP image embedding

        Returns:
            g: [B, num_stages] in [gate_min, gate_max]
        """
        g_raw = self.mlp(e_img)          # [B, S]
        g_sig = torch.sigmoid(g_raw)     # (0,1)

        # affine scaling to [gate_min, gate_max]
        g = self.gate_min + (self.gate_max - self.gate_min) * g_sig
        return g
