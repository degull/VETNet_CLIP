# E:\VETNet_CLIP\models\backbone\volterra_layer.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Utility Function ----------------- #

def circular_shift(x, shift_x, shift_y):
    """
    Feature Map을 2D 평면에서 순환 이동(Circular Shift)시키는 함수.
    x: (B, C, H, W)
    shift_x: X축 이동량
    shift_y: Y축 이동량
    """
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))

# ----------------- Core Layer: Rank Decomposition Volterra ----------------- #

class VolterraLayer2D(nn.Module):
    """
    2차 볼테라 필터링을 구현하는 레이어.
    출력 = H1 * x + H2 * x^2 (선형 항 + 2차 비선형 항)
    use_lossless=False 버전 (Rank Decomposition 기반).
    in_channels == out_channels == dim 으로 사용하는 걸 기본으로 가정.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=4, bias=True):
        super().__init__()
        assert in_channels == out_channels, \
            "현재 구현은 in_channels == out_channels 인 경우를 주로 가정합니다."
        self.rank = rank
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 1. 선형 항 (H1 * x)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=self.padding, bias=bias)

        # 2. 2차 비선형 항 (H2 * x^2) - Rank Decomposition
        self.W2a = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=self.padding, bias=bias)
            for _ in range(rank)
        ])
        self.W2b = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=self.padding, bias=bias)
            for _ in range(rank)
        ])

    def forward(self, x):
        # 1. 선형 항
        linear_term = self.conv1(x)

        # 2. 2차 항
        quadratic_term = 0
        for a, b in zip(self.W2a, self.W2b):
            qa = torch.clamp(a(x), min=-1.0, max=1.0)
            qb = torch.clamp(b(x), min=-1.0, max=1.0)
            quadratic_term += qa * qb

        return linear_term + quadratic_term


# ----------------- 선택적: Lossless Volterra ----------------- #

class VolterraLayer2D_Lossless(nn.Module):
    """
    circular_shift를 이용해 lossless 볼테라 연산을 구현하는 버전 (참고용).
    기본 백본에서는 VolterraLayer2D만 사용해도 충분함.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        assert in_channels == out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=self.padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=self.padding, bias=bias)
        self.shifts = self._generate_shifts(kernel_size)

    def _generate_shifts(self, k):
        P = k // 2
        shifts = []
        for s1 in range(-P, P + 1):
            for s2 in range(-P, P + 1):
                if s1 == 0 and s2 == 0:
                    continue
                if (s1, s2) < (0, 0):
                    continue
                shifts.append((s1, s2))
        return shifts

    def forward(self, x):
        linear_term = self.conv1(x)
        quadratic_term = 0

        for s1, s2 in self.shifts:
            x_shifted = circular_shift(x, s1, s2)
            prod = x * x_shifted
            prod = torch.clamp(prod, min=-1.0, max=1.0)
            quadratic_term += self.conv2(prod)

        return linear_term + quadratic_term


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    dummy_input = torch.randn(1, 32, 64, 64)
    in_c = 32
    out_c = 32

    print("=== VolterraLayer2D Rank Decomposition Test ===")
    model_rank = VolterraLayer2D(in_channels=in_c, out_channels=out_c, rank=4)
    output_rank = model_rank(dummy_input)
    print(f"Input  Shape: {dummy_input.shape}")
    print(f"Output Shape (Rank): {output_rank.shape}")

    print("\n=== VolterraLayer2D Lossless Test ===")
    model_lossless = VolterraLayer2D_Lossless(in_channels=in_c, out_channels=out_c)
    output_lossless = model_lossless(dummy_input)
    print(f"Output Shape (Lossless): {output_lossless.shape}")
