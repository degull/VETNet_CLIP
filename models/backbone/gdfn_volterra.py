# E:\VETNet_CLIP\models\backbone\gdfn_volterra.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GDFN(nn.Module):
    """
    Restormer 공식 GDFN 구현 (오류 수정 버전)
    depthwise conv는 x1, x2로 split 하기 전에 전체 채널(H*2)에 적용해야 한다.
    """
    def __init__(self, dim, expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        # 1×1 projection → 2H channels
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)

        # depthwise convolution (channels = 2H)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_dim * 2,  # depthwise
            bias=bias
        )

        # output projection: H channels
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        x : (B, C, H, W)
        """
        x_in = self.project_in(x)     # (B, 2H, H, W)

        x_dw = self.dwconv(x_in)      # depthwise conv 전체 채널에 적용

        # split AFTER depthwise conv
        x1, x2 = x_dw.chunk(2, dim=1) # each becomes (B, H, H, W)

        # gated feed-forward
        out = F.gelu(x1) * x2

        # project back to original dim
        out = self.project_out(out)

        return out


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    print("=== GDFN Test ===")
    x = torch.randn(1, 48, 64, 64)
    gdfn = GDFN(dim=48)
    y = gdfn(x)
    print(f"Input  Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
