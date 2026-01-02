# E:\VETNet_CLIP\models\backbone\mdta_strategy.py
# phase -1 (vetnet backbone)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):

    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W
        C_head = C // self.num_heads

        # 1. QKV 생성
        q, k, v = self.qkv(x).chunk(3, dim=1)  # (B, C, H, W)

        # 2. (B, C, H, W) → (B, Head, C_head, N)
        q = q.view(B, self.num_heads, C_head, N)
        k = k.view(B, self.num_heads, C_head, N)
        v = v.view(B, self.num_heads, C_head, N)

        # 3. 채널 방향(C_head)에서 어텐션 계산
        #    attn: (B, Head, C_head, C_head)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        attn = torch.matmul(q_norm, k_norm.transpose(-1, -2)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 4. attn을 v에 적용 → (B, Head, C_head, N)
        out = torch.matmul(attn, v)

        # 5. (B, Head, C_head, N) → (B, C, H, W)
        out = out.view(B, C, H, W)
        out = self.project_out(out)
        return out


# -----------------------------------------------------------
# 아래는 Phase2용 Strategy Token 버전 (지금은 안 써도 됨)
# -----------------------------------------------------------

class MDTAWithStrategy(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.project_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, X_flat, S_tokens=None):
        if S_tokens is not None:
            X_in = torch.cat([X_flat, S_tokens], dim=1)  # (B, N+K, C)
        else:
            X_in = X_flat

        B, N_tot, C = X_in.shape

        qkv = self.qkv(X_in)
        q, k, v = qkv.chunk(3, dim=-1)

        # 여기서는 간단한 token-wise attention (N_tot × N_tot)
        # → Phase2에서 windowing / 축소 고려
        q = q.view(B, N_tot, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, N_tot, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, N_tot, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # (B, H, N_tot, d)
        out = out.transpose(1, 2).contiguous().view(B, N_tot, C)
        out = self.project_out(out)

        # 이미지 토큰 부분만 반환 (앞의 N 개)
        return out[:, :X_flat.shape[1], :]


# ----------------- 테스트 코드 ----------------- #
if __name__ == "__main__":
    print("=== MDTA (Channel-wise Transposed) Test ===")
    x = torch.randn(1, 48, 256, 256)
    attn = MDTA(dim=48, num_heads=8)
    y = attn(x)
    print(f"Input  Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")

    print("\n=== MDTAWithStrategy (Linear-based) Test ===")
    B, N, C = 2, 64, 48
    K = 4
    X_flat = torch.randn(B, N, C)
    S = torch.randn(B, K, C)
    attn_s = MDTAWithStrategy(dim=C)
    Y_flat = attn_s(X_flat, S_tokens=S)
    print(f"X_flat Shape: {X_flat.shape}")
    print(f"S_tokens Shape: {S.shape}")
    print(f"Y_flat Shape: {Y_flat.shape}")
