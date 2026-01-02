# E:\VETNet_CLIP\tools\test_clip_encoder.py
import os, sys
from PIL import Image
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.vision.clip_image_encoder import CLIPImageEncoder

def main():
    encoder = CLIPImageEncoder(
        model_name="openai/clip-vit-large-patch14",
        frozen=True,
    )

    # 아무 이미지 1장 경로로 바꿔서 테스트
    img_path = "E:/VETNet_CLIP/data/RESIDE-6K/test/hazy/0001_0.8_0.2.jpg"
    img = Image.open(img_path).convert("RGB")

    with torch.no_grad():
        e = encoder([img])  # [1, D]

    print("[OK] e_img shape:", e.shape)
    print("[OK] dtype:", e.dtype, "device:", e.device)
    print("[OK] norm:", e.norm(dim=-1))

if __name__ == "__main__":
    main()
