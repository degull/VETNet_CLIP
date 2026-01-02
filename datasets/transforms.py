# G:\VETNet_pilot\datasets\transforms.py
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import os


class TrainTransform:
    """
    PNG 캐시 생성 시 이미 256×256로 resize되므로
    여기서는 flip + to_tensor만 수행해야 한다.
    """
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img_in, img_gt):

        # Random horizontal flip
        if random.random() < 0.5:
            img_in = F.hflip(img_in)
            img_gt = F.hflip(img_gt)

        # ❌ Resize 절대 금지 (2중 보간 아티팩트 발생함)
        # img_in = F.resize(img_in, (self.size, self.size))
        # img_gt = F.resize(img_gt, (self.size, self.size))

        # ToTensor
        img_in = F.to_tensor(img_in)
        img_gt = F.to_tensor(img_gt)

        return img_in, img_gt


class TestTransform:
    """
    테스트용도도 마찬가지로 resize 제거해야 함.
    PNG 캐시가 이미 정규화된 사이즈를 가짐.
    """
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img_in, img_gt):

        # ❌ resize 제거
        # img_in = F.resize(img_in, (self.size, self.size))
        # img_gt = F.resize(img_gt, (self.size, self.size))

        img_in = F.to_tensor(img_in)
        img_gt = F.to_tensor(img_gt)

        return img_in, img_gt


# ============================================================
# Self-Test Code (요청한 부분)
# transforms.py 단독 실행 시 동작함
# ============================================================
if __name__ == "__main__":
    print("=== Transform Self-Test ===")

    # 테스트용 256×256 RGB dummy 이미지 생성
    dummy_in = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    dummy_gt = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    print("Created dummy images:", dummy_in.size, dummy_gt.size)

    # TrainTransform 테스트
    train_tf = TrainTransform(size=256)
    t_in, t_gt = train_tf(dummy_in, dummy_gt)

    print("\n[TrainTransform]")
    print("input tensor shape :", t_in.shape)   # torch.Size([3,256,256])
    print("gt tensor shape    :", t_gt.shape)
    print("input min/max      :", t_in.min().item(), t_in.max().item())
    print("gt min/max         :", t_gt.min().item(), t_gt.max().item())

    # TestTransform 테스트
    test_tf = TestTransform(size=256)
    v_in, v_gt = test_tf(dummy_in, dummy_gt)

    print("\n[TestTransform]")
    print("input tensor shape :", v_in.shape)
    print("gt tensor shape    :", v_gt.shape)

    print("\nAll tests completed. Transform logic looks OK.")
