# CLIP image encoder 모듈
# E:\VETNet_CLIP\models\vision\clip_image_encoder.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import List, Union, Optional

import torch
import torch.nn as nn
from PIL import Image

from transformers import CLIPModel, CLIPProcessor


class CLIPImageEncoder(nn.Module):
    """
    CLIP Image Encoder wrapper.
    - Model: openai/clip-vit-large-patch14 (recommended)
    - Output: global image embedding [B, D] (D=768 for ViT-L/14)
    - Frozen by default (recommended at first)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        frozen: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        # cast
        self.model = self.model.to(dtype=self.dtype)

        if frozen:
            self.freeze()

        self.eval()

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: Union[torch.Tensor, List[Image.Image]]):
        """
        Args:
            images:
              - list of PIL Images, OR
              - torch.Tensor in [B,3,H,W] with values in [0,1] or [0,255]
        Returns:
            e_img: [B, D] float tensor
        """
        if isinstance(images, torch.Tensor):
            # Convert tensor -> PIL list (safe, but slower). Prefer PIL list for offline cache.
            # Expect: [B,3,H,W]
            imgs = []
            x = images.detach().cpu()
            if x.max() <= 1.5:
                x = (x * 255.0).clamp(0, 255)
            x = x.to(torch.uint8)
            for i in range(x.shape[0]):
                pil = Image.fromarray(x[i].permute(1, 2, 0).numpy())
                imgs.append(pil)
            images = imgs

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # CLIP forward
        image_features = self.model.get_image_features(**inputs)  # [B, D]

        # Normalize (recommended for similarity / stable controller)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)

        return image_features

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str], batch_size: int = 32):
        """
        Convenience function for offline caching.
        """
        all_embeds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
            emb = self.forward(batch_imgs)  # [B, D]
            all_embeds.append(emb.detach().cpu())
        return torch.cat(all_embeds, dim=0)  # [N, D]
