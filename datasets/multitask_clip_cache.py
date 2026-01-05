# E:\VETNet_CLIP\datasets\multitask_clip_cache.py
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _norm_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def _is_image(p: str) -> bool:
    return p.lower().endswith(IMAGE_EXTS)


def _pil_to_tensor_01(img: Image.Image) -> torch.Tensor:
    import numpy as np
    arr = np.array(img, dtype="uint8")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def _resize_if_needed(img: Image.Image, min_size: int) -> Image.Image:
    w, h = img.size
    if min(w, h) >= min_size:
        return img
    scale = float(min_size) / float(min(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), resample=Image.BICUBIC)


def _random_crop_pair(x: Image.Image, y: Image.Image, size: int):
    x = _resize_if_needed(x, size)
    y = _resize_if_needed(y, size)
    w, h = x.size
    left = random.randint(0, w - size)
    top = random.randint(0, h - size)
    return (
        x.crop((left, top, left + size, top + size)),
        y.crop((left, top, left + size, top + size)),
    )


def _center_crop_pair(x: Image.Image, y: Image.Image, size: int):
    x = _resize_if_needed(x, size)
    y = _resize_if_needed(y, size)
    w, h = x.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return (
        x.crop((left, top, left + size, top + size)),
        y.crop((left, top, left + size, top + size)),
    )


class MultiTaskCLIPCacheDataset(Dataset):
    """
    Phase-2 dataset
    returns:
      - input  : degraded image tensor [3,H,W] in [0,1]
      - gt     : clean image tensor [3,H,W] in [0,1]
      - e_img  : CLIP embedding tensor [D] (float32)
      - dataset / task / paths
    """

    def __init__(
        self,
        preload_cache_root: str,
        datasets: Dict[str, Dict],
        crop_size: int = 256,
        train: bool = True,
        max_per_dataset: Optional[int] = None,
        seed: int = 123,
    ):
        super().__init__()
        self.preload_cache_root = _norm_path(preload_cache_root)
        self.datasets_cfg = datasets
        self.crop_size = int(crop_size)
        self.train = bool(train)

        random.seed(seed)

        self.items: List[Dict] = []

        # 🔥 핵심 추가: CLIP embedding dim
        self.clip_dim: Optional[int] = None

        for ds_name, cfg in self.datasets_cfg.items():
            ds_dir = _norm_path(os.path.join(self.preload_cache_root, ds_name))
            clip_pt = _norm_path(os.path.join(ds_dir, "clip_image_embeddings.pt"))

            if not os.path.isfile(clip_pt):
                raise FileNotFoundError(f"Missing CLIP embedding file: {clip_pt}")

            clip_obj = torch.load(clip_pt, map_location="cpu")

            paths: List[str] = [_norm_path(p) for p in clip_obj["paths"]]
            embeds: torch.Tensor = clip_obj["embeddings"]  # [N, D]

            if embeds.dim() != 2 or len(paths) != embeds.size(0):
                raise RuntimeError(f"Invalid CLIP embedding format in {clip_pt}")

            # 🔥 CLIP dim 설정 및 검증
            if self.clip_dim is None:
                self.clip_dim = embeds.size(1)
            else:
                assert self.clip_dim == embeds.size(1), (
                    f"CLIP dim mismatch: {self.clip_dim} vs {embeds.size(1)} "
                    f"in dataset {ds_name}"
                )

            task = cfg.get("task", "unknown")

            idxs = list(range(len(paths)))
            if max_per_dataset is not None and len(idxs) > max_per_dataset:
                random.shuffle(idxs)
                idxs = idxs[:max_per_dataset]
                idxs.sort()

            for i in idxs:
                x_path = paths[i]
                gt_path = self._infer_gt_path(ds_name, x_path)
                if gt_path is None:
                    continue
                self.items.append({
                    "dataset": ds_name,
                    "task": task,
                    "x_path": x_path,
                    "gt_path": gt_path,
                    "e_img": embeds[i].clone(),
                })

        if len(self.items) == 0:
            raise RuntimeError("No valid items collected.")

        assert self.clip_dim is not None, "CLIP dim was not set."

    # --------------------------
    # Dataset-specific GT mapping
    # --------------------------
    def _infer_gt_path(self, ds_name: str, x_path: str) -> Optional[str]:
        p = _norm_path(x_path)

        if ds_name == "CSD" and "/Snow/" in p:
            return p.replace("/Snow/", "/Gt/") if os.path.isfile(p.replace("/Snow/", "/Gt/")) else None

        if ds_name == "DayRainDrop":
            if "/Drop/" in p:
                return p.replace("/Drop/", "/Clear/") if os.path.isfile(p.replace("/Drop/", "/Clear/")) else None
            if "/Blur/" in p:
                return p.replace("/Blur/", "/Clear/") if os.path.isfile(p.replace("/Blur/", "/Clear/")) else None

        if ds_name == "NightRainDrop" and "/Drop/" in p:
            return p.replace("/Drop/", "/Clear/") if os.path.isfile(p.replace("/Drop/", "/Clear/")) else None

        if ds_name in ("rain100H", "rain100L") and "/rain/" in p:
            return p.replace("/rain/", "/norain/") if os.path.isfile(p.replace("/rain/", "/norain/")) else None

        if ds_name == "RESIDE-6K" and "/hazy/" in p:
            return p.replace("/hazy/", "/GT/") if os.path.isfile(p.replace("/hazy/", "/GT/")) else None

        return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]

        x_img = Image.open(it["x_path"]).convert("RGB")
        gt_img = Image.open(it["gt_path"]).convert("RGB")

        if self.crop_size > 0:
            if self.train:
                x_img, gt_img = _random_crop_pair(x_img, gt_img, self.crop_size)
            else:
                x_img, gt_img = _center_crop_pair(x_img, gt_img, self.crop_size)

        x = _pil_to_tensor_01(x_img)
        gt = _pil_to_tensor_01(gt_img)

        e_img = it["e_img"].float()

        return {
            "input": x,
            "gt": gt,
            "e_img": e_img,
            "dataset": it["dataset"],
            "task": it["task"],
            "x_path": it["x_path"],
            "gt_path": it["gt_path"],
        }
