# E:\VETNet_CLIP\tools\generate_clip_image_embeddings.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
from tqdm import tqdm
from PIL import Image

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.vision.clip_image_encoder import CLIPImageEncoder


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

SAVE_ROOT = "E:/VETNet_CLIP/preload_cache"

DATASETS = {
    "CSD": {
        "task": "desnow",
        "paths": [
            "E:/VETNet_CLIP/data/CSD/Train/Snow",
            "E:/VETNet_CLIP/data/CSD/Test/Snow",
        ],
    },
    "DayRainDrop": {
        "task": "deraindrop",
        "paths": [
            "E:/VETNet_CLIP/data/DayRainDrop/Blur",
            "E:/VETNet_CLIP/data/DayRainDrop/Drop",
        ],
    },
    "NightRainDrop": {
        "task": "deraindrop",
        "paths": [
            "E:/VETNet_CLIP/data/NightRainDrop/Drop",
        ],
    },
    "rain100H": {
        "task": "derain",
        "paths": [
            "E:/VETNet_CLIP/data/rain100H/train/rain",
            "E:/VETNet_CLIP/data/rain100H/test/rain",
        ],
    },
    "rain100L": {
        "task": "derain",
        "paths": [
            "E:/VETNet_CLIP/data/rain100L/train/rain",
            "E:/VETNet_CLIP/data/rain100L/test/rain",
        ],
    },
    "RESIDE-6K": {
        "task": "dehaze",
        "paths": [
            "E:/VETNet_CLIP/data/RESIDE-6K/train/hazy",
            "E:/VETNet_CLIP/data/RESIDE-6K/test/hazy",
        ],
    },
}


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTENSIONS)


def collect_image_paths(root_dirs):
    paths = []
    for root in root_dirs:
        for r, _, files in os.walk(root):
            for f in files:
                if is_image_file(f):
                    paths.append(os.path.join(r, f))
    return sorted(paths)


def main():
    encoder = CLIPImageEncoder(
        model_name="openai/clip-vit-large-patch14",
        frozen=True,
    )

    os.makedirs(SAVE_ROOT, exist_ok=True)

    # speed params
    BATCH = 32

    for dataset_name, cfg in DATASETS.items():
        task = cfg["task"]
        img_paths = collect_image_paths(cfg["paths"])

        print("\n" + "=" * 90)
        print(f"[DATASET] {dataset_name}")
        print(f"  Task       : {task}")
        print(f"  Num Images : {len(img_paths)}")
        print("=" * 90)

        save_dir = os.path.join(SAVE_ROOT, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        out_pt = os.path.join(save_dir, "clip_image_embeddings.pt")
        out_meta = os.path.join(save_dir, "clip_image_embeddings_meta.json")

        all_embeds = []
        all_paths = []

        for i in tqdm(range(0, len(img_paths), BATCH), desc=f"CLIP [{dataset_name}]", ncols=110):
            batch_paths = img_paths[i:i+BATCH]
            batch_imgs = []
            ok_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(img)
                    ok_paths.append(p)
                except Exception as e:
                    print(f"[WARN] load fail: {p} ({e})")

            if len(batch_imgs) == 0:
                continue

            with torch.no_grad():
                e = encoder(batch_imgs)  # [b, D] on encoder.device
                e = e.detach().cpu().to(torch.float16)  # store compact

            all_embeds.append(e)
            all_paths.extend(ok_paths)

        embeds = torch.cat(all_embeds, dim=0) if len(all_embeds) > 0 else torch.empty(0, 768)

        torch.save(
            {
                "model": encoder.model_name,
                "dataset": dataset_name,
                "task": task,
                "paths": all_paths,
                "embeddings": embeds,  # [N,D] float16
            },
            out_pt
        )

        meta = {
            "model": encoder.model_name,
            "dataset": dataset_name,
            "task": task,
            "num_items": int(embeds.shape[0]),
            "dim": int(embeds.shape[1]) if embeds.numel() > 0 else 0,
            "dtype": str(embeds.dtype),
            "file": out_pt.replace("\\", "/"),
        }
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] {out_pt}")
        print(f"[META ] {out_meta}")
        print(f"[OK] {dataset_name}: {embeds.shape}")

    print("\n[FINISHED] CLIP image embedding cache generation done.")


if __name__ == "__main__":
    main()
