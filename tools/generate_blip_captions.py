# ============================================================
# BLIP Caption Generator (FINAL VERSION)
# - Pure image captioning (NO prompt)
# - Generates degradation-aware pseudo-text
# - Saves per-dataset JSON files
# - Training-time only (teacher signal)
# ============================================================

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
from tqdm import tqdm
from PIL import Image

import torch
from torch.cuda.amp import autocast
from transformers import BlipProcessor, BlipForConditionalGeneration


# =========================
# Config
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

BLIP_MODEL = "Salesforce/blip-image-captioning-base"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# JSON 저장 루트
SAVE_ROOT = "E:/VETNet_CLIP/preload_cache"


# =========================
# Dataset Definitions
# (ONLY degraded images)
# =========================

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


# =========================
# Utils
# =========================

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


# =========================
# Main
# =========================

def main():
    print("[INFO] Device:", DEVICE)
    print("[INFO] Loading BLIP model:", BLIP_MODEL)

    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    os.makedirs(SAVE_ROOT, exist_ok=True)

    for dataset_name, cfg in DATASETS.items():
        task = cfg["task"]
        img_paths = collect_image_paths(cfg["paths"])

        print("\n" + "=" * 80)
        print(f"[DATASET] {dataset_name}")
        print(f"  Task       : {task}")
        print(f"  Num Images : {len(img_paths)}")
        print("=" * 80)

        save_dir = os.path.join(SAVE_ROOT, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "blip_caption.json")

        captions = {}

        for img_path in tqdm(img_paths, desc=f"BLIP [{dataset_name}]", ncols=100):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to load image: {img_path} ({e})")
                continue

            inputs = processor(
                images=image,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                with autocast(enabled=(DEVICE == "cuda")):
                    output_ids = model.generate(
                        **inputs,
                        max_length=40,
                        num_beams=3,
                        do_sample=False,
                    )

            caption = processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            captions[img_path] = {
                "caption": caption,
                "dataset": dataset_name,
                "task": task,
            }

        # ---------- Save JSON (dataset finished) ----------
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] {save_path}")
        print(f"[OK] {dataset_name}: {len(captions)} captions written")

    print("\n[FINISHED] BLIP caption generation completed successfully.")


if __name__ == "__main__":
    main()
