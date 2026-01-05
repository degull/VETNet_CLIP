# ============================================================
# BLIP caption ↔ CLIP image embedding alignment checker
# - Verifies 1:1 path-level correspondence
# - Reports missing / extra / matched samples
# ============================================================

import os
import json
import torch
from collections import defaultdict

# =========================
# Config
# =========================

DATASETS = [
    "CSD",
    "DayRainDrop",
    "NightRainDrop",
    "rain100H",
    "rain100L",
    "RESIDE-6K",
]

CACHE_ROOT = "E:/VETNet_CLIP/preload_cache"

# number of example paths to print
NUM_EXAMPLES = 5


# =========================
# Utils
# =========================

def normalize_path(p: str) -> str:
    """Normalize path for cross-platform consistency"""
    return os.path.normpath(p).replace("\\", "/")


# =========================
# Main
# =========================

def main():
    print("\n================ BLIP ↔ CLIP ALIGNMENT CHECK ================\n")

    summary = {}

    for dataset in DATASETS:
        print("=" * 80)
        print(f"[DATASET] {dataset}")

        ds_root = os.path.join(CACHE_ROOT, dataset)

        blip_json_path = os.path.join(ds_root, "blip_caption.json")
        clip_pt_path = os.path.join(ds_root, "clip_image_embeddings.pt")

        if not os.path.isfile(blip_json_path):
            print(f"[ERROR] Missing BLIP json: {blip_json_path}")
            continue

        if not os.path.isfile(clip_pt_path):
            print(f"[ERROR] Missing CLIP embeddings: {clip_pt_path}")
            continue

        # -------------------------
        # Load BLIP captions
        # -------------------------
        with open(blip_json_path, "r", encoding="utf-8") as f:
            blip_data = json.load(f)

        blip_paths = {normalize_path(p) for p in blip_data.keys()}

        # -------------------------
        # Load CLIP embeddings
        # -------------------------
        clip_data = torch.load(clip_pt_path, map_location="cpu")

        clip_paths = {normalize_path(p) for p in clip_data["paths"]}

        # -------------------------
        # Set operations
        # -------------------------
        matched = blip_paths & clip_paths
        only_blip = blip_paths - clip_paths
        only_clip = clip_paths - blip_paths

        # -------------------------
        # Report
        # -------------------------
        print(f"Total BLIP captions : {len(blip_paths)}")
        print(f"Total CLIP embeds   : {len(clip_paths)}")
        print(f"Matched             : {len(matched)}")
        print(f"Only in BLIP        : {len(only_blip)}")
        print(f"Only in CLIP        : {len(only_clip)}")

        if len(only_blip) > 0:
            print("\n[Examples] Only in BLIP:")
            for p in list(only_blip)[:NUM_EXAMPLES]:
                print("  ", p)

        if len(only_clip) > 0:
            print("\n[Examples] Only in CLIP:")
            for p in list(only_clip)[:NUM_EXAMPLES]:
                print("  ", p)

        if len(matched) > 0:
            print("\n[Examples] Matched:")
            for p in list(matched)[:NUM_EXAMPLES]:
                print("  ", p)

        summary[dataset] = {
            "blip": len(blip_paths),
            "clip": len(clip_paths),
            "matched": len(matched),
            "only_blip": len(only_blip),
            "only_clip": len(only_clip),
        }

        print()

    # -------------------------
    # Final summary
    # -------------------------
    print("\n================ SUMMARY ================\n")
    for ds, s in summary.items():
        print(
            f"{ds:12s} | "
            f"BLIP: {s['blip']:6d} | "
            f"CLIP: {s['clip']:6d} | "
            f"Matched: {s['matched']:6d} | "
            f"BLIP-only: {s['only_blip']:6d} | "
            f"CLIP-only: {s['only_clip']:6d}"
        )

    print("\n[FINISHED] Alignment check completed.\n")


if __name__ == "__main__":
    main()
