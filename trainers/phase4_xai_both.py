# E:\VETNet_CLIP\trainers\phase4_xai_both.py
# ============================================================
# Phase-4: XAI Export (BOTH)
# - Caption (BLIP, optional)
# - Restoration-aware explanation (gate/mask-based, text-free)
# - Save:
#   (A) triplet image: [Input | Restored | GT]
#   (B) figure subplot: Input/Restored/GT + Mask heatmap + Stage gate bar + Text
#   (C) meta.json + meta.csv
# ============================================================

import os
import sys
import json
import math
import csv
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# matplotlib (figure export)  ※ seaborn 금지
import matplotlib.pyplot as plt

# PIL for triplet saving
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# skimage SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# transformers BLIP
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _HAS_BLIP = True
except Exception:
    _HAS_BLIP = False

# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.backbone.vetnet_backbone import VETNetBackbone
from models.degradation.degradation_estimator import DegradationEstimator
from models.controller.condition_translator import (
    ConditionTranslator,
    ConditionTranslatorConfig,
)
from datasets.multitask_clip_cache import MultiTaskCLIPCacheDataset


# ============================================================
# Config
# ============================================================

@dataclass
class XAIConfig:
    # ---- input ckpt ----
    ckpt_path: str = "E:\VETNet_CLIP\checkpoints\phase3_restore\epoch_001_L0.0336_P27.55_S0.8157.pth"

    # ---- dataset cache ----
    cache_root: str = "E:/VETNet_CLIP/preload_cache"
    crop_size: int = 256
    num_stages: int = 8

    # Explicit datasets list (recommended)
    datasets_cfg: Optional[Dict[str, Dict]] = None

    # ---- model toggles (must match Phase-3 if you used these) ----
    enable_film: bool = False
    film_mode: str = "stage_scalar"       # "stage_scalar" | "stage_channel"
    enable_spatial_gate: bool = False     # if you trained with spatial gate

    # ---- export ----
    out_root: str = "E:/VETNet_CLIP/results/phase4_xai_both"
    max_items: int = 300
    batch_size: int = 1
    num_workers: int = 0

    # ---- BLIP caption ----
    enable_caption: bool = True
    blip_name: str = "Salesforce/blip-image-captioning-base"
    caption_max_new_tokens: int = 24

    # ---- device ----
    use_amp: bool = False   # XAI export는 보통 False 권장 (안정/재현성)
    device: str = "cuda"


cfg = XAIConfig()
cfg.datasets_cfg = {
    "CSD": {},
    "DayRainDrop": {},
    "NightRainDrop": {},
    "rain100H": {},
    "rain100L": {},
    "RESIDE-6K": {},
}


# ============================================================
# Utils
# ============================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)

def _tensor_to_uint8_hwc(x01: torch.Tensor) -> np.ndarray:
    # x01: [3,H,W] in [0,1]
    x = (x01 * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return x

def _psnr_torch(x01: torch.Tensor, y01: torch.Tensor) -> float:
    mse = F.mse_loss(x01, y01).item()
    return 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)

def _ssim_skimage(x01: torch.Tensor, y01: torch.Tensor) -> Optional[float]:
    if not _HAS_SKIMAGE:
        return None
    x = x01.detach().cpu().permute(1,2,0).numpy()
    y = y01.detach().cpu().permute(1,2,0).numpy()
    # multichannel=True 대체: channel_axis=-1
    return float(sk_ssim(x, y, data_range=1.0, channel_axis=-1))

def _save_triplet_side_by_side(inp01: torch.Tensor, out01: torch.Tensor, gt01: torch.Tensor, save_path: str):
    if not _HAS_PIL:
        return
    _ensure_dir(os.path.dirname(save_path))
    a = _tensor_to_uint8_hwc(inp01)
    b = _tensor_to_uint8_hwc(out01)
    c = _tensor_to_uint8_hwc(gt01)
    H, W = a.shape[0], a.shape[1]
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, 0:W] = a
    canvas[:, W:2*W] = b
    canvas[:, 2*W:3*W] = c
    Image.fromarray(canvas).save(save_path)

def _mask_ratio(M: Optional[torch.Tensor], thr: float = 0.5) -> Optional[float]:
    if M is None:
        return None
    if not torch.is_tensor(M):
        return None
    # M: [B,1,H,W] or [B,C,H,W]
    m = M[:, :1]
    return float((m > thr).float().mean().item())

def _topk_stage(g8: List[float], k: int = 3) -> List[int]:
    idx = np.argsort(np.array(g8))[::-1][:k]
    return [int(i) for i in idx.tolist()]

def _restoration_explain_text(
    g_stage_8: List[float],
    mask_ratio_val: Optional[float],
    caption: str,
) -> str:
    # policy-style explanation (text-free + caption optional)
    top3 = _topk_stage(g_stage_8, k=3)
    g_str = ", ".join([f"{v:.2f}" for v in g_stage_8])

    lines = []
    if caption.strip() != "":
        lines.append(f"[Caption] {caption.strip()}")

    # mask / local artifact
    if mask_ratio_val is not None:
        if mask_ratio_val > 0.08:
            lines.append(f"[Local artifacts] Spatial mask ratio≈{mask_ratio_val:.2f} (localized degradation likely).")
            lines.append("[Policy] Prioritize localized artifact removal while preserving background structures.")
        else:
            lines.append(f"[Local artifacts] Spatial mask ratio≈{mask_ratio_val:.2f} (weak/limited localization).")
            lines.append("[Policy] Emphasize global restoration with minimal local suppression.")
    else:
        lines.append("[Local artifacts] Spatial mask not available (spatial gating OFF or estimator returns global only).")
        lines.append("[Policy] Rely on global stage-wise modulation.")

    # gates
    lines.append(f"[Stage gates] g_stage=[{g_str}]")
    lines.append(f"[Activated stages] top-3 stages={top3} (higher gates indicate stronger usage).")

    return "\n".join(lines)

def _try_load_blip(device: str):
    if not (_HAS_BLIP and cfg.enable_caption):
        print("[BLIP] disabled or transformers not installed -> caption OFF")
        return None
    try:
        processor = BlipProcessor.from_pretrained(cfg.blip_name)
        model = BlipForConditionalGeneration.from_pretrained(cfg.blip_name).to(device).eval()
        print(f"[BLIP] loaded: {cfg.blip_name}")
        return (processor, model)
    except Exception as e:
        print("[BLIP] load failed -> caption OFF. Reason:", str(e))
        return None

def _blip_caption(blip_pack, pil_img, device: str) -> str:
    if blip_pack is None:
        return ""
    processor, model = blip_pack
    try:
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg.caption_max_new_tokens)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""

def _make_subplot_figure(
    inp01: torch.Tensor,
    out01: torch.Tensor,
    gt01: torch.Tensor,
    M_1hw: Optional[torch.Tensor],
    g_stage_8: List[float],
    text: str,
    save_path: str
):
    """
    2x3:
      [Input | Restored | GT]
      [Mask  | StageGateBar | Text]
    """
    _ensure_dir(os.path.dirname(save_path))

    inp = _tensor_to_uint8_hwc(inp01)
    out = _tensor_to_uint8_hwc(out01)
    gt  = _tensor_to_uint8_hwc(gt01)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2,3,1); ax1.imshow(inp); ax1.set_title("Input"); ax1.axis("off")
    ax2 = fig.add_subplot(2,3,2); ax2.imshow(out); ax2.set_title("Restored"); ax2.axis("off")
    ax3 = fig.add_subplot(2,3,3); ax3.imshow(gt);  ax3.set_title("GT"); ax3.axis("off")

    ax4 = fig.add_subplot(2,3,4)
    if M_1hw is not None:
        m = M_1hw.detach().cpu().float().clamp(0,1).numpy()
        ax4.imshow(m, cmap="jet")  # heatmap only, fine for figure
        ax4.set_title("Mask / Spatial map")
    else:
        ax4.text(0.5, 0.5, "Mask: N/A", ha="center", va="center")
        ax4.set_title("Mask / Spatial map")
    ax4.axis("off")

    ax5 = fig.add_subplot(2,3,5)
    xs = np.arange(len(g_stage_8))
    ax5.bar(xs, np.array(g_stage_8, dtype=np.float32))
    ax5.set_ylim(0, 1.0)
    ax5.set_xticks(xs)
    ax5.set_title("Stage gate g_stage (8)")
    ax5.set_xlabel("stage")
    ax5.set_ylabel("gate")

    ax6 = fig.add_subplot(2,3,6)
    ax6.axis("off")
    ax6.set_title("XAI (caption + rationale)")
    ax6.text(0.0, 1.0, text, va="top", wrap=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def _write_json(path: str, obj: Any):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_csv(path: str, rows: List[Dict[str, Any]]):
    _ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Main
# ============================================================

def main():
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # output dirs
    triplet_dir = os.path.join(cfg.out_root, "images_triplet")
    fig_dir     = os.path.join(cfg.out_root, "figures")
    _ensure_dir(cfg.out_root)
    _ensure_dir(triplet_dir)
    _ensure_dir(fig_dir)

    # dataset
    dataset = MultiTaskCLIPCacheDataset(
        preload_cache_root=cfg.cache_root,
        datasets=cfg.datasets_cfg,
        crop_size=cfg.crop_size,
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    print("[Data] total items:", len(dataset))

    # build models
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    backbone = VETNetBackbone(
        in_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=(4,6,6,8),
        heads=(1,2,4,8),
        volterra_rank=4,          # ★ Phase-3와 동일
        ffn_expansion_factor=2.66,
        bias=False,
    ).to(device)

    degrader = DegradationEstimator().to(device)

    # clip_dim inference
    sample = dataset[0]
    clip_dim = int(sample["e_img"].numel())

    translator_cfg = ConditionTranslatorConfig(
        clip_dim=clip_dim,
        deg_dim=getattr(degrader, "out_dim", 5),
        num_stages=cfg.num_stages,
        enable_film=cfg.enable_film,
        film_mode=cfg.film_mode,
    )
    translator = ConditionTranslator(translator_cfg).to(device)

    # load
    backbone.load_state_dict(ckpt["backbone"], strict=True)
    degrader.load_state_dict(ckpt["degrader"], strict=True)
    translator.load_state_dict(ckpt["translator"], strict=True)

    backbone.eval(); degrader.eval(); translator.eval()

    # BLIP
    blip_pack = _try_load_blip(device)

    # meta
    meta: Dict[str, Any] = {
        "cfg": cfg.__dict__,
        "has_blip": (blip_pack is not None),
        "has_skimage": _HAS_SKIMAGE,
        "items": []
    }
    csv_rows: List[Dict[str, Any]] = []

    n = 0
    pbar = tqdm(loader, desc="[Phase4 BOTH] exporting", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            if n >= cfg.max_items:
                break

            x = batch["input"].to(device, non_blocking=True)     # [B,3,H,W]
            y = batch["gt"].to(device, non_blocking=True)        # [B,3,H,W]
            e = batch["e_img"].to(device, non_blocking=True)     # [B,768] etc.

            # restore forward
            # DegradationEstimator may return (M, v) or v only
            deg = degrader(x)
            if isinstance(deg, (tuple, list)) and len(deg) >= 2:
                M, v = deg[0], deg[1]
            else:
                M, v = None, deg

            out = translator(e, v)
            g_stage = out["g_stage"]                # [B,8]
            film = out.get("film", None)

            # NOTE: spatial gate visualization uses M directly here
            # (you can switch to "G after SpatialGate" if you want later)
            y_hat = backbone(x, g_stage=g_stage, film=film, spatial_gate=None)

            # take first item in batch
            inp01 = _to_01(x[0])
            out01 = _to_01(y_hat[0])
            gt01  = _to_01(y[0])

            # metrics
            psnr_val = _psnr_torch(out01, gt01)
            ssim_val = _ssim_skimage(out01, gt01)

            # triplet save
            trip_path = os.path.join(triplet_dir, f"sample_{n:06d}.png")
            _save_triplet_side_by_side(inp01, out01, gt01, trip_path)

            # caption on restored
            cap = ""
            if blip_pack is not None and _HAS_PIL:
                pil_rest = Image.fromarray(_tensor_to_uint8_hwc(out01))
                cap = _blip_caption(blip_pack, pil_rest, device=device)

            # explanation text
            g8 = [float(v) for v in g_stage[0].detach().cpu().float().tolist()]
            mr = _mask_ratio(M, thr=0.5)
            explain = _restoration_explain_text(g8, mr, cap)

            # subplot fig save
            fig_path = os.path.join(fig_dir, f"fig_{n:06d}.png")
            M1 = None
            if M is not None and torch.is_tensor(M):
                M1 = M[0, 0].detach()  # [H,W]
            _make_subplot_figure(inp01, out01, gt01, M1, g8, explain, fig_path)

            # record meta
            item = {
                "index": n,
                "triplet_path": trip_path,
                "figure_path": fig_path,
                "psnr": float(psnr_val),
                "ssim": (None if ssim_val is None else float(ssim_val)),
                "caption": cap,
                "explain": explain,
                "g_stage": g8,
                "mask_ratio": mr,
            }
            meta["items"].append(item)

            csv_rows.append({
                "index": n,
                "psnr": float(psnr_val),
                "ssim": ("" if ssim_val is None else float(ssim_val)),
                "mask_ratio": ("" if mr is None else float(mr)),
                "triplet_path": trip_path,
                "figure_path": fig_path,
                "caption": cap,
            })

            n += 1
            pbar.set_postfix({
                "P": f"{psnr_val:.2f}",
                "S": ("NA" if ssim_val is None else f"{ssim_val:.4f}"),
                "mask": ("NA" if mr is None else f"{mr:.3f}")
            })

    # write meta
    json_path = os.path.join(cfg.out_root, "meta.json")
    csv_path  = os.path.join(cfg.out_root, "meta.csv")
    _write_json(json_path, meta)
    _write_csv(csv_path, csv_rows)

    print("\n[DONE] Phase-4 BOTH Export")
    print(" - triplets :", triplet_dir)
    print(" - figures  :", fig_dir)
    print(" - meta.json:", json_path)
    print(" - meta.csv :", csv_path)
    if blip_pack is None:
        print(" - caption  : OFF (transformers/BLIP not available or disabled)")
    if not _HAS_SKIMAGE:
        print(" - SSIM     : OFF (skimage not available)")

if __name__ == "__main__":
    main()


# AR-Gate 기반 step-wise XAI + JSON 요약(policy_summary) + interpretation
# E:\VETNet_CLIP\trainers\phase4_xai_both.py
# ============================================================
# Phase-4: XAI Export (BOTH)
# - Caption (BLIP, optional)
# - Restoration-aware explanation (AR gate/mask-based, text-free)
# - Save:
#   (A) triplet image: [Input | Restored | GT]
#   (B) figure subplot: Input/Restored/GT + Mask heatmap + Stage gate bar + Text
#   (C) meta.json + meta.csv
# ============================================================

import os
import sys
import json
import math
import csv
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# matplotlib (figure export)  ※ seaborn 금지
import matplotlib.pyplot as plt

# PIL for triplet saving
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# skimage SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# transformers BLIP
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _HAS_BLIP = True
except Exception:
    _HAS_BLIP = False

# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.backbone.vetnet_backbone import VETNetBackbone
from models.degradation.degradation_estimator import DegradationEstimator
from models.controller.condition_translator import (
    ConditionTranslator,
    ConditionTranslatorConfig,
)
from datasets.multitask_clip_cache import MultiTaskCLIPCacheDataset


# ============================================================
# Config
# ============================================================

@dataclass
class XAIConfig:
    # ---- input ckpt ----
    ckpt_path: str = r"E:\VETNet_CLIP\checkpoints\phase3_restore\epoch_001_L0.0336_P27.55_S0.8157.pth"

    # ---- dataset cache ----
    cache_root: str = "E:/VETNet_CLIP/preload_cache"
    crop_size: int = 256
    num_stages: int = 8

    # Explicit datasets list (recommended)
    datasets_cfg: Optional[Dict[str, Dict]] = None

    # ---- model toggles (must match Phase-3 if you used these) ----
    enable_film: bool = False
    film_mode: str = "stage_scalar"       # "stage_scalar" | "stage_channel"
    enable_spatial_gate: bool = False     # if you trained with spatial gate

    # ---- export ----
    out_root: str = "E:/VETNet_CLIP/results/phase4_xai_both"
    max_items: int = 300
    batch_size: int = 1
    num_workers: int = 0

    # ---- BLIP caption ----
    enable_caption: bool = True
    blip_name: str = "Salesforce/blip-image-captioning-base"
    caption_max_new_tokens: int = 24

    # ---- device ----
    use_amp: bool = False   # XAI export는 보통 False 권장 (안정/재현성)
    device: str = "cuda"

    # ---- AR-XAI knobs ----
    # gate change threshold for "transition" detection
    ar_drop_thr: float = 0.08


cfg = XAIConfig()
cfg.datasets_cfg = {
    "CSD": {},
    "DayRainDrop": {},
    "NightRainDrop": {},
    "rain100H": {},
    "rain100L": {},
    "RESIDE-6K": {},
}


# ============================================================
# Utils
# ============================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)

def _tensor_to_uint8_hwc(x01: torch.Tensor) -> np.ndarray:
    # x01: [3,H,W] in [0,1]
    x = (x01 * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return x

def _psnr_torch(x01: torch.Tensor, y01: torch.Tensor) -> float:
    mse = F.mse_loss(x01, y01).item()
    return 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)

def _ssim_skimage(x01: torch.Tensor, y01: torch.Tensor) -> Optional[float]:
    if not _HAS_SKIMAGE:
        return None
    x = x01.detach().cpu().permute(1,2,0).numpy()
    y = y01.detach().cpu().permute(1,2,0).numpy()
    return float(sk_ssim(x, y, data_range=1.0, channel_axis=-1))

def _save_triplet_side_by_side(inp01: torch.Tensor, out01: torch.Tensor, gt01: torch.Tensor, save_path: str):
    if not _HAS_PIL:
        return
    _ensure_dir(os.path.dirname(save_path))
    a = _tensor_to_uint8_hwc(inp01)
    b = _tensor_to_uint8_hwc(out01)
    c = _tensor_to_uint8_hwc(gt01)
    H, W = a.shape[0], a.shape[1]
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, 0:W] = a
    canvas[:, W:2*W] = b
    canvas[:, 2*W:3*W] = c
    Image.fromarray(canvas).save(save_path)

def _mask_ratio(M: Optional[torch.Tensor], thr: float = 0.5) -> Optional[float]:
    if M is None:
        return None
    if not torch.is_tensor(M):
        return None
    m = M[:, :1]
    return float((m > thr).float().mean().item())

def _try_load_blip(device: str):
    if not (_HAS_BLIP and cfg.enable_caption):
        print("[BLIP] disabled or transformers not installed -> caption OFF")
        return None
    try:
        processor = BlipProcessor.from_pretrained(cfg.blip_name)
        model = BlipForConditionalGeneration.from_pretrained(cfg.blip_name).to(device).eval()
        print(f"[BLIP] loaded: {cfg.blip_name}")
        return (processor, model)
    except Exception as e:
        print("[BLIP] load failed -> caption OFF. Reason:", str(e))
        return None

def _blip_caption(blip_pack, pil_img, device: str) -> str:
    if blip_pack is None:
        return ""
    processor, model = blip_pack
    try:
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg.caption_max_new_tokens)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


# ============================================================
# AR-Gate XAI (NEW)
# ============================================================

def _strength_label(x: float) -> str:
    # simple binning for paper/demo readability
    if x >= 0.80:
        return "high"
    if x >= 0.55:
        return "medium"
    return "mild"

def _detect_transition_stage(g8: List[float], drop_thr: float = 0.08) -> int:
    """
    Returns 1-based stage index where a major drop happens (largest negative delta),
    or the first drop exceeding threshold; fallback to largest drop.
    """
    if len(g8) < 2:
        return 1
    g = np.array(g8, dtype=np.float32)
    d = g[1:] - g[:-1]  # delta
    # find first "big drop"
    for i, di in enumerate(d, start=2):  # stage index of g[i-1]->g[i]
        if di <= -drop_thr:
            return int(i)
    # fallback: largest negative delta
    j = int(np.argmin(d)) + 2
    return j

def _policy_summary(g8: List[float]) -> Dict[str, Any]:
    g = np.array(g8, dtype=np.float32)
    early = float(g[:3].mean()) if len(g) >= 3 else float(g.mean())
    late  = float(g[-3:].mean()) if len(g) >= 3 else float(g.mean())
    mid_t = _detect_transition_stage(g8, drop_thr=cfg.ar_drop_thr)
    return {
        "early_strength": _strength_label(early),
        "mid_transition": f"stage-{mid_t}",
        "late_refinement": _strength_label(late),
        "_early_mean": round(early, 4),
        "_late_mean": round(late, 4),
    }

def _interpretation_from_policy(g8: List[float], mr: Optional[float]) -> str:
    ps = _policy_summary(g8)
    mid = ps["mid_transition"]
    # optional local/global note
    local_note = ""
    if mr is not None:
        if mr > 0.08:
            local_note = " Localized artifacts are indicated by the spatial mask, so the policy prioritizes local cleanup early while preserving background structures."
        else:
            local_note = " The spatial mask suggests weak localization, so the policy focuses on global restoration."
    return (
        f"The model applies strong restoration in early stages ({ps['early_strength']}), "
        f"then transitions around {mid} to reduce restoration strength, "
        f"and finishes with {ps['late_refinement']} refinement to avoid over-restoration."
        f"{local_note}"
    )

def _stepwise_policy_text(
    g8: List[float],
    mr: Optional[float],
    caption: str,
) -> str:
    """
    Outputs the requested 'Step-wise Restoration Policy Explanation' style text.
    """
    g = [float(v) for v in g8]
    ps = _policy_summary(g)
    mid_stage = int(ps["mid_transition"].split("-")[1])

    lines: List[str] = []
    if caption.strip() != "":
        lines.append(f"[Caption] {caption.strip()}")
        lines.append("")

    lines.append("Step-wise Restoration Policy Explanation")
    lines.append("")

    # 1) Stage-1
    lines.append(
        f"- At Stage-1, the gate value is {g[0]:.2f}, indicating strong activation to recover global structure."
    )

    # 2) Stage-2/3 trend
    if len(g) >= 3:
        lines.append(
            f"- Stages 2 and 3 remain {('highly active' if (g[1] >= 0.80 and g[2] >= 0.80) else 'active')}, "
            f"suggesting persistent degradation across the image."
        )
    elif len(g) == 2:
        lines.append(
            f"- Stage 2 remains active (gate={g[1]:.2f}), suggesting persistent degradation across the image."
        )

    # 3) transition
    if mid_stage >= 2 and mid_stage <= len(g):
        lines.append(
            f"- From {ps['mid_transition']} onward, the gate value decreases, implying that major artifacts have been removed."
        )
    else:
        lines.append(
            "- The gate sequence indicates a gradual reduction of restoration strength after the initial correction."
        )

    # 4) late refinement
    lines.append(
        f"- Later stages focus on {ps['late_refinement']} refinement while avoiding over-restoration."
    )

    # optional mask note
    if mr is not None:
        lines.append("")
        lines.append(f"[Mask] spatial mask ratio≈{mr:.2f}")
        if mr > 0.08:
            lines.append("[Mask] localized degradation likely (e.g., raindrops/snow blobs).")
        else:
            lines.append("[Mask] weak/limited localization; degradation likely global.")

    # raw gates
    lines.append("")
    lines.append("[AR Gate Trace] " + str([round(v, 2) for v in g]))

    return "\n".join(lines)


# ============================================================
# Figure
# ============================================================

def _make_subplot_figure(
    inp01: torch.Tensor,
    out01: torch.Tensor,
    gt01: torch.Tensor,
    M_1hw: Optional[torch.Tensor],
    g_stage_8: List[float],
    text: str,
    save_path: str
):
    """
    2x3:
      [Input | Restored | GT]
      [Mask  | StageGateBar | Text]
    """
    _ensure_dir(os.path.dirname(save_path))

    inp = _tensor_to_uint8_hwc(inp01)
    out = _tensor_to_uint8_hwc(out01)
    gt  = _tensor_to_uint8_hwc(gt01)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2,3,1); ax1.imshow(inp); ax1.set_title("Input"); ax1.axis("off")
    ax2 = fig.add_subplot(2,3,2); ax2.imshow(out); ax2.set_title("Restored"); ax2.axis("off")
    ax3 = fig.add_subplot(2,3,3); ax3.imshow(gt);  ax3.set_title("GT"); ax3.axis("off")

    ax4 = fig.add_subplot(2,3,4)
    if M_1hw is not None:
        m = M_1hw.detach().cpu().float().clamp(0,1).numpy()
        ax4.imshow(m, cmap="jet")
        ax4.set_title("Mask / Spatial map")
    else:
        ax4.text(0.5, 0.5, "Mask: N/A", ha="center", va="center")
        ax4.set_title("Mask / Spatial map")
    ax4.axis("off")

    ax5 = fig.add_subplot(2,3,5)
    xs = np.arange(len(g_stage_8))
    ax5.bar(xs, np.array(g_stage_8, dtype=np.float32))
    ax5.set_ylim(0, 1.0)
    ax5.set_xticks(xs)
    ax5.set_title("AR Stage gate trace (8)")
    ax5.set_xlabel("stage")
    ax5.set_ylabel("gate")

    ax6 = fig.add_subplot(2,3,6)
    ax6.axis("off")
    ax6.set_title("XAI (AR policy + caption)")
    ax6.text(0.0, 1.0, text, va="top", wrap=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# IO
# ============================================================

def _write_json(path: str, obj: Any):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_csv(path: str, rows: List[Dict[str, Any]]):
    _ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Helpers: image_id
# ============================================================

def _get_image_id_from_batch(batch: Dict[str, Any], fallback: str) -> str:
    """
    MultiTaskCLIPCacheDataset implementations vary.
    Try common keys; fallback to provided string.
    """
    for k in ["image_id", "id", "key", "name", "input_path", "path", "inp_path"]:
        if k in batch:
            v = batch[k]
            if isinstance(v, (list, tuple)) and len(v) > 0:
                v = v[0]
            if torch.is_tensor(v):
                continue
            if isinstance(v, str) and len(v) > 0:
                # shorten
                return os.path.basename(v).replace("\\", "/")
    return fallback


# ============================================================
# Main
# ============================================================

def main():
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # output dirs
    triplet_dir = os.path.join(cfg.out_root, "images_triplet")
    fig_dir     = os.path.join(cfg.out_root, "figures")
    _ensure_dir(cfg.out_root)
    _ensure_dir(triplet_dir)
    _ensure_dir(fig_dir)

    # dataset
    dataset = MultiTaskCLIPCacheDataset(
        preload_cache_root=cfg.cache_root,
        datasets=cfg.datasets_cfg,
        crop_size=cfg.crop_size,
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    print("[Data] total items:", len(dataset))

    # build models
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    backbone = VETNetBackbone(
        in_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=(4,6,6,8),
        heads=(1,2,4,8),
        volterra_rank=4,          # ★ Phase-3와 동일
        ffn_expansion_factor=2.66,
        bias=False,
    ).to(device)

    degrader = DegradationEstimator().to(device)

    # clip_dim inference
    sample = dataset[0]
    clip_dim = int(sample["e_img"].numel())

    translator_cfg = ConditionTranslatorConfig(
        clip_dim=clip_dim,
        deg_dim=getattr(degrader, "out_dim", 5),
        num_stages=cfg.num_stages,
        enable_film=cfg.enable_film,
        film_mode=cfg.film_mode,
    )
    translator = ConditionTranslator(translator_cfg).to(device)

    # load
    backbone.load_state_dict(ckpt["backbone"], strict=True)
    degrader.load_state_dict(ckpt["degrader"], strict=True)
    translator.load_state_dict(ckpt["translator"], strict=True)

    backbone.eval(); degrader.eval(); translator.eval()

    # BLIP
    blip_pack = _try_load_blip(device)

    # meta
    meta: Dict[str, Any] = {
        "cfg": cfg.__dict__,
        "has_blip": (blip_pack is not None),
        "has_skimage": _HAS_SKIMAGE,
        "items": []
    }
    csv_rows: List[Dict[str, Any]] = []

    n = 0
    pbar = tqdm(loader, desc="[Phase4 BOTH] exporting", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            if n >= cfg.max_items:
                break

            x = batch["input"].to(device, non_blocking=True)     # [B,3,H,W]
            y = batch["gt"].to(device, non_blocking=True)        # [B,3,H,W]
            e = batch["e_img"].to(device, non_blocking=True)     # [B,D]

            # restore forward
            deg = degrader(x)
            if isinstance(deg, (tuple, list)) and len(deg) >= 2:
                M, v = deg[0], deg[1]
            else:
                M, v = None, deg

            out = translator(e, v)
            g_stage = out["g_stage"]                # [B,8]
            film = out.get("film", None)

            # NOTE: spatial gate visualization uses M directly here
            y_hat = backbone(x, g_stage=g_stage, film=film, spatial_gate=None)

            # first item
            inp01 = _to_01(x[0])
            out01 = _to_01(y_hat[0])
            gt01  = _to_01(y[0])

            # metrics
            psnr_val = _psnr_torch(out01, gt01)
            ssim_val = _ssim_skimage(out01, gt01)

            # triplet save
            trip_path = os.path.join(triplet_dir, f"sample_{n:06d}.png")
            _save_triplet_side_by_side(inp01, out01, gt01, trip_path)

            # caption on restored
            cap = ""
            if blip_pack is not None and _HAS_PIL:
                pil_rest = Image.fromarray(_tensor_to_uint8_hwc(out01))
                cap = _blip_caption(blip_pack, pil_rest, device=device)

            # AR gate trace
            g8 = [float(v) for v in g_stage[0].detach().cpu().float().tolist()]
            mr = _mask_ratio(M, thr=0.5)

            # NEW: step-wise policy text + policy_summary + interpretation
            explain = _stepwise_policy_text(g8, mr, cap)
            policy_sum = _policy_summary(g8)
            interpretation = _interpretation_from_policy(g8, mr)

            # subplot fig save
            fig_path = os.path.join(fig_dir, f"fig_{n:06d}.png")
            M1 = None
            if M is not None and torch.is_tensor(M):
                M1 = M[0, 0].detach()  # [H,W]
            _make_subplot_figure(inp01, out01, gt01, M1, g8, explain, fig_path)

            # image id
            image_id = _get_image_id_from_batch(batch, fallback=f"sample_{n:06d}")

            # record meta (JSON item includes requested fields)
            item = {
                "image_id": image_id,
                "index": n,

                "triplet_path": trip_path,
                "figure_path": fig_path,

                "psnr": float(psnr_val),
                "ssim": (None if ssim_val is None else float(ssim_val)),

                "caption": cap,

                # --- requested AR-XAI outputs ---
                "ar_gate_trace": [float(f"{v:.2f}") for v in g8],
                "policy_summary": {
                    "early_strength": policy_sum["early_strength"],
                    "mid_transition": policy_sum["mid_transition"],
                    "late_refinement": policy_sum["late_refinement"],
                },
                "interpretation": interpretation,

                # keep useful raw info too
                "explain": explain,
                "g_stage": g8,
                "mask_ratio": mr,
            }
            meta["items"].append(item)

            # CSV row (compact)
            csv_rows.append({
                "image_id": image_id,
                "index": n,
                "psnr": float(psnr_val),
                "ssim": ("" if ssim_val is None else float(ssim_val)),
                "mask_ratio": ("" if mr is None else float(mr)),
                "triplet_path": trip_path,
                "figure_path": fig_path,
                "caption": cap,
                "ar_gate_trace": json.dumps([float(f"{v:.2f}") for v in g8]),
                "mid_transition": policy_sum["mid_transition"],
                "early_strength": policy_sum["early_strength"],
                "late_refinement": policy_sum["late_refinement"],
                "interpretation": interpretation,
            })

            n += 1
            pbar.set_postfix({
                "P": f"{psnr_val:.2f}",
                "S": ("NA" if ssim_val is None else f"{ssim_val:.4f}"),
                "mask": ("NA" if mr is None else f"{mr:.3f}")
            })

    # write meta
    json_path = os.path.join(cfg.out_root, "meta.json")
    csv_path  = os.path.join(cfg.out_root, "meta.csv")
    _write_json(json_path, meta)
    _write_csv(csv_path, csv_rows)

    print("\n[DONE] Phase-4 BOTH Export (AR-XAI)")
    print(" - triplets :", triplet_dir)
    print(" - figures  :", fig_dir)
    print(" - meta.json:", json_path)
    print(" - meta.csv :", csv_path)
    if blip_pack is None:
        print(" - caption  : OFF (transformers/BLIP not available or disabled)")
    if not _HAS_SKIMAGE:
        print(" - SSIM     : OFF (skimage not available)")

if __name__ == "__main__":
    main()
