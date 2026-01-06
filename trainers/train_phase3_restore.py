""" # ============================================================
# Phase-3: Restoration-aware Training (FINAL) + Metrics + Preview
#
# 1) Gate distillation (Phase-2 → Phase-3)  [DEFAULT ON]
# 2) FiLM ablation: stage-scalar / channel-wise
# 3) Spatial gate from M(x) (raindrop / snow)
#
# + PSNR/SSIM logging (iter + epoch)
# + Preview saving: (Distorted | Restored | GT) in ONE image
#
# Text-free inference guaranteed.
# ============================================================

import os, sys, math, datetime
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# (metrics + saving)
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


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
class Config:
    cache_root: str = "E:/VETNet_CLIP/preload_cache"
    backbone_ckpt: str = "E:/VETNet_CLIP/checkpoints/phase1_backbone/epoch_021_L0.0204_P31.45_S0.9371.pth"
    phase2_gate_ckpt_dir: str = "E:/VETNet_CLIP/checkpoints/phase2_gate"

    # checkpoints
    save_root: str = "E:/VETNet_CLIP/checkpoints/phase3_restore"
    # previews
    results_root: str = "E:/VETNet_CLIP/results/phase3_restore"

    epochs: int = 150
    batch_size: int = 2
    num_workers: int = 0

    lr_backbone: float = 5e-5
    lr_ctrl: float = 2e-4
    weight_decay: float = 0.0

    crop_size: int = 256
    num_stages: int = 8

    # (name -> cfg dict). If None, auto-detect from cache_root
    datasets_cfg: Optional[Dict[str, Dict]] = None

    # gate distill
    lambda_g: float = 0.1
    lambda_warmup: int = 10

    # ---- ablation switches ----
    enable_film: bool = False
    film_mode: str = "stage_scalar"   # "stage_scalar" | "stage_channel"
    enable_spatial_gate: bool = False # M(x) → G_s

    # ---- training ----
    use_amp: bool = True
    grad_clip: float = 1.0

    # debug / safety
    strict_cache_check: bool = True   # True: raise if missing pt, False: skip dataset

    # ---- logging / preview ----
    log_every: int = 200            # print metrics every N iterations
    preview_every: int = 100        # save (input|output|gt) every N iterations
    preview_max_items: int = 1      # number of samples per preview image (keep 1 to avoid too many files)

    # normalization assumption
    assume_0_1: bool = True         # if True, clamp to [0,1] for metrics/save


cfg = Config()

# Prefer explicit list you provided (stable & reproducible).
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

def ramp_lambda(epoch: int) -> float:
    if epoch <= 1:
        return 0.0
    if epoch >= cfg.lambda_warmup:
        return cfg.lambda_g
    return cfg.lambda_g * (epoch - 1) / (cfg.lambda_warmup - 1)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _pick_latest_file(folder: str) -> str:
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {folder}")
    return max(files, key=os.path.getmtime)

def _validate_cache_root(cache_root: str, datasets_cfg: Dict[str, Dict], strict: bool = True) -> Dict[str, Dict]:
    ok = {}
    missing = []
    for name, dc in datasets_cfg.items():
        droot = os.path.join(cache_root, name)
        clip_pt = os.path.join(droot, "clip_image_embeddings.pt")
        if not os.path.isdir(droot):
            missing.append((name, f"missing folder: {droot}"))
            continue
        if not os.path.exists(clip_pt):
            missing.append((name, f"missing file: {clip_pt}"))
            continue
        ok[name] = dc

    if len(missing) > 0:
        msg = "\n".join([f"- {n}: {why}" for (n, why) in missing])
        if strict:
            raise FileNotFoundError(
                "[CacheCheck] Some datasets are missing required cache files:\n"
                f"{msg}\n\n"
                "Fix: generate clip_image_embeddings.pt for those datasets OR remove them from cfg.datasets_cfg."
            )
        else:
            print("[CacheCheck] Skipping missing datasets:\n" + msg)

    if len(ok) == 0:
        raise RuntimeError("[CacheCheck] No valid cached datasets found. Nothing to train on.")
    return ok

def _to_01(x: torch.Tensor) -> torch.Tensor:
    # We assume cache images are already in [0,1]. Clamp for safety.
    if cfg.assume_0_1:
        return x.clamp(0, 1)
    return x

def _calc_psnr_ssim_batch(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    if not _HAS_SKIMAGE:
        return float("nan"), float("nan")

    pred = _to_01(pred).detach().cpu().numpy()
    gt   = _to_01(gt).detach().cpu().numpy()

    psnr_list, ssim_list = [], []
    for p, g in zip(pred, gt):
        p = p.transpose(1, 2, 0)
        g = g.transpose(1, 2, 0)
        psnr_list.append(peak_signal_noise_ratio(g, p, data_range=1.0))
        ssim_list.append(structural_similarity(g, p, channel_axis=2, data_range=1.0))
    return float(sum(psnr_list) / len(psnr_list)), float(sum(ssim_list) / len(ssim_list))

def _tensor_to_pil(img3chw: torch.Tensor) -> "Image.Image":
    assert _HAS_PIL, "PIL not available"
    img = _to_01(img3chw).detach().cpu()
    img = (img * 255.0).round().byte()
    img = img.permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray(img)

def _save_triplet_side_by_side(
    x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor,
    save_path: str,
    title: Optional[str] = None
):

    if not _HAS_PIL:
        return

    x0 = _tensor_to_pil(x[0])
    o0 = _tensor_to_pil(y_hat[0])
    y0 = _tensor_to_pil(y[0])

    w, h = x0.size
    canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
    canvas.paste(x0, (0, 0))
    canvas.paste(o0, (w, 0))
    canvas.paste(y0, (w * 2, 0))

    if title is not None:
        draw = ImageDraw.Draw(canvas)
        # Minimal overlay (avoid font dependency issues)
        draw.rectangle([0, 0, w * 3, 20], fill=(0, 0, 0))
        draw.text((6, 2), title, fill=(255, 255, 255))

        # labels
        draw.text((6, 22), "Input", fill=(255, 255, 255))
        draw.text((w + 6, 22), "Restored", fill=(255, 255, 255))
        draw.text((w * 2 + 6, 22), "GT", fill=(255, 255, 255))

    _ensure_dir(os.path.dirname(save_path))
    canvas.save(save_path)

# ============================================================
# Teacher (Phase-2 gate oracle)
# ============================================================

class GateTeacher(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)   # keep same name as phase-2

    def forward(self, x):
        return torch.sigmoid(self.mlp(x))

def load_teacher(device: str) -> nn.Module:
    ckpt = _pick_latest_file(cfg.phase2_gate_ckpt_dir)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["gate_controller"] if "gate_controller" in sd else sd

    idx = sorted(int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith("weight"))
    if len(idx) == 0:
        raise KeyError(
            "[Teacher] Cannot find keys like 'mlp.{i}.weight' in checkpoint. "
            "Please check the phase2 checkpoint format."
        )

    dims = [sd[f"mlp.{idx[0]}.weight"].shape[1]]
    for i in idx:
        dims.append(sd[f"mlp.{i}.weight"].shape[0])

    net = GateTeacher(dims).to(device)
    net.load_state_dict(sd, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    print("[Teacher] ckpt:", ckpt)
    print("[Teacher] dims:", dims)
    return net

# ============================================================
# Spatial Gate (M(x) → G_s)
# ============================================================

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, m: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        g = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
        return self.conv(g)

# ============================================================
# Training
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # ---- validate cache ----
    if cfg.datasets_cfg is None:
        cfg.datasets_cfg = {d: {} for d in os.listdir(cfg.cache_root) if os.path.isdir(os.path.join(cfg.cache_root, d))}
        print("[Config] Auto-detected datasets:", list(cfg.datasets_cfg.keys()))

    cfg.datasets_cfg = _validate_cache_root(cfg.cache_root, cfg.datasets_cfg, strict=cfg.strict_cache_check)
    print("[Config] Using datasets:", list(cfg.datasets_cfg.keys()))

    _ensure_dir(cfg.save_root)
    _ensure_dir(cfg.results_root)

    dataset = MultiTaskCLIPCacheDataset(
        preload_cache_root=cfg.cache_root,
        datasets=cfg.datasets_cfg,
        crop_size=cfg.crop_size,
        train=True,
    )
    loader = DataLoader(
        dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda")
    )

    sample_e = dataset[0]["e_img"]
    clip_dim = int(sample_e.shape[-1]) if hasattr(sample_e, "shape") else int(sample_e.numel())
    print("[Data] clip_dim:", clip_dim)
    print("[Data] total items:", len(dataset))

    backbone = VETNetBackbone(
        in_channels=3, out_channels=3,
        dim=64, num_blocks=(4,6,6,8),
        heads=(1,2,4,8), volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ).to(device)

    ckpt = torch.load(cfg.backbone_ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    backbone.load_state_dict(sd, strict=True)
    print("[Backbone] Loaded:", cfg.backbone_ckpt)

    degrader = DegradationEstimator().to(device)

    translator_cfg = ConditionTranslatorConfig(
        clip_dim=clip_dim,
        deg_dim=getattr(degrader, "out_dim", 5),
        num_stages=cfg.num_stages,
        enable_film=cfg.enable_film,
        film_mode=cfg.film_mode,
    )
    translator = ConditionTranslator(translator_cfg).to(device)

    spatial_gate = SpatialGate().to(device) if cfg.enable_spatial_gate else None
    teacher = load_teacher(device)

    # ---- optimizer: separate LR (backbone vs ctrl) ----
    optim = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": degrader.parameters(), "lr": cfg.lr_ctrl},
            {"params": translator.parameters(), "lr": cfg.lr_ctrl},
            *(([{"params": spatial_gate.parameters(), "lr": cfg.lr_ctrl}]) if spatial_gate is not None else []),
        ],
        weight_decay=cfg.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    if not _HAS_SKIMAGE:
        print("[WARN] skimage not found -> PSNR/SSIM will be NaN. Install: pip install scikit-image")
    if not _HAS_PIL:
        print("[WARN] PIL/numpy not found -> preview images will not be saved.")

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        lam = ramp_lambda(epoch)
        print(f"\n[Epoch {epoch:03d}/{cfg.epochs}] lambda_g={lam:.3f}  film={cfg.enable_film}({cfg.film_mode})  spatial={cfg.enable_spatial_gate}")

        backbone.train()
        degrader.train()
        translator.train()
        if spatial_gate is not None:
            spatial_gate.train()

        # ---- epoch accumulators ----
        loss_sum = 0.0
        rec_sum = 0.0
        g_sum = 0.0

        psnr_sum = 0.0
        ssim_sum = 0.0
        n_img = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            global_step += 1

            x = batch["input"].to(device, non_blocking=True)
            y = batch["gt"].to(device, non_blocking=True)
            e = batch["e_img"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                with torch.no_grad():
                    g_t = teacher(e)

                deg = degrader(x)
                M, v = deg if isinstance(deg, (tuple, list)) else (None, deg)

                out = translator(e, v)
                g_s = out["g_stage"]
                film = out.get("film", None)

                if cfg.enable_spatial_gate and (spatial_gate is not None) and (M is not None):
                    G = spatial_gate(M[:, :1], x.shape[-2:])
                else:
                    G = None

                y_hat = backbone(x, g_stage=g_s, film=film, spatial_gate=G)

                # metrics assume [0,1]
                y_hat_01 = _to_01(y_hat)
                y_01 = _to_01(y)

                L_rec = F.l1_loss(y_hat_01, y_01)
                L_g = torch.mean(torch.abs(g_s - g_t))
                loss = L_rec + lam * L_g

            scaler.scale(loss).backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(degrader.parameters()) + list(translator.parameters()),
                    cfg.grad_clip
                )
                if spatial_gate is not None:
                    torch.nn.utils.clip_grad_norm_(spatial_gate.parameters(), cfg.grad_clip)

            scaler.step(optim)
            scaler.update()

            # ---- update accumulators ----
            bs = int(x.size(0))
            loss_sum += float(loss.item()) * bs
            rec_sum += float(L_rec.item()) * bs
            g_sum += float(L_g.item()) * bs
            n_img += bs

            # batch psnr/ssim (costly -> do it with no_grad)
            if _HAS_SKIMAGE:
                with torch.no_grad():
                    p_b, s_b = _calc_psnr_ssim_batch(y_hat_01, y_01)
                psnr_sum += p_b * bs
                ssim_sum += s_b * bs
                psnr_now = p_b
                ssim_now = s_b
            else:
                psnr_now = float("nan")
                ssim_now = float("nan")

            # ---- periodic preview saving ----
            if (cfg.preview_every > 0) and (global_step % cfg.preview_every == 0):
                if _HAS_PIL:
                    save_dir = os.path.join(cfg.results_root, f"epoch_{epoch:03d}")
                    tag = f"ep{epoch:03d}_it{global_step:07d}"
                    save_path = os.path.join(save_dir, f"{tag}.png")
                    title = (
                        f"{tag}  "
                        f"L={loss.item():.4f}  "
                        f"Rec={L_rec.item():.4f}  "
                        f"G={L_g.item():.4f}  "
                        f"PSNR={psnr_now:.2f}  "
                        f"SSIM={ssim_now:.4f}"
                    )

                    _save_triplet_side_by_side(_to_01(x), y_hat_01, y_01, save_path, title=title)

            # ---- tqdm postfix ----
            postfix = {
                "L": f"{loss.item():.4f}",
                "Rec": f"{L_rec.item():.4f}",
                "G": f"{L_g.item():.4f}",
            }
            if _HAS_SKIMAGE:
                postfix["P"] = f"{psnr_now:.2f}"
                postfix["S"] = f"{ssim_now:.3f}"
            pbar.set_postfix(postfix)

            # ---- periodic console log ----
            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                p_avg = (psnr_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")
                s_avg = (ssim_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")
                print(
                    f"[Iter {global_step:07d}] "
                    f"L={loss_sum/n_img:.4f} Rec={rec_sum/n_img:.4f} G={g_sum/n_img:.4f} "
                    f"PSNR={p_avg:.2f} SSIM={s_avg:.4f}"
                )

        # ---- epoch summary ----
        loss_avg = loss_sum / max(1, n_img)
        rec_avg = rec_sum / max(1, n_img)
        g_avg = g_sum / max(1, n_img)
        psnr_avg = (psnr_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")
        ssim_avg = (ssim_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")

        print(
            f"[Epoch {epoch:03d} End] "
            f"L={loss_avg:.4f} Rec={rec_avg:.4f} G={g_avg:.4f} "
            f"PSNR={psnr_avg:.2f} SSIM={ssim_avg:.4f}"
        )

        # ---- save per-epoch checkpoint (Phase-1 style filename) ----
        save_name = (
            f"epoch_{epoch:03d}_"
            f"L{loss_avg:.4f}_"
            f"P{psnr_avg:.2f}_"
            f"S{ssim_avg:.4f}.pth"
        )
        save_path = os.path.join(cfg.save_root, save_name)

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "cfg": cfg.__dict__,

                "backbone": backbone.state_dict(),
                "degrader": degrader.state_dict(),
                "translator": translator.state_dict(),
                "spatial_gate": (spatial_gate.state_dict() if spatial_gate is not None else None),

                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),

                "epoch_metrics": {
                    "loss": loss_avg,
                    "rec": rec_avg,
                    "gate": g_avg,
                    "psnr": psnr_avg,
                    "ssim": ssim_avg,
                }
            },
            save_path
        )
        print("[CKPT] saved:", save_path)

    print("[DONE] Phase-3 Final")

if __name__ == "__main__":
    train()
 """

# 이어서 학습
# ============================================================
# Phase-3: Restoration-aware Training (FINAL) + Metrics + Preview
#
# 1) Gate distillation (Phase-2 → Phase-3)  [DEFAULT ON]
# 2) FiLM ablation: stage-scalar / channel-wise
# 3) Spatial gate from M(x) (raindrop / snow)
#
# + PSNR/SSIM logging (iter + epoch)
# + Preview saving: (Distorted | Restored | GT) in ONE image
#
# Text-free inference guaranteed.
# + RESUME from Phase-3 checkpoint supported (optimizer+scaler+global_step)
# ============================================================

import os, sys, math, datetime
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# (metrics + saving)
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


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
class Config:
    cache_root: str = "E:/VETNet_CLIP/preload_cache"
    backbone_ckpt: str = "E:/VETNet_CLIP/checkpoints/phase1_backbone/epoch_021_L0.0204_P31.45_S0.9371.pth"
    phase2_gate_ckpt_dir: str = "E:/VETNet_CLIP/checkpoints/phase2_gate"

    # checkpoints
    save_root: str = "E:/VETNet_CLIP/checkpoints/phase3_restore"
    # previews
    results_root: str = "E:/VETNet_CLIP/results/phase3_restore"

    # ---- RESUME (Phase-3) ----
    # Set to None to train from scratch
    resume_ckpt: Optional[str] = "E:/VETNet_CLIP/checkpoints/phase3_restore/epoch_001_L0.0336_P27.55_S0.8157.pth"

    epochs: int = 150
    batch_size: int = 2
    num_workers: int = 0

    lr_backbone: float = 5e-5
    lr_ctrl: float = 2e-4
    weight_decay: float = 0.0

    crop_size: int = 256
    num_stages: int = 8

    # (name -> cfg dict). If None, auto-detect from cache_root
    datasets_cfg: Optional[Dict[str, Dict]] = None

    # gate distill
    lambda_g: float = 0.1
    lambda_warmup: int = 10

    # ---- ablation switches ----
    enable_film: bool = False
    film_mode: str = "stage_scalar"   # "stage_scalar" | "stage_channel"
    enable_spatial_gate: bool = False # M(x) → G_s

    # ---- training ----
    use_amp: bool = True
    grad_clip: float = 1.0

    # debug / safety
    strict_cache_check: bool = True   # True: raise if missing pt, False: skip dataset

    # ---- logging / preview ----
    log_every: int = 200            # print metrics every N iterations
    preview_every: int = 100        # save (input|output|gt) every N iterations
    preview_max_items: int = 1      # number of samples per preview image (keep 1 to avoid too many files)

    # normalization assumption
    assume_0_1: bool = True         # if True, clamp to [0,1] for metrics/save


cfg = Config()

# Prefer explicit list you provided (stable & reproducible).
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

def ramp_lambda(epoch: int) -> float:
    if epoch <= 1:
        return 0.0
    if epoch >= cfg.lambda_warmup:
        return cfg.lambda_g
    return cfg.lambda_g * (epoch - 1) / (cfg.lambda_warmup - 1)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _pick_latest_file(folder: str) -> str:
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {folder}")
    return max(files, key=os.path.getmtime)

def _validate_cache_root(cache_root: str, datasets_cfg: Dict[str, Dict], strict: bool = True) -> Dict[str, Dict]:
    """
    Ensures each dataset folder exists and contains clip_image_embeddings.pt.
    If strict=False, datasets missing the pt file are skipped.
    """
    ok = {}
    missing = []
    for name, dc in datasets_cfg.items():
        droot = os.path.join(cache_root, name)
        clip_pt = os.path.join(droot, "clip_image_embeddings.pt")
        if not os.path.isdir(droot):
            missing.append((name, f"missing folder: {droot}"))
            continue
        if not os.path.exists(clip_pt):
            missing.append((name, f"missing file: {clip_pt}"))
            continue
        ok[name] = dc

    if len(missing) > 0:
        msg = "\n".join([f"- {n}: {why}" for (n, why) in missing])
        if strict:
            raise FileNotFoundError(
                "[CacheCheck] Some datasets are missing required cache files:\n"
                f"{msg}\n\n"
                "Fix: generate clip_image_embeddings.pt for those datasets OR remove them from cfg.datasets_cfg."
            )
        else:
            print("[CacheCheck] Skipping missing datasets:\n" + msg)

    if len(ok) == 0:
        raise RuntimeError("[CacheCheck] No valid cached datasets found. Nothing to train on.")
    return ok

def _to_01(x: torch.Tensor) -> torch.Tensor:
    # We assume cache images are already in [0,1]. Clamp for safety.
    if cfg.assume_0_1:
        return x.clamp(0, 1)
    return x

def _calc_psnr_ssim_batch(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    """
    pred, gt: [B,3,H,W] in [0,1]
    returns: (psnr_mean, ssim_mean)
    """
    if not _HAS_SKIMAGE:
        return float("nan"), float("nan")

    pred = _to_01(pred).detach().cpu().numpy()
    gt   = _to_01(gt).detach().cpu().numpy()

    psnr_list, ssim_list = [], []
    for p, g in zip(pred, gt):
        p = p.transpose(1, 2, 0)
        g = g.transpose(1, 2, 0)
        psnr_list.append(peak_signal_noise_ratio(g, p, data_range=1.0))
        ssim_list.append(structural_similarity(g, p, channel_axis=2, data_range=1.0))
    return float(sum(psnr_list) / len(psnr_list)), float(sum(ssim_list) / len(ssim_list))

def _tensor_to_pil(img3chw: torch.Tensor) -> "Image.Image":
    """
    img3chw: [3,H,W] float in [0,1]
    """
    assert _HAS_PIL, "PIL not available"
    img = _to_01(img3chw).detach().cpu()
    img = (img * 255.0).round().byte()
    img = img.permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray(img)

def _save_triplet_side_by_side(
    x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor,
    save_path: str,
    title: Optional[str] = None
):
    """
    Save ONE image: [input | output | gt] side-by-side for first sample in batch.
    """
    if not _HAS_PIL:
        return

    x0 = _tensor_to_pil(x[0])
    o0 = _tensor_to_pil(y_hat[0])
    y0 = _tensor_to_pil(y[0])

    w, h = x0.size
    canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
    canvas.paste(x0, (0, 0))
    canvas.paste(o0, (w, 0))
    canvas.paste(y0, (w * 2, 0))

    if title is not None:
        draw = ImageDraw.Draw(canvas)
        # Minimal overlay (avoid font dependency issues)
        draw.rectangle([0, 0, w * 3, 20], fill=(0, 0, 0))
        draw.text((6, 2), title, fill=(255, 255, 255))

        # labels
        draw.text((6, 22), "Input", fill=(255, 255, 255))
        draw.text((w + 6, 22), "Restored", fill=(255, 255, 255))
        draw.text((w * 2 + 6, 22), "GT", fill=(255, 255, 255))

    _ensure_dir(os.path.dirname(save_path))
    canvas.save(save_path)

# ============================================================
# Teacher (Phase-2 gate oracle)
# ============================================================

class GateTeacher(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)   # keep same name as phase-2

    def forward(self, x):
        return torch.sigmoid(self.mlp(x))

def load_teacher(device: str) -> nn.Module:
    ckpt = _pick_latest_file(cfg.phase2_gate_ckpt_dir)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["gate_controller"] if "gate_controller" in sd else sd

    idx = sorted(int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith("weight"))
    if len(idx) == 0:
        raise KeyError(
            "[Teacher] Cannot find keys like 'mlp.{i}.weight' in checkpoint. "
            "Please check the phase2 checkpoint format."
        )

    dims = [sd[f"mlp.{idx[0]}.weight"].shape[1]]
    for i in idx:
        dims.append(sd[f"mlp.{i}.weight"].shape[0])

    net = GateTeacher(dims).to(device)
    net.load_state_dict(sd, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    print("[Teacher] ckpt:", ckpt)
    print("[Teacher] dims:", dims)
    return net

# ============================================================
# Spatial Gate (M(x) → G_s)
# ============================================================

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, m: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        g = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
        return self.conv(g)

# ============================================================
# Training
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # ---- validate cache ----
    if cfg.datasets_cfg is None:
        cfg.datasets_cfg = {d: {} for d in os.listdir(cfg.cache_root) if os.path.isdir(os.path.join(cfg.cache_root, d))}
        print("[Config] Auto-detected datasets:", list(cfg.datasets_cfg.keys()))

    cfg.datasets_cfg = _validate_cache_root(cfg.cache_root, cfg.datasets_cfg, strict=cfg.strict_cache_check)
    print("[Config] Using datasets:", list(cfg.datasets_cfg.keys()))

    _ensure_dir(cfg.save_root)
    _ensure_dir(cfg.results_root)

    dataset = MultiTaskCLIPCacheDataset(
        preload_cache_root=cfg.cache_root,
        datasets=cfg.datasets_cfg,
        crop_size=cfg.crop_size,
        train=True,
    )
    loader = DataLoader(
        dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda")
    )

    sample_e = dataset[0]["e_img"]
    clip_dim = int(sample_e.shape[-1]) if hasattr(sample_e, "shape") else int(sample_e.numel())
    print("[Data] clip_dim:", clip_dim)
    print("[Data] total items:", len(dataset))

    backbone = VETNetBackbone(
        in_channels=3, out_channels=3,
        dim=64, num_blocks=(4,6,6,8),
        heads=(1,2,4,8), volterra_rank=4,
        ffn_expansion_factor=2.66,
        bias=False,
    ).to(device)

    ckpt = torch.load(cfg.backbone_ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    backbone.load_state_dict(sd, strict=True)
    print("[Backbone] Loaded:", cfg.backbone_ckpt)

    degrader = DegradationEstimator().to(device)

    translator_cfg = ConditionTranslatorConfig(
        clip_dim=clip_dim,
        deg_dim=getattr(degrader, "out_dim", 5),
        num_stages=cfg.num_stages,
        enable_film=cfg.enable_film,
        film_mode=cfg.film_mode,
    )
    translator = ConditionTranslator(translator_cfg).to(device)

    spatial_gate = SpatialGate().to(device) if cfg.enable_spatial_gate else None
    teacher = load_teacher(device)

    # ---- optimizer: separate LR (backbone vs ctrl) ----
    optim = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": degrader.parameters(), "lr": cfg.lr_ctrl},
            {"params": translator.parameters(), "lr": cfg.lr_ctrl},
            *(([{"params": spatial_gate.parameters(), "lr": cfg.lr_ctrl}]) if spatial_gate is not None else []),
        ],
        weight_decay=cfg.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    if not _HAS_SKIMAGE:
        print("[WARN] skimage not found -> PSNR/SSIM will be NaN. Install: pip install scikit-image")
    if not _HAS_PIL:
        print("[WARN] PIL/numpy not found -> preview images will not be saved.")

    # ============================================================
    # Resume from Phase-3 checkpoint (if provided)
    # ============================================================
    start_epoch = 1
    global_step = 0

    if cfg.resume_ckpt is not None and os.path.isfile(cfg.resume_ckpt):
        print("[Resume] Loading checkpoint:", cfg.resume_ckpt)
        rckpt = torch.load(cfg.resume_ckpt, map_location="cpu")

        # NOTE: backbone was already initialized from Phase-1 ckpt,
        # but we override it with Phase-3 checkpoint to truly resume.
        backbone.load_state_dict(rckpt["backbone"], strict=True)
        degrader.load_state_dict(rckpt["degrader"], strict=True)
        translator.load_state_dict(rckpt["translator"], strict=True)

        if spatial_gate is not None and rckpt.get("spatial_gate") is not None:
            spatial_gate.load_state_dict(rckpt["spatial_gate"], strict=True)

        # optimizer/scaler
        if "optim" in rckpt and rckpt["optim"] is not None:
            optim.load_state_dict(rckpt["optim"])
        if "scaler" in rckpt and rckpt["scaler"] is not None:
            scaler.load_state_dict(rckpt["scaler"])

        start_epoch = int(rckpt.get("epoch", 0)) + 1
        global_step = int(rckpt.get("global_step", 0))

        print(
            f"[Resume] Resumed from epoch {int(rckpt.get('epoch', 0))} "
            f"(next epoch={start_epoch}), global_step={global_step}"
        )
    else:
        if cfg.resume_ckpt is not None:
            print("[Resume] resume_ckpt not found ->", cfg.resume_ckpt)
        print("[Resume] Training from scratch (Phase-1 init for backbone).")

    for epoch in range(start_epoch, cfg.epochs + 1):
        lam = ramp_lambda(epoch)
        print(f"\n[Epoch {epoch:03d}/{cfg.epochs}] lambda_g={lam:.3f}  film={cfg.enable_film}({cfg.film_mode})  spatial={cfg.enable_spatial_gate}")

        backbone.train()
        degrader.train()
        translator.train()
        if spatial_gate is not None:
            spatial_gate.train()

        # ---- epoch accumulators ----
        loss_sum = 0.0
        rec_sum = 0.0
        g_sum = 0.0

        psnr_sum = 0.0
        ssim_sum = 0.0
        n_img = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            global_step += 1

            x = batch["input"].to(device, non_blocking=True)
            y = batch["gt"].to(device, non_blocking=True)
            e = batch["e_img"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                with torch.no_grad():
                    g_t = teacher(e)

                deg = degrader(x)
                M, v = deg if isinstance(deg, (tuple, list)) else (None, deg)

                out = translator(e, v)
                g_s = out["g_stage"]
                film = out.get("film", None)

                if cfg.enable_spatial_gate and (spatial_gate is not None) and (M is not None):
                    G = spatial_gate(M[:, :1], x.shape[-2:])
                else:
                    G = None

                y_hat = backbone(x, g_stage=g_s, film=film, spatial_gate=G)

                # metrics assume [0,1]
                y_hat_01 = _to_01(y_hat)
                y_01 = _to_01(y)

                L_rec = F.l1_loss(y_hat_01, y_01)
                L_g = torch.mean(torch.abs(g_s - g_t))
                loss = L_rec + lam * L_g

            scaler.scale(loss).backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(degrader.parameters()) + list(translator.parameters()),
                    cfg.grad_clip
                )
                if spatial_gate is not None:
                    torch.nn.utils.clip_grad_norm_(spatial_gate.parameters(), cfg.grad_clip)

            scaler.step(optim)
            scaler.update()

            # ---- update accumulators ----
            bs = int(x.size(0))
            loss_sum += float(loss.item()) * bs
            rec_sum += float(L_rec.item()) * bs
            g_sum += float(L_g.item()) * bs
            n_img += bs

            # batch psnr/ssim (costly -> do it with no_grad)
            if _HAS_SKIMAGE:
                with torch.no_grad():
                    p_b, s_b = _calc_psnr_ssim_batch(y_hat_01, y_01)
                psnr_sum += p_b * bs
                ssim_sum += s_b * bs
                psnr_now = p_b
                ssim_now = s_b
            else:
                psnr_now = float("nan")
                ssim_now = float("nan")

            # ---- periodic preview saving ----
            if (cfg.preview_every > 0) and (global_step % cfg.preview_every == 0):
                if _HAS_PIL:
                    save_dir = os.path.join(cfg.results_root, f"epoch_{epoch:03d}")
                    tag = f"ep{epoch:03d}_it{global_step:07d}"
                    save_path = os.path.join(save_dir, f"{tag}.png")
                    title = (
                        f"{tag}  "
                        f"L={loss.item():.4f}  "
                        f"Rec={L_rec.item():.4f}  "
                        f"G={L_g.item():.4f}  "
                        f"PSNR={psnr_now:.2f}  "
                        f"SSIM={ssim_now:.4f}"
                    )

                    _save_triplet_side_by_side(_to_01(x), y_hat_01, y_01, save_path, title=title)

            # ---- tqdm postfix ----
            postfix = {
                "L": f"{loss.item():.4f}",
                "Rec": f"{L_rec.item():.4f}",
                "G": f"{L_g.item():.4f}",
            }
            if _HAS_SKIMAGE:
                postfix["P"] = f"{psnr_now:.2f}"
                postfix["S"] = f"{ssim_now:.3f}"
            pbar.set_postfix(postfix)

            # ---- periodic console log ----
            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                p_avg = (psnr_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")
                s_avg = (ssim_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")
                print(
                    f"[Iter {global_step:07d}] "
                    f"L={loss_sum/n_img:.4f} Rec={rec_sum/n_img:.4f} G={g_sum/n_img:.4f} "
                    f"PSNR={p_avg:.2f} SSIM={s_avg:.4f}"
                )

        # ---- epoch summary ----
        loss_avg = loss_sum / max(1, n_img)
        rec_avg = rec_sum / max(1, n_img)
        g_avg = g_sum / max(1, n_img)
        psnr_avg = (psnr_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")
        ssim_avg = (ssim_sum / n_img) if (_HAS_SKIMAGE and n_img > 0) else float("nan")

        print(
            f"[Epoch {epoch:03d} End] "
            f"L={loss_avg:.4f} Rec={rec_avg:.4f} G={g_avg:.4f} "
            f"PSNR={psnr_avg:.2f} SSIM={ssim_avg:.4f}"
        )

        # ---- save per-epoch checkpoint (Phase-1 style filename) ----
        save_name = (
            f"epoch_{epoch:03d}_"
            f"L{loss_avg:.4f}_"
            f"P{psnr_avg:.2f}_"
            f"S{ssim_avg:.4f}.pth"
        )
        save_path = os.path.join(cfg.save_root, save_name)

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "cfg": cfg.__dict__,

                "backbone": backbone.state_dict(),
                "degrader": degrader.state_dict(),
                "translator": translator.state_dict(),
                "spatial_gate": (spatial_gate.state_dict() if spatial_gate is not None else None),

                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),

                "epoch_metrics": {
                    "loss": loss_avg,
                    "rec": rec_avg,
                    "gate": g_avg,
                    "psnr": psnr_avg,
                    "ssim": ssim_avg,
                }
            },
            save_path
        )
        print("[CKPT] saved:", save_path)

    print("[DONE] Phase-3 Final")

if __name__ == "__main__":
    train()
