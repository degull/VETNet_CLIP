# ============================================================
# Phase-3: Restoration-aware Training (Final)
# - Teacher: Phase-2 GateController (CLIP e_img -> g_phase2), frozen (teacher only)
# - Student: DegradationEstimator D(x) + ConditionTranslator T(e_clip ⊕ v_deg) -> (film, g)
# - Backbone: VETNetBackbone, (default: finetune for practical final training)
#
# Loss (FIXED):
#   L = L_restoration + lambda_g * |g - g_phase2|_1
#
# Notes:
# - BLIP captions are NOT used here.
# - Inference remains text-free: uses CLIP image encoder + D(x) + translator.
# ============================================================

import os
import sys
import time
import math
import json
import random
import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ------------------------------------------------------------
# ROOT for imports
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.backbone.vetnet_backbone import VETNetBackbone
from models.controller.gate_controller import GateController
from models.degradation.degradation_estimator import DegradationEstimator
from models.controller.condition_translator import ConditionTranslator
from datasets.multitask_clip_cache import MultiTaskCLIPCacheDataset


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # cache root includes per-dataset folders:
    # preload_cache/<DS>/clip_image_embeddings.pt
    cache_root: str = "E:/VETNet_CLIP/preload_cache"

    # Phase-1 backbone ckpt (starting point)
    backbone_ckpt: str = "E:/VETNet_CLIP/checkpoints/phase1_backbone/epoch_021_L0.0204_P31.45_S0.9371.pth"

    # Phase-2 gate teacher ckpt directory (contains epoch_XXX.pth)
    phase2_gate_ckpt_dir: str = "E:/VETNet_CLIP/checkpoints/phase2_gate"

    # outputs
    save_root: str = "E:/VETNet_CLIP/checkpoints/phase3_restore"
    results_root: str = "E:/VETNet_CLIP/results/phase3_restore"

    # training
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 0
    lr_backbone: float = 5e-5
    lr_controller: float = 2e-4
    weight_decay: float = 0.0

    # crop
    crop_size: int = 256

    # Loss fixed
    lambda_g: float = 0.10  # gate distill weight

    # AMP
    use_amp: bool = True

    # finetune options (practical)
    freeze_backbone: bool = False     # practical default = finetune backbone
    freeze_degradation: bool = False  # train D(x)
    freeze_translator: bool = False   # train translator
    # teacher is always frozen

    # log / save
    log_every: int = 50
    save_every_epochs: int = 1
    eval_images_per_batch: int = 1   # quick PSNR/SSIM trend

    # datasets config (must match your preload_cache subfolders)
    datasets_cfg: Dict[str, Dict[str, str]] = None

    # stage count (fixed)
    num_stages: int = 8


cfg = Config()
if cfg.datasets_cfg is None:
    cfg.datasets_cfg = {
        "CSD": {"task": "desnow"},
        "DayRainDrop": {"task": "deraindrop"},
        "NightRainDrop": {"task": "deraindrop"},
        "rain100H": {"task": "derain"},
        "rain100L": {"task": "derain"},
        "RESIDE-6K": {"task": "dehaze"},
    }


# ============================================================
# Utilities
# ============================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

class TeeLogger:
    def __init__(self, log_path: str):
        _ensure_dir(os.path.dirname(log_path))
        self.fp = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str):
        self._stdout.write(msg)
        self.fp.write(msg)

    def flush(self):
        self._stdout.flush()
        self.fp.flush()

    def close(self):
        try:
            self.fp.close()
        except:
            pass


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_simple(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # lightweight SSIM-ish proxy (trend only)
    x = pred.detach().float().clamp(0, 1)
    y = gt.detach().float().clamp(0, 1)
    mu_x = x.mean().item()
    mu_y = y.mean().item()
    var_x = x.var(unbiased=False).item()
    var_y = y.var(unbiased=False).item()
    cov = ((x - x.mean()) * (y - y.mean())).mean().item()
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2) + 1e-12)
    return float(ssim)


def find_latest_epoch_ckpt(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    best_ep = -1
    best_path = None
    for fn in os.listdir(ckpt_dir):
        if fn.startswith("epoch_") and fn.endswith(".pth"):
            try:
                ep = int(fn.split("_")[1].split(".")[0])
            except:
                continue
            if ep > best_ep:
                best_ep = ep
                best_path = os.path.join(ckpt_dir, fn)
    return best_path


def load_backbone_ckpt(backbone: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    backbone.load_state_dict(state_dict, strict=True)


def save_phase3_ckpt(
    path: str,
    epoch: int,
    backbone: nn.Module,
    degrader: nn.Module,
    translator: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler_state: Optional[dict],
    extra: Dict[str, Any],
):
    _ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "degradation_estimator": degrader.state_dict(),
            "condition_translator": translator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler_state,
            "extra": extra,
        },
        path
    )


def try_resume_phase3(
    ckpt_dir: str,
    backbone: nn.Module,
    degrader: nn.Module,
    translator: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> int:
    latest = find_latest_epoch_ckpt(ckpt_dir)
    if latest is None:
        return 1
    ckpt = torch.load(latest, map_location="cpu")
    backbone.load_state_dict(ckpt["backbone"], strict=True)
    degrader.load_state_dict(ckpt["degradation_estimator"], strict=True)
    translator.load_state_dict(ckpt["condition_translator"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler", None) is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except:
            pass
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"[RESUME] Loaded: {latest}")
    print(f"[RESUME] Start from epoch {start_epoch}")
    return start_epoch


# ============================================================
# Build models
# ============================================================

def build_teacher_gate_controller(clip_dim: int, num_stages: int, device: str) -> GateController:
    """
    Teacher only:
    - load Phase-2 ckpt
    - frozen
    """
    teacher = GateController(
        in_dim=clip_dim,
        num_stages=num_stages,
        hidden_dim=512,
    ).to(device)

    ckpt_path = find_latest_epoch_ckpt(cfg.phase2_gate_ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"Cannot find Phase-2 gate ckpt in: {cfg.phase2_gate_ckpt_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "gate_controller" in ckpt:
        teacher.load_state_dict(ckpt["gate_controller"], strict=True)
    else:
        # fallback: maybe saved as raw state_dict
        teacher.load_state_dict(ckpt, strict=True)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[Teacher] Phase-2 gate loaded: {ckpt_path}")
    return teacher


def set_requires_grad(m: nn.Module, req: bool):
    for p in m.parameters():
        p.requires_grad = req


# ============================================================
# Training
# ============================================================

def train_phase3_restore():
    _ensure_dir(cfg.save_root)
    _ensure_dir(cfg.results_root)
    _ensure_dir(os.path.join(cfg.results_root, "logs"))

    # log file
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg.results_root, "logs", f"train_phase3_restore_{now}.txt")
    tee = TeeLogger(log_path)
    sys.stdout = tee

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[Phase3-Restore] Device:", device)
        print("[Phase3-Restore] AMP:", cfg.use_amp)
        print("[Phase3-Restore] freeze_backbone:", cfg.freeze_backbone)
        print("[Phase3-Restore] freeze_degradation:", cfg.freeze_degradation)
        print("[Phase3-Restore] freeze_translator:", cfg.freeze_translator)
        print("[Phase3-Restore] lambda_g:", cfg.lambda_g)

        # ---------------- Dataset ----------------
        dataset = MultiTaskCLIPCacheDataset(
            preload_cache_root=cfg.cache_root,
            datasets=cfg.datasets_cfg,
            crop_size=cfg.crop_size,
            train=True,
        )
        print("[Phase3-Restore] Total items:", len(dataset))

        # infer clip dim safely
        sample0 = dataset[0]
        clip_dim = int(sample0["e_img"].numel())
        print("[Phase3-Restore] CLIP dim:", clip_dim)

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # ---------------- Models ----------------
        # backbone
        backbone = VETNetBackbone(
            in_channels=3, out_channels=3,
            dim=64,
            num_blocks=(4, 6, 6, 8),
            heads=(1, 2, 4, 8),
            volterra_rank=4,
            ffn_expansion_factor=2.66,
            bias=False,
        ).to(device)

        load_backbone_ckpt(backbone, cfg.backbone_ckpt)
        print("[Phase3-Restore] Backbone loaded:", cfg.backbone_ckpt)

        # degradation estimator
        degrader = DegradationEstimator().to(device)

        # condition translator (outputs: film + gate)
        # NOTE: must match your implementation signature
        translator = ConditionTranslator(
            clip_dim=clip_dim,
            deg_dim=getattr(degrader, "out_dim", 8),  # fallback if your degrader exposes out_dim
            num_stages=cfg.num_stages,
        ).to(device)

        # teacher gate controller (phase2)
        teacher_gate = build_teacher_gate_controller(clip_dim, cfg.num_stages, device)

        # freeze options
        if cfg.freeze_backbone:
            backbone.eval()
            set_requires_grad(backbone, False)
        else:
            backbone.train()
            set_requires_grad(backbone, True)

        if cfg.freeze_degradation:
            degrader.eval()
            set_requires_grad(degrader, False)
        else:
            degrader.train()
            set_requires_grad(degrader, True)

        if cfg.freeze_translator:
            translator.eval()
            set_requires_grad(translator, False)
        else:
            translator.train()
            set_requires_grad(translator, True)

        # ---------------- Optimizer ----------------
        params = []
        if not cfg.freeze_backbone:
            params += [{"params": backbone.parameters(), "lr": cfg.lr_backbone}]
        if not cfg.freeze_degradation:
            params += [{"params": degrader.parameters(), "lr": cfg.lr_controller}]
        if not cfg.freeze_translator:
            params += [{"params": translator.parameters(), "lr": cfg.lr_controller}]

        if len(params) == 0:
            raise RuntimeError("No trainable parameters. Set at least one of freeze_* = False.")

        optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))

        # resume
        start_epoch = try_resume_phase3(cfg.save_root, backbone, degrader, translator, optimizer, scaler)

        # ---------------- Train Loop ----------------
        global_step = 0

        for epoch in range(start_epoch, cfg.epochs + 1):
            t0 = time.time()

            # set modes
            if cfg.freeze_backbone:
                backbone.eval()
            else:
                backbone.train()

            if cfg.freeze_degradation:
                degrader.eval()
            else:
                degrader.train()

            if cfg.freeze_translator:
                translator.eval()
            else:
                translator.train()

            loss_sum = 0.0
            rec_sum = 0.0
            gdist_sum = 0.0
            psnr_sum = 0.0
            ssim_sum = 0.0
            cnt = 0

            # For quick gate monitoring
            gate_mean_accum = torch.zeros(cfg.num_stages, dtype=torch.float64)
            gate_count = 0

            pbar = tqdm(loader, ncols=120, desc=f"Phase3 Epoch {epoch:03d}/{cfg.epochs}")

            for batch in pbar:
                global_step += 1
                x = batch["input"].to(device, non_blocking=True)     # [B,3,H,W] in [0,1]
                y = batch["gt"].to(device, non_blocking=True)        # [B,3,H,W]
                e_clip = batch["e_img"].to(device, non_blocking=True)  # [B,D]

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device == "cuda")):
                    # ---- Teacher (Phase-2 gate) ----
                    with torch.no_grad():
                        g_teacher = teacher_gate(e_clip)  # [B,8], already sigmoid/clamped by teacher model

                    # ---- Student controller ----
                    # degrader should return (M, v) or dict; we handle common cases robustly
                    deg_out = degrader(x)
                    if isinstance(deg_out, (tuple, list)) and len(deg_out) >= 2:
                        M, v = deg_out[0], deg_out[1]
                    elif isinstance(deg_out, dict):
                        M = deg_out.get("map", None)
                        v = deg_out.get("vec", None)
                        if v is None:
                            v = deg_out.get("v", None)
                    else:
                        # fallback: degrader returns only vector
                        M, v = None, deg_out

                    if v is None:
                        raise RuntimeError("DegradationEstimator must provide a global severity vector v.")

                    # translator should output (film, g_student) or dict
                    tr_out = translator(e_clip, v)
                    if isinstance(tr_out, (tuple, list)) and len(tr_out) >= 2:
                        film, g_student = tr_out[0], tr_out[1]
                    elif isinstance(tr_out, dict):
                        film = tr_out.get("film", None)
                        g_student = tr_out.get("g", None)
                        if g_student is None:
                            g_student = tr_out.get("g_stage", None)
                    else:
                        raise RuntimeError("ConditionTranslator must return (film, g_student) or dict with keys.")

                    if g_student is None:
                        raise RuntimeError("ConditionTranslator did not return g_student.")

                    # ---- Restoration ----
                    y_hat = backbone(x, g_stage=g_student, film=film)

                    # ---- Loss (FIXED) ----
                    L_rec = F.l1_loss(y_hat, y)
                    L_gdist = torch.mean(torch.abs(g_student - g_teacher))
                    loss = L_rec + cfg.lambda_g * L_gdist

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # metrics (trend)
                with torch.no_grad():
                    K = min(cfg.eval_images_per_batch, x.size(0))
                    for i in range(K):
                        psnr_sum += compute_psnr(y_hat[i].clamp(0, 1), y[i].clamp(0, 1))
                        ssim_sum += compute_ssim_simple(y_hat[i].clamp(0, 1), y[i].clamp(0, 1))

                    # gate stats
                    g_cpu = g_student.detach().float().mean(dim=0).cpu()
                    gate_mean_accum += g_cpu.double()
                    gate_count += 1

                # accumulate
                loss_sum += float(loss.item())
                rec_sum += float(L_rec.item())
                gdist_sum += float(L_gdist.item())
                cnt += 1

                denom = max(1, cnt * min(cfg.eval_images_per_batch, cfg.batch_size))
                avg_loss = loss_sum / max(1, cnt)
                avg_rec = rec_sum / max(1, cnt)
                avg_gd = gdist_sum / max(1, cnt)
                avg_psnr = psnr_sum / denom
                avg_ssim = ssim_sum / denom

                pbar.set_postfix({
                    "L": f"{avg_loss:.4f}",
                    "Rec": f"{avg_rec:.4f}",
                    "Gd": f"{avg_gd:.4f}",
                    "P": f"{avg_psnr:.2f}",
                    "S": f"{avg_ssim:.3f}",
                })

                if (global_step % cfg.log_every) == 0:
                    gmean = (gate_mean_accum / max(1, gate_count)).numpy()
                    print(f"[E{epoch:03d} step {global_step:06d}] "
                          f"L={avg_loss:.4f} Rec={avg_rec:.4f} Gdist={avg_gd:.4f} "
                          f"P={avg_psnr:.2f} S={avg_ssim:.3f} "
                          f"g_mean={np.round(gmean, 4).tolist()}")

            # ---- epoch done ----
            denom = max(1, cnt * min(cfg.eval_images_per_batch, cfg.batch_size))
            epoch_loss = loss_sum / max(1, cnt)
            epoch_rec = rec_sum / max(1, cnt)
            epoch_gd = gdist_sum / max(1, cnt)
            epoch_psnr = psnr_sum / denom
            epoch_ssim = ssim_sum / denom

            gmean_epoch = (gate_mean_accum / max(1, gate_count)).numpy()
            elapsed = time.time() - t0

            print("\n" + "=" * 90)
            print(f"[Phase3][Epoch {epoch:03d}] time={elapsed/60:.1f}m  "
                  f"L={epoch_loss:.4f} Rec={epoch_rec:.4f} Gdist={epoch_gd:.4f} "
                  f"P={epoch_psnr:.2f} S={epoch_ssim:.3f}")
            print(f"[Phase3][Epoch {epoch:03d}] g_mean={np.round(gmean_epoch, 4).tolist()}")
            print("=" * 90)

            # ---- save ----
            if (epoch % cfg.save_every_epochs) == 0:
                ckpt_name = f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.3f}.pth"
                ckpt_path = os.path.join(cfg.save_root, ckpt_name)

                extra = {
                    "epoch_loss": epoch_loss,
                    "epoch_rec": epoch_rec,
                    "epoch_gdist": epoch_gd,
                    "epoch_psnr": epoch_psnr,
                    "epoch_ssim": epoch_ssim,
                    "g_mean": np.round(gmean_epoch, 6).tolist(),
                    "lambda_g": cfg.lambda_g,
                    "freeze_backbone": cfg.freeze_backbone,
                    "freeze_degradation": cfg.freeze_degradation,
                    "freeze_translator": cfg.freeze_translator,
                    "backbone_ckpt_init": cfg.backbone_ckpt,
                    "phase2_gate_ckpt_dir": cfg.phase2_gate_ckpt_dir,
                }

                save_phase3_ckpt(
                    ckpt_path,
                    epoch,
                    backbone,
                    degrader,
                    translator,
                    optimizer,
                    scaler.state_dict() if (cfg.use_amp and device == "cuda") else None,
                    extra
                )

                print(f"[CKPT] Saved: {ckpt_path}")

        print("\n[FINISHED] Phase-3 restore training completed.")

    finally:
        sys.stdout = tee._stdout
        tee.close()
        print(f"[LOG] Saved log to: {log_path}")


if __name__ == "__main__":
    train_phase3_restore()
