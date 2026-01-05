# ============================================================
# Phase-2: Gate-only Controller Training (CLIP -> g_stage)
# - Backbone frozen
# - No FiLM (gamma/beta), No BLIP text input to student
# - Objective: learn stage-wise policy gates g in R^{B×8}
# - Saves logs + gate statistics + visualizations (dataset/task-wise)
# ============================================================

import os
import sys
import time
import json
import math
import random
import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- matplotlib for plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# ROOT for imports
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- your modules ----
from models.backbone.vetnet_backbone import VETNetBackbone
from models.controller.gate_controller import GateController
from datasets.multitask_clip_cache import MultiTaskCLIPCacheDataset


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # cache root includes per-dataset folders:
    #   preload_cache/CSD/clip_image_embeddings.pt
    #   preload_cache/CSD/clip_image_embeddings_meta.json
    #   preload_cache/CSD/blip_caption.json   (not used in Phase-2)
    cache_root: str = "E:/VETNet_CLIP/preload_cache"

    # Phase-1 backbone checkpoint (you already have)
    backbone_ckpt: str = "E:/VETNet_CLIP/checkpoints/phase1_backbone/epoch_021_L0.0204_P31.45_S0.9371.pth"

    # outputs
    save_root: str = "E:/VETNet_CLIP/checkpoints/phase2_gate"
    results_root: str = "E:/VETNet_CLIP/results/phase2_gate"

    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 0.0

    # loss weights
    lambda_gate: float = 0.10  # 0.05 ~ 0.2 recommended

    # gate constraints
    gate_min: float = 0.0
    gate_max: float = 1.0

    # AMP
    use_amp: bool = True

    # logging / visualization frequency
    log_every: int = 50               # steps
    stats_every_epochs: int = 1       # epoch
    viz_every_epochs: int = 1         # epoch

    # gate visualization sampling (avoid huge memory)
    # accumulate up to N samples per epoch for stats/viz (random subsample)
    max_gate_samples_per_epoch: int = 60000

    # eval metrics on a subset (for speed)
    metric_images_per_batch: int = 1  # compute PSNR/SSIM on first k samples in batch


cfg = Config()


# ============================================================
# Simple PSNR/SSIM (no skimage dependency)
# ============================================================

def _to_uint8_img(t: torch.Tensor) -> np.ndarray:
    # t: [3,H,W] in [0,1]
    x = t.detach().float().cpu().clamp(0, 1).numpy()
    x = np.transpose(x, (1, 2, 0))
    return (x * 255.0 + 0.5).astype(np.uint8)

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # pred, gt: [3,H,W] in [0,1]
    mse = F.mse_loss(pred, gt).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def compute_ssim_simple(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # lightweight SSIM-ish proxy (not exact SSIM)
    # For logging trend only.
    # If you already use skimage in other scripts, you can swap it.
    x = pred.detach().float().cpu().clamp(0, 1)
    y = gt.detach().float().cpu().clamp(0, 1)
    mu_x = x.mean().item()
    mu_y = y.mean().item()
    var_x = x.var(unbiased=False).item()
    var_y = y.var(unbiased=False).item()
    cov = ((x - x.mean()) * (y - y.mean())).mean().item()
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2) + 1e-12)
    return float(ssim)


# ============================================================
# Logging helper (console + file)
# ============================================================

class TeeLogger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
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


# ============================================================
# Gate stats / visualization
# ============================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_gate_histogram(all_gates_np: np.ndarray, out_path: str, title: str):
    # all_gates_np: [N, 8]
    plt.figure()
    plt.hist(all_gates_np.flatten(), bins=50)
    plt.title(title)
    plt.xlabel("gate value")
    plt.ylabel("count")
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_stage_bar(means: np.ndarray, out_path: str, title: str, ylabel: str = "mean gate"):
    # means: [8]
    plt.figure()
    x = np.arange(len(means))
    plt.bar(x, means)
    plt.xticks(x, [f"s{i}" for i in range(len(means))], rotation=0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.ylabel(ylabel)
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_grouped_stage_bar(group_to_means: Dict[str, np.ndarray], out_path: str, title: str):
    # group_to_means: key -> [8]
    groups = list(group_to_means.keys())
    if len(groups) == 0:
        return
    S = 8
    plt.figure(figsize=(12, 4))
    x = np.arange(S)
    width = 0.8 / max(1, len(groups))
    for i, g in enumerate(groups):
        plt.bar(x + i * width, group_to_means[g], width=width, label=g)
    plt.xticks(x + 0.4, [f"s{i}" for i in range(S)])
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend(fontsize=8, ncol=min(4, len(groups)))
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def dump_gate_stats_csv(rows: List[Dict[str, Any]], out_path: str):
    # simple csv writer without pandas
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


# ============================================================
# Checkpoint utils
# ============================================================

def save_checkpoint(path: str, epoch: int, gate_controller: nn.Module, optimizer: torch.optim.Optimizer):
    _ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "gate_controller": gate_controller.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path
    )

def try_load_checkpoint(ckpt_dir: str, gate_controller: nn.Module, optimizer: torch.optim.Optimizer):
    # resume latest "epoch_XXX.pth"
    if not os.path.exists(ckpt_dir):
        return 1
    best_epoch = 0
    best_path = None
    for fn in os.listdir(ckpt_dir):
        if fn.startswith("epoch_") and fn.endswith(".pth"):
            try:
                ep = int(fn.split("_")[1])
                if ep > best_epoch:
                    best_epoch = ep
                    best_path = os.path.join(ckpt_dir, fn)
            except:
                pass
    if best_path is None:
        return 1

    ckpt = torch.load(best_path, map_location="cpu")
    gate_controller.load_state_dict(ckpt["gate_controller"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", best_epoch)) + 1
    print(f"[RESUME] Loaded: {best_path}")
    print(f"[RESUME] Start from epoch {start_epoch}")
    return start_epoch


# ============================================================
# Training
# ============================================================

def train_phase2_gate():
    # outputs
    _ensure_dir(cfg.save_root)
    _ensure_dir(cfg.results_root)
    _ensure_dir(os.path.join(cfg.results_root, "logs"))
    _ensure_dir(os.path.join(cfg.results_root, "stats"))
    _ensure_dir(os.path.join(cfg.results_root, "viz"))

    # log file
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg.results_root, "logs", f"train_phase2_gate_{now}.txt")
    tee = TeeLogger(log_path)
    sys.stdout = tee  # redirect prints

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[Phase2-Gate] Device:", device)

        # dataset / loader
        DATASETS_CFG = {
            "CSD": {"task": "desnow"},
            "DayRainDrop": {"task": "deraindrop"},
            "NightRainDrop": {"task": "deraindrop"},
            "rain100H": {"task": "derain"},
            "rain100L": {"task": "derain"},
            "RESIDE-6K": {"task": "dehaze"},
        }

        dataset = MultiTaskCLIPCacheDataset(
            cfg.cache_root,
            DATASETS_CFG,
        )


        print("[Phase2-Gate] Total items:", len(dataset))

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

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

        # load phase-1 checkpoint
        ckpt = torch.load(cfg.backbone_ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        backbone.load_state_dict(state_dict, strict=True)
        print("[Phase2-Gate] Backbone loaded:", cfg.backbone_ckpt)

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        print("[Phase2-Gate] Backbone frozen:", True)

        # controller
        gate_controller = GateController(
            in_dim=dataset.clip_dim,
            num_stages=backbone.NUM_MACRO_STAGES,
            hidden_dim=512,
            num_layers=3,
            gate_min=cfg.gate_min,
            gate_max=cfg.gate_max,
        ).to(device)

        optimizer = torch.optim.AdamW(
            gate_controller.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        # AMP scaler (keep torch.cuda.amp for compatibility; warning is OK)
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))

        # resume (optional)
        start_epoch = try_load_checkpoint(cfg.save_root, gate_controller, optimizer)

        # main loop
        for epoch in range(start_epoch, cfg.epochs + 1):
            gate_controller.train()

            loss_sum = 0.0
            rec_sum = 0.0
            gate_sum = 0.0
            psnr_sum = 0.0
            ssim_sum = 0.0
            cnt = 0

            # gate accumulation for stats/viz (cpu numpy)
            # store: gates, dataset_name, task
            gates_buf = []
            ds_buf = []
            task_buf = []

            pbar = tqdm(loader, ncols=120, desc=f"Phase2 Epoch {epoch:03d}/{cfg.epochs}")
            for step, batch in enumerate(pbar, start=1):
                # batch fields expected from MultiTaskCLIPCacheDataset:
                #   "input": [B,3,H,W] float
                #   "gt":    [B,3,H,W] float
                #   "e_img": [B,D] float
                #   "dataset": list[str] length B
                #   "task": list[str] length B
                x = batch["input"].to(device, non_blocking=True)
                y = batch["gt"].to(device, non_blocking=True)
                e = batch["e_img"].to(device, non_blocking=True)

                ds_names = batch["dataset"]
                tasks = batch["task"]

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device == "cuda")):
                    g = gate_controller(e)  # [B,8] in [0,1]
                    y_hat = backbone(x, g_stage=g)
                    L_rec = F.l1_loss(y_hat, y)

                    # gate regularization to keep near 1 (all-on default policy)
                    L_gate = torch.mean(torch.abs(g - 1.0))

                    loss = L_rec + cfg.lambda_gate * L_gate

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # metrics (subset)
                with torch.no_grad():
                    # compute on first K samples
                    K = min(cfg.metric_images_per_batch, x.size(0))
                    for i in range(K):
                        psnr_sum += compute_psnr(y_hat[i].clamp(0, 1), y[i].clamp(0, 1))
                        ssim_sum += compute_ssim_simple(y_hat[i].clamp(0, 1), y[i].clamp(0, 1))

                # accumulate
                loss_sum += float(loss.item())
                rec_sum += float(L_rec.item())
                gate_sum += float(L_gate.item())
                cnt += 1

                # buffer gates for stats/viz (subsample if huge)
                with torch.no_grad():
                    g_cpu = g.detach().float().cpu().numpy()  # [B,8]
                    for bi in range(g_cpu.shape[0]):
                        gates_buf.append(g_cpu[bi])
                        ds_buf.append(str(ds_names[bi]))
                        task_buf.append(str(tasks[bi]))

                # limit memory (random downsample)
                if len(gates_buf) > cfg.max_gate_samples_per_epoch:
                    # keep last chunk to reduce CPU cost (simple)
                    gates_buf = gates_buf[-cfg.max_gate_samples_per_epoch:]
                    ds_buf = ds_buf[-cfg.max_gate_samples_per_epoch:]
                    task_buf = task_buf[-cfg.max_gate_samples_per_epoch:]

                # log
                avg_loss = loss_sum / cnt
                avg_rec = rec_sum / cnt
                avg_gate = gate_sum / cnt
                denom = max(1, cnt * min(cfg.metric_images_per_batch, cfg.batch_size))
                avg_psnr = psnr_sum / denom
                avg_ssim = ssim_sum / denom

                pbar.set_postfix({
                    "L": f"{avg_loss:.4f}",
                    "Rec": f"{avg_rec:.4f}",
                    "G": f"{avg_gate:.4f}",
                    "P": f"{avg_psnr:.2f}",
                    "S": f"{avg_ssim:.3f}",
                })

                if (step % cfg.log_every) == 0:
                    print(f"[E{epoch:03d} step {step:05d}] "
                          f"L={avg_loss:.4f} Rec={avg_rec:.4f} G={avg_gate:.4f} "
                          f"P={avg_psnr:.2f} S={avg_ssim:.3f}")

            # ---- epoch done ----
            denom = max(1, cnt * min(cfg.metric_images_per_batch, cfg.batch_size))
            epoch_loss = loss_sum / max(1, cnt)
            epoch_rec = rec_sum / max(1, cnt)
            epoch_gate = gate_sum / max(1, cnt)
            epoch_psnr = psnr_sum / denom
            epoch_ssim = ssim_sum / denom

            print("\n" + "=" * 80)
            print(f"[Phase2][Epoch {epoch:03d}] "
                  f"L={epoch_loss:.4f} Rec={epoch_rec:.4f} G={epoch_gate:.4f} "
                  f"P={epoch_psnr:.2f} S={epoch_ssim:.3f}")
            print("=" * 80)

            # ---- save ckpt ----
            ckpt_path = os.path.join(cfg.save_root, f"epoch_{epoch:03d}.pth")
            save_checkpoint(ckpt_path, epoch, gate_controller, optimizer)
            print(f"[CKPT] Saved: {ckpt_path}")

            # ---- stats + viz ----
            if (epoch % cfg.stats_every_epochs) == 0 or (epoch % cfg.viz_every_epochs) == 0:
                all_gates = np.stack(gates_buf, axis=0) if len(gates_buf) > 0 else None
                if all_gates is None:
                    print("[WARN] No gates collected for stats/viz.")
                    continue

                # overall means
                overall_mean = all_gates.mean(axis=0)
                overall_std = all_gates.std(axis=0)

                # dataset-wise
                ds_to_vals: Dict[str, List[np.ndarray]] = {}
                task_to_vals: Dict[str, List[np.ndarray]] = {}

                for g_vec, dsn, tsk in zip(gates_buf, ds_buf, task_buf):
                    ds_to_vals.setdefault(dsn, []).append(g_vec)
                    task_to_vals.setdefault(tsk, []).append(g_vec)

                ds_means = {k: np.stack(v, 0).mean(0) for k, v in ds_to_vals.items()}
                task_means = {k: np.stack(v, 0).mean(0) for k, v in task_to_vals.items()}

                # ----- print readable summary -----
                print("\n[Gate Stats] Overall mean:", np.round(overall_mean, 4).tolist())
                print("[Gate Stats] Overall std :", np.round(overall_std, 4).tolist())

                # dataset summary
                print("\n[Gate Stats] Dataset-wise mean (first 8 stages):")
                for k in sorted(ds_means.keys()):
                    print(f"  - {k:12s} : {np.round(ds_means[k], 4).tolist()}  (n={len(ds_to_vals[k])})")

                # task summary (derain / dehaze / desnow / deraindrop)
                print("\n[Gate Stats] Task-wise mean:")
                for k in sorted(task_means.keys()):
                    print(f"  - {k:12s} : {np.round(task_means[k], 4).tolist()}  (n={len(task_to_vals[k])})")

                # ----- save CSV -----
                rows = []
                rows.append({
                    "group": "OVERALL",
                    "name": "ALL",
                    "count": all_gates.shape[0],
                    **{f"s{i}_mean": float(overall_mean[i]) for i in range(8)},
                    **{f"s{i}_std": float(overall_std[i]) for i in range(8)},
                })
                for k in sorted(ds_means.keys()):
                    vals = np.stack(ds_to_vals[k], 0)
                    rows.append({
                        "group": "DATASET",
                        "name": k,
                        "count": vals.shape[0],
                        **{f"s{i}_mean": float(ds_means[k][i]) for i in range(8)},
                        **{f"s{i}_std": float(vals.std(0)[i]) for i in range(8)},
                    })
                for k in sorted(task_means.keys()):
                    vals = np.stack(task_to_vals[k], 0)
                    rows.append({
                        "group": "TASK",
                        "name": k,
                        "count": vals.shape[0],
                        **{f"s{i}_mean": float(task_means[k][i]) for i in range(8)},
                        **{f"s{i}_std": float(vals.std(0)[i]) for i in range(8)},
                    })

                csv_path = os.path.join(cfg.results_root, "stats", f"gate_stats_epoch_{epoch:03d}.csv")
                dump_gate_stats_csv(rows, csv_path)
                print(f"[STATS] Saved CSV: {csv_path}")

                # ----- save VIZ -----
                if (epoch % cfg.viz_every_epochs) == 0:
                    # overall histogram
                    hist_path = os.path.join(cfg.results_root, "viz", f"epoch_{epoch:03d}_gate_hist_all.png")
                    save_gate_histogram(all_gates, hist_path, title=f"Gate histogram (all) - epoch {epoch:03d}")

                    # overall stage mean bar
                    bar_path = os.path.join(cfg.results_root, "viz", f"epoch_{epoch:03d}_stage_mean_all.png")
                    save_stage_bar(overall_mean, bar_path, title=f"Stage mean gates (all) - epoch {epoch:03d}")

                    # dataset grouped bar
                    ds_bar_path = os.path.join(cfg.results_root, "viz", f"epoch_{epoch:03d}_stage_mean_by_dataset.png")
                    save_grouped_stage_bar(ds_means, ds_bar_path, title=f"Stage mean gates by dataset - epoch {epoch:03d}")

                    # task grouped bar (derain/noise/haze...)
                    task_bar_path = os.path.join(cfg.results_root, "viz", f"epoch_{epoch:03d}_stage_mean_by_task.png")
                    save_grouped_stage_bar(task_means, task_bar_path, title=f"Stage mean gates by task - epoch {epoch:03d}")

                    print(f"[VIZ] Saved:\n  {hist_path}\n  {bar_path}\n  {ds_bar_path}\n  {task_bar_path}")

        print("\n[FINISHED] Phase-2 Gate training completed.")

    finally:
        # restore stdout
        sys.stdout = tee._stdout
        tee.close()
        print(f"[LOG] Saved log to: {log_path}")


if __name__ == "__main__":
    train_phase2_gate()
