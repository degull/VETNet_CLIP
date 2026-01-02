""" # PNG 읽어오기
# E:\VETNet_CLIP\trainers\train_phase1_backbone.py
import os, sys, time, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[DEBUG] Using ROOT:", ROOT)

from datasets.multitask_dataset_cache import MultiTaskDatasetCache
from models.backbone.vetnet_backbone import VETNetBackbone
from torch.amp import autocast, GradScaler

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except:
    USE_SKIMAGE = False


# ============================================================
class Config:
    cache_root = "E:/VETNet_CLIP/preload_cache"

    save_root = "E:/VETNet_CLIP/checkpoints/phase1_backbone"
    results_root = "E:/VETNet_CLIP/results/phase1_backbone"

    epochs = 100
    batch_size = 2
    num_workers = 0
    lr = 3e-4

    in_channels = 3
    out_channels = 3
    dim = 64
    num_blocks = (4, 6, 6, 8)
    heads = (1, 2, 4, 8)
    volterra_rank = 4
    ffn_expansion_factor = 2.66
    bias = False

    metric_images_per_batch = 2
    use_amp = True

    preview_count = 3
    iter_save_interval = 150   # 🔵 추가: iteration 저장 주기


cfg = Config()


# ============================================================
def tensor_to_img(t):
    t = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype("uint8")


def save_triplet(input, pred, gt, path):
    inp = tensor_to_img(input)
    pr = tensor_to_img(pred)
    gt_img = tensor_to_img(gt)

    H, W, _ = inp.shape
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, 0:W] = inp
    canvas[:, W:2*W] = pr
    canvas[:, 2*W:3*W] = gt_img

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)


def save_preview_images(inputs, preds, gts, epoch, save_dir, count=3):
    os.makedirs(save_dir, exist_ok=True)

    total = inputs.size(0)
    count = min(count, total)
    idxs = np.random.choice(total, count, replace=False)

    for i, idx in enumerate(idxs):
        path = os.path.join(save_dir, f"epoch_{epoch:03d}_preview_{i:02d}.png")
        save_triplet(inputs[idx], preds[idx], gts[idx], path)


def compute_psnr_ssim(pred, gt):
    if not USE_SKIMAGE:
        return 0, 0
    p = tensor_to_img(pred[0])
    g = tensor_to_img(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2)
    return psnr, ssim


# ============================================================
def train_phase1():

    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)
    os.makedirs(os.path.join(cfg.results_root, "iter"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    dataset = MultiTaskDatasetCache(cfg.cache_root, size=256)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    print("[Data] Total cached samples =", len(dataset))

    model = VETNetBackbone(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        dim=cfg.dim,
        num_blocks=cfg.num_blocks,
        heads=cfg.heads,
        volterra_rank=cfg.volterra_rank,
        ffn_expansion_factor=cfg.ffn_expansion_factor,
        bias=cfg.bias,
    ).to(device)

    print("[Model Params]", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler()

    best_ssim = -1

    # ============================================================
    for epoch in range(1, cfg.epochs + 1):

        model.train()
        loss_sum = 0
        psnr_sum = 0
        ssim_sum = 0
        cnt = 0

        pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch}")

        preview_inp = None
        preview_gt = None
        preview_pred = None

        for it, batch in enumerate(pbar, start=1):

            inp = batch["input"].to(device)
            gt = batch["gt"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                pred = model(inp)
                loss = F.l1_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred_c = pred.clamp(0, 1)

            if preview_inp is None:
                preview_inp = inp.detach().cpu()
                preview_gt = gt.detach().cpu()
                preview_pred = pred_c.detach().cpu()

            # 🔵 150 iteration마다 저장
            if it % cfg.iter_save_interval == 0:
                iter_path = os.path.join(
                    cfg.results_root,
                    "iter",
                    f"epoch_{epoch:03d}_iter_{it:05d}.png"
                )
                save_triplet(inp[0], pred_c[0], gt[0], iter_path)

            ps, ss = compute_psnr_ssim(pred_c, gt)

            loss_sum += loss.item()
            psnr_sum += ps
            ssim_sum += ss
            cnt += 1

            pbar.set_postfix({
                "L": f"{loss_sum/cnt:.4f}",
                "P": f"{psnr_sum/cnt:.2f}",
                "S": f"{ssim_sum/cnt:.3f}",
            })

        epoch_loss = loss_sum / cnt
        epoch_psnr = psnr_sum / cnt
        epoch_ssim = ssim_sum / cnt

        scheduler.step()

        print(f"\n[Epoch {epoch}] Loss={epoch_loss:.4f}  PSNR={epoch_psnr:.2f}  SSIM={epoch_ssim:.4f}")

        save_preview_images(
            preview_inp, preview_pred, preview_gt,
            epoch, cfg.results_root, count=cfg.preview_count
        )

        img_path = os.path.join(
            cfg.results_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.png",
        )
        save_triplet(preview_inp[0], preview_pred[0], preview_gt[0], img_path)

        ckpt_path = os.path.join(
            cfg.save_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_path,
        )

        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            torch.save(model.state_dict(), os.path.join(cfg.save_root, "best_phase1_backbone.pth"))
            print("[BEST] Updated best SSIM model")

    print("\nTraining Finished.")
    print("Best SSIM:", best_ssim)


if __name__ == "__main__":
    train_phase1()
 """

# 이어서 학습
# E:\VETNet_CLIP\trainers\train_phase1_backbone.py
import os, sys, time, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import re # 정규 표현식 모듈 추가

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[DEBUG] Using ROOT:", ROOT)

from datasets.multitask_dataset_cache import MultiTaskDatasetCache
from models.backbone.vetnet_backbone import VETNetBackbone
from torch.cuda.amp import autocast, GradScaler # torch.amp 대신 torch.cuda.amp 사용 권장

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except:
    USE_SKIMAGE = False


# ============================================================
class Config:
    cache_root = "E:/VETNet_CLIP/preload_cache"

    save_root = "E:/VETNet_CLIP/checkpoints/phase1_backbone"
    results_root = "E:/VETNet_CLIP/results/phase1_backbone"

    epochs = 100 # 전체 목표 에포크 (이어서 훈련 시 시작 에포크에 따라 실제 훈련 횟수 결정)
    batch_size = 2
    num_workers = 0 
    lr = 3e-4

    in_channels = 3
    out_channels = 3
    dim = 64
    num_blocks = (4, 6, 6, 8)
    heads = (1, 2, 4, 8)
    volterra_rank = 4
    ffn_expansion_factor = 2.66
    bias = False

    metric_images_per_batch = 2
    use_amp = True

    # 🔵 새로 추가: 미리보기로 저장할 이미지 수
    preview_count = 3


cfg = Config()


# ============================================================
def tensor_to_img(t):
    # detach(), cpu(), clamp(0, 1), permute(1, 2, 0) (C, H, W -> H, W, C), numpy()
    t = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype("uint8")


def save_triplet(input, pred, gt, path):
    inp = tensor_to_img(input)
    pr = tensor_to_img(pred)
    gt_img = tensor_to_img(gt)

    H, W, _ = inp.shape
    # Input | Prediction | Ground Truth 순서로 이미지를 나열
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, 0:W] = inp
    canvas[:, W:2*W] = pr
    canvas[:, 2*W:3*W] = gt_img

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)


# 🔵 랜덤 미리보기 저장 기능
def save_preview_images(inputs, preds, gts, epoch, save_dir, count=3):
    os.makedirs(save_dir, exist_ok=True)

    total = inputs.size(0)
    count = min(count, total)

    # 랜덤 선택
    idxs = np.random.choice(total, count, replace=False)

    for i, idx in enumerate(idxs):
        path = os.path.join(save_dir, f"epoch_{epoch:03d}_preview_{i:02d}.png")
        # idx는 랜덤으로 선택된 배치 인덱스
        save_triplet(inputs[idx], preds[idx], gts[idx], path)


def compute_psnr_ssim(pred, gt):
    if not USE_SKIMAGE:
        return 0, 0
    # 배치에서 첫 번째 이미지 사용
    p = tensor_to_img(pred[0])
    g = tensor_to_img(gt[0])
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    # channel_axis=2는 (H, W, C) 형식임을 지정
    ssim = structural_similarity(g, p, channel_axis=2)
    return psnr, ssim


# 🔵 체크포인트 로드 함수 (추가)
def load_checkpoint(save_root, model, optimizer, scheduler):
    start_epoch = 1
    best_ssim = -1.0
    latest_ckpt_path = None
    latest_epoch = 0

    # 체크포인트 파일 목록 검색
    if os.path.exists(save_root):
        files = os.listdir(save_root)
        
        # 'epoch_XXX...' 형식의 파일 중 가장 큰 에포크 번호를 찾습니다.
        pattern = re.compile(r"epoch_(\d{3})_L.*\.pth")
        
        for file in files:
            match = pattern.match(file)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_ckpt_path = os.path.join(save_root, file)

    if latest_ckpt_path:
        print(f"\n[INFO] Latest checkpoint found: {latest_ckpt_path}")
        try:
            checkpoint = torch.load(latest_ckpt_path)
            
            # 모델 상태 로드
            model.load_state_dict(checkpoint["state_dict"])
            
            # 옵티마이저 상태 로드
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            
            # 스케줄러 상태 로드
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
                
            # 시작 에포크 및 최고 SSIM 업데이트
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1 # 다음 에포크부터 시작
            
            # 파일 이름에서 SSIM 값을 추출하여 best_ssim 업데이트 시도
            ssim_match = re.search(r"S([\d\.]+)\.pth$", latest_ckpt_path)
            if ssim_match:
                best_ssim = float(ssim_match.group(1))
            
            print(f"[INFO] Resuming training from Epoch {start_epoch}, Current Best SSIM: {best_ssim:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint {latest_ckpt_path}: {e}")
            start_epoch = 1 # 로드 실패 시 에포크 1부터 다시 시작
            best_ssim = -1.0
    else:
        print("[INFO] No previous checkpoint found. Starting training from Epoch 1.")

    return start_epoch, best_ssim


# ============================================================
def train_phase1():

    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # ============================================================
    # 1. 데이터 로더 설정
    # ============================================================
    dataset = MultiTaskDatasetCache(cfg.cache_root, size=256)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers, # num_workers=0 설정 사용
        pin_memory=True,
        drop_last=True,
    )

    print("[Data] Total cached samples =", len(dataset))

    # ============================================================
    # 2. 모델, 옵티마이저, 스케줄러, 스케일러 설정
    # ============================================================
    model = VETNetBackbone(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        dim=cfg.dim,
        num_blocks=cfg.num_blocks,
        heads=cfg.heads,
        volterra_rank=cfg.volterra_rank,
        ffn_expansion_factor=cfg.ffn_expansion_factor,
        bias=cfg.bias,
    ).to(device)

    print("[Model Params]", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.use_amp) # GradScaler에 enabled 인자 추가

    # ============================================================
    # 3. 체크포인트 로드 (추가된 부분)
    # ============================================================
    start_epoch, best_ssim = load_checkpoint(cfg.save_root, model, optimizer, scheduler)
    
    # 훈련 시작 에포크부터 전체 에포크까지 반복
    for epoch in range(start_epoch, cfg.epochs + 1):

        model.train()
        loss_sum = 0
        psnr_sum = 0
        ssim_sum = 0
        cnt = 0

        pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch}")

        # 🔵 미리보기 저장용 임시 버퍼 (새 에포크마다 초기화)
        preview_inp = None
        preview_gt = None
        preview_pred = None

        for batch in pbar:
            inp = batch["input"].to(device) # 입력 이미지 (저화질/노이즈 등)
            gt = batch["gt"].to(device)     # 정답 이미지 (고화질)

            optimizer.zero_grad(set_to_none=True)

            # AMP(자동 혼합 정밀도) 사용
            with autocast(dtype=torch.float16, enabled=cfg.use_amp):
                pred = model(inp)
                loss = F.l1_loss(pred, gt)

            # 역전파 및 가중치 업데이트 (GradScaler 사용)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred_c = pred.clamp(0, 1) # 예측 결과를 0~1 사이로 클램핑

            # 🔵 미리보기용 첫 배치를 저장
            if preview_inp is None:
                # 훈련 중이지만, detach/cpu 하여 이후 이미지 저장에 사용
                preview_inp = inp.detach().cpu()
                preview_gt = gt.detach().cpu()
                preview_pred = pred_c.detach().cpu()

            # 평가 지표 계산 (첫 번째 샘플에 대해서만 계산)
            ps, ss = compute_psnr_ssim(pred_c, gt)

            loss_sum += loss.item()
            psnr_sum += ps
            ssim_sum += ss
            cnt += 1

            # tqdm 막대에 현재 평균 지표 표시
            pbar.set_postfix({
                "L": f"{loss_sum/cnt:.4f}", # L1 Loss 평균
                "P": f"{psnr_sum/cnt:.2f}", # PSNR 평균
                "S": f"{ssim_sum/cnt:.3f}", # SSIM 평균
            })

        epoch_loss = loss_sum / cnt
        epoch_psnr = psnr_sum / cnt
        epoch_ssim = ssim_sum / cnt

        scheduler.step() # 에포크 종료 후 스케줄러 업데이트

        print(f"\n[Epoch {epoch}] Loss={epoch_loss:.4f}  PSNR={epoch_psnr:.2f}  SSIM={epoch_ssim:.4f}")

        # ======================================================
        # 🔵 랜덤 Preview 이미지 저장 (preview_count 수만큼)
        # ======================================================
        save_preview_images(preview_inp, preview_pred, preview_gt,
                             epoch, cfg.results_root, count=cfg.preview_count)

        # ======================================================
        # 대표 이미지 저장 (첫 1장)
        # ======================================================
        img_path = os.path.join(
            cfg.results_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.png",
        )
        save_triplet(preview_inp[0], preview_pred[0], preview_gt[0], img_path)
        # [Image of Triplet image: Input, Prediction, Ground Truth]

        # ======================================================
        # checkpoint 저장
        # ======================================================
        ckpt_path = os.path.join(
            cfg.save_root,
            f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                # "best_ssim": best_ssim # best_ssim도 저장할 수 있으나, 여기서는 파일 이름 기반으로 처리
            },
            ckpt_path,
        )

        # 최고 SSIM 모델 저장
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            # 주의: 모델의 state_dict만 저장합니다.
            torch.save(model.state_dict(), os.path.join(cfg.save_root, "best_phase1_backbone.pth"))
            print("[BEST] Updated best SSIM model")

    print("\nTraining Finished.")
    print("Best SSIM:", best_ssim)


if __name__ == "__main__":
    train_phase1()