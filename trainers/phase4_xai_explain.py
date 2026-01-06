# gate / mask / degradation 기반 “왜 이렇게 복원했는지” 설명
# BLIP ❌
# 완전 text-free
# phase4_xai_explain.py
import os, torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm

from models.backbone.vetnet_backbone import VETNetBackbone
from models.degradation.degradation_estimator import DegradationEstimator
from models.controller.condition_translator import ConditionTranslator, ConditionTranslatorConfig
from datasets.multitask_clip_cache import MultiTaskCLIPCacheDataset

CKPT_PATH = "E:/VETNet_CLIP/checkpoints/phase3_restore/epoch_050_....pth"
SAVE_ROOT = "E:/VETNet_CLIP/results/phase4_xai/explain"
DEVICE = "cuda"

dataset = MultiTaskCLIPCacheDataset("E:/VETNet_CLIP/preload_cache", train=False)
loader = DataLoader(dataset, batch_size=1)

ckpt = torch.load(CKPT_PATH, map_location="cpu")

backbone = VETNetBackbone().to(DEVICE)
degrader = DegradationEstimator().to(DEVICE)
translator = ConditionTranslator(
    ConditionTranslatorConfig(clip_dim=dataset[0]["e_img"].numel(), num_stages=8)
).to(DEVICE)

backbone.load_state_dict(ckpt["backbone"])
degrader.load_state_dict(ckpt["degrader"])
translator.load_state_dict(ckpt["translator"])

backbone.eval(); degrader.eval(); translator.eval()
os.makedirs(SAVE_ROOT, exist_ok=True)

for i, batch in enumerate(tqdm(loader)):
    x = batch["input"].to(DEVICE)
    e = batch["e_img"].to(DEVICE)

    with torch.no_grad():
        M, v = degrader(x)
        out = translator(e, v)
        g = out["g_stage"]
        y = backbone(x, g_stage=g)

    g_txt = ", ".join([f"{v:.2f}" for v in g[0].cpu().tolist()])
    m_ratio = (M[0,0] > 0.5).float().mean().item()

    explanation = (
        f"Restoration Explanation:\n"
        f"- Detected spatial degradation (ratio={m_ratio:.2f})\n"
        f"- Stage gates activated: [{g_txt}]\n"
        f"- Policy: prioritize local artifact removal, suppress over-sharpening"
    )

    img = (y[0].clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0,0,pil.size[0],80], fill=(0,0,0))
    draw.text((5,5), explanation, fill=(255,255,255))

    pil.save(os.path.join(SAVE_ROOT, f"xai_{i:04d}.png"))
