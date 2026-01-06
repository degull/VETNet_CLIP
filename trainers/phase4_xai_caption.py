# BLIP 기반 자연어 캡션 중심
# what is visible
# 장면/객체 중심
# phase4_xai_caption.py
import os, torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm

from transformers import BlipProcessor, BlipForConditionalGeneration

from models.backbone.vetnet_backbone import VETNetBackbone
from models.degradation.degradation_estimator import DegradationEstimator
from models.controller.condition_translator import ConditionTranslator, ConditionTranslatorConfig
from datasets.multitask_clip_cache import MultiTaskCLIPCacheDataset

# ---------------- CONFIG ----------------
CKPT_PATH = "E:/VETNet_CLIP/checkpoints/phase3_restore/epoch_050_....pth"
CACHE_ROOT = "E:/VETNet_CLIP/preload_cache"
SAVE_ROOT  = "E:/VETNet_CLIP/results/phase4_xai/caption"
MAX_ITEMS  = 200
DEVICE = "cuda"

# ---------------- BLIP ----------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE).eval()

def caption(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip.generate(**inputs, max_new_tokens=25)
    return processor.decode(out[0], skip_special_tokens=True)

# ---------------- MODEL ----------------
dataset = MultiTaskCLIPCacheDataset(CACHE_ROOT, datasets=None, train=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

ckpt = torch.load(CKPT_PATH, map_location="cpu")

backbone = VETNetBackbone().to(DEVICE)
degrader = DegradationEstimator().to(DEVICE)

clip_dim = dataset[0]["e_img"].numel()
translator = ConditionTranslator(
    ConditionTranslatorConfig(clip_dim=clip_dim, num_stages=8)
).to(DEVICE)

backbone.load_state_dict(ckpt["backbone"])
degrader.load_state_dict(ckpt["degrader"])
translator.load_state_dict(ckpt["translator"])

backbone.eval(); degrader.eval(); translator.eval()
os.makedirs(SAVE_ROOT, exist_ok=True)

# ---------------- LOOP ----------------
for i, batch in enumerate(tqdm(loader)):
    if i >= MAX_ITEMS: break

    x = batch["input"].to(DEVICE)
    e = batch["e_img"].to(DEVICE)

    with torch.no_grad():
        v = degrader(x)
        g = translator(e, v)["g_stage"]
        y = backbone(x, g_stage=g)

    img = (y[0].clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()
    pil = Image.fromarray(img)

    cap = caption(pil)

    draw = ImageDraw.Draw(pil)
    draw.rectangle([0,0,pil.size[0],30], fill=(0,0,0))
    draw.text((5,5), cap, fill=(255,255,255))

    pil.save(os.path.join(SAVE_ROOT, f"xai_{i:04d}.png"))
