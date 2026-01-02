# G:/VETNet_pilot/datasets/multitask_dataset_cache.py
import os
from PIL import Image
from torch.utils.data import Dataset
from transforms import TrainTransform

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def is_img(path):
    return os.path.splitext(path)[-1].lower() in IMG_EXT


class MultiTaskDatasetCache(Dataset):
    """
    PNG 캐시 기반 Dataset
    preload_cache/
        DayRainDrop/
            000000_in.png
            000000_gt.png
            ...
        NightRainDrop/
            000171_in.png
            000171_gt.png
        CSD/
        RESIDE-6K/
        rain100H/
    """

    def __init__(self, cache_root, transform=None, size=256):
        super().__init__()
        self.cache_root = cache_root
        self.transform = transform if transform else TrainTransform(size=size)

        self.pairs = []   # (dataset_name, in_path, gt_path)

        datasets = sorted(os.listdir(cache_root))
        for ds_name in datasets:
            ds_dir = os.path.join(cache_root, ds_name)
            if not os.path.isdir(ds_dir):
                continue

            files = sorted([f for f in os.listdir(ds_dir) if f.endswith("_in.png")])
            for f in files:
                idx = f.replace("_in.png", "")
                in_path = os.path.join(ds_dir, f)
                gt_path = os.path.join(ds_dir, f"{idx}_gt.png")

                if not os.path.isfile(gt_path):
                    continue

                self.pairs.append((ds_name, in_path, gt_path))

        print(f"[CACHE DATASET] Total cached pairs = {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ds_name, in_path, gt_path = self.pairs[idx]

        img_in = Image.open(in_path).convert("RGB")
        img_gt = Image.open(gt_path).convert("RGB")

        img_in, img_gt = self.transform(img_in, img_gt)

        return {
            "input": img_in,
            "gt": img_gt,
            "dataset": ds_name,
        }


if __name__ == "__main__":
    ds = MultiTaskDatasetCache("E:/VETNet_CLIP/preload_cache")
    print(len(ds))
    sample = ds[0]
    print(sample["input"].shape, sample["dataset"])
