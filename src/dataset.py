import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Expects a directory structure like:

        root/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg
                img4.jpg
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        self.class_to_idx = self._find_classes()
        self._scan_files()

    def _find_classes(self) -> dict:
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def _scan_files(self):
        for cls_name, idx in self.class_to_idx.items():
            class_dir = self.root_dir / cls_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
