"""
Custom dataset utilities for loading image data for classification.

This module defines a lightweight dataset class compatible with PyTorch's
DataLoader. It expects a directory structure where each subdirectory represents
a class and contains the corresponding images. The dataset automatically assigns
integer labels based on alphabetical class ordering and supports optional
transform pipelines.

The dataset is designed to work with configurable transform pipelines, including
optional data augmentation applied during training. Augmentation is not handled
inside the dataset itself but is passed in through the `transform` argument.
Typical augmentations include random flips, rotations, and color jitter, while
validation transforms remain deterministic to ensure consistent evaluation.
"""

from pathlib import Path
from typing import Callable, Optional, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Dataset for loading images from a directory structured by class folders.

    Expected directory layout:
        root/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg
                img4.jpg

    The dataset:
        - assigns class indices based on alphabetical folder order
        - collects all valid image files recursively
        - applies an optional transform to each image (including augmentation
          when enabled in the training pipeline)

    Args:
        root_dir (str): Path to the dataset root directory.
        transform (Callable, optional): Optional preprocessing transform applied
            to each loaded image. This may include data augmentation during
            training or deterministic preprocessing during validation.

    Attributes:
        root_dir (Path): Normalized dataset root path.
        transform (Callable | None): Transform applied to each image.
        samples (list[tuple[Path, int]]): List of (image_path, class_index) pairs.
        classes (dict[str, int]): Mapping from class name to integer label.
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        self.classes = self._store_class_info_in_dict()
        self._add_samples()

    def _store_class_info_in_dict(self) -> dict:
        """
        Scan the dataset directory and assign integer labels to each class.

        Returns:
            dict[str, int]: Mapping from class names to integer indices.
        """
        classes = sorted([sub_dir.name for sub_dir in self.root_dir.iterdir() if sub_dir.is_dir()])
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def _add_samples(self):
        """
        Populate the internal sample list with all valid image paths.

        Only files with extensions .jpg, .jpeg, or .png are included.
        """
        for cls_name, idx in self.classes.items():
            class_dir = self.root_dir / cls_name #Use Path’s "/"" operator for joining two strings
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
