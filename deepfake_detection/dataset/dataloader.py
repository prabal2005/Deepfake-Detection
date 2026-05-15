"""
Dataset & Data Preprocessing for Deepfake Detection
=====================================================
Supports: FaceForensics++, DFDC, Celeb-DF, or any custom dataset
organized as:
    dataset/
        train/
            real/   ← real face images
            fake/   ← deepfake images
        val/
            real/
            fake/
        test/
            real/
            fake/
"""

import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
#  1. Data Augmentation & Normalization
# ─────────────────────────────────────────────

# ImageNet mean/std (used because we use pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split: str = "train", img_size: int = 224):
    """
    Returns torchvision transforms for train / val / test splits.

    Training: heavy augmentation to prevent overfitting.
    Val/Test: only resize + normalize.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────
#  2. Custom Dataset Class
# ─────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for deepfake detection.

    Labels:
        0 → REAL
        1 → FAKE (deepfake)
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str, split: str = "train",
                 img_size: int = 224, transform=None):
        """
        Args:
            root_dir : path to dataset root (contains train/val/test folders)
            split    : "train", "val", or "test"
            img_size : input image size for the model
            transform: optional custom transform (overrides default)
        """
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform or get_transforms(split, img_size)
        self.samples   = []   # list of (image_path, label)

        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(self.split_dir, label_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Folder not found: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.VALID_EXTENSIONS:
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx)
                    )

        print(f"[Dataset] Split='{split}' | Total samples: {len(self.samples)}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        print(f"          Real: {real_count} | Fake: {fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image (PIL is required for torchvision transforms)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────
#  3. DataLoader Factory
# ─────────────────────────────────────────────

def get_dataloaders(root_dir: str, batch_size: int = 32,
                    img_size: int = 224, num_workers: int = 4):
    """
    Creates DataLoaders for train, val, and test splits.

    Args:
        root_dir   : path to dataset root folder
        batch_size : number of images per batch
        img_size   : resize all images to (img_size × img_size)
        num_workers: parallel data loading workers

    Returns:
        dict with keys "train", "val", "test"
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = DeepfakeDataset(root_dir, split=split, img_size=img_size)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    return loaders


# ─────────────────────────────────────────────
#  4. Face Extractor Utility (OpenCV)
# ─────────────────────────────────────────────

class FaceExtractor:
    """
    Extracts and crops faces from raw images using OpenCV's
    Haar Cascade face detector before feeding into the CNN.
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def extract_face(self, image_path: str, output_size: int = 224):
        """
        Detects the largest face in the image, crops & resizes it.

        Returns:
            PIL.Image of the cropped face, or the full image if no face found.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read: {image_path}")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            # No face detected → use full image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        else:
            # Pick the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            margin = int(0.15 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_bgr.shape[1], x + w + margin)
            y2 = min(img_bgr.shape[0], y + h + margin)
            face_bgr = img_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img  = Image.fromarray(face_rgb)

        return pil_img.resize((output_size, output_size), Image.LANCZOS)


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Transform test (train split):")
    t = get_transforms("train")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    tensor = t(dummy_img)
    print(f"  Output tensor shape: {tensor.shape}")   # (3, 224, 224)
    print(f"  Min/Max values: {tensor.min():.3f} / {tensor.max():.3f}")
