"""
Deepfake Detection using Convolutional Neural Networks (CNN)
============================================================
Model Architecture: Custom CNN + Transfer Learning (ResNet50)
Authors: [Your Team Names]
Project: Deepfake Detection Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────
#  1. Custom CNN Architecture (Built from Scratch)
# ─────────────────────────────────────────────

class DeepfakeCNN(nn.Module):
    """
    A custom CNN architecture for binary classification:
    Real (0) vs Fake (1) face images.

    Architecture:
        - 4 Convolutional Blocks (Conv → BN → ReLU → MaxPool)
        - Global Average Pooling
        - Fully Connected layers with Dropout
        - Binary output (Sigmoid)
    """

    def __init__(self, num_classes=1):
        super(DeepfakeCNN, self).__init__()

        # Block 1: 3 → 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 224 → 112
            nn.Dropout2d(0.1)
        )

        # Block 2: 32 → 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 112 → 56
            nn.Dropout2d(0.1)
        )

        # Block 3: 64 → 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 56 → 28
            nn.Dropout2d(0.2)
        )

        # Block 4: 128 → 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 28 → 14
            nn.Dropout2d(0.2)
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x  # Raw logits (use BCEWithLogitsLoss during training)


# ─────────────────────────────────────────────
#  2. Transfer Learning Model (ResNet50-based)
# ─────────────────────────────────────────────

class DeepfakeResNet(nn.Module):
    """
    Transfer Learning approach using pretrained ResNet50.
    Fine-tunes the last few layers for deepfake detection.
    """

    def __init__(self, num_classes=1, freeze_layers=True):
        super(DeepfakeResNet, self).__init__()

        # Load pretrained ResNet50
        backbone = models.resnet50(pretrained=True)

        # Optionally freeze early layers
        if freeze_layers:
            for name, param in backbone.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False

        # Remove original FC layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Custom classification head
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out


# ─────────────────────────────────────────────
#  3. Model Factory
# ─────────────────────────────────────────────

def get_model(model_name: str = "custom_cnn", device: str = "cpu"):
    """
    Factory function to create a model by name.

    Args:
        model_name: "custom_cnn" or "resnet50"
        device: "cuda" or "cpu"

    Returns:
        model: PyTorch model moved to device
    """
    if model_name == "custom_cnn":
        model = DeepfakeCNN(num_classes=1)
    elif model_name == "resnet50":
        model = DeepfakeResNet(num_classes=1, freeze_layers=True)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'custom_cnn' or 'resnet50'.")

    model = model.to(device)
    print(f"[Model] '{model_name}' loaded on {device}")
    print(f"[Model] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Test Custom CNN
    model = get_model("custom_cnn", device)
    dummy = torch.randn(4, 3, 224, 224).to(device)   # batch of 4 images
    out = model(dummy)
    print(f"Custom CNN output shape: {out.shape}\n")  # Expected: (4, 1)

    # Test ResNet50
    model2 = get_model("resnet50", device)
    out2 = model2(dummy)
    print(f"ResNet50 output shape:   {out2.shape}\n")  # Expected: (4, 1)
