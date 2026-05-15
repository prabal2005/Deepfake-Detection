"""
Grad-CAM Visualization — Deepfake Detection CNN
================================================
Generates Gradient-weighted Class Activation Maps (Grad-CAM)
to visually explain WHICH regions the CNN focuses on when
classifying an image as Real or Fake.

Reference: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations
           from Deep Networks via Gradient-based Localization"
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from models.cnn_model import get_model
from dataset.dataloader import get_transforms


# ─────────────────────────────────────────────
#  Grad-CAM Implementation
# ─────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM heatmaps for any target convolutional layer.

    Usage:
        gcam = GradCAM(model, target_layer=model.block4)
        heatmap = gcam.generate(input_tensor)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes the Grad-CAM heatmap for the given input.

        Args:
            input_tensor: preprocessed image tensor (1, 3, H, W)

        Returns:
            heatmap: numpy array of shape (H, W), values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logit = self.model(input_tensor)
        score = torch.sigmoid(logit)

        # Backward pass (gradient w.r.t. target class)
        self.model.zero_grad()
        score.backward()

        # Pool gradients over spatial dimensions
        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weight activations by pooled gradients
        weighted = (pooled_grads * self.activations).sum(dim=1, keepdim=True)
        heatmap  = F.relu(weighted).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────
#  Overlay Heatmap on Image
# ─────────────────────────────────────────────

def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap on the original image.

    Args:
        original_img: RGB image as numpy array (H, W, 3), uint8
        heatmap     : normalized heatmap (H, W), float [0,1]
        alpha       : blending factor for heatmap

    Returns:
        blended RGB image (H, W, 3), uint8
    """
    # Resize heatmap to image size
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = np.uint8(
        alpha * heatmap_rgb + (1 - alpha) * original_img
    )
    return blended


# ─────────────────────────────────────────────
#  Full Visualization Pipeline
# ─────────────────────────────────────────────

def visualize_gradcam(image_path: str, checkpoint_path: str,
                      save_path: str = None, device: str = None):
    """
    Full pipeline: load model, compute Grad-CAM, plot & save.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model  = get_model(config["model"], device)
    model.load_state_dict(ckpt["model_state"])

    # Pick target layer (last conv block)
    if hasattr(model, "block4"):
        target_layer = model.block4   # Custom CNN
    else:
        target_layer = model.backbone.layer4  # ResNet50

    # ── Preprocess image ──────────────────────
    transform = get_transforms("test", img_size=224)
    pil_img   = Image.open(image_path).convert("RGB")
    tensor    = transform(pil_img).unsqueeze(0).to(device)

    # ── Compute Grad-CAM ──────────────────────
    gcam    = GradCAM(model, target_layer)
    heatmap = gcam.generate(tensor)
    gcam.remove_hooks()

    # ── Get prediction ────────────────────────
    model.eval()
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    label = "FAKE" if prob >= 0.5 else "REAL"
    conf  = prob if prob >= 0.5 else (1 - prob)

    # ── Prepare images for plot ───────────────
    orig_np = np.array(pil_img.resize((224, 224)))
    overlay = overlay_heatmap(orig_np, heatmap)

    # ── Plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    color = "red" if label == "FAKE" else "green"
    axes[2].set_title(
        f"Overlay — Prediction: {label}\nConfidence: {conf*100:.1f}%",
        fontsize=12, fontweight="bold", color=color
    )
    axes[2].axis("off")

    plt.suptitle("Grad-CAM: CNN Attention Visualization for Deepfake Detection",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Grad-CAM] Saved → {save_path}")
    plt.show()
    plt.close()

    return {"label": label, "confidence": conf, "prob_fake": prob}


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save",       default="results/gradcam_output.png")
    args = parser.parse_args()

    visualize_gradcam(args.image, args.checkpoint, args.save)
