"""
Predict — Single Image Deepfake Detection
==========================================
Usage:
    python predict.py --image path/to/face.jpg --checkpoint results/best_model.pth

Returns whether the image is REAL or FAKE along with a confidence score.
"""

import os
import sys
import argparse
import torch
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from models.cnn_model import get_model
from dataset.dataloader import get_transforms


# ─────────────────────────────────────────────
#  Predictor Class
# ─────────────────────────────────────────────

class DeepfakePredictor:
    """
    Loads a trained checkpoint and predicts Real/Fake for a single image.
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_transforms("test", img_size=224)
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt["config"]
        self.model = get_model(config["model"], self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[Predictor] Model '{config['model']}' loaded from: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Predicts if an image is Real or Fake.

        Args:
            image_path: path to a face image (jpg, png, etc.)

        Returns:
            dict with keys: label, confidence, probability_fake
        """
        # Load and preprocess
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Forward pass
        logit = self.model(tensor)
        prob_fake = torch.sigmoid(logit).item()
        is_fake   = prob_fake >= 0.5

        label      = "FAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else (1.0 - prob_fake)

        return {
            "label"           : label,
            "confidence"      : round(confidence * 100, 2),   # as %
            "probability_fake": round(prob_fake, 4),
            "probability_real": round(1.0 - prob_fake, 4),
        }

    def predict_batch(self, image_paths: list) -> list:
        """Predict on a list of image paths."""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result["image"] = path
                results.append(result)
            except Exception as e:
                results.append({"image": path, "error": str(e)})
        return results


# ─────────────────────────────────────────────
#  Pretty Print
# ─────────────────────────────────────────────

def print_result(result: dict):
    label = result.get("label", "ERROR")
    conf  = result.get("confidence", 0)
    p_fake = result.get("probability_fake", 0)
    p_real = result.get("probability_real", 0)

    bar_len  = 30
    fake_bar = "█" * int(p_fake * bar_len)
    real_bar = "█" * int(p_real * bar_len)

    print("\n" + "="*50)
    print(f"  PREDICTION : {label}  ({conf:.2f}% confident)")
    print("="*50)
    print(f"  P(Fake): [{fake_bar:<{bar_len}}] {p_fake:.4f}")
    print(f"  P(Real): [{real_bar:<{bar_len}}] {p_real:.4f}")
    print("="*50 + "\n")


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection — Single Image Prediction")
    parser.add_argument("--image",      required=True,  help="Path to the image file")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pth model checkpoint")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[Error] Image not found: {args.image}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    predictor = DeepfakePredictor(args.checkpoint)
    result    = predictor.predict(args.image)
    print_result(result)
