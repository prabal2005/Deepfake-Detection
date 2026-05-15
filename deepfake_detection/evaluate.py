"""
Evaluation & Metrics — Deepfake Detection CNN
==============================================
Computes on the held-out test set:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC with curve plot
  - Confusion Matrix
  - Per-sample predictions CSV
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

import sys
sys.path.append(os.path.dirname(__file__))
from models.cnn_model import get_model
from dataset.dataloader import get_dataloaders


# ─────────────────────────────────────────────
#  Load checkpoint
# ─────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Loads a saved model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model  = get_model(config["model"], device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} "
          f"| Val Loss: {ckpt['val_loss']:.4f} | Val Acc: {ckpt['val_acc']:.4f}")
    return model, config


# ─────────────────────────────────────────────
#  Inference on test set
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Runs inference on a DataLoader.
    Returns true labels, predicted probabilities, and predicted classes.
    """
    all_labels, all_probs, all_preds = [], [], []

    for images, labels in tqdm(loader, desc="[Inference]"):
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        preds  = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().astype(int))

    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds)
    )


# ─────────────────────────────────────────────
#  Plot: Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        linewidths=0.5, linecolor="gray"
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix — Deepfake Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
#  Plot: AUC-ROC Curve
# ─────────────────────────────────────────────

def plot_roc_curve(y_true, y_probs, auc_score: float, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
             label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] ROC curve saved → {save_path}")


# ─────────────────────────────────────────────
#  Full Evaluation Pipeline
# ─────────────────────────────────────────────

def evaluate_model(checkpoint_path: str, dataset_path: str,
                   results_dir: str = "./results"):
    os.makedirs(results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)

    # Load test data
    loaders = get_dataloaders(
        root_dir   = dataset_path,
        batch_size = config["batch_size"],
        img_size   = config["img_size"],
        num_workers= 4
    )

    # Run inference
    y_true, y_probs, y_pred = run_inference(model, loaders["test"], device)

    # ── Compute Metrics ───────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_probs)

    print(f"\n{'='*50}")
    print("  TEST SET EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"{'='*50}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Real", "Fake"]))

    # ── Save metrics JSON ─────────────────────
    metrics = {
        "accuracy" : round(acc, 6), "precision": round(prec, 6),
        "recall"   : round(rec, 6), "f1_score" : round(f1, 6),
        "auc_roc"  : round(auc, 6),
        "total_samples": int(len(y_true)),
        "real_samples" : int((y_true == 0).sum()),
        "fake_samples" : int((y_true == 1).sum()),
    }
    import json
    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(results_dir, "confusion_matrix.png")
    )
    plot_roc_curve(
        y_true, y_probs, auc,
        os.path.join(results_dir, "roc_curve.png")
    )

    # ── Save per-sample predictions ───────────
    pred_path = os.path.join(results_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_label", "pred_label",
                         "pred_prob", "correct"])
        for i, (tl, pl, pp) in enumerate(zip(y_true, y_pred, y_probs)):
            writer.writerow([i, int(tl), int(pl),
                             round(float(pp), 4), int(tl == pl)])
    print(f"[CSV] Per-sample predictions saved → {pred_path}")

    return metrics


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection CNN")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved .pth checkpoint")
    parser.add_argument("--dataset",    required=True,
                        help="Path to dataset root folder")
    parser.add_argument("--results",    default="./results",
                        help="Directory to save evaluation outputs")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.dataset, args.results)
