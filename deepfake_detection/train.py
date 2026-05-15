"""
Training Script — Deepfake Detection CNN
=========================================
Features:
  - Binary Cross Entropy loss with class-weighting
  - Adam optimizer with ReduceLROnPlateau scheduler
  - Early stopping
  - Best model checkpointing
  - Per-epoch metrics: Loss, Accuracy, AUC-ROC
  - Training curve export
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.cnn_model import get_model
from dataset.dataloader import get_dataloaders


# ─────────────────────────────────────────────
#  Config / Hyperparameters
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model"       : "custom_cnn",   # "custom_cnn" or "resnet50"
    "dataset_path": "./data",
    "img_size"    : 224,
    "batch_size"  : 32,
    "num_epochs"  : 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience"    : 7,              # Early stopping patience
    "checkpoint"  : "./results/best_model.pth",
    "results_dir" : "./results",
}


# ─────────────────────────────────────────────
#  Training Helper Functions
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one full epoch over the training set."""
    model.train()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B,) → (B,1)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
    epoch_auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return epoch_loss, epoch_acc, epoch_auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates model on val/test set. Returns loss, accuracy, AUC-ROC."""
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
    epoch_auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return epoch_loss, epoch_acc, epoch_auc


# ─────────────────────────────────────────────
#  Plot Training Curves
# ─────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str):
    """Saves a 3-panel plot: Loss | Accuracy | AUC-ROC."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("train_loss",  "val_loss",  "Loss",     "Loss"),
        ("train_acc",   "val_acc",   "Accuracy", "Accuracy"),
        ("train_auc",   "val_auc",   "AUC-ROC",  "AUC-ROC Score"),
    ]

    for ax, (train_key, val_key, title, ylabel) in zip(axes, metrics):
        ax.plot(epochs, history[train_key], "b-o", label="Train", markersize=3)
        ax.plot(epochs, history[val_key],   "r-o", label="Val",   markersize=3)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Deepfake Detection CNN — Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


# ─────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────

def train(config: dict):
    os.makedirs(config["results_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  Deepfake Detection — CNN Training")
    print(f"  Model      : {config['model']}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {config['batch_size']}")
    print(f"  Epochs     : {config['num_epochs']}")
    print(f"  LR         : {config['learning_rate']}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────
    loaders = get_dataloaders(
        root_dir    = config["dataset_path"],
        batch_size  = config["batch_size"],
        img_size    = config["img_size"],
        num_workers = 4
    )

    # ── Model ─────────────────────────────────
    model = get_model(config["model"], device)

    # ── Loss (pos_weight handles class imbalance) ──
    pos_weight = torch.tensor([2.0]).to(device)  # upweight fake class
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer & Scheduler ─────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, verbose=True)

    # ── History ───────────────────────────────
    history = {k: [] for k in
               ["train_loss", "val_loss", "train_acc",
                "val_acc", "train_auc", "val_auc"]}

    best_val_loss = float("inf")
    patience_counter = 0

    # ── Epoch Loop ────────────────────────────
    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_auc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        vl_loss, vl_acc, vl_auc = evaluate(
            model, loaders["val"], criterion, device)

        scheduler.step(vl_loss)
        elapsed = time.time() - t0

        # Log
        print(f"Epoch [{epoch:02d}/{config['num_epochs']}] "
              f"| Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  AUC: {tr_auc:.4f} "
              f"| Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  AUC: {vl_auc:.4f} "
              f"| {elapsed:.1f}s")

        for k, v in zip(history.keys(),
                        [tr_loss, vl_loss, tr_acc, vl_acc, tr_auc, vl_auc]):
            history[k].append(v)

        # Checkpoint
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_loss"   : vl_loss,
                "val_acc"    : vl_acc,
                "val_auc"    : vl_auc,
                "config"     : config,
            }, config["checkpoint"])
            print(f"  ✓ Best model saved (val_loss={vl_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n[Early Stop] No improvement for {config['patience']} epochs.")
                break

    # ── Save history & plots ──────────────────
    history_path = os.path.join(config["results_dir"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(
        history,
        os.path.join(config["results_dir"], "training_curves.png")
    )

    print(f"\n[Done] Best Val Loss: {best_val_loss:.4f}")
    return history


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection CNN")
    parser.add_argument("--model",     default="custom_cnn",
                        choices=["custom_cnn", "resnet50"])
    parser.add_argument("--dataset",   default="./data")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--results",   default="./results")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "model"        : args.model,
        "dataset_path" : args.dataset,
        "num_epochs"   : args.epochs,
        "batch_size"   : args.batch,
        "learning_rate": args.lr,
        "results_dir"  : args.results,
        "checkpoint"   : os.path.join(args.results, "best_model.pth"),
    })

    train(config)
