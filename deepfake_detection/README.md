# Deepfake Detection using CNN

A deep learning system for detecting AI-generated (deepfake) face images using
Convolutional Neural Networks. Supports a custom CNN built from scratch and a
fine-tuned ResNet50 via transfer learning.

---

## Project Structure

```
deepfake_detection/
│
├── models/
│   └── cnn_model.py          ← CNN architectures (Custom + ResNet50)
│
├── dataset/
│   └── dataloader.py         ← Dataset class, transforms, DataLoader factory
│
├── utils/
│   └── gradcam.py            ← Grad-CAM explainability visualization
│
├── train.py                  ← Full training loop with checkpointing
├── evaluate.py               ← Test-set evaluation, metrics, plots
├── predict.py                ← Single-image inference
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone / download the project
cd deepfake_detection

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Structure

Organize your dataset as follows (supports FaceForensics++, DFDC, Celeb-DF):

```
data/
├── train/
│   ├── real/   ← real face images (.jpg / .png)
│   └── fake/   ← deepfake images
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

## Usage

### 1. Train the Model

```bash
# Train Custom CNN (default)
python train.py --dataset ./data --epochs 30 --batch 32

# Train with ResNet50 (Transfer Learning)
python train.py --model resnet50 --dataset ./data --epochs 20 --lr 5e-5
```

Training automatically saves:
- `results/best_model.pth`  — best checkpoint
- `results/history.json`    — per-epoch metrics
- `results/training_curves.png` — loss/accuracy/AUC plots

### 2. Evaluate on Test Set

```bash
python evaluate.py --checkpoint results/best_model.pth --dataset ./data
```

Outputs:
- Accuracy, Precision, Recall, F1, AUC-ROC
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/predictions.csv`

### 3. Predict a Single Image

```bash
python predict.py --image path/to/face.jpg --checkpoint results/best_model.pth
```

### 4. Grad-CAM Visualization

```bash
python utils/gradcam.py --image path/to/face.jpg --checkpoint results/best_model.pth
```

---

## Model Architecture

### Custom CNN
| Block | Layers | Output Size |
|-------|--------|------------|
| Block 1 | Conv(3→32) × 2, BN, ReLU, MaxPool | 112×112 |
| Block 2 | Conv(32→64) × 2, BN, ReLU, MaxPool | 56×56 |
| Block 3 | Conv(64→128) × 2, BN, ReLU, MaxPool | 28×28 |
| Block 4 | Conv(128→256) × 2, BN, ReLU, MaxPool | 14×14 |
| Head | GlobalAvgPool → FC(256→512→128→1) | scalar |

### Transfer Learning (ResNet50)
- Pretrained on ImageNet
- Frozen early layers (layer1, layer2, layer3)
- Fine-tuned: layer4 + custom classification head

---

## Results (Expected)

| Metric | Custom CNN | ResNet50 |
|--------|-----------|---------|
| Accuracy | ~88% | ~93% |
| AUC-ROC | ~0.92 | ~0.97 |
| F1-Score | ~0.87 | ~0.93 |

---

## Training Details

- **Loss:** Binary Cross-Entropy with Logits (class-weighted)
- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping:** patience=7 epochs
- **Augmentation:** Random crop, flip, rotation, color jitter

---

## References

1. Rossler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019
2. Dolhansky et al., *The DeepFake Detection Challenge (DFDC)*, 2020
3. Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020
4. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
5. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017
