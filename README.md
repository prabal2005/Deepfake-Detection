# Deepfake Detection using CNN

A deep learning-based research project for detecting manipulated images and videos (deepfakes) using Convolutional Neural Networks (CNNs). This project focuses on identifying fake visual content by analyzing spatial features, facial inconsistencies, and manipulation artifacts in digital media.

---

## 📌 Overview

With the rapid growth of Artificial Intelligence and Generative Models, deepfake technology has become capable of producing highly realistic fake media. Although deepfakes have useful applications in entertainment and media, they also create serious concerns such as misinformation, identity theft, cyber fraud, and privacy violations.

This research proposes a CNN-based deepfake detection system that can distinguish between real and manipulated content with high accuracy.

---

## 🎯 Objectives

- Detect manipulated images and videos using deep learning
- Improve accuracy of deepfake detection systems
- Analyze visual inconsistencies and facial artifacts
- Compare performance with existing detection methods
- Study applications, limitations, and future scope of deepfake detection

---

## 🧠 Proposed Model

The proposed system uses a **Convolutional Neural Network (CNN)** model.

### Why CNN?
- Effective in image and video frame analysis
- Automatically extracts important visual features
- Detects facial inconsistencies and manipulation artifacts
- Faster and less computationally expensive than hybrid models

---

## ⚙️ Methodology

1. Data Collection  
2. Preprocessing  
   - Face Detection  
   - Frame Extraction  
   - Resizing & Normalization  
3. Feature Extraction using CNN  
4. Classification (Real / Fake)  
5. Performance Evaluation

---

## 📂 Dataset Used

The model is trained and evaluated using publicly available datasets:

- **FaceForensics++**
- **DeepFake Detection Challenge (DFDC)**

These datasets contain thousands of real and manipulated videos generated using multiple deepfake techniques.

---

## 📊 Evaluation Metrics

The performance of the model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC (Area Under Curve)
- EER (Equal Error Rate)

---

## 📈 Results

| Metric | Performance |
|--------|-------------|
| Accuracy | ~94.5% |
| Precision | ~93.2% |
| Recall | ~92.8% |
| F1-Score | ~93.0% |
| AUC | ~96% |

The confusion matrix and ROC analysis show that the model performs efficiently with very low error rates.

---

## 📌 Applications

Deepfake detection can be used in:

- Cybersecurity & Fraud Prevention
- Social Media Monitoring
- Digital Forensics
- Media Verification
- Law Enforcement
- Biometric Security
- Finance & Banking

---

## ⚠️ Limitations

- Difficulty in detecting highly realistic deepfakes
- Performance decreases on low-quality videos
- Requires large datasets and computational power
- Real-time detection remains challenging

---

## 🚀 Future Work

Future improvements may include:

- Real-time deepfake detection
- Multi-modal analysis (audio + video + text)
- Transformer-based architectures
- Improved robustness against advanced deepfakes
- Better generalization across unseen datasets

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📚 References

1. Nguyen, T. T., et al. *Deep Learning for Deepfakes Creation and Detection*, IEEE Access, 2022.  
2. Tolosana, R., et al. *Deepfakes and Beyond*, Information Fusion, 2020.  
3. Rossler, A., et al. *FaceForensics++*, ICCV, 2019.  
4. Mirsky, Y., & Lee, W. *The Creation and Detection of Deepfakes: A Survey*, 2020.

---

## 👨‍💻 Author

**Prabal Rai**  
**Rahul Choudhary**
**Piyush Kapoor**
**Raghav Baluni**
Research Project – Deepfake Detection using CNN
