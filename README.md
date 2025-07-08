# BACH – Breast Cancer Histology Image Classification

## 📌 Project Overview

This project addresses the classification of breast cancer histology images using a Convolutional Neural Network (CNN). It aims to assist early cancer diagnosis by automating the classification of tissue images into four classes: **Normal, Benign, In Situ carcinoma,** and **Invasive carcinoma**.

Breast cancer remains a leading cause of cancer-related deaths worldwide. Since manual examination of histopathological images is time-consuming and subjective, deep learning offers a promising alternative to support and speed up diagnosis.

---

## ⚠️ Key Challenge: High-Resolution Images

The BACH dataset provides **very high-resolution images** (2048 × 1536 pixels), which posed a major challenge during model training. Most CNN architectures are designed for smaller input sizes (e.g., 128×128 or 224×224), so we had to **downscale images to 384×384**, resulting in **loss of important morphological features**. This affected the model's ability to distinguish between visually similar classes such as *In Situ* and *Normal*.

To address this, we explored the **sliding window technique**, where large images are split into smaller patches (e.g., 128×128) to preserve local information and improve feature extraction. Although this significantly improved the quality of input data, the number of generated patches per image (~16,500 per class) made training computationally intensive and **infeasible without GPU acceleration** (one epoch > 3 hours).

### ➕ Proposed Solution:
- Use **pretrained models** (e.g., ResNet, Inception-v3) with **transfer learning**
- Apply **patch-based classification** with **sliding windows** selectively on informative regions
- Use **adaptive architectures** like RANet for multi-scale input handling

---

## 🧠 Model Summary

- **Type**: Custom CNN (built from scratch)
- **Framework**: TensorFlow + Keras
- **Structure**:
  - 4 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPooling)
  - Dropout + GlobalAveragePooling
  - Dense (128) + Softmax output
- **Input Size**: 384 × 384
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## 📊 Dataset Information

- **Source**: [Kaggle – BACH Challenge 2018](https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images)
- **Classes**: Normal, Benign, InSitu, Invasive (100 images per class)
- **Total**: 400 images (17 GB original; 7 GB used after filtering)
- **Preprocessing**:
  - Resize to 384×384
  - Normalize pixel values
  - Augmentations: Rotation, translation, zoom, shear, flip

---

## 🔁 Program / Algorithm Flow

```text
1. Load and preprocess images (resize, normalize, augment)
2. Split dataset (70% train / 15% val / 15% test)
3. Define and compile CNN architecture
4. Train model (25 epochs, batch size = 16)
5. Evaluate performance (accuracy, F1-score)
6. Infer class for new uploaded histology image
