# BACH – Breast Cancer Histology Image Classification

## 📌 Project Overview

This project focuses on classifying breast cancer histology images using a Convolutional Neural Network (CNN). It aims to assist early diagnosis by automatically identifying tissue types in microscopic images into four categories: **Normal, Benign, In Situ carcinoma,** and **Invasive carcinoma**.

Breast cancer is a major global health issue, and traditional diagnosis based on visual assessment is time-consuming and subjective. This project proposes an AI-based solution to support and speed up the diagnostic process.

---

## 🧠 Model Summary

- **Architecture**: Custom CNN with 4 convolutional blocks
- **Framework**: TensorFlow + Keras
- **Layers**:
  - Conv2D (filters: 32, 64, 128, 256; kernel: 3×3; L2 regularization)
  - BatchNormalization
  - ReLU activation
  - MaxPooling2D
  - Dropout
  - GlobalAveragePooling2D
  - Dense (128) + Softmax output layer
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Input Image Size**: Resized to 384×384
- **Training Parameters**:
  - Epochs: 25
  - Batch Size: 16
  - Data Split: 70% Train / 15% Validation / 15% Test

---

## 📊 Dataset Information

- **Dataset**: [Kaggle - BACH 2018](https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images)
- **Image Resolution**: Original: 2048 × 1536 pixels → Resized: 384 × 384
- **Total Images**: 400 (100 per class)
- **Classes**:
  - Normal
  - Benign
  - In Situ carcinoma
  - Invasive carcinoma
- **Preprocessing**:
  - Augmentation: Rotation, shift, zoom, shear, horizontal flip
  - Resizing and normalization

---

## 🔁 Program / Algorithm Flow

1. Load and preprocess dataset
2. Split dataset into train, validation, and test sets
3. Define CNN model architecture
4. Compile the model
5. Train the model on training data
6. Evaluate model on validation and test sets
7. Use model for inference on new input images

---

## 📈 Performance Results

| Metric                 | Value     |
|------------------------|-----------|
| Training Accuracy      | ~60%      |
| Validation Accuracy    | 65%       |
| Test Accuracy          | 50%       |
| Best F1-score (Benign) | 0.62      |
| Worst F1-score (InSitu)| 0.20      |

- Overfitting observed after epoch 20
- Confusion between InSitu, Normal, and Benign classes
- Dataset imbalance and downscaling affected performance

---

## 🧪 Inference Demo

The model allows users to upload histopathological images of breast tissue. After preprocessing, it outputs the predicted class and a confidence score.

### Steps:
1. Upload histology image
2. Image is resized, normalized
3. Model predicts class: **Benign / InSitu / Invasive / Normal**
4. Display prediction with confidence

---
## 🔍 Sliding Window Approach (Exploratory)

Due to the **very high resolution** of original images (2048×1536), resizing to 384×384 caused significant information loss.

To overcome this, we experimented with a **sliding window technique**, where each image was split into smaller patches (e.g., 299×299). This allowed better local feature extraction and preserved more morphological information.

- ✅ Generated ~16,500 patches **per class**
- ❌ However, training on this large patch set was **computationally infeasible** using our environment (Google Colab):
  - One epoch ≈ 3 hours
  - 3-hour session limit interrupted training

While the method showed promise, it was ultimately abandoned due to resource limitations. It is strongly recommended for future implementations with better hardware (GPU clusters or TPUs).

---

## 📉 Limitations

- **High-resolution image problem**: Resizing led to loss of fine-grained tissue structures.
- **Sliding Window Training** was not feasible due to runtime constraints.
- **Overfitting** was observed after ~20 epochs.
- **Poor performance** on subtle or underrepresented classes (e.g. `InSitu`).
- **No model interpretability tools** (e.g., Grad-CAM) implemented.

---

## 🧭 Potential Improvements

- Use **transfer learning** (e.g. ResNet, DenseNet, EfficientNet)
- Apply **nuclear-based patch extraction**
- Adopt **Resolution Adaptive Networks (RANet)** for variable image complexity
- Leverage **semi-supervised learning** for better generalization on limited data
- Incorporate **Grad-CAM** for model explainability

---
## 👨‍💻 Authors

- Adna Hajdarević
- Elma Hodžić
- Nedim Kalajdžija  
**Supervisor**: Merjem Bećirović

---
