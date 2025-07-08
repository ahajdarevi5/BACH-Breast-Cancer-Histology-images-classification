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

## ⚠️ Limitations

- Limited generalization due to small dataset
- High-resolution images downscaled → loss of detail
- Difficulty in distinguishing visually similar classes (e.g., InSitu vs. Normal)

---

## ✅ Future Improvements

- Use of pretrained models (e.g., ResNet, DenseNet, Inception-v3)
- Apply patch-based classification (sliding window)
- Employ attention mechanisms (e.g., Grad-CAM)
- Increase dataset size and explore semi-supervised learning

---

## 👨‍💻 Authors

- Adna Hajdarević
- Elma Hodžić
- Nedim Kalajdžija  
**Supervisor**: Merjem Bećirović

---

