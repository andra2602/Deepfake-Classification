# Deepfake Image Classification - CNN & MLP Models

This repository contains two machine learning models developed for the [**Deepfake Classification - UniBuc Competition**](https://www.kaggle.com/competitions/deepfake-classification-unibuc/overview) hosted on Kaggle.

The challenge focused on detecting **AI-generated (deepfake) faces**, requiring participants to train classifiers that distinguish between authentic and synthetically generated images.  

This is a **multi-class classification** task, where each image must be classified into one of **five classes** (0 to 4).

## 🗂️ Dataset Overview

- **Training set:** 12,500 images  
- **Validation set:** 1,250 images  
- **Test set:** 6,500 images  

### 📄 Metadata files:
- `train.csv` – image filenames + labels
- `validation.csv` – validation filenames + labels
- `test.csv` – test image filenames (no labels)
- `sample_submission.csv` – submission format reference

All images are provided in **.png format** and are paired with metadata files in **CSV format** (`image_id,label`).

## 📁 Project Structure
- `model_andra_cnn_final.py` – Deep Convolutional Neural Network for image classification
- `model_andra_mlp.py` – Multilayer Perceptron trained on flattened image data
- `Andruta_AndraMihaela_232_doc.pdf` – 📄 Full documentation with technical explanation and experiments

## Model Architectures
### 🧠 Convolutional Neural Network (CNN) Architecture

The CNN model is designed to extract spatial features from RGB images and classify them into one of 5 deepfake classes.

It consists of 4 convolutional blocks, each followed by Batch Normalization, ReLU activation, and MaxPooling. Dropout regularization is applied before and after the first fully connected layer to reduce overfitting.

### 🔧 Architecture Overview:

- **Input:** RGB image (3 channels)
- **Conv Block 1:**  
  - `Conv2D(3 → 16)` → `BatchNorm2D(16)` → `ReLU` → `MaxPool(2x2)`
- **Conv Block 2:**  
  - `Conv2D(16 → 32)` → `BatchNorm2D(32)` → `ReLU` → `MaxPool(2x2)`
- **Conv Block 3:**  
  - `Conv2D(32 → 64)` → `BatchNorm2D(64)` → `ReLU` → `MaxPool(2x2)`
- **Conv Block 4:**  
  - `Conv2D(64 → 128)` → `BatchNorm2D(128)` → `ReLU` → `MaxPool(2x2)`
- **Flatten:**  
  - Output of size `128 x 8 x 8` flattened to `8192` features
- **Fully Connected Layer 1:**  
  - `Linear(8192 → 256)` → `ReLU` → `Dropout(0.3)`
- **Fully Connected Layer 2 (Output):**  
  - `Linear(256 → 5)` (for 5-class classification)

### 🧪 Regularization:
- **Batch Normalization:** After every convolution for stable training
- **Dropout:** 30% dropout rate to reduce overfitting in FC layers

### 🗂️ Output:
- Final layer outputs 5 logits corresponding to the deepfake class predictions.

---

### 🧠 Multilayer Perceptron (MLP) Architecture

This MLP model acts as a baseline for deepfake image classification by learning directly from flattened pixel data.  

The model gradually reduces dimensionality across 3 hidden layers and uses LeakyReLU activations to mitigate the "dying ReLU" problem. Dropout is applied to reduce overfitting during training.

### 🔧 Architecture Overview:

- **Input:** RGB image (3 channels, 128x128 pixels)  
  → Flattened to a `49,152-dimensional vector` (3 × 128 × 128)

- **Hidden Layer 1:**  
  - `Linear(49,152 → 1024)`  
  - `LeakyReLU(α=0.01)`

- **Hidden Layer 2:**  
  - `Linear(1024 → 512)`  
  - `LeakyReLU(α=0.01)`

- **Hidden Layer 3:**  
  - `Linear(512 → 256)`  
  - `LeakyReLU(α=0.01)`

- **Dropout:**  
  - `Dropout(p=0.3)` before final layer (used only during training)

- **Output Layer:**  
  - `Linear(256 → 5)` → outputs raw logits for 5-class prediction

### 🧪 Activation & Regularization:

- **LeakyReLU:** Used throughout to preserve gradient flow and avoid dead neurons  
- **Dropout:** 30% of neurons are randomly deactivated during training to reduce overfitting

### 🗂️ Output:

The final layer outputs a vector of 5 logits corresponding to the probability distribution over the deepfake image classes (0–4).

## 🧪 Training & Evaluation
Both models — CNN and MLP — were trained using CrossEntropyLoss.
Evaluation was done on the validation set, using the following metrics:
- ✅ Accuracy
- 📉 Loss over epochs
- 📈 Accuracy over epochs
- 📊 Confusion Matrix

## ⚙️ How to Run
Prepare your local folders and files:

```bash
/train          # training images
/validation     # validation images
/test           # test images


train.csv
validation.csv
test.csv
sample_submission.csv
```

## Train & Evaluate the Models
Run either of the scripts below:

```bash
model_andra_cnn_final.py     # trains & evaluates CNN model
model_andra_mlp.py     # trains & evaluates MLP model
```
### Outputs
- 📉 Confusion matrix
- 📄 Final submission file


## 📄 Full Documentation
See [**Andruta_AndraMihaela_232_doc.pdf**](https://github.com/andra2602/Deepfake-Classification/blob/main/Andruta_AndraMihaela_232_doc.pdf) for a complete technical breakdown:
- Dataset exploration
- Model architecture choices
- Hyperparameter tuning
- Performance analysis

## 📌 Technologies
- Python 3.11
- PyTorch
- torchvision
- pandas, scikit-learn
- matplotlib


