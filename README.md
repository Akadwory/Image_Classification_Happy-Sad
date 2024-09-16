# CNN-Based Emotion Classification - Professional Machine Learning Solution

A highly optimized Convolutional Neural Network (CNN) designed for binary emotion classification from facial images. This project uses TensorFlow and Keras, adhering to industry standards and best practices in deep learning for scalability and production readiness.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Model Architecture](#model-architecture)
4. [Training & Validation](#training--validation)
5. [Results](#results)
6. [Installation & Requirements](#installation--requirements)
7. [Usage](#usage)
8. [Evaluation & Inference](#evaluation--inference)
9. [Project Structure](#project-structure)
10. [License](#license)

## Introduction

This project addresses the task of classifying human emotions from facial images into two categories: `Happy` and `Sad`. The model uses a well-established Convolutional Neural Network (CNN) architecture, which is fine-tuned for high accuracy and efficient deployment. This CNN is built for real-time inference, making it suitable for deployment in production environments such as emotion detection in mobile or web-based applications.

## Data Overview

The dataset used in this project is pre-sorted into two classes:
- **Happy**: Contains images of individuals displaying happiness.
- **Sad**: Contains images of individuals expressing sadness.

### Preprocessing
- **Normalization**: All images are resized to 256x256 and pixel values are normalized to a [0, 1] range.
- **Data Splits**:
  - **Training**: 70%
  - **Validation**: 15%
  - **Test**: 15%

## Model Architecture

The model is a robust CNN-based architecture optimized for binary classification:

- **Input Layer**: Image input of shape (256, 256, 3).
- **Conv2D Layers**: Three layers with increasing filters (32, 64, 128) using `ReLU` activation.
- **MaxPooling2D Layers**: Reducing the dimensionality of feature maps.
- **Fully Connected Layers**: Includes a hidden dense layer with 512 neurons.
- **Dropout**: Added for regularization to prevent overfitting (drop rate: 0.5).
- **Output Layer**: A single sigmoid-activated neuron for binary classification.

### Optimizer and Loss
- **Optimizer**: Adam with a learning rate of `1e-5`.
- **Loss**: Binary Cross-Entropy (due to binary classification task).
- **Metrics**: Accuracy, Precision, Recall to measure performance.

### Model Summary
Layer (type) Output Shape Param #
conv2d (Conv2D) (None, 254, 254, 32) 896
max_pooling2d (MaxPooling2D) (None, 127, 127, 32) 0
conv2d_1 (Conv2D) (None, 125, 125, 64) 18496
max_pooling2d_1 (MaxPooling2D) (None, 62, 62, 64) 0
conv2d_2 (Conv2D) (None, 60, 60, 128) 73856
max_pooling2d_2 (MaxPooling2D) (None, 30, 30, 128) 0
flatten (Flatten) (None, 115200) 0
dense (Dense) (None, 512) 58982912
dropout (Dropout) (None, 512) 0
dense_1 (Dense) (None, 1) 513
Total params: 59,076,673 Trainable params: 59,076,673 Non-trainable params: 0

## Training & Validation

The model is trained for 20 epochs with early stopping and model checkpoints enabled to prevent overfitting. Real-time validation metrics such as loss and accuracy are tracked to ensure optimal learning.

### Training Plot
- **Accuracy**: The model achieves a high accuracy, close to 98% on the training set and ~97% on the validation set.
- **Loss**: The training and validation loss reduce significantly over time, demonstrating model convergence.

### Plots
- **Accuracy Plot**:
![accuracy](results/accuracy.png)
  
- **Loss Plot**:
![loss](results/loss.png)

## Results

After training the model, the following performance metrics were observed:

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~97%
- **Precision**: ~97%
- **Recall**: ~97%

These results demonstrate the model's high reliability for emotion classification tasks.
## Installation & Requirements

### Setup Environment

First, clone the repository and install the required packages:
git clone https://github.com/your-repository/cnn-emotion-classification.git
cd cnn-emotion-classification
pip install -r requirements.txt
The following major libraries are required:

TensorFlow
Keras
OpenCV
Pandas
Matplotlib

TensorFlow Installation:
Ensure you have TensorFlow installed for model training:
pip install tensorflow

---
## Usage

### Training the Model

To train the model from scratch, run:


python src/train.py

Model Evaluation
To evaluate the model on the test dataset:
python src/evaluate.py

Running Inference
You can perform inference on any new image with:
python src/inference.py --image <path_to_image>

Sample Inference
An example of running inference:
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model('models/cnn_emotion_model.h5')
img = cv2.imread('path_to_image')
img = cv2.resize(img, (256, 256))
img = np.expand_dims(img/255.0, axis=0)

prediction = model.predict(img)
print("Happy" if prediction > 0.5 else "Sad")


---
## Project Structure

```yaml
CNN-Emotion-Classification/
  ├── data/
  │   ├── happy/                     # Folder containing images of happy faces
  │   └── sad/                       # Folder containing images of sad faces
  │
  ├── models/
  │   └── cnn_emotion_model.h5       # Saved CNN model for future inference
  │
  ├── src/
  │   ├── __init__.py                # Package initializer
  │   ├── data_loader.py             # Code for loading and preprocessing data
  │   ├── model.py                   # CNN model definition
  │   ├── train.py                   # Training logic and saving the model
  │   ├── inference.py               # Running inference on new images
  │   └── evaluate.py                # Evaluation metrics (precision, recall, accuracy)
  │
  ├── notebooks/
  │   └── cnn_image_classification.ipynb  # Jupyter notebook for experimentation and analysis
  │
  ├── results/
  │   ├── accuracy.png               # Accuracy plot over epochs
  │   └── loss.png                   # Loss plot over epochs
  │
  ├── requirements.txt               # Python dependencies
  ├── README.md                      # Project documentation (This file)
  ├── .gitignore                     # Ignore data files, models, and other unnecessary files

