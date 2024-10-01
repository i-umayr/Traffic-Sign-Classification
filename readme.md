# Traffic Sign Detection and Classification

This repository contains an end-to-end solution for traffic sign detection and classification, leveraging both traditional machine learning techniques and deep learning approaches. The project focuses on feature extraction, model training, and evaluation using various methods such as Histogram of Oriented Gradients (HOG) and Convolutional Neural Networks (CNNs) with transfer learning.

## Data Preprocessing

- Traffic sign labels are loaded from CSV files.
- Images from different traffic sign classes are collected, labeled, and preprocessed.
- All images are resized and prepared for subsequent feature extraction and model training.

## Traffic Sign Detection Using HOG Features

- Features are extracted from images using Histogram of Oriented Gradients (HOG).
- A Random Forest classifier is trained to detect traffic signs based on these HOG features.
- Model performance is evaluated using accuracy metrics and a detailed classification report.

## Data Splitting & Augmentation

- The dataset is divided into training and validation sets for deep learning tasks.
- Data augmentation is applied to the training set to improve model generalization.

## Traffic Sign Classification with Transfer Learning

- The InceptionV3 architecture, pre-trained on ImageNet, is fine-tuned for traffic sign classification.
- Custom layers are added to adapt the model for the traffic sign dataset.
- The model is trained and validated to ensure optimal performance.

## Results

- The InceptionV3-based deep learning model achieves an accuracy of **97.62%**.
- The Random Forest classifier using HOG features attains an accuracy of **92.07%**.

## Conclusion

This repository illustrates a complete workflow for traffic sign detection and classification, combining traditional machine learning methods with modern deep learning techniques. The project highlights the effectiveness of both HOG-based feature extraction with Random Forest and transfer learning with CNNs.
