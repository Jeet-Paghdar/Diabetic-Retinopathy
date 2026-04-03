# Diabetic Retinopathy Detection System

## Project Overview
We are trying to develop a deep learning-based diagnostic tool designed to detect and grade **Diabetic Retinopathy (DR)** from retinal fundus images. Unlike standard CNN approaches, this project utilizes a EfficientNetB4 training model and Vision Transformers (ViT) to capture both local lesions and global retinal features.

The system has tried to include an **Explainable AI (XAI)** module using Grad-CAM to generate heatmaps, helping clinicians trust the AI's predictions by visualizing the infected regions.

## Key Features
- **Training Model:** Uses EfficientNetB4 Training Model which generates an accuracy of 82.02%.
- **Ben Graham's Preprocessing:** Advanced image enhancement techniques.
- **Explainability:** Grad-CAM heatmaps to visualize disease severity.
- **Web Interface:** User-friendly Streamlit dashboard for real-time diagnosis.

## Dataset
We utilize a unified dataset merging:
1. **APTOS 2019 Blindness Detection**
2. **Kaggle 2015 Diabetic Retinopathy Detection**

## Tech Stack
- **Deep Learning:** TensorFlow, Keras, EfficientNet, ViT
- **Image Processing:** OpenCV, Albumentations
- **Web App:** Streamlit
- **Experiment Tracking:** Weights & Biases
