# Bone Fracture Detection using Custom CNN in PyTorch

This repository contains a Deep Learning project that classifies X-ray images to detect bone fractures. I built a custom Convolutional Neural Network (CNN) from scratch using PyTorch to successfully classify images into two categories: `fractured` and `not fractured`.

## 🚀 Project Highlights
* **Framework:** PyTorch
* **Model Architecture:** Custom CNN (`BoneCNN`) with 3 Convolutional blocks, Max Pooling, and Dropout for regularization.
* **Optimization:** Adam optimizer with Cosine Annealing Learning Rate Scheduler for smooth convergence.
* **Custom Callbacks:** Implemented a custom `EarlyStopping` class to prevent overfitting and save the best model weights.
* **Performance:** Achieved **~94% Accuracy** on the unseen test dataset.

## 📂 Dataset
The dataset consists of X-ray images split into `train`, `val`, and `test` folders. 
* To run this code, place your dataset in the `./dataset/` directory.
* Applied transformations: Grayscaling (converted to 3-channels), Random Horizontal Flips, Random Rotations, and Normalization.

## 📈 Results
The model successfully converged around epoch 18 (triggered early stopping). The confusion matrix and classification report in the notebook demonstrate high precision and recall for both classes.
