# Chest X-Ray Thoracic Pathology Classification

## Overview

This project focuses on automated detection of thoracic pathologies from chest X-ray images using deep learning. The objective is to classify chest radiographs into one of 20 pathology categories, including a **No Finding** class.

The solution leverages transfer learning with **DenseNet121**, advanced image preprocessing techniques, class imbalance handling, and a custom loss function designed to align with the competition's asymmetric evaluation metric.

Submitted as part of Intro to Deep Learning And Generative AI course in the IITM BS degree. 

---

## Problem Statement

Chest X-rays are among the most widely used medical imaging modalities for diagnosing lung and thoracic diseases. Manual interpretation can be time-consuming and subject to variability among radiologists.

This project aims to build a machine learning system capable of identifying thoracic abnormalities directly from chest radiographs.

### Challenges

* Severe class imbalance across disease categories
* High cost of false negatives in medical diagnosis
* Subtle visual differences between pathologies
* Large-scale image processing requirements

---

## Dataset

The dataset consists of chest X-ray images annotated with 20 thoracic pathology classes, including:

* Atelectasis
* Cardiomegaly
* Consolidation
* Edema
* Effusion
* Emphysema
* Fibrosis
* Hernia
* Infiltration
* Mass
* Nodule
* Pleural Thickening
* Pneumonia
* Pneumothorax
* No Finding
* and other thoracic conditions

---

## Methodology

### Image Preprocessing

* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Image resizing to 224 × 224
* Normalization using ImageNet statistics

### Data Handling

* Stratified train-validation split
* WeightedRandomSampler to address class imbalance
* Data augmentation for improved generalization

### Model Architecture

* DenseNet121 (ImageNet pretrained)
* Custom classification head
* Mixed precision training for faster convergence

### Loss Function

A custom **Asymmetric Cost Loss** was implemented to reflect the competition scoring system, where:

* True Positive = +1
* False Positive = -1
* False Negative = -5

This encourages the model to prioritize disease detection and reduce missed diagnoses.

---

## Training Configuration

| Component  | Configuration         |
| ---------- | --------------------- |
| Backbone   | DenseNet121           |
| Optimizer  | AdamW                 |
| Scheduler  | Cosine Annealing LR   |
| Batch Size | 32                    |
| Image Size | 224 × 224             |
| Precision  | Mixed Precision (AMP) |

---

## Competition Metric

The competition uses a macro-averaged asymmetric scoring function that heavily penalizes false negatives.

The implementation includes probability calibration and threshold optimization to better align predictions with the evaluation metric.

---

## Results

The pipeline successfully:

* Trains a DenseNet121-based classifier on chest X-ray images
* Handles class imbalance using weighted sampling
* Incorporates medical-domain-specific preprocessing
* Generates competition-ready submission files

---

## Repository Structure

```text
├── notebook.ipynb
├── submission.csv
├── README.md
└── assets/
```

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* Matplotlib

---

## Future Improvements

* Multi-label classification formulation
* Class-wise threshold optimization
* Test Time Augmentation (TTA)
* Ensemble learning
* Medical-image-specific pretrained models (CheXNet, BioViL, etc.)

---

## Author

Pranay Aggarwal

BS in Data Science and Applications, IIT Madras
