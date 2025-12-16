# 🇪🇬 Egyptian New Currency Recognition: A Deep Learning Approach

[![GitHub Stars](https://img.shields.io/github/stars/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning?style=social)](https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow%2FKeras-blueviolet)](https://www.tensorflow.org/)
[![Project Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning)

## 1. 📚 Project Overview

This repository documents a robust solution for the automatic recognition and classification of Egyptian currency denominations using Deep Learning. Developed as a graduation project by the Faculty of Computers and Artificial Intelligence at **Banha University**, the system is designed to accurately classify both traditional paper banknotes and the **newly introduced polymer banknotes (10 EGP and 20 EGP)**.

The primary objective is to develop a Convolutional Neural Network (CNN) model that exhibits high accuracy and resilience against complex real-world image variations, making it suitable for integration into automated currency handling systems.

---

## 2. 🎯 Problem Statement: Currency Classification

The core challenge addressed is a high-stakes, multi-class image classification task involving 8 visually similar categories of Egyptian banknotes.

### 2.1. Classification Scope (8 Classes)
* 5 Egyptian Pound Banknote (EGP)
* 10 EGP Banknote
* **New 10 EGP Banknote (Polymer)**
* 20 EGP Banknote
* **New 20 EGP Banknote (Polymer)**
* 50 EGP Banknote
* 100 EGP Banknote
* 200 EGP Banknote

### 2.2. Robustness Requirements
The model must achieve **viewpoint invariance** and **lighting robustness**. This necessitates a solution capable of maintaining high performance despite variations in:
* **Viewing Angle:** Angled and Rotated Views.
* **Background:** Plain laboratory settings vs. complex textured environments.
* **Illumination:** Diverse lighting conditions affecting banknote texture and color.

---

## 3. 💾 Dataset Description

Our solution is built upon a meticulously curated, high-resolution dataset specifically designed to maximize model generalization.

### 3.1. Data Acquisition Strategy
To ensure a robust training environment, images for each denomination were captured with variations in:
* **Capture Positions:** Frontal views, variable angled shots, and rotated orientations.
* **Backgrounds:** Controlled plain backgrounds and natural, textured surfaces.
* **Technical Details:** High-resolution images in standardized formats.

---

## 4. 🧠 Methodology: CNN Architecture

A tailored Convolutional Neural Network (CNN) architecture was implemented for feature extraction and classification.

### 4.1. Model Structure
The architecture [Specify Base Model, e.g., is based on a **custom sequential model** or **fine-tuned ResNet-50**], incorporating standard building blocks optimized for visual classification tasks:

$$
\text{Input} \rightarrow \underbrace{\text{Conv Layers} + \text{Pooling}}_{\text{Feature Extraction}} \rightarrow \underbrace{\text{Flatten} \rightarrow \text{Dense Layers}}_{\text{Classification Head}} \rightarrow \text{Output (8 Classes)}
$$

**[INSERT PROFESSIONAL FIGURE/DIAGRAM OF THE CNN MODEL ARCHITECTURE HERE]**

### 4.2. Training Parameters
* **Loss Function:** [Specify Loss Function, e.g., Categorical Cross-Entropy]
* **Optimizer:** [Specify Optimizer, e.g., Adam]
* **Epochs:** [N]
* **Batch Size:** [M]

---

## 5. 📈 Results and Evaluation

The model was rigorously tested on a dedicated, unseen test set to validate its performance and generalization capability.

### 5.1. Training Performance Curves

The following plots illustrate the model's convergence and stability throughout the training process:

#### Training and Validation Loss
**[INSERT PLOT OF TRAINING & VALIDATION LOSS HERE]**

#### Training and Validation Accuracy
**[INSERT PLOT OF TRAINING & VALIDATION ACCURACY HERE]**

### 5.2. Quantitative Evaluation

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Overall Accuracy** | [INSERT ACCURACY VALUE]% | Classification accuracy on the final test set. |
| **Macro Precision** | [INSERT PRECISION VALUE] | Average precision across all classes, indicating low false positives. |
| **Macro Recall** | [INSERT RECALL VALUE] | Average recall across all classes, indicating low false negatives. |
| **F1-Score** | [INSERT F1-SCORE VALUE] | Harmonic mean of precision and recall, serving as a balanced performance measure. |

### 5.3. Confusion Matrix
The confusion matrix provides a granular view of per-class performance, crucial for identifying specific misclassification tendencies (e.g., confusion between old and new denominations).

**[INSERT CONFUSION MATRIX PLOT HERE]**

### 5.4. Prediction Examples
Visual examples demonstrating the model's successful classification of challenging, real-world images from the test set.

**[INSERT EXAMPLES OF PREDICTED IMAGES (e.g., Image, Predicted Class, True Class, Confidence Score) HERE]**

---

## 6. ⚙️ System Setup and Reproducibility

This section outlines the steps required to set up the environment and reproduce the results.

### 6.1. Prerequisites
* Python 3.x
* Git

### 6.2. Installation
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning.git](https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning.git)
    cd Egyptian-New-Currency-Using-Deep-Learning
    ```

2.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Acquisition:**
    * (Instruction on how to download and structure the dataset, e.g., "Place the dataset files into the `./data/` directory.")

4.  **Run Training Script:**
    ```bash
    python train_model.py
    ```

---

## 7. 📝 Challenges and Learnings

### 7.1. Key Difficulties Encountered
* **Generalization under Augmentation:** Balancing aggressive data augmentation (rotation, lighting shifts) necessary for real-world robustness without introducing noise that hinders feature learning.
* **Fine-Grained Classification:** Developing the model depth required to differentiate subtle security features separating the old and new polymer banknotes.

### 7.2. Core Learnings
* **Data Quality is Paramount:** The success of the project was highly dependent on the initial comprehensive data acquisition strategy, emphasizing variability in all physical dimensions.
* **Iterative Model Refinement:** Performance improvement was achieved through iterative experimentation with different pre-trained CNN backbones and tuning of the final classification layers.
