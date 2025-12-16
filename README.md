# Egyptian New Currency Recognition using Deep Learning

[![GitHub Stars](https://img.shields.io/github/stars/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning?style=social)](https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Project Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning)

## 🌟 Overview

This repository hosts a Deep Learning project developed as a graduation requirement by the **Faculty of Computers and Artificial Intelligence at Banha University**. The primary goal is to accurately recognize and classify various denominations of Egyptian currency, with a special emphasis on the recently introduced **polymer banknotes of 10 EGP and 20 EGP**.

We utilize a Convolutional Neural Network (CNN) to address the challenges posed by real-world image variations, including diverse lighting, viewing angles, and backgrounds, making it a robust solution for currency recognition systems.

## 💰 Dataset

The project relies on a meticulously curated, high-quality image dataset of Egyptian currency, captured under diverse conditions to ensure model robustness.

### Currency Denominations:
The dataset covers a total of **8** distinct classes:
* 5 Egyptian Pound Banknote
* 10 Egyptian Pound Banknote
* 20 Egyptian Pound Banknote
* **New 10 Egyptian Pound Banknote (Polymer)**
* **New 20 Egyptian Pound Banknote (Polymer)**
* 50 Egyptian Pound Banknote
* 100 Egyptian Pound Banknote
* 200 Egyptian Pound Banknote

### Image Diversity and Robustness:
To simulate real-world scenarios and enhance the dataset's utility, each currency denomination is captured from multiple perspectives:
* **Capture Positions:** Frontal Views, Variably Angled Shots, and Rotated Positions.
* **Backgrounds:** Plain backgrounds for controlled environments and natural settings with textured surfaces.
* **Lighting:** Varied lighting conditions to account for potential challenges.

### Technical Details:
* **Image Resolution:** High-resolution images for detailed analysis.
* **Format:** Standardized image formats facilitating seamless integration into machine learning pipelines.

## 💡 Clear Explanation of the Problem

The core challenge is **multi-class image classification** under varying real-world conditions. Specifically, the model must:
1.  **Handle Visual Similarity:** Accurately classify between 8 visually similar categories, including the subtle differences between old and new 10 EGP and 20 EGP notes.
2.  **Achieve Invariance:** Maintain high accuracy despite significant variations in image capture, such as different viewing angles (`Angled Shots`), rotation, diverse lighting, and complex backgrounds.
3.  **Robust Feature Extraction:** Deep learning is essential to automatically learn the intricate features and security details necessary for reliable currency identification.

## 🏗️ Description of the CNN Architecture

A Convolutional Neural Network (CNN) was implemented to automatically learn spatial hierarchies of features from the banknote images.

**[INSERT FIGURE OF THE CNN MODEL ARCHITECTURE HERE]**

### Architecture Details:
(Describe the architecture here, e.g., "The model is based on the [VGG16/ResNet50] architecture, pre-trained on ImageNet, with the final classification layers replaced to accommodate the 8 currency classes. We fine-tuned...")

## 📊 Training Loss and Accuracy Curves

The model was trained over [N] epochs. The following plots illustrate the convergence and generalization capability of the model by tracking the loss and accuracy on both the training and validation sets.

### Training & Validation Loss Curve
**[INSERT PLOT OF TRAINING & VALIDATION LOSS HERE]**

### Training & Validation Accuracy Curve
**[INSERT PLOT OF TRAINING & VALIDATION ACCURACY HERE]**

## 🔬 Testing Results

The final model performance was evaluated on a dedicated, unseen test set to measure its real-world effectiveness.

### Confusion Matrix
The confusion matrix shows the number of correct and incorrect predictions made for each class, highlighting potential misclassification patterns.

**[INSERT CONFUSION MATRIX PLOT HERE]**

### Evaluation Metrics
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Accuracy** | [INSERT ACCURACY VALUE] | Overall classification accuracy on the test set. |
| **Precision (Macro)** | [INSERT PRECISION VALUE] | Average precision across all classes. |
| **Recall (Macro)** | [INSERT RECALL VALUE] | Average recall (sensitivity) across all classes. |
| **F1-Score (Macro)** | [INSERT F1-SCORE VALUE] | Harmonic mean of precision and recall. |

## 🖼️ Example Predictions on Unseen Images

The following examples showcase the model's performance on images not seen during the training or validation phase.

**[INSERT EXAMPLES OF PREDICTED IMAGES (e.g., Image, Predicted Class, True Class, Confidence Score) HERE]**

## 🚧 Difficulties Faced and What Was Learned

### Difficulties Faced
1.  **Handling Image Variation:** The need to train on `Angled Shots` and `Rotated Positions` required extensive data augmentation to prevent the model from overfitting to simple frontal views.
2.  **Distinguishing New vs. Old Notes:** Fine-tuning the model's depth and complexity was crucial to ensure it could differentiate between the old and new 10 EGP/20 EGP notes, which are visually similar in value but different in material and design.
3.  **Data Quality:** Ensuring consistency across diverse lighting and backgrounds added complexity to the pre-processing pipeline.

### Lessons Learned
1.  **Transfer Learning Efficacy:** Using pre-trained models proved highly effective for achieving high accuracy quickly, leveraging features learned from millions of general images.
2.  **Importance of a Diverse Dataset:** The project validated the initial hypothesis that a dataset incorporating variations in perspective, lighting, and background is non-negotiable for real-world currency recognition applications.
3.  **Evaluation Focus:** The confusion matrix was critical in identifying the specific denominations that the model struggled to separate, guiding necessary adjustments in training parameters or data balance.

## 🛠️ Setup and Installation

To reproduce the environment and run the code locally, ensure you have Python 3.x installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning.git](https://github.com/Mohamed-Teba/Egyptian-New-Currency-Using-Deep-Learning.git)
    cd Egyptian-New-Currency-Using-Deep-Learning
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Dataset:**
    (Instructions on where to download the dataset and place it in the project structure, e.g., `data/`)

4.  **Run the main script (e.g., Training):**
    ```bash
    python train_model.py
    ```
