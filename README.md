# CNN for GI Disease Classification & Correlation Analysis

A deep learning pipeline for classifying gastrointestinal (GI) endoscopy images across 23 anatomical disease labels, with secondary binary prediction of clinical findings using extracted CNN embeddings.

**Course**: CSC 483 – Applied Deep Learning, DePaul University  
**Team**: Bhuvanesh Kantamuneni, Mohammed Musaddiq Vavartar, Vishakha Kamothi, Afshaan Fathima Syeda

---

## Overview

This project tackles multi-class classification of GI endoscopy images using a custom-built CNN trained on a clinically diverse dataset. After validating model performance, we extract 256-dimensional feature embeddings from the trained network and use them to predict additional clinical findings — specifically H. pylori infection and pre-pyloric ulcers — via logistic regression. We also perform co-occurrence and Pearson correlation analysis to uncover statistical relationships between GI disease labels.

### Key results
- Multi-class classification across **23 GI anatomical disease labels**
- Feature embeddings used for **binary disease prediction** (H. pylori, pre-pyloric ulcer)
- Co-occurrence and **Pearson correlation analysis** across all disease label pairs
- Evaluated with Accuracy, Weighted F1, MCC, AUC (One-vs-Rest per class)

---

## Project Structure

```
├── csc483_Bhuvanesh_Kantamuneni_final_project.ipynb   # Main notebook
├── splits/
│   └── image_classification.csv                        # Image filenames + GI disease labels
├── Labeled Images/                                      # GI endoscopy image dataset
├── gastrohun-videoendoscopy-metadata.json               # Patient metadata (H. pylori, diagnoses)
├── cnn_classification_report.csv                        # Per-class classification metrics (generated)
├── confusion_matrix_cnn.png                             # Normalized confusion matrix (generated)
└── all_image_features.csv                               # 256-dim CNN embeddings per image (generated)
```

---

## Pipeline

```
Raw Images + CSV Labels
        ↓
  Data Cleaning & Label Encoding (23 classes)
        ↓
  70 / 15 / 15 Train / Val / Test Split (stratified)
        ↓
  Custom Keras Data Generator (batch=32, 224×224, augmentation)
        ↓
  CNN Training (Conv2D → MaxPool → Dense(512) → Dense(256) → Softmax)
        ↓
  Evaluation: Accuracy, F1, MCC, AUC per class, Confusion Matrix
        ↓
  Feature Extraction (256-dim embeddings from penultimate layer)
        ↓
  Binary Disease Prediction via Logistic Regression (H. pylori, Pre-pyloric Ulcer)
        ↓
  Co-occurrence & Pearson Correlation Analysis across disease labels
```

---

## Model Architecture

Custom CNN built with Keras Sequential API:

| Layer | Details |
|---|---|
| Conv2D × 3 blocks | Filters: 32 → 64 → 128, ReLU, L2 regularization |
| MaxPooling2D | After each conv block |
| Flatten | — |
| Dense(512) | ReLU, Dropout |
| Dense(256) | ReLU, Dropout — used as feature embedding layer |
| Dense(num_classes) | Softmax output (23 classes) |

- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping, ModelCheckpoint
- Class imbalance handled via `compute_class_weight`

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy, Pandas |
| Image Loading | PIL (Pillow) |
| ML & Evaluation | scikit-learn (LogisticRegression, LabelEncoder, metrics) |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab (GPU), Python 3.x |

---

## How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/<your-username>/gi-disease-classification.git
   cd gi-disease-classification
   ```

2. Install dependencies
   ```bash
   pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn pillow
   ```

3. Set paths in the notebook — update `project_directory` and `Image_directory` to your local or Drive paths

4. Open and run `csc483_Bhuvanesh_Kantamuneni_final_project.ipynb` top to bottom

> **Note**: The notebook was developed on Google Colab with GPU runtime. If running locally, GPU is recommended for training. Dataset files are not included in this repo due to size — see the dataset section below.

---

## Dataset

This project uses the **GastroHUN Video Endoscopy dataset**, a clinically annotated GI endoscopy image dataset with:
- Multi-label disease annotations per patient (`FG agreement` column)
- Patient metadata including H. pylori status and diagnoses

Dataset is not redistributed here. If you have access, place the labeled images under `Labeled Images/` and the CSV under `splits/`.

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Weighted F1 | Handles class imbalance across 23 labels |
| MCC (Matthews Correlation Coefficient) | Robust metric for imbalanced multi-class |
| OvR AUC per class | Per-disease distinguishability score |
| AUROC + Average Precision | For binary disease prediction tasks |

---

## Disease Correlation Analysis

Beyond classification, we analyze which GI disease labels tend to co-occur in the same patient using:
- **Co-occurrence matrix**: raw counts of label pairs appearing in the same patient
- **Pearson correlation**: statistical strength of association between label pairs (+1 always together, 0 unrelated, −1 never together)

This analysis surfaces clinically meaningful patterns between anatomical sites and disease presentations.

---

## Authors

- **Bhuvanesh Kantamuneni** — DePaul University, MS Artificial Intelligence
- Mohammed Musaddiq Vavartar
- Vishakha Kamothi
- Afshaan Fathima Syeda
