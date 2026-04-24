
# Seismic Signal Classification Project

This project focuses on the classification of seismic signals using Machine Learning models based on different feature sets (36 and 201 features). Two primary algorithms were implemented and compared: Support Vector Machine (SVM) and XGBoost.

## 🚀 Project Overview

The seismic data was processed and evaluated in two different dimensions to observe the impact of feature depth on model performance:

- **36 Feature Model:** Classification using a compact set of extracted features.
- **201 Feature Model:** Classification using an extended, high-dimensional feature set.

## 📊 Model Performance Results

The training and testing results achieved in the current environment are as follows:

### 1. XGBoost Classifier

The XGBoost model demonstrated perfect performance across both datasets.

| Model | Accuracy | Misclassified Samples |
|-------|----------|----------------------|
| XGBoost (36 features) | 1.000000 (100%) | 0 |
| XGBoost (201 features) | 1.000000 (100%) | 0 |

### 2. Support Vector Machine (SVM)

The SVM models showed high accuracy but were slightly outperformed by the XGBoost implementation.

| Model | Accuracy | Misclassified Samples |
|-------|----------|----------------------|
| SVM (36 features) | 0.902439 (~90.2%) | 4 |
| SVM (201 features) | 0.952381 (~95.2%) | 2 |

## 📂 Project Structure

```
/
├── source/
│   ├── train_svm.py
│   ├── train_xgboost.py
│   ├── Classify_svm.py
│   └── Classify_xgboost.py
├── data/
│   └── *.csv (Dataset files for training and testing)
├── models/
│   └── *.pkl (Serialized model files for deployment)
└── README.md
```

## ⚙️ Installation & Usage

To run this project locally, ensure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### Training the Models

To retrain the models and override the existing `.pkl` files:

```bash
python source/train_svm.py
python source/train_xgboost.py
```

### Running Inference

To evaluate the saved models against the test data:

```bash
python source/Classify_svm.py
python source/Classify_xgboost.py
```

## 📝 Conclusion

Based on the experimental results, **XGBoost proved to be the superior algorithm** for this specific seismic signal classification task, achieving zero misclassifications on both the 36 and 201 feature sets. While SVM performed admirably, it showed a higher sensitivity to feature dimensions compared to the gradient boosting approach.

---

**Author:** Rafi Veyisov  
**Date:** April 2026
```