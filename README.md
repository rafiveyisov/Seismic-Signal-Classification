Seismic Signal Classification Project
This project focuses on the classification of seismic signals using Machine Learning models based on different feature sets (36 and 201 features). Two primary algorithms were implemented and compared: Support Vector Machine (SVM) and XGBoost.

🚀 Project Overview
The seismic data was processed and evaluated in two different dimensions to observe the impact of feature depth on model performance:

36 Feature Model: Classification using a compact set of extracted features.

201 Feature Model: Classification using an extended, high-dimensional feature set.

📊 Model Performance Results
The training and testing results achieved in the current environment are as follows:

1. XGBoost Classifier
The XGBoost model demonstrated perfect performance across both datasets.

XGBoost (36 features):

Accuracy: 1.000000 (100%)

Misclassified Samples: 0

XGBoost (201 features):

Accuracy: 1.000000 (100%)

Misclassified Samples: 0

2. Support Vector Machine (SVM)
The SVM models showed high accuracy but were slightly outperformed by the XGBoost implementation.

SVM (36 features):

Accuracy: 0.902439 (~90.2%)

Misclassified Samples: 4

SVM (201 features):

Accuracy: 0.952381 (~95.2%)

Misclassified Samples: 2

📂 Project Structure
/source: Python scripts for model training (train_*.py) and inference (Classify_*.py).

/data: Dataset files in .csv format used for training and testing.

/models: Serialized model files saved in .pkl format for deployment.

⚙️ Installation & Usage
To run this project locally, ensure you have the following dependencies installed:

Bash
pip install pandas numpy scikit-learn xgboost joblib
Training the Models
To retrain the models and override the existing .pkl files:

Bash
python source/train_svm.py
python source/train_xgboost.py
Running Inference
To evaluate the saved models against the test data:

Bash
python source/Classify_svm.py
python source/Classify_xgboost.py
📝 Conclusion
Based on the experimental results, XGBoost proved to be the superior algorithm for this specific seismic signal classification task, achieving zero misclassifications on both the 36 and 201 feature sets. While SVM performed admirably, it showed a higher sensitivity to feature dimensions compared to the gradient boosting approach.

Author: Rafi Veyisov

Date: April 2026