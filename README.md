# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent transactions using both **Logistic Regression** and **Convolutional Neural Networks (CNN)**. This project explores how traditional models compare with deep learning techniques when applied to imbalanced datasets, such as credit card transactions.

---

## 🧠 Models Implemented
- **Logistic Regression** – A simple, interpretable linear model
- **Convolutional Neural Network (CNN)** – For learning complex non-linear patterns in the data

---

## 📁 Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Total Transactions: 284,807  
- Fraudulent Transactions: 492  
- Highly imbalanced dataset

---

## 📊 Preprocessing Steps
- Handled **class imbalance** by undersampling legitimate transactions.
- Applied **StandardScaler** for feature scaling.
- Reshaped data for CNN: `(samples, features, 1)` format.
- Imputed missing values using **KNNImputer** (for the logistic model variant).
- Train-Test split with `stratify=y` to preserve class distribution.

---

## 🧪 Logistic Regression Model

- **Library:** scikit-learn
- Accuracy:
  - Training: ~96%
  - Test: ~94%
- Steps:
  - Sampled equal number of fraud and non-fraud cases.
  - Applied logistic regression directly to imbalanced dataset.
  - Evaluated with accuracy scores.

---

## 🧠 CNN Model Architecture

```python
Sequential([
    Conv1D(32, 2, activation='relu'),
    BatchNormalization(),
    MaxPool1D(2),
    Dropout(0.2),

    Conv1D(64, 2, activation='relu'),
    BatchNormalization(),
    MaxPool1D(2),
    Dropout(0.5),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
