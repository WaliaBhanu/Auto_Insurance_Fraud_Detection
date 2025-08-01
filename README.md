# 🚨 Insurance Fraud Detection App

This repository contains an end-to-end machine learning pipeline to detect fraudulent insurance claims using **Random Forest** and **Streamlit** for deployment. It handles real-world preprocessing (like frequency, target, and ordinal encoding) and makes predictions interactively.



## 📦 Features

- Full pipeline including:
  - Frequency Encoding
  - Target Encoding
  - One-Hot Encoding
  - Ordinal Encoding
  - Standard Scaling
- Trained with `RandomForestClassifier` (custom hyperparameters)
- Resampling handled with `SMOTENC` for class imbalance
- Live web app built using Streamlit
- Interactive form inputs to classify claims as **fraud** or **not fraud**


## 🧠 Tech Stack

- Python 3.8+
- Scikit-learn
- Imbalanced-learn
- Pandas / NumPy
- Streamlit
- Pickle (for saving model)
- Custom Transformers



