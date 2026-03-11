# Smart Return Risk Predictor

## Overview

The **Smart Return Risk Predictor** is a machine learning system that predicts whether an order is likely to result in a **High Return Risk** or **Low Return Risk** based on historical order and customer data.

The project applies a structured data science pipeline including **data preprocessing, feature engineering, class balancing, model training, and evaluation**.

This system can help e-commerce businesses **identify orders that are more likely to be returned**, enabling better decision-making and cost reduction.

---

# Problem Statement

Product returns significantly impact operational costs in e-commerce.
The objective of this project is to build a **machine learning model capable of identifying high-risk return orders** using historical purchase and customer behavior data.

---

# Data Science Pipeline

## 1. Data Preprocessing

The dataset was cleaned and prepared by:

* Handling missing values
* Removing duplicates
* Encoding categorical variables using **Label Encoding**

---

## 2. Feature Engineering

A new feature called **Return Ratio** was created:

Return_Ratio = Refunded Item Count / (Purchased Item Count + 1)

This feature helps quantify the **likelihood of a product being returned**.

---

## 3. Risk Label Creation

A target variable **ReturnRisk** was generated based on the return ratio.

* Return_Ratio > 0.5 → High Risk
* Return_Ratio ≤ 0.5 → Low Risk

---

## 4. Handling Class Imbalance

Since real-world return data can be imbalanced, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to balance the dataset.

---

## 5. Model Training

A **Random Forest Classifier** was used for prediction.

Model configuration:

* n_estimators = 300
* max_depth = 12
* class_weight = balanced

Random Forest was chosen for its **robust performance and ability to handle complex feature interactions**.

---

# Model Evaluation

The model was evaluated using standard classification metrics.

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~90%  |
| Precision | ~89%  |
| Recall    | ~91%  |
| F1 Score  | ~90%  |

A **confusion matrix** was also generated to visualize prediction performance.

---

# Technologies Used

| Category           | Tools                    |
| ------------------ | ------------------------ |
| Programming        | Python                   |
| Data Processing    | Pandas, NumPy            |
| Machine Learning   | Scikit-learn             |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Visualization      | Matplotlib               |
| Model Saving       | Joblib                   |
| Deployment         | Streamlit                |

---

# Project Structure

```
Smart_Return_Risk_Predictor
│
├── step1_load_dataset.py
├── step2_clean_dataset.py
├── step3_feature_engineering.py
├── step4_model_building.py
├── app.py
├── engineered_order_dataset.csv
├── return_risk_model.pkl
├── scaler.pkl
├── model_features.pkl
├── requirements.txt
└── README.md
```

---

# Running the Project

Install dependencies

```
pip install -r requirements.txt
```

Train the model

```
python step4_model_building.py
```

Run the Streamlit application

```
streamlit run app.py
```

---

# Future Improvements

* Use advanced models such as **XGBoost or Gradient Boosting**
* Integrate **real-time order data**
* Deploy the model as a **cloud-based prediction service**

---

# Author

**Nikhila Devaraj**

