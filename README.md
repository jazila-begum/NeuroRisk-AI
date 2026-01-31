
# NeuroRisk AI

**Early detection, timely intervention**

NeuroRisk AI is a **machine learning–based stroke risk prediction system** that estimates an individual’s **probability of stroke risk as a percentage** using clinical and lifestyle features.
The project focuses on **early risk awareness** to support timely medical intervention and preventive care.

---

## 🔍 Problem Statement

Stroke remains one of the leading causes of long-term disability and mortality worldwide. Many risk factors—such as hypertension, diabetes, smoking habits, and lifestyle patterns—are measurable well before a stroke occurs.

NeuroRisk AI aims to:

* Analyze patient data using machine learning
* Predict **stroke risk as a continuous probability score**
* Support early awareness and informed decision-making

> ⚠️ This system is **not a diagnostic tool**. It is intended for **risk assessment and educational purposes only**.

---

## 💡 Solution Overview

The project uses a **regression-based machine learning approach** to estimate stroke risk as a **percentage score (0–100%)**, rather than a simple yes/no classification. This provides a more nuanced understanding of risk levels.

---

## 🧠 Model Approach

* **Problem Type:** Supervised Machine Learning (Regression)
* **Target Variable:** Stroke risk probability (%)
* **Models Explored:**

  * Linear Regression
  * Random Forest Regressor
  * Gradient Boosting Regressor *(if applicable)*

Models were evaluated and selected based on performance, stability, and interpretability.

---

## 📊 Dataset & Features

Typical features used include:

* Age
* Gender
* Hypertension
* Heart disease
* BMI
* Average glucose level
* Smoking status
* Lifestyle indicators

Data preprocessing steps:

* Missing value handling
* Categorical encoding
* Feature scaling
* Class imbalance awareness

---

## ⚙️ Training & Evaluation

* **Train/Test Split:** Standard supervised learning split
* **Evaluation Metrics:**

  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
  * R² Score

Predictions are converted into a **percentage-based risk score** for intuitive interpretation.

---

## 📈 Output Example

| Patient | Predicted Stroke Risk |
| ------- | --------------------- |
| User A  | 12.6%                 |
| User B  | 38.9%                 |
| User C  | 71.4%                 |

Risk levels can be interpreted as:

* **Low Risk:** < 20%
* **Moderate Risk:** 20–50%
* **High Risk:** > 50%

*(Thresholds are configurable and for demonstration purposes.)*

---

## 🚀 Project Highlights

* Percentage-based risk estimation (not just binary prediction)
* Focus on interpretability and early awareness
* Clean preprocessing and feature engineering pipeline
* Extendable to web or mobile deployment

---

## 🛠 Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Modeling:** Regression-based ML algorithms
* **Tools:** Jupyter Notebook, Git, GitHub



