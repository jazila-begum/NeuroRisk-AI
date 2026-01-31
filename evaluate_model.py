import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib  # For loading the trained model

# Load the trained model
model = joblib.load("stroke_model.pkl")  # Ensure your model is saved as a .pkl file

# Load the new dataset for evaluation
df = pd.read_excel("stroke.xlsx")

# Drop 'id' column as it's not useful
df.drop(columns=['id'], inplace=True)

# Handling missing values
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Encoding categorical variables
le = LabelEncoder()
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Splitting features and target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Standardizing features
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Apply the same transformation used during training

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]  # Get probability scores for AUC-ROC calculation

# Model Evaluation
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC Score: {roc_auc:.4f}')
print("\nClassification Report:\n", classification_report(y, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
