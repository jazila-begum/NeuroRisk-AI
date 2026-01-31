import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
def load_data():
    df = pd.read_excel("stroke.xlsx")
    df.drop(columns=["id"], inplace=True)

    imputer = SimpleImputer(strategy="mean")
    df["bmi"] = imputer.fit_transform(df[["bmi"]])
    df["avg_glucose_level"] = imputer.fit_transform(df[["avg_glucose_level"]])
    
    for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Binning numerical features based on feature importance
    df["age_bin"] = pd.cut(df["age"], bins=[0, 30, 60, 100], labels=["Young", "Middle-Aged", "Senior"])
    df["glucose_bin"] = pd.cut(df["avg_glucose_level"], bins=[0, 90, 140, 300], labels=["Low", "Normal", "High"])
    df["bmi_bin"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 50], labels=["Underweight", "Normal", "Overweight", "Obese"])
    
    return df

df = load_data()

# Encode categorical features based on feature importance order
categorical_features = ["age_bin", "hypertension", "heart_disease", "glucose_bin", "bmi_bin", "smoking_status", "ever_married", "work_type", "Residence_type", "gender"]
label_encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in categorical_features}

for col, encoder in label_encoders.items():
    df[col] = encoder.transform(df[col].astype(str))

# Prioritizing features based on importance order
X = df[["age_bin", "hypertension", "heart_disease", "glucose_bin", "bmi_bin", "smoking_status", "ever_married", "work_type", "Residence_type", "gender"]]
y = df["stroke"]

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Hybrid Scaling based on feature importance
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

X_train[["bmi_bin"]] = standard_scaler.fit_transform(X_train[["bmi_bin"]])
X_test[["bmi_bin"]] = standard_scaler.transform(X_test[["bmi_bin"]])

X_train[["age_bin", "glucose_bin"]] = minmax_scaler.fit_transform(X_train[["age_bin", "glucose_bin"]])
X_test[["age_bin", "glucose_bin"]] = minmax_scaler.transform(X_test[["age_bin", "glucose_bin"]])

# Define models with class weighting
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42, class_weight='balanced'))
xgb = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=class_weights[1])

# Create an ensemble model
model = VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)], voting='soft')
model.fit(X_train, y_train)

# Save the trained model
with open("stroke_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save label encoders and scalers
with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)

with open("standard_scaler.pkl", "wb") as scaler_file:
    pickle.dump(standard_scaler, scaler_file)

with open("minmax_scaler.pkl", "wb") as minmax_file:
    pickle.dump(minmax_scaler, minmax_file)

print("Model training complete with prioritized feature selection, hybrid scaling, and class weighting.")
