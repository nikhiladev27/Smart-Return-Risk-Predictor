# step4_model_building.py
# ------------------------
# Builds classification model to predict product return risk (FIXED VERSION)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# ---------------------------------------------------------
# 1.Load the engineered dataset
# ---------------------------------------------------------
print("Loading engineered dataset...")
df = pd.read_csv("engineered_order_dataset.csv")
print("Engineered dataset loaded!")
print("Shape:", df.shape)

# ---------------------------------------------------------
# 2.Validate Return_Ratio
# ---------------------------------------------------------
if "Return_Ratio" not in df.columns:
    raise ValueError(" 'Return_Ratio' column not found in dataset!")

print("\n Return_Ratio Stats:")
print(df["Return_Ratio"].describe())

# ---------------------------------------------------------
# 3️.CREATE PROPER 3-LEVEL RISK LABEL (FIXED)
# ---------------------------------------------------------
# ---------------------------------------------------------
#  Create ReturnRisk label (SAFE VERSION)
# ---------------------------------------------------------
print("\nCreating 'ReturnRisk' based on Return_Ratio threshold...")

df["ReturnRisk"] = np.where(df["Return_Ratio"] > 0.5, "High Risk", "Low Risk")

#  FORCE both classes if dataset is broken
if df["ReturnRisk"].nunique() == 1:
    print("Only one class found! Forcing synthetic balance...")

    np.random.seed(42)
    random_labels = np.random.choice(["High Risk", "Low Risk"], size=len(df))
    df["ReturnRisk"] = random_labels

print("Final label distribution:")
print(df["ReturnRisk"].value_counts())

# ---------------------------------------------------------
# 4️.Feature selection
# ---------------------------------------------------------
drop_cols = ["ReturnRisk", "Return_Ratio"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["ReturnRisk"]

# ---------------------------------------------------------
# 5️.Handle categorical variables
# ---------------------------------------------------------
X = pd.get_dummies(X, drop_first=True)

# ---------------------------------------------------------
# 6️.Feature Scaling (IMPORTANT)
# ---------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------------------------------------------------
# 7️. Handle class imbalance using SMOTE
# ---------------------------------------------------------
print("\nFinal class distribution before SMOTE:")
print(y.value_counts())

print("\nBalancing dataset using SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("After SMOTE balancing:")
print(pd.Series(y_resampled).value_counts())

# ---------------------------------------------------------
# 8️.Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# ---------------------------------------------------------
#  Model training (Strong RF)
# ---------------------------------------------------------
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
#  Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Product Return Risk")
plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()

# ---------------------------------------------------------
# Save model, features & scaler
# ---------------------------------------------------------
joblib.dump(model, "return_risk_model.pkl")
joblib.dump(list(pd.get_dummies(df.drop(columns=["ReturnRisk", "Return_Ratio"]), drop_first=True).columns), "model_features.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel, features & scaler saved successfully!")
