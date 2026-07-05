# ==========================================
# Plant Disease Detection using SVM
# ==========================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# ==========================================
# Load Dataset
# ==========================================

print("Loading Dataset...")

df = pd.read_csv("plant_disease.csv")

print("\nDataset Shape:", df.shape)
print(df.head())

# ==========================================
# Features & Target
# ==========================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nClasses:")
print(y.value_counts())

# ==========================================
# Encode Labels
# ==========================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================
# Feature Scaling
# ==========================================

print("\nScaling Features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Train SVM Model
# ==========================================

print("\nTraining SVM Model...")

model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================
# Confusion Matrix
# ==========================================

print("\nGenerating Confusion Matrix...")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Greens"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, "plant_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nFiles Saved Successfully")

print("plant_disease_model.pkl")
print("scaler.pkl")
print("label_encoder.pkl")

print("\nProject Completed Successfully!")
