# ==========================================
# Plant Disease Prediction
# ==========================================

import joblib
import pandas as pd

# ==========================================
# Load Saved Files
# ==========================================

model = joblib.load("plant_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Model Loaded Successfully!")

# ==========================================
# Load Test Sample
# ==========================================

sample = pd.read_csv("sample.csv")

# Scale Features
sample = scaler.transform(sample)

# Predict
prediction = model.predict(sample)

# Convert Label
result = label_encoder.inverse_transform(prediction)

print("\nPredicted Disease:")
print(result[0])
