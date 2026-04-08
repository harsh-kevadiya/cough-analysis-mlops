import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os

X = np.load("data/processed/features/X.npy")
y = np.load("data/processed/features/y.npy")

if len(X.shape) == 3:
    X = X.reshape(-1, 64, 80, 3)

MODEL_PATH = "models/cough_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# SHAPE VERIFICATION
expected_channels = model.input_shape[-1]
actual_channels = X.shape[-1]

if expected_channels != actual_channels:
    print(f"❌ MISMATCH: Model expects {expected_channels} channels, but X has {actual_channels}.")
    exit()

print(f"🔮 Predicting on {actual_channels} channels...")
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n📊 LAB REPORT")
print(classification_report(y, y_pred, target_names=['Healthy', 'Symptomatic']))
print("\n🧩 CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))