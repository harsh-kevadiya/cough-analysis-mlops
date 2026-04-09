import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports", exist_ok=True)

# 1. Load Data & Reshape
X = np.load("data/processed/features/X.npy")
y = np.load("data/processed/features/y.npy")
X = X.reshape(-1, 64, 80, 3)

# 2. Load Model & Predict
model = tf.keras.models.load_model("models/cough_model.keras")
y_pred = (model.predict(X) > 0.5).astype(int)

# 3. Save Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Symptomatic'], yticklabels=['Healthy', 'Symptomatic'])
plt.title('Final Model: Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("reports/confusion_matrix.png", dpi=300)

print("\n" + "="*30)
print(classification_report(y, y_pred, target_names=['Healthy', 'Symptomatic']))
print("✅ Confusion Matrix saved to reports/confusion_matrix.png")