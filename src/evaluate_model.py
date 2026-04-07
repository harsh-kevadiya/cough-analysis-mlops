import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X = np.load("data/processed/features/x.npy")
y = np.load("data/processed/features/y.npy")

# Normalize
if np.max(X) > 0:
    X = X / np.max(X)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Load model
model = load_model("models/cough_model.keras")

# Predictions
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# Plot
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()