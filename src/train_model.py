import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os

# 1. Load Data
X = np.load("data/processed/features/X.npy")
y = np.load("data/processed/features/y.npy")

# Force 4D Shape (Samples, 64, 80, 3)
if len(X.shape) == 3:
    X = X.reshape(-1, 64, 80, 3)

print(f"📊 Training with data shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Build Model with EXPLICIT 3-channel input
model = models.Sequential([
    layers.Input(shape=(64, 80, 3)), # <--- THE 3 CHANNELS
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 3. Class Weights (The 'Guessing' Fix)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = {0: weights[0], 1: weights[1]}

# 4. Fit & Save
os.makedirs("models", exist_ok=True)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), class_weight=cw)
model.save("models/cough_model.keras")
print("✅ New 3-channel model saved successfully.")