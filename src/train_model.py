import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os

# 1. Load Data
X = np.load("data/processed/features/X.npy")
y = np.load("data/processed/features/y.npy")

if len(X.shape) == 3:
    X = X.reshape(-1, 64, 80, 3)

# Shuffle is True by default in train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Build Bottleneck CNN
def build_bottleneck_model():
    model = models.Sequential([
        # Feature Extraction
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 80, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # THE FIX: Global Average Pooling instead of Flatten
        # This reduces the spatial dimensions to a single vector, killing "noise fingerprints"
        layers.GlobalAveragePooling2D(), 

        # Classification Head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_bottleneck_model()

# 3. Optimization
# Using a standard learning rate to ensure we escape the 50% "local minima"
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

# MLOps Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Handle Class Imbalance
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = {0: weights[0], 1: weights[1]}

# 4. Train
print("🚀 Training Bottleneck CNN (Anti-Overfit Mode)...")
model.fit(X_train, y_train, 
          epochs=60, 
          batch_size=32, 
          validation_data=(X_test, y_test), 
          class_weight=cw,
          callbacks=[early_stop, reduce_lr])

# Save output
os.makedirs("models", exist_ok=True)
model.save("models/cough_model.keras")
print("✅ New bottleneck model saved!")