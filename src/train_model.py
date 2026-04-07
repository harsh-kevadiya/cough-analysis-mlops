import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load features
X = np.load("data/processed/features/x.npy")
y = np.load("data/processed/features/y.npy")

print("📊 Loaded X shape:", X.shape)

# 🔥 STEP 1: FIX DATA SCALE (VERY IMPORTANT)
# Flatten for scaling
n_samples, h, w = X.shape
X = X.reshape(n_samples, -1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape back
X = X.reshape(n_samples, h, w, 1)

# 🔥 STEP 2: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)

# 🔥 STEP 3: IMPROVED CNN (STABLE)
model = Sequential([
    Input(shape=(h, w, 1)),

    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')  # 🔥 binary classification
])

# 🔥 STEP 4: CORRECT LOSS FUNCTION
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 🔥 STEP 5: EARLY STOPPING (PREVENT OVERFIT)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 🔥 STEP 6: TRAIN
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# 🔥 STEP 7: EVALUATE
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save("models/cough_model.keras")
print("💾 Model saved!")