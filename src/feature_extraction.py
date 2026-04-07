import os
import numpy as np
import pandas as pd
import librosa

df = pd.read_csv("data/processed/balanced_labels.csv")

OUTPUT_DIR = "data/processed/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X = []
y = []

max_len = 128

print("🔄 Extracting MFCC features...")

for i, row in df.iterrows():
    try:
        file_path = row["file_path"]
        label = row["label"]

        # Load audio
        signal, sr = librosa.load(file_path, sr=16000)

        # Skip silent audio
        if np.max(np.abs(signal)) < 0.01:
            continue

        # Normalize
        signal = signal / np.max(np.abs(signal))

        # Extract MFCC (MORE STABLE)
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40
        )

        # Resize to fixed size
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
        else:
            mfcc = mfcc[:, :max_len]

        X.append(mfcc)
        y.append(label)

        if i % 500 == 0:
            print(f"Processed {i} files...")

    except Exception:
        continue

X = np.array(X)
y = np.array(y)

print(f"\n📊 Max value in X: {np.max(X)}")
print(f"📊 Min value in X: {np.min(X)}")

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

print("\n✅ Feature extraction completed!")
print(f"📊 Shape of X: {X.shape}")
print(f"📊 Shape of y: {y.shape}")