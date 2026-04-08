import os
import numpy as np
import pandas as pd
import librosa

PROJECT_ROOT = os.getcwd()
CSV_PATH = os.path.join(PROJECT_ROOT, "data/processed/balanced_labels.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_feature(feat):
    return (feat - np.mean(feat)) / (np.std(feat) + 1e-6)

df = pd.read_csv(CSV_PATH)
X, y = [], []
max_len = 80 
n_mels = 64

print(f"🔄 Extracting 3-Channel Features...")

for i, row in df.iterrows():
    audio_path = os.path.join(PROJECT_ROOT, row['file_path'])
    if not os.path.exists(audio_path): continue

    try:
        y_audio, sr = librosa.load(audio_path, sr=16000)
        y_audio, _ = librosa.effects.trim(y_audio, top_db=20)
        
        # 1. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=n_mels)
        mel = librosa.power_to_db(mel)
        
        # 2. MFCC
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mels)
        
        # 3. Delta MFCC
        delta = librosa.feature.delta(mfcc)

        def fix_shape(feat):
            if feat.shape[1] < max_len:
                return np.pad(feat, ((0, 0), (0, max_len - feat.shape[1])))
            return feat[:, :max_len]

        # Explicitly stack into (64, 80, 3)
        # We use float32 to save memory and match Keras defaults
        f1 = normalize_feature(fix_shape(mel))
        f2 = normalize_feature(fix_shape(mfcc))
        f3 = normalize_feature(fix_shape(delta))
        
        stacked = np.stack([f1, f2, f3], axis=-1).astype(np.float32)
        
        X.append(stacked)
        y.append(row['label'])

        if i % 200 == 0: print(f"✅ Processed {i}...")
    except Exception as e:
        continue

# CRITICAL: Convert list to a single 4D NumPy array
X_final = np.array(X) 
y_final = np.array(y)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X_final)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_final)
print(f"🎉 Final Shape: {X_final.shape}") # Should be (4770, 64, 80, 3)