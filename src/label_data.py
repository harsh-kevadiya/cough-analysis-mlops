import os
import json
import pandas as pd

RAW_DIR = "data/raw"
AUDIO_DIR = "data/processed/audio_wav"

data = []

count = 0
skipped = 0

for file in os.listdir(RAW_DIR):
    if file.endswith(".json"):
        json_path = os.path.join(RAW_DIR, file)

        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)

            audio_filename = file.replace(".json", ".wav")
            audio_path = os.path.join(AUDIO_DIR, audio_filename)

            if not os.path.exists(audio_path):
                skipped += 1
                continue

            status = metadata.get("status", "unknown")

            if status == "healthy":
                label = 0   # Healthy
            elif status == "symptomatic":
                label = 1   # COVID / respiratory issue
            else:
                skipped += 1
                continue

            data.append({
                "file_path": audio_path,
                "label": label
            })

            count += 1

        except:
            skipped += 1

df = pd.DataFrame(data)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/labels.csv", index=False)

print("✅ Labels created successfully!")
print(f"📊 Total samples: {len(df)}")
print(f"⚠️ Skipped files: {skipped}")

print("\n📊 Label Distribution:\n")
print(df["label"].value_counts())
