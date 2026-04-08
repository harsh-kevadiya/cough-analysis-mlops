import os
import pandas as pd

# Update these paths to match your folder structure
METADATA_CSV = "data/raw/metadata_compiled.csv" 
AUDIO_DIR = "data/processed/audio_wav"

data = []
skipped_no_audio = 0
skipped_low_quality = 0
skipped_unknown_status = 0

print("🔍 Extracting labels from your custom CSV...")

if not os.path.exists(METADATA_CSV):
    print(f"❌ ERROR: {METADATA_CSV} not found!")
else:
    # We read it without assuming names if the header is messy
    df_master = pd.read_csv(METADATA_CSV)
    
    # Standardizing column names based on your paste
    # We assume: Col 0 = UUID, Col 2 = cough_detected, Col 7 = status
    # But it's safer to use the names if they exist:
    uuid_col = 'uuid' if 'uuid' in df_master.columns else df_master.columns[0]
    score_col = 'cough_detected' if 'cough_detected' in df_master.columns else df_master.columns[2]
    status_col = 'status' if 'status' in df_master.columns else df_master.columns[7]

    for _, row in df_master.iterrows():
        uuid = row[uuid_col]
        audio_path = os.path.join(AUDIO_DIR, f"{uuid}.wav")

        # 1. Quality Filter: Must be > 0.8
        if row[score_col] < 0.8:
            skipped_low_quality += 1
            continue

        # 2. Check if file exists
        if not os.path.exists(audio_path):
            skipped_no_audio += 1
            continue

        # 3. Mapping your 'status' column
        status = str(row[status_col]).lower()
        if status == 'healthy':
            label = 0
        elif 'symptomatic' in status or 'covid' in status:
            label = 1
        else:
            skipped_unknown_status += 1
            continue

        data.append({"file_path": audio_path, "label": label})

    if data:
        df = pd.DataFrame(data)
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/labels.csv", index=False)
        print(f"✅ Success! Created labels.csv with {len(df)} samples.")
        print(f"📊 Distribution -> Healthy: {len(df[df.label==0])} | Sick: {len(df[df.label==1])}")
        print(f"⚠️ Debug -> Low Quality: {skipped_low_quality} | Missing Audio: {skipped_no_audio}")
    else:
        print("❌ No matches found! Ensure WAV filenames match the UUIDs in column 1.")