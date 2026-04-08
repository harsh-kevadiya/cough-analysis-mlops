import pandas as pd
import os

df = pd.read_csv("data/processed/labels.csv")

df_healthy = df[df["label"] == 0]
df_symptom = df[df["label"] == 1]

# Find the smaller class size
n_samples = min(len(df_healthy), len(df_symptom))

if n_samples == 0:
    print(f"❌ Balance failed. Healthy: {len(df_healthy)}, Sick: {len(df_symptom)}")
else:
    df_h_balanced = df_healthy.sample(n_samples, random_state=42)
    df_s_balanced = df_symptom.sample(n_samples, random_state=42)

    df_final = pd.concat([df_h_balanced, df_s_balanced]).sample(frac=1, random_state=42)
    df_final.to_csv("data/processed/balanced_labels.csv", index=False)
    print(f"✅ Balanced! {len(df_final)} total samples (50/50 split).")