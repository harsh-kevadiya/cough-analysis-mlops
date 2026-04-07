import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/labels.csv")

# Separate classes
df_healthy = df[df["label"] == 0]
df_symptom = df[df["label"] == 1]

# Downsample healthy class
df_healthy_downsampled = df_healthy.sample(len(df_symptom), random_state=42)

# Combine
df_balanced = pd.concat([df_healthy_downsampled, df_symptom])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Save
df_balanced.to_csv("data/processed/balanced_labels.csv", index=False)

print("✅ Balanced dataset created!")
print(df_balanced["label"].value_counts())