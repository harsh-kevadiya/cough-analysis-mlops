import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load the processed features
X = np.load("data/processed/features/X.npy")
y = np.load("data/processed/features/y.npy")

# Select a "Symptomatic" sample (label 1) for the best visual
sample_idx = np.where(y == 1)[0][0] 
sample = X[sample_idx] # Shape should be (64, 80, 3)

# 2. Create the Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Channel 1: Mel Spectrogram', 'Channel 2: Delta', 'Channel 3: Delta-Delta']

for i in range(3):
    img = axes[i].imshow(sample[:, :, i], aspect='auto', origin='lower', cmap='magma')
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Time Windows')
    axes[i].set_ylabel('Mel Bins')
    plt.colorbar(img, ax=axes[i])

plt.tight_layout()

# 3. Save for PPT
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/feature_visualization.png", dpi=300)
plt.show()

print("✅ Feature visualization saved to reports/feature_visualization.png")