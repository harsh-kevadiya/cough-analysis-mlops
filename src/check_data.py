import numpy as np

# 1. Load your feature and label files
X = np.load("data/processed/features/X.npy")
y = np.load("data/processed/features/y.npy")

# 2. Check total count
num_samples = X.shape[0]
print(f"📊 Total Samples: {num_samples}")

# 3. Check Class Balance
healthy_count = np.sum(y == 0)
symptomatic_count = np.sum(y == 1)

print(f"✅ Healthy Samples: {healthy_count} ({healthy_count/num_samples:.1%})")
print(f"⚠️ Symptomatic Samples: {symptomatic_count} ({symptomatic_count/num_samples:.1%})")

# 4. Check for potential duplicates (Data Leakage check)
# We flatten the images to check if any are exactly identical
X_flattened = X.reshape(num_samples, -1)
unique_samples = np.unique(X_flattened, axis=0).shape[0]

if unique_samples < num_samples:
    print(f"🚨 ALERT: Found {num_samples - unique_samples} duplicate feature sets!")
    print("This means the same audio clip is likely in both your Train and Test sets.")
else:
    print("💎 No exact duplicates found in feature sets.")