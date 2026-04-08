import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/processed/features/X.npy")
y = np.load("data/processed/features/y.npy")

idx = np.random.randint(0, len(X))
plt.imshow(X[idx], aspect='auto', origin='lower', cmap='magma')
plt.title(f"Label: {y[idx]} (0=Healthy, 1=Sick)")
plt.colorbar()
plt.show()