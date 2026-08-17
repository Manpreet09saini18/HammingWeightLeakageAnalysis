import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load traces
with h5py.File("dataset/ASCAD.h5", "r") as f:
    traces = np.array(f["Profiling_traces"]["traces"])

labels = np.load("dataset/hw_labels.npy")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    traces,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Normalize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

print("\nNormalization Completed Successfully!")

# Save processed files
np.save("dataset/X_train.npy", X_train)
np.save("dataset/X_test.npy", X_test)

np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)

print("\nProcessed Dataset Saved Successfully!")