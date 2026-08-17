import h5py
import numpy as np

# Load dataset
with h5py.File("dataset/ASCAD.h5", "r") as f:
    labels = np.array(f["Profiling_traces"]["labels"])

print("Total Labels :", len(labels))
print("Unique Labels :", np.unique(labels))
print("Minimum Label :", labels.min())
print("Maximum Label :", labels.max())

# Count how many samples belong to each label
unique, counts = np.unique(labels, return_counts=True)

print("\nLabel Distribution")
for u, c in zip(unique, counts):
    print(f"Hamming Weight {u}: {c} traces")