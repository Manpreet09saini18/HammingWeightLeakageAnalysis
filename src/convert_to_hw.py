import h5py
import numpy as np

# Function to compute Hamming Weight
def hamming_weight(x):
    return bin(x).count("1")

with h5py.File("dataset/ASCAD.h5", "r") as f:
    labels = np.array(f["Profiling_traces"]["labels"])

# Convert 0-255 labels to Hamming Weight (0-8)
hw_labels = np.array([hamming_weight(x) for x in labels])

print("Original Labels:")
print(np.unique(labels)[:10], "...")

print("\nHamming Weight Labels:")
print(np.unique(hw_labels))

print("\nMinimum HW:", hw_labels.min())
print("Maximum HW:", hw_labels.max())

# Distribution
unique, counts = np.unique(hw_labels, return_counts=True)

print("\nDistribution:")
for u, c in zip(unique, counts):
    print(f"HW {u}: {c} traces")