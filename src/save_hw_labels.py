import h5py
import numpy as np

from leakage_model import compute_hw_labels


DATASET_PATH = "dataset/ASCAD.h5"
OUTPUT_PATH = "dataset/hw_labels.npy"

TARGET_BYTE = 2


print("=" * 60)
print("GENERATING HAMMING WEIGHT LABELS")
print("=" * 60)

print("\n[1/3] Loading profiling metadata...")

with h5py.File(DATASET_PATH, "r") as f:

    metadata = f["Profiling_traces"]["metadata"]

    plaintexts = np.array(
        metadata["plaintext"]
    )

    keys = np.array(
        metadata["key"]
    )

print("Plaintexts shape :", plaintexts.shape)
print("Keys shape       :", keys.shape)

print("\n[2/3] Computing Hamming Weight labels...")

hw_labels = compute_hw_labels(
    plaintexts,
    keys,
    target_byte_index=TARGET_BYTE
)

print("HW labels shape  :", hw_labels.shape)
print("Unique labels    :", np.unique(hw_labels))
print(
    "Label counts     :",
    np.unique(hw_labels, return_counts=True)
)

print("\nFirst 20 labels:")
print(hw_labels[:20])

print("\n[3/3] Saving labels...")

np.save(
    OUTPUT_PATH,
    hw_labels
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("Saved :", OUTPUT_PATH)
print("Shape :", hw_labels.shape)