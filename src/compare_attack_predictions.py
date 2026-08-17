import h5py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load actual attack labels
with h5py.File("dataset/ASCAD.h5", "r") as f:
    attack_labels = np.array(f["Attack_traces"]["labels"])

# Convert labels to Hamming Weight
actual_hw = np.array([bin(x).count("1") for x in attack_labels])

# Load predicted HW
predicted_hw = np.load("results/attack_predictions.npy")

print("=" * 50)
print("Attack Trace Prediction Evaluation")
print("=" * 50)

accuracy = accuracy_score(actual_hw, predicted_hw)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(actual_hw, predicted_hw))

# Confusion Matrix
cm = confusion_matrix(actual_hw, predicted_hw)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("Attack Trace Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted HW")
plt.ylabel("Actual HW")

plt.xticks(range(9))
plt.yticks(range(9))

plt.savefig("results/attack_confusion_matrix.png")
plt.show()

print("\nConfusion Matrix saved successfully!")