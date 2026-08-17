import numpy as np
import matplotlib.pyplot as plt

# Load predictions
predictions = np.load("results/attack_predictions.npy")

print("=" * 40)
print("Attack Prediction Analysis")
print("=" * 40)

print("Total Predictions :", len(predictions))
print("Minimum HW :", predictions.min())
print("Maximum HW :", predictions.max())

# Distribution
unique, counts = np.unique(predictions, return_counts=True)

print("\nPrediction Distribution:\n")

for u, c in zip(unique, counts):
    print(f"HW {u}: {c} traces")

# Plot
plt.figure(figsize=(8,5))
plt.bar(unique, counts)

plt.title("Predicted Hamming Weight Distribution")
plt.xlabel("Hamming Weight")
plt.ylabel("Number of Traces")

plt.grid(axis="y")

plt.savefig("results/prediction_distribution.png")
plt.show()

print("\nPrediction distribution graph saved successfully!")