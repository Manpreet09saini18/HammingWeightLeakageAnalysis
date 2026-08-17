import numpy as np
import joblib

print("=" * 60)
print("Hamming Weight Leakage Analysis")
print("Final Summary Report")
print("=" * 60)

# Load best Random Forest model
rf = joblib.load("models/best_random_forest.pkl")

# Load attack predictions
predictions = np.load("results/attack_predictions.npy")

print("\nModel Information")
print("-" * 40)
print("Algorithm :", type(rf).__name__)
print("Number of Trees :", rf.n_estimators)
print("Maximum Depth :", rf.max_depth)

print("\nAttack Prediction Information")
print("-" * 40)
print("Total Attack Traces :", len(predictions))
print("Minimum HW :", predictions.min())
print("Maximum HW :", predictions.max())

unique, counts = np.unique(predictions, return_counts=True)

print("\nPredicted Hamming Weight Distribution")
print("-" * 40)

for u, c in zip(unique, counts):
    print(f"HW {u}: {c}")

print("\nTop 10 Important Trace Points")
print("-" * 40)

importance = rf.feature_importances_
top10 = np.argsort(importance)[::-1][:10]

for i, idx in enumerate(top10, start=1):
    print(f"{i}. Trace Point {idx} : {importance[idx]:.6f}")

print("\nSummary Completed Successfully!")