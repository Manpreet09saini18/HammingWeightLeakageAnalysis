import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load best model
rf = joblib.load("models/best_random_forest.pkl")

# Get feature importance
importance = rf.feature_importances_

print("Number of Features :", len(importance))

# Top 10 important features
indices = np.argsort(importance)[::-1]

print("\nTop 10 Important Trace Points")

for i in range(10):
    print(f"Rank {i+1}: Trace Point {indices[i]}  Importance = {importance[indices[i]]:.6f}")

# Plot
plt.figure(figsize=(15,5))
plt.plot(importance)
plt.title("Feature Importance of Random Forest")
plt.xlabel("Trace Sample Index")
plt.ylabel("Importance")

plt.grid(True)

plt.savefig("results/feature_importance.png", dpi=300)

plt.show()

print("\nFeature importance graph saved!")