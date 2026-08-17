import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X_train = np.load("dataset/X_train.npy")
X_test = np.load("dataset/X_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

experiments = [
    (100, None),
    (200, None),
    (300, None),
    (100, 20),
    (200, 20),
    (300, 20)
]

results = []

best_accuracy = 0
best_model = None

print("="*50)
print("Random Forest Hyperparameter Tuning")
print("="*50)

for trees, depth in experiments:

    print(f"\nTraining RF (Trees={trees}, MaxDepth={depth})")

    rf = RandomForestClassifier(
        n_estimators=trees,
        max_depth=depth,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy :", accuracy)

    results.append({
        "Trees": trees,
        "MaxDepth": depth,
        "Accuracy": accuracy
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = rf

# Save comparison
df = pd.DataFrame(results)

df.to_csv("results/random_forest_comparison.csv", index=False)

# Save best model
joblib.dump(best_model, "models/best_random_forest.pkl")

print("\n===========================")
print("Best Accuracy :", best_accuracy)
print("===========================")

print("\nComparison saved.")
print("Best model saved.")