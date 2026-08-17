import pandas as pd
import numpy as np
import joblib

# Load model
rf = joblib.load("models/best_random_forest.pkl")

# Load predictions
predictions = np.load("results/attack_predictions.npy")

# Feature importance
importance = rf.feature_importances_
top10 = np.argsort(importance)[::-1][:10]

rows = []

rows.append(["Algorithm", "Random Forest"])
rows.append(["Trees", rf.n_estimators])
rows.append(["Maximum Depth", rf.max_depth])
rows.append(["Attack Traces", len(predictions)])

rows.append(["", ""])

rows.append(["Top Feature", "Importance"])

for i in top10:
    rows.append([f"Trace Point {i}", importance[i]])

df = pd.DataFrame(rows)

df.to_csv("results/final_results.csv", index=False, header=False)

print("Final results exported successfully!")