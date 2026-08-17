import h5py
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load attack traces
with h5py.File("dataset/ASCAD.h5", "r") as f:
    attack_traces = np.array(f["Attack_traces"]["traces"])

print("Attack Traces Shape:", attack_traces.shape)

# Normalize attack traces
scaler = StandardScaler()
attack_traces = scaler.fit_transform(attack_traces)

# Load trained model
rf = joblib.load("models/best_random_forest.pkl")

# Predict Hamming Weight
predictions = rf.predict(attack_traces)

print("\nFirst 20 Predictions:")
print(predictions[:20])

# Save predictions
np.save("results/attack_predictions.npy", predictions)

print("\nPredictions saved successfully!")