import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Load data
X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")

# Load best model
rf = joblib.load("models/best_random_forest.pkl")

# Prediction
y_pred = rf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Random Forest Evaluation")
print("=" * 50)
print(f"\nAccuracy : {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues", values_format="d")

plt.title("Random Forest Confusion Matrix")

plt.savefig("results/random_forest_confusion_matrix.png", dpi=300)

plt.show()

print("\nConfusion Matrix saved successfully!")