import h5py
import joblib
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main():

    print("=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load ASCAD profiling traces
    # --------------------------------------------------

    print("\n[1/6] Loading profiling traces...")

    with h5py.File("dataset/ASCAD.h5", "r") as f:

        traces = np.array(
            f["Profiling_traces"]["traces"]
        )

        metadata = f["Profiling_traces"]["metadata"]

        plaintext = np.array(
            metadata["plaintext"]
        )

        keys = np.array(
            metadata["key"]
        )

    print("Traces shape    :", traces.shape)
    print("Plaintexts shape:", plaintext.shape)
    print("Keys shape      :", keys.shape)

    # --------------------------------------------------
    # 2. Load Hamming Weight labels
    # --------------------------------------------------

    print("\n[2/6] Loading Hamming Weight labels...")

    labels = np.load(
        "dataset/hw_labels.npy"
    )

    print("Labels shape :", labels.shape)

    print(
        "Unique labels :",
        np.unique(labels)
    )

    # Safety check
    if len(traces) != len(labels):

        raise ValueError(
            f"Number of traces ({len(traces)}) "
            f"does not match number of labels "
            f"({len(labels)})"
        )

    # --------------------------------------------------
    # 3. Train/test split
    # --------------------------------------------------

    print("\n[3/6] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        traces,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    print(
        "Training samples :",
        X_train.shape
    )

    print(
        "Testing samples  :",
        X_test.shape
    )

    # --------------------------------------------------
    # 4. Scale features
    # --------------------------------------------------

    print("\n[4/6] Scaling features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # IMPORTANT:
    # Save the exact scaler used during training.
    # success_rate.py will use this same scaler
    # on the attack traces.

    scaler_path = (
        MODEL_DIR /
        "rf_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        "Scaler saved :",
        scaler_path
    )

    # --------------------------------------------------
    # 5. Train Random Forest
    # --------------------------------------------------

    print("\n[5/6] Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print(
        "Number of trees    :",
        rf.n_estimators
    )

    print(
        "Number of features :",
        X_train.shape[1]
    )

    rf.fit(
        X_train,
        y_train
    )

    print("Training completed!")

    # --------------------------------------------------
    # 6. Evaluate model
    # --------------------------------------------------

    print("\n[6/6] Evaluating model...")

    predictions = rf.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )

    print(
        "\nAccuracy :",
        accuracy
    )

    print(
        "\nClassification Report:"
    )

    print(report)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = (
        MODEL_DIR /
        "best_random_forest.pkl"
    )

    joblib.dump(
        rf,
        model_path
    )

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    np.save(
        RESULTS_DIR /
        "rf_predictions.npy",
        predictions
    )

    # --------------------------------------------------
    # Save accuracy
    # --------------------------------------------------

    accuracy_path = (
        RESULTS_DIR /
        "rf_accuracy.txt"
    )

    with open(
        accuracy_path,
        "w"
    ) as f:

        f.write(
            f"Accuracy : {accuracy:.6f}\n"
        )

        f.write(
            f"Accuracy Percentage : "
            f"{accuracy * 100:.2f}%\n"
        )

        f.write(
            "\nClassification Report:\n"
        )

        f.write(report)

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print("\nGenerated files:")
    print("------------------------------")
    print(
        "models/best_random_forest.pkl"
    )
    print(
        "models/rf_scaler.pkl"
    )
    print(
        "results/rf_predictions.npy"
    )
    print(
        "results/rf_accuracy.txt"
    )


if __name__ == "__main__":
    main()