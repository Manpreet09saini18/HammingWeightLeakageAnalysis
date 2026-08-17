import time
from pathlib import Path

import h5py
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def train_svm():

    start_time = time.time()

    print("=" * 60)
    print("SVM TRAINING PIPELINE")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Load ASCAD dataset
    # -------------------------------------------------

    print("\n[1/6] Loading dataset...")

    with h5py.File("dataset/ASCAD.h5", "r") as f:

        X_train = np.array(
            f["Profiling_traces"]["traces"]
        )

        X_attack = np.array(
            f["Attack_traces"]["traces"]
        )

    # Correct 9-class Hamming Weight labels
    y_train = np.load(
        "dataset/hw_labels.npy"
    )

    print("Profiling traces :", X_train.shape)
    print("Attack traces    :", X_attack.shape)

    print("Training labels  :", y_train.shape)
    print("Unique HW labels :", np.unique(y_train))

    # Safety check
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Training traces ({len(X_train)}) "
            f"and labels ({len(y_train)}) do not match."
        )

    if not np.all(np.isin(y_train, np.arange(9))):
        raise ValueError(
            "Training labels are not valid 0-8 Hamming Weight labels."
        )

    # -------------------------------------------------
    # 2. Load selected features
    # -------------------------------------------------

    print("\n[2/6] Loading selected features...")

    features = np.load(
        RESULTS_DIR / "features.npy"
    )

    features = np.asarray(
        features,
        dtype=int
    )

    print("Selected features :", len(features))

    X_train = X_train[:, features]
    X_attack = X_attack[:, features]

    print("Training shape :", X_train.shape)
    print("Attack shape   :", X_attack.shape)

    # -------------------------------------------------
    # Use smaller subset for SVM
    # -------------------------------------------------

    N = 10000

    if N > len(X_train):
        N = len(X_train)

    X_train = X_train[:N]
    y_train = y_train[:N]

    print("\nSVM training subset :", N)

    print(
        "Training HW distribution:"
    )

    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):
        print(
            f"  HW {label}: {count}"
        )

    # -------------------------------------------------
    # 3. Scale features
    # -------------------------------------------------

    print("\n[3/6] Scaling selected features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_attack = scaler.transform(
        X_attack
    )

    scaler_path = (
        MODEL_DIR /
        "svm_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        "Scaler saved :",
        scaler_path
    )

    # -------------------------------------------------
    # 4. Train SVM
    # -------------------------------------------------

    print("\n[4/6] Training SVM...")

    print(
        "This may take some time..."
    )

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",

        # IMPORTANT:
        # Required for attack key ranking
        probability=True,

        random_state=42
    )

    print(
        "Training samples :",
        len(X_train)
    )

    print(
        "Feature count    :",
        X_train.shape[1]
    )

    print(
        "Kernel           : RBF"
    )

    print(
        "C                : 10"
    )

    print(
        "Probability      : True"
    )

    svm.fit(
        X_train,
        y_train
    )

    print(
        "\nSVM training finished!"
    )

    # -------------------------------------------------
    # 5. Predict attack traces
    # -------------------------------------------------

    print(
        "\n[5/6] Predicting attack traces..."
    )

    # Class predictions
    predictions = svm.predict(
        X_attack
    )

    # Probability predictions
    probabilities = svm.predict_proba(
        X_attack
    )

    print(
        "Prediction shape  :",
        predictions.shape
    )

    print(
        "Probability shape :",
        probabilities.shape
    )

    print(
        "SVM classes       :",
        svm.classes_
    )

    # Safety check
    if probabilities.shape[1] != 9:

        raise ValueError(
            "SVM did not produce 9 Hamming Weight classes."
        )

    # -------------------------------------------------
    # 6. Save outputs
    # -------------------------------------------------

    print(
        "\n[6/6] Saving outputs..."
    )

    # Save model
    model_path = (
        MODEL_DIR /
        "svm_model.pkl"
    )

    joblib.dump(
        svm,
        model_path
    )

    # Save predicted HW classes
    np.save(
        RESULTS_DIR /
        "svm_predictions.npy",
        predictions
    )

    # Save probabilities
    np.save(
        RESULTS_DIR /
        "svm_probabilities.npy",
        probabilities
    )

    # Save classes
    np.save(
        RESULTS_DIR /
        "svm_classes.npy",
        svm.classes_
    )

    # -------------------------------------------------
    # Training information
    # -------------------------------------------------

    training_time = (
        time.time() -
        start_time
    )

    log_path = (
        RESULTS_DIR /
        "svm_training_log.txt"
    )

    with open(
        log_path,
        "w"
    ) as f:

        f.write(
            "========== SVM Training ==========\n"
        )

        f.write(
            f"Training Samples : {len(X_train)}\n"
        )

        f.write(
            f"Attack Samples   : {len(X_attack)}\n"
        )

        f.write(
            f"Selected Features: {len(features)}\n"
        )

        f.write(
            "Kernel           : RBF\n"
        )

        f.write(
            "C                : 10\n"
        )

        f.write(
            "Gamma            : scale\n"
        )

        f.write(
            "Probability      : True\n"
        )

        f.write(
            f"Training Time    : "
            f"{training_time:.2f} sec\n"
        )

        f.write(
            "\nHW Classes:\n"
        )

        f.write(
            str(svm.classes_)
        )

        f.write(
            "\n"
        )

    # -------------------------------------------------
    # Final output
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("SVM TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Training Time : "
        f"{training_time:.2f} sec"
    )

    print("\nGenerated Files")
    print("------------------------------")

    print(
        "models/svm_model.pkl"
    )

    print(
        "models/svm_scaler.pkl"
    )

    print(
        "results/svm_predictions.npy"
    )

    print(
        "results/svm_probabilities.npy"
    )

    print(
        "results/svm_classes.npy"
    )

    print(
        "results/svm_training_log.txt"
    )

    print("\nIMPORTANT:")
    print(
        "SVM classification accuracy on the attack set "
        "was NOT calculated because attack labels are "
        "not 9-class HW labels."
    )

    print(
        "\nThe probability matrix is ready "
        "for AES key ranking / success-rate analysis."
    )


if __name__ == "__main__":
    train_svm()