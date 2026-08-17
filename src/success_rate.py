import joblib
import h5py
import numpy as np

from attack_utils import compute_key_scores, rank_keys


MODEL_PATH = "models/best_random_forest.pkl"
SCALER_PATH = "models/rf_scaler.pkl"


def main():

    print("=" * 60)
    print("SUCCESS RATE ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------
    # 1. Load Random Forest model AND scaler
    # ---------------------------------------------------

    print("\n[1/5] Loading Random Forest model...")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Model loaded successfully!")
    print("Scaler loaded successfully!")

    print(
        "Random Forest expected features:",
        model.n_features_in_
    )

    # ---------------------------------------------------
    # 2. Load attack traces and metadata
    # ---------------------------------------------------

    print("\n[2/5] Loading attack traces and metadata...")

    with h5py.File("dataset/ASCAD.h5", "r") as f:

        X_attack = np.array(
            f["Attack_traces"]["traces"]
        )

        metadata = f["Attack_traces"]["metadata"]

        plaintext = np.array(
            metadata["plaintext"]
        )

        keys = np.array(
            metadata["key"]
        )

    print("Attack traces :", X_attack.shape)
    print("Plaintexts    :", plaintext.shape)
    print("Keys          :", keys.shape)

    # ---------------------------------------------------
    # Attack byte
    # ---------------------------------------------------

    TARGET_BYTE = 2

    pt_attack = plaintext[:, TARGET_BYTE]

    true_key = int(
        keys[0, TARGET_BYTE]
    )

    print("Target byte   :", TARGET_BYTE)
    print("Correct key byte:", true_key)

    # ---------------------------------------------------
    # 3. Prepare attack traces
    # ---------------------------------------------------

    print("\n[3/5] Preparing attack traces...")

    # Check feature count
    if X_attack.shape[1] != model.n_features_in_:

        raise ValueError(
            f"Feature mismatch! "
            f"Random Forest expects "
            f"{model.n_features_in_} features, "
            f"but attack traces contain "
            f"{X_attack.shape[1]} features."
        )

    print(
        "Attack features before scaling:",
        X_attack.shape[1]
    )

    # IMPORTANT:
    # Training used StandardScaler.
    # Therefore attack traces MUST use
    # the SAME saved scaler.

    X_attack = scaler.transform(
        X_attack
    )

    print(
        "Attack features after scaling:",
        X_attack.shape[1]
    )

    # ---------------------------------------------------
    # 4. Predict Hamming Weight probabilities
    # ---------------------------------------------------

    print(
        "\n[4/5] Predicting Hamming Weight probabilities..."
    )

    probabilities = model.predict_proba(
        X_attack
    )

    print(
        "Probability matrix shape:",
        probabilities.shape
    )

    print(
        "Model classes:",
        model.classes_
    )

    # Safety check
    if probabilities.shape[1] != 9:

        raise ValueError(
            "Expected 9 Hamming Weight classes "
            "(0-8), but model returned "
            f"{probabilities.shape[1]} classes."
        )

    # ---------------------------------------------------
    # 5. Success Rate analysis
    # ---------------------------------------------------

    print(
        "\n[5/5] Computing Success Rate..."
    )

    trace_counts = []
    success = []
    ranks = []

    for n in range(
        100,
        len(X_attack) + 1,
        100
    ):

        # Use first n attack traces
        current_probabilities = probabilities[:n]
        current_plaintext = pt_attack[:n]

        # Calculate score for all 256 key guesses
        key_scores = compute_key_scores(
            current_probabilities,
            current_plaintext
        )

        # Rank keys from best to worst
        ranking = rank_keys(
            key_scores
        )

        # Find rank of the real key
        rank_positions = np.where(
            ranking == true_key
        )[0]

        if len(rank_positions) == 0:

            raise RuntimeError(
                f"Correct key {true_key} "
                "was not found in ranking."
            )

        rank = int(
            rank_positions[0] + 1
        )

        # Success means correct key is rank 1
        if rank == 1:
            success_value = 1
        else:
            success_value = 0

        trace_counts.append(n)
        ranks.append(rank)
        success.append(success_value)

        print(
            f"Traces: {n:5d} | "
            f"Rank: {rank:3d} | "
            f"Success: {success_value}"
        )

    # ---------------------------------------------------
    # Convert to numpy arrays
    # ---------------------------------------------------

    success = np.array(
        success,
        dtype=np.int32
    )

    ranks = np.array(
        ranks,
        dtype=np.int32
    )

    trace_counts = np.array(
        trace_counts,
        dtype=np.int32
    )

    # ---------------------------------------------------
    # Save results
    # ---------------------------------------------------

    np.save(
        "results/success_rate.npy",
        success
    )

    np.save(
        "results/key_ranks.npy",
        ranks
    )

    np.save(
        "results/trace_counts.npy",
        trace_counts
    )

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Correct key       : {true_key}"
    )

    print(
        f"Final key rank    : {ranks[-1]}"
    )

    print(
        f"Best key rank     : {ranks.min()}"
    )

    best_index = np.argmin(ranks)

    print(
        f"Best rank traces  : "
        f"{trace_counts[best_index]}"
    )

    print(
        f"Success rate      : "
        f"{success.mean():.4f}"
    )

    print(
        f"Successful points : "
        f"{success.sum()}"
    )

    print(
        "\nResults saved successfully!"
    )


if __name__ == "__main__":
    main()