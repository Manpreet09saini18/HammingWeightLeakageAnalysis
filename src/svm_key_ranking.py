import h5py
import joblib
import numpy as np
from pathlib import Path

from attack_utils import compute_key_scores, rank_keys


MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main():

    print("=" * 60)
    print("SVM KEY RANKING / SUCCESS RATE ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load SVM model
    # ---------------------------------------------------------

    print("\n[1/7] Loading SVM model...")

    svm = joblib.load(
        MODEL_DIR / "svm_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "svm_scaler.pkl"
    )

    print("SVM model loaded successfully!")
    print("Scaler loaded successfully!")

    expected_features = svm.n_features_in_

    print(
        "SVM expected features:",
        expected_features
    )

    # ---------------------------------------------------------
    # 2. Load probabilities
    # ---------------------------------------------------------

    print("\n[2/7] Loading SVM probabilities...")

    probabilities = np.load(
        RESULTS_DIR / "svm_probabilities.npy"
    )

    classes = np.load(
        RESULTS_DIR / "svm_classes.npy"
    )

    print(
        "Probability matrix:",
        probabilities.shape
    )

    print(
        "SVM classes:",
        classes
    )

    if probabilities.shape[1] != 9:

        raise ValueError(
            "SVM probability matrix must contain "
            "9 Hamming Weight classes."
        )

    # ---------------------------------------------------------
    # 3. Load attack traces and metadata
    # ---------------------------------------------------------

    print("\n[3/7] Loading attack traces and metadata...")

    with h5py.File(
        "dataset/ASCAD.h5",
        "r"
    ) as f:

        attack_group = f["Attack_traces"]

        traces = np.array(
            attack_group["traces"]
        )

        metadata = np.array(
            attack_group["metadata"]
        )

        print(
            "Attack traces:",
            traces.shape
        )

        print(
            "Metadata shape:",
            metadata.shape
        )

        # -----------------------------------------------------
        # Inspect metadata fields
        # -----------------------------------------------------

        print(
            "Metadata fields:",
            metadata.dtype.names
        )

        # -----------------------------------------------------
        # Get plaintext
        # -----------------------------------------------------

        if "plaintext" in metadata.dtype.names:

            plaintexts = np.array(
                metadata["plaintext"]
            )

        elif "plaintext" in metadata.dtype.names:

            plaintexts = np.array(
                metadata["plaintext"]
            )

        else:

            raise KeyError(
                "Plaintext field was not found in "
                "Attack_traces/metadata."
            )

        # -----------------------------------------------------
        # Get key
        # -----------------------------------------------------

        if "key" in metadata.dtype.names:

            keys = np.array(
                metadata["key"]
            )

        else:

            keys = None

    print(
        "Plaintexts shape:",
        plaintexts.shape
    )

    if keys is not None:

        print(
            "Keys shape:",
            keys.shape
        )

    # ---------------------------------------------------------
    # 4. Target byte
    # ---------------------------------------------------------

    TARGET_BYTE = 2

    print(
        "\nTarget byte:",
        TARGET_BYTE
    )

    if keys is not None:

        correct_key = int(
            keys[0][TARGET_BYTE]
        )

        print(
            "Correct key byte:",
            correct_key
        )

    else:

        print(
            "WARNING: Key is not available "
            "in metadata."
        )

        correct_key = None

    # ---------------------------------------------------------
    # 5. Verify dimensions
    # ---------------------------------------------------------

    print("\n[4/7] Checking dimensions...")

    features = np.load(
        RESULTS_DIR / "features.npy"
    )

    print(
        "Selected features:",
        len(features)
    )

    if len(features) != expected_features:

        raise ValueError(
            f"Feature mismatch!\n"
            f"SVM expects {expected_features} features\n"
            f"features.npy contains {len(features)}"
        )

    X_attack = traces[:, features]

    print(
        "Attack features:",
        X_attack.shape
    )

    # ---------------------------------------------------------
    # Scale attack traces
    # ---------------------------------------------------------

    X_attack = scaler.transform(
        X_attack
    )

    print(
        "Attack features scaled:",
        X_attack.shape
    )

    # ---------------------------------------------------------
    # 6. AES key ranking
    # ---------------------------------------------------------

    print("\n[5/7] Computing AES key ranking...")

    # We only use the number of probability rows
    # available from the SVM.

    N = min(
        len(probabilities),
        len(plaintexts)
    )

    probabilities = probabilities[:N]
    plaintexts = plaintexts[:N]

    plaintext_bytes = plaintexts[:, TARGET_BYTE]

    print(
        "Traces used:",
        N
    )

    # ---------------------------------------------------------
    # Compute scores for 256 key hypotheses
    # ---------------------------------------------------------

    key_scores = compute_key_scores(
        probabilities,
        plaintext_bytes
    )

    # ---------------------------------------------------------
    # Rank keys
    # ---------------------------------------------------------

    key_ranking = rank_keys(
        key_scores
    )

    # Position of correct key
    # rank 1 = best

    if correct_key is not None:

        correct_positions = np.where(
            key_ranking == correct_key
        )[0]

        if len(correct_positions) > 0:

            final_rank = (
                int(correct_positions[0])
                + 1
            )

        else:

            final_rank = 256

    else:

        final_rank = None

    # ---------------------------------------------------------
    # Print top candidates
    # ---------------------------------------------------------

    print("\nTop 10 key candidates:")

    print(
        "-" * 40
    )

    for i in range(10):

        candidate = int(
            key_ranking[i]
        )

        score = key_scores[candidate]

        marker = ""

        if (
            correct_key is not None
            and candidate == correct_key
        ):

            marker = "  <-- CORRECT KEY"

        print(
            f"Rank {i + 1:3d} | "
            f"Key {candidate:3d} | "
            f"Score {score:.4f}"
            f"{marker}"
        )

    # ---------------------------------------------------------
    # 7. Save results
    # ---------------------------------------------------------

    print("\n[6/7] Saving key-ranking results...")

    np.save(
        RESULTS_DIR / "svm_key_scores.npy",
        key_scores
    )

    np.save(
        RESULTS_DIR / "svm_key_ranking.npy",
        key_ranking
    )

    # ---------------------------------------------------------
    # Save human-readable report
    # ---------------------------------------------------------

    with open(
        RESULTS_DIR / "svm_key_ranking.txt",
        "w"
    ) as f:

        f.write(
            "========== SVM AES KEY RANKING ==========\n\n"
        )

        f.write(
            f"Target byte      : {TARGET_BYTE}\n"
        )

        if correct_key is not None:

            f.write(
                f"Correct key      : {correct_key}\n"
            )

            f.write(
                f"Final key rank   : {final_rank}\n"
            )

        f.write(
            f"Traces used      : {N}\n"
        )

        f.write(
            "\nTop 10 Candidates\n"
        )

        f.write(
            "------------------------------\n"
        )

        for i in range(10):

            candidate = int(
                key_ranking[i]
            )

            score = key_scores[candidate]

            f.write(
                f"Rank {i + 1:3d} | "
                f"Key {candidate:3d} | "
                f"Score {score:.6f}\n"
            )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\n[7/7] Analysis completed!")

    print(
        "\n" + "=" * 60
    )

    print(
        "SVM KEY RANKING COMPLETED"
    )

    print(
        "=" * 60
    )

    if correct_key is not None:

        print(
            f"Correct key : {correct_key}"
        )

        print(
            f"Final rank  : {final_rank}"
        )

    print(
        "\nGenerated files:"
    )

    print(
        "results/svm_key_scores.npy"
    )

    print(
        "results/svm_key_ranking.npy"
    )

    print(
        "results/svm_key_ranking.txt"
    )


if __name__ == "__main__":
    main()