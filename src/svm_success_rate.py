import h5py
import numpy as np
from pathlib import Path


RESULTS_DIR = Path("results")


TARGET_BYTE = 2
STEP = 100


# AES S-box
AES_SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)


HW_TABLE = np.array(
    [bin(i).count("1") for i in range(256)],
    dtype=np.uint8
)


def hypothetical_hw(plaintext_bytes, key_guess):
    """
    Calculate hypothetical Hamming Weight:
    HW(SBOX(plaintext XOR key))
    """
    sbox_input = np.bitwise_xor(
        plaintext_bytes,
        key_guess
    )

    sbox_output = AES_SBOX[sbox_input]

    return HW_TABLE[sbox_output]


def calculate_rank(probabilities, plaintext_bytes, correct_key):
    """
    Calculate the rank of the correct AES key.
    """

    scores = np.zeros(256)

    for key_guess in range(256):

        hw = hypothetical_hw(
            plaintext_bytes,
            key_guess
        )

        scores[key_guess] = np.sum(
            np.log(
                probabilities[
                    np.arange(len(hw)),
                    hw
                ] + 1e-12
            )
        )

    ranking = np.argsort(scores)[::-1]

    rank = np.where(
        ranking == correct_key
    )[0][0] + 1

    return rank, scores, ranking


def main():

    print("=" * 60)
    print("SVM SUCCESS RATE ANALYSIS")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Load probabilities
    # -------------------------------------------------

    print("\n[1/5] Loading SVM probabilities...")

    probabilities = np.load(
        RESULTS_DIR / "svm_probabilities.npy"
    )

    classes = np.load(
        RESULTS_DIR / "svm_classes.npy"
    )

    print("Probability matrix :", probabilities.shape)
    print("SVM classes        :", classes)

    # -------------------------------------------------
    # 2. Load metadata
    # -------------------------------------------------

    print("\n[2/5] Loading attack metadata...")

    with h5py.File(
        "dataset/ASCAD.h5",
        "r"
    ) as f:

        metadata = np.array(
            f["Attack_traces"]["metadata"]
        )

    plaintexts = np.array(
        metadata["plaintext"]
    )

    keys = np.array(
        metadata["key"]
    )

    print("Plaintexts :", plaintexts.shape)
    print("Keys       :", keys.shape)

    # -------------------------------------------------
    # 3. Target byte
    # -------------------------------------------------

    correct_key = int(
        keys[0, TARGET_BYTE]
    )

    print("\nTarget byte      :", TARGET_BYTE)
    print("Correct key byte :", correct_key)

    # -------------------------------------------------
    # 4. Success-rate calculation
    # -------------------------------------------------

    print("\n[3/5] Computing success rate...")

    total_traces = len(probabilities)

    trace_counts = []
    ranks = []
    successes = []

    success_count = 0

    for n in range(
        STEP,
        total_traces + 1,
        STEP
    ):

        current_probabilities = probabilities[:n]

        current_plaintexts = (
            plaintexts[:n, TARGET_BYTE]
        )

        rank, scores, ranking = calculate_rank(
            current_probabilities,
            current_plaintexts,
            correct_key
        )

        success = 1 if rank == 1 else 0

        if success == 1:
            success_count += 1

        trace_counts.append(n)
        ranks.append(rank)
        successes.append(success)

        print(
            f"Traces: {n:5d} | "
            f"Rank: {rank:3d} | "
            f"Success: {success}"
        )

    # -------------------------------------------------
    # 5. Save results
    # -------------------------------------------------

    trace_counts = np.array(
        trace_counts
    )

    ranks = np.array(
        ranks
    )

    successes = np.array(
        successes
    )

    success_rate = (
        np.mean(successes)
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        "Correct key       :",
        correct_key
    )

    print(
        "Final key rank    :",
        ranks[-1]
    )

    print(
        "Best key rank     :",
        np.min(ranks)
    )

    print(
        "Success rate      :",
        f"{success_rate:.4f}"
    )

    print(
        "Successful points :",
        success_count
    )

    # Save arrays

    np.save(
        RESULTS_DIR / "svm_trace_counts.npy",
        trace_counts
    )

    np.save(
        RESULTS_DIR / "svm_key_ranks.npy",
        ranks
    )

    np.save(
        RESULTS_DIR / "svm_success_points.npy",
        successes
    )

    with open(
        RESULTS_DIR / "svm_success_rate.txt",
        "w"
    ) as f:

        f.write(
            "========== SVM SUCCESS RATE ==========\n\n"
        )

        f.write(
            f"Target byte      : {TARGET_BYTE}\n"
        )

        f.write(
            f"Correct key      : {correct_key}\n"
        )

        f.write(
            f"Final key rank   : {ranks[-1]}\n"
        )

        f.write(
            f"Best key rank    : {np.min(ranks)}\n"
        )

        f.write(
            f"Success rate     : {success_rate:.4f}\n"
        )

        f.write(
            f"Successful points: {success_count}\n"
        )

        f.write(
            "\nTrace Count | Key Rank | Success\n"
        )

        f.write(
            "--------------------------------\n"
        )

        for trace_count, rank, success in zip(
            trace_counts,
            ranks,
            successes
        ):

            f.write(
                f"{trace_count:10d} | "
                f"{rank:8d} | "
                f"{success}\n"
            )

    print("\nResults saved successfully!")

    print(
        "\nGenerated files:"
    )

    print(
        "results/svm_trace_counts.npy"
    )

    print(
        "results/svm_key_ranks.npy"
    )

    print(
        "results/svm_success_points.npy"
    )

    print(
        "results/svm_success_rate.txt"
    )


if __name__ == "__main__":
    main()