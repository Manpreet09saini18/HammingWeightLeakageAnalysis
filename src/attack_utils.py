import numpy as np

from leakage_model import AES_SBOX

# -------------------------------
# Hamming Weight Lookup Table
# -------------------------------

HW_TABLE = np.array(
    [bin(i).count("1") for i in range(256)],
    dtype=np.uint8,
)


def hypothetical_hw(plaintext_bytes, key_guess):
    """
    Compute hypothetical Hamming Weight leakage.
    """

    sbox_input = np.bitwise_xor(
        plaintext_bytes,
        key_guess,
    )

    sbox_output = AES_SBOX[sbox_input]

    return HW_TABLE[sbox_output]


def log_sum(scores, hw_labels, probabilities):
    """
    Update cumulative log-likelihood for one key guess.
    """

    eps = 1e-12

    scores += np.log(
        probabilities[
            np.arange(len(hw_labels)),
            hw_labels,
        ]
        + eps
    )

    return scores


def compute_key_scores(probabilities, plaintext_bytes):
    """
    Compute log-likelihood score for all 256 key guesses.
    """

    key_scores = np.zeros(256)

    for key_guess in range(256):

        hw = hypothetical_hw(
            plaintext_bytes,
            key_guess,
        )

        key_scores[key_guess] = np.sum(
            np.log(
                probabilities[
                    np.arange(len(hw)),
                    hw,
                ]
                + 1e-12
            )
        )

    return key_scores


def rank_keys(scores):
    """
    Return key guesses sorted from highest score to lowest.
    """

    return np.argsort(scores)[::-1]
