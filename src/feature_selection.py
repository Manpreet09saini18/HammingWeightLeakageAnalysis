import numpy as np


def select_features(snr, k=50):

    idx = np.argsort(snr)[-k:]

    idx = np.sort(idx)

    return idx


if __name__ == "__main__":

    snr = np.load("results/snr.npy")

    features = select_features(snr)

    np.save("results/features.npy", features)

    print(features)
