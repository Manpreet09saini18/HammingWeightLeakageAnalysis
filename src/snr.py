import h5py
import numpy as np
import matplotlib.pyplot as plt


def compute_snr(traces, labels):

    n_samples = traces.shape[1]

    classes = np.unique(labels)

    signal = np.zeros(n_samples)
    noise = np.zeros(n_samples)

    means = []

    for c in classes:
        means.append(
            traces[labels == c].mean(axis=0)
        )

    means = np.array(means)

    signal = np.var(means, axis=0)

    for c in classes:
        noise += traces[labels == c].var(axis=0)

    noise /= len(classes)

    snr = signal / (noise + 1e-12)

    return snr


if __name__ == "__main__":

    # Load profiling traces directly
    with h5py.File("dataset/ASCAD.h5", "r") as f:

        traces = np.array(
            f["Profiling_traces"]["traces"]
        )

        labels = np.array(
            f["Profiling_traces"]["labels"]
        )

    print("Profiling Traces Shape :", traces.shape)
    print("Profiling Labels Shape :", labels.shape)

    # Calculate SNR
    snr = compute_snr(traces, labels)

    # Save SNR values
    np.save("results/snr.npy", snr)

    # Plot SNR
    plt.figure(figsize=(12, 5))

    plt.plot(snr)

    plt.title("Signal-to-Noise Ratio")
    plt.xlabel("Sample Index")
    plt.ylabel("SNR")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "results/snr_plot.png",
        dpi=300
    )

    plt.show()

    print("\nSNR Analysis Completed Successfully!")
    print("Top leakage sample:", np.argmax(snr))
    print("SNR values saved to results/snr.npy")