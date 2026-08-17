import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data


def analyze_traces(num_traces=5):

    data = load_data()

    traces = data["X_train"]

    print("=" * 50)
    print("Trace Statistics")
    print("=" * 50)

    print(f"Number of traces : {traces.shape[0]}")
    print(f"Samples per trace: {traces.shape[1]}")

    print(f"Minimum value : {np.min(traces):.2f}")
    print(f"Maximum value : {np.max(traces):.2f}")
    print(f"Mean          : {np.mean(traces):.2f}")
    print(f"Std Deviation : {np.std(traces):.2f}")

    plt.figure(figsize=(12,5))

    for i in range(num_traces):
        plt.plot(traces[i], label=f"Trace {i}")

    plt.title("Power Traces")
    plt.xlabel("Sample Index")
    plt.ylabel("Power")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    analyze_traces()