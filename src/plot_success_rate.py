import numpy as np
import matplotlib.pyplot as plt

# Load Success Rate values
sr = np.load("results/success_rate.npy")

# Number of traces
traces = np.arange(100, 100 * len(sr) + 1, 100)

plt.figure(figsize=(8, 5))

plt.plot(
    traces,
    sr,
    marker="o",
)

plt.xlabel("Number of traces")
plt.ylabel("Success Rate")
plt.title("Success Rate Curve")

plt.grid()
plt.tight_layout()

plt.savefig(
    "results/success_rate_curve.png",
    dpi=300,
)

plt.show()
