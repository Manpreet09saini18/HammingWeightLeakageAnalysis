import matplotlib.pyplot as plt
import numpy as np

# Guessing Entropy values
ge = np.load("results/guessing_entropy.npy")

# Number of traces
traces = np.arange(
    100,
    100 * (len(ge) + 1),
    100,
)

plt.figure(figsize=(8, 5))

plt.plot(
    traces,
    ge,
    linewidth=2,
)

plt.xlabel("Number of Attack Traces")
plt.ylabel("Guessing Entropy (Key Rank)")
plt.title("Guessing Entropy vs Number of Traces")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/guessing_entropy_curve.png",
    dpi=300,
)

plt.show()
