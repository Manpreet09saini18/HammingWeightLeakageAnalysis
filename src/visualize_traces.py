import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("dataset/ASCAD.h5", "r") as f:
    traces = np.array(f["Profiling_traces"]["traces"])

print("Shape:", traces.shape)

# Plot first 5 traces
for i in range(5):
    plt.plot(traces[i], label=f"Trace {i+1}")

plt.title("First 5 Power Traces")
plt.xlabel("Sample Point")
plt.ylabel("Power")
plt.legend()
plt.show()