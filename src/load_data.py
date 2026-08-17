import h5py
import numpy as np

dataset_path = "dataset/ASCAD.h5"

with h5py.File(dataset_path, "r") as f:

    print("Dataset Loaded Successfully!\n")

    # Load Profiling Data
    X_profiling = np.array(f["Profiling_traces"]["traces"])
    Y_profiling = np.array(f["Profiling_traces"]["labels"])

    # Load Attack Data
    X_attack = np.array(f["Attack_traces"]["traces"])
    Y_attack = np.array(f["Attack_traces"]["labels"])

    print("Profiling Traces Shape :", X_profiling.shape)
    print("Profiling Labels Shape :", Y_profiling.shape)

    print("Attack Traces Shape :", X_attack.shape)
    print("Attack Labels Shape :", Y_attack.shape)