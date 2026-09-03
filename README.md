# Hamming Weight Leakage Analysis

This project uses **Machine Learning to analyze AES power traces** using the **Hamming Weight (HW) leakage model**.

We compare:

* Random Forest
* SVM

The goal is to use power traces to identify the correct AES key byte.

## Project Flow

```text
ASCAD Dataset
      ↓
HW Labels
      ↓
SNR Analysis
      ↓
Feature Selection
      ↓
Random Forest / SVM
      ↓
256 Key Guesses
      ↓
Key Ranking
      ↓
Success Rate
```

## How to Run

Run commands from:

```powershell
C:\HammingWeightLeakageAnalysis
```

Make sure `(venv)` is active.

### 1. Create HW labels

```powershell
python src\save_hw_labels.py
```

Creates:

```text
dataset/hw_labels.npy
```

**Purpose:** Creates Hamming Weight labels from the AES data.

### 2. SNR Analysis

```powershell
python src\snr.py
```

Creates:

```text
results/snr.npy
results/snr_plot.png
```

**Purpose:** Finds useful leakage points in the power traces.

### 3. Select Features

```powershell
python src\feature_selection.py
```

Creates:

```text
results/features.npy
```

**Purpose:** Selects the most useful trace points.

### 4. Train Random Forest

```powershell
python src\train_random_forest.py
```

Creates the Random Forest model.

**Purpose:** Learns the relationship between power traces and Hamming Weight.

### 5. Tune Random Forest

```powershell
python src\tune_random_forest.py
```

**Purpose:** Tests different Random Forest settings and finds the best one.

Best result in this experiment:

```text
300 Trees
Maximum Depth = 20
Accuracy = 27.54%
```

### 6. Train SVM

```powershell
python src\svm_model.py
```

**Purpose:** Trains the SVM model and produces Hamming Weight probabilities.

### 7. SVM Key Ranking

```powershell
python src\svm_key_ranking.py
```

**Purpose:** Tests all 256 possible AES key bytes and ranks them.

### 8. SVM Success Rate

```powershell
python src\svm_success_rate.py
```

**Purpose:** Checks the key rank as more attack traces are used.

### 9. Create Graphs

```powershell
python src\plot_svm_key_ranking.py
python src\plot_svm_success_rate.py
```

## Important Files

```text
dataset/ASCAD.h5
        ↓
dataset/hw_labels.npy
        ↓
results/features.npy
        ↓
models/
        ↓
results/
```

`leakage_model.py` and `attack_utils.py` are **helper files**. Do not run them directly.

## Final Results

### Random Forest

```text
Accuracy = 27.54%
Correct Key = 224
Best Rank = 1
Rank 1 at = 9,800 traces
```

### SVM

```text
Correct Key = 224
Best Rank = 67
Final Rank = 224
Success Rate = 0%
```

### Conclusion

For this experiment, **Random Forest performed better than SVM for AES key ranking**.

The basic idea is:

```text
Power Traces
   ↓
Machine Learning
   ↓
Hamming Weight
   ↓
Test 256 Keys
   ↓
Rank Keys
   ↓
Find Correct Key
```
