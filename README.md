*# Hamming Weight Leakage Analysis*



*A classical machine-learning approach for analyzing AES side-channel leakage using Hamming Weight classification and key-ranking techniques.*



*## Overview*



*This project investigates whether machine-learning models can learn information leaked through AES power-consumption traces.*



*The project uses the \*\*ASCAD dataset\*\* and evaluates two classical machine-learning approaches:*



*\* Random Forest*

*\* Support Vector Machine (SVM)*



*The main objective is not only classification accuracy, but also \*\*AES key recovery through key ranking\*\*.*



*## Dataset*



*\* Dataset: ASCAD*

*\* Profiling traces: 50,000*

*\* Attack traces: 10,000*

*\* Target AES byte: 2*

*\* Correct key byte: 224*

*\* Selected features: 50*



*The AES intermediate values are converted into \*\*Hamming Weight classes (0–8)\*\*.*



*## Methodology*



*```text*

*ASCAD Dataset*

&#x20;     *↓*

*Data Preprocessing*

&#x20;     *↓*

*Hamming Weight Labels*

&#x20;     *↓*

*SNR Analysis*

&#x20;     *↓*

*Feature Selection*

&#x20;     *↓*

*Machine Learning Models*

&#x20;  *↙              ↘*

*Random Forest     SVM*

&#x20;  *↓                ↓*

*Predictions      Probabilities*

&#x20;  *↓                ↓*

*AES Key Ranking and Evaluation*

&#x20;     *↓*

*Guessing / Success Rate Analysis*

*```*



*## Random Forest Results*



*The Random Forest model was tuned using different numbers of trees and maximum depths.*



*Best configuration:*



*\* Trees: 300*

*\* Maximum Depth: 20*

*\* Selected Features: 50*

*\* Attack Traces: 10,000*

*\* Hamming Weight Accuracy: \*\*27.54%\*\**



*### Key Ranking*



*\* Correct key: \*\*224\*\**

*\* Best key rank: \*\*1\*\**

*\* Final key rank: \*\*1\*\**

*\* First rank-1 result: \*\*9,800 traces\*\**

*\* Final evaluation: \*\*10,000 traces\*\**



*### Success Rate*



*\* Success rate: \*\*2.00%\*\**

*\* Successful checkpoints: \*\*2\*\**



*The Random Forest model successfully ranked the correct AES key byte first during the attack analysis.*



*## SVM Results*



*The SVM model used an RBF kernel.*



*Configuration:*



*\* Kernel: RBF*

*\* C: 10*

*\* Training samples: 10,000*

*\* Selected features: 50*

*\* Attack traces: 10,000*



*Results:*



*\* Correct key: \*\*224\*\**

*\* Best key rank: \*\*67\*\**

*\* Final key rank: \*\*224\*\**

*\* Success rate: \*\*0.00%\*\**



*The SVM did not recover the correct key within the tested attack traces.*



*## Model Comparison*



*| Metric           | Random Forest |            SVM |*

*| ---------------- | ------------: | -------------: |*

*| Features         |            50 |             50 |*

*| Training Samples |        50,000 |         10,000 |*

*| Attack Traces    |        10,000 |         10,000 |*

*| Accuracy         |        27.54% | Not calculated |*

*| Best Key Rank    |         \*\*1\*\* |             67 |*

*| Final Key Rank   |         \*\*1\*\* |            224 |*

*| Success Rate     |     \*\*2.00%\*\* |          0.00% |*



*## Important Results*



*The strongest result was obtained using the tuned Random Forest model.*



*The correct AES key byte \*\*224\*\* reached \*\*rank 1\*\* at 9,800 attack traces and remained at rank 1 at 10,000 traces.*



*This indicates that, for this experiment, Random Forest provided a stronger key-ranking result than SVM.*



*## Repository Structure*



*```text*

*HammingWeightLeakageAnalysis/*

*│*

*├── src/*

*│   ├── train\_random\_forest.py*

*│   ├── tune\_random\_forest.py*

*│   ├── evaluate\_random\_forest.py*

*│   ├── svm\_model.py*

*│   ├── svm\_key\_ranking.py*

*│   ├── svm\_success\_rate.py*

*│   ├── snr.py*

*│   ├── feature\_selection.py*

*│   ├── guessing\_entropy.py*

*│   └── ...*

*│*

*├── results/*

*│   ├── final\_model\_comparison.txt*

*│   ├── random\_forest\_tuning\_results.csv*

*│   ├── key\_ranks.npy*

*│   ├── success\_rate.npy*

*│   ├── key\_scores\_plot.png*

*│   ├── snr\_plot.png*

*│   ├── feature\_importance.png*

*│   ├── random\_forest\_confusion\_matrix.png*

*│   ├── svm\_key\_ranking\_final.txt*

*│   ├── svm\_success\_rate\_final.txt*

*│   └── svm\_classification\_report.txt*

*│*

*├── README.md*

*├── requirements.txt*

*├── LICENSE*

*└── .gitignore*

*```*



*## How to Run*



*Create and activate a Python virtual environment:*



*```bash*

*python -m venv venv*

*venv\\Scripts\\activate*

*```*



*Install dependencies:*



*```bash*

*pip install -r requirements.txt*

*```*



*Place the ASCAD dataset in the appropriate local dataset directory.*



*Run the required preprocessing, feature-selection, training, and evaluation scripts from the `src/` directory.*



*Example:*



*```bash*

*python src/train\_random\_forest.py*

*python src/svm\_model.py*

*python src/svm\_key\_ranking.py*

*python src/svm\_success\_rate.py*

*```*



*## Notes*



*The dataset and trained model files are not included in this repository because of their large size. They are excluded through `.gitignore`.*



*## Technologies*



*\* Python*

*\* NumPy*

*\* Pandas*

*\* Scikit-learn*

*\* Matplotlib*

*\* Random Forest*

*\* Support Vector Machine*

*\* AES Side-Channel Analysis*

*\* Hamming Weight Leakage Model*

*\* SNR-based Feature Selection*



*## License*



*This project is provided for educational and research purposes.*



