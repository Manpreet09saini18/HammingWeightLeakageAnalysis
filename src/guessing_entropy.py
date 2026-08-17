import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("Guessing Entropy Analysis")
print("=" * 50)

# Load key scores
scores = np.load("results/key_scores.npy")

# Rank keys from highest score to lowest
ranking = np.argsort(scores)[::-1]

print("\nTop 10 Key Candidates:\n")

for i in range(10):
    print(f"Rank {i+1}: Key {ranking[i]}  Score = {scores[ranking[i]]}")

# Assume recovered key from previous step
true_key = ranking[0]

# Guessing Entropy
ge = np.where(ranking == true_key)[0][0] + 1

print("\nGuessing Entropy :", ge)

# Plot Scores
plt.figure(figsize=(10,5))
plt.plot(scores)
plt.title("Key Candidate Scores")
plt.xlabel("Key Candidate")
plt.ylabel("Score")
plt.grid(True)

plt.savefig("results/key_scores_plot.png")
plt.show()

print("\nKey score graph saved successfully!")