import joblib

rf = joblib.load("models/best_random_forest.pkl")

print("="*50)
print("PROJECT INFORMATION")
print("="*50)

print("Dataset : ASCAD")
print("Algorithm : Random Forest")
print("Classification : Hamming Weight")
print("Classes : 9 (0-8)")
print("Trees :", rf.n_estimators)
print("Maximum Depth :", rf.max_depth)
print("Features : 700")
print("="*50)