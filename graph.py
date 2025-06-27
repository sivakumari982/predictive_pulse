results = {
    "LogReg":    {"accuracy": 0.967, "f1": 0.967},
    "KNN":       {"accuracy": 0.991, "f1": 0.993},
    "Naive Bayes": {"accuracy": 0.923, "f1": 0.934},
    "DT":        {"accuracy": 1.000, "f1": 1.000},
    "RF":        {"accuracy": 1.000, "f1": 1.000},
    "GB":        {"accuracy": 1.000, "f1": 1.000},
    "AdaBoost":  {"accuracy": 1.000, "f1": 1.000},
    "ANN":       {"accuracy": 1.000, "f1": 1.000},
    "SVM":       {"accuracy": 0.989, "f1": 0.991}
}
import matplotlib.pyplot as plt
import pandas as pd

# Convert dictionary to DataFrame
df = pd.DataFrame(results).T  # Transpose to get models as index
df = df.sort_values(by='accuracy', ascending=False)  # optional

# Plot grouped bar chart
ax = df.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'sandybrown'])

# Title and labels
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0.0, 1.05)  # limit y-axis to 0–1.05
plt.xticks(rotation=45)
plt.legend(["Accuracy", "F1 Score"])
plt.tight_layout()
plt.show()
